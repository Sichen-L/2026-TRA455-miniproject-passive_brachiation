"""Generic shooting utilities and passive-brachiation stride maps.

The top-level Newton and grid-search routines are model agnostic: they only
need a stride map ``P(x)``.  The passive-brachiation factory below builds one
such map from the current two-link model, slope geometry, and collision model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np

from .kinematics import Slope, forward_kinematics
from .model import BrachiationParameters, BrachiationState, TwoLinkBrachiationModel
from .simulation import SimulationSample, simulate
from .switching import CollisionMode, CollisionModel, SwitchDecision


StrideMap = Callable[[np.ndarray], np.ndarray]
FeasibilityCheck = Callable[[np.ndarray], bool]
ParameterizedStrideMapFactory = Callable[[float], StrideMap]
ParameterizedFeasibilityFactory = Callable[[float], FeasibilityCheck | None]


@dataclass(frozen=True)
class ShootingResult:
    """Result of Newton shooting."""

    x: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    history: list[np.ndarray]
    residual_history: list[float]


@dataclass(frozen=True)
class GridSearchResult:
    """One candidate found by grid-search initialization."""

    x: np.ndarray
    residual_norm: float
    p_of_x: np.ndarray


@dataclass(frozen=True)
class FixedPointBracket:
    """One residual sign-change interval for a scalar fixed-point problem."""

    lower: float
    upper: float
    residual_lower: float
    residual_upper: float


@dataclass(frozen=True)
class ResidualScanResult:
    """Grid residuals and sign-change brackets for a 1-D fixed-point map."""

    grid: np.ndarray
    residuals: np.ndarray
    brackets: list[FixedPointBracket]
    exact_roots: list[float]


@dataclass(frozen=True)
class FixedPoint1DResult:
    """Result from a scalar fixed-point root solver."""

    x: float
    p_of_x: float
    residual: float
    residual_norm: float
    method: str
    converged: bool
    bracket: tuple[float, float] | None = None
    iterations: int | None = None
    function_calls: int | None = None


@dataclass(frozen=True)
class StrideMapEvaluation:
    """Optional diagnostic record for one passive-brachiation stride."""

    x: np.ndarray
    p_of_x: np.ndarray
    samples: list[SimulationSample]


@dataclass(frozen=True)
class ElbowSectionLegality:
    """Whether the elbow stays below the slope section over a trajectory."""

    legal: bool
    max_signed_distance: float
    min_signed_distance: float
    violation_count: int


@dataclass(frozen=True)
class ContinuationPoint:
    """One point on a parameterized fixed-point branch."""

    parameter: float
    x: np.ndarray
    p_of_x: np.ndarray | None
    residual: np.ndarray | None
    residual_norm: float
    converged: bool
    iterations: int
    spectral_radius: float | None
    eigenvalues: np.ndarray | None
    jacobian: np.ndarray | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class ContinuationResult:
    """A warm-started continuation trace over a scalar parameter."""

    points: list[ContinuationPoint]
    stopped_early: bool
    stop_reason: str | None = None


def _as_vector(x: Iterable[float], dim: int | None = None) -> np.ndarray:
    vector = np.asarray(x, dtype=float).reshape(-1)
    if dim is not None and vector.size != dim:
        raise ValueError(f"Expected vector dimension {dim}, got {vector.size}.")
    return vector


def finite_difference_jacobian(
    P: StrideMap,
    x: Iterable[float],
    dim: int | None = None,
    delta: float = 1e-6,
    feasibility_check: FeasibilityCheck | None = None,
    p_at_x: np.ndarray | None = None,
) -> np.ndarray:
    """Return a forward finite-difference Jacobian for ``P`` at ``x``."""
    x = _as_vector(x, dim)
    dim = x.size
    p0 = _as_vector(P(x), dim) if p_at_x is None else _as_vector(p_at_x, dim)
    J = np.zeros((dim, dim), dtype=float)

    for i in range(dim):
        x_step = x.copy()
        x_step[i] += delta
        if feasibility_check is not None and not feasibility_check(x_step):
            x_step[i] = x[i] - delta
            if not feasibility_check(x_step):
                raise ValueError(f"Finite-difference step {i} is infeasible.")
            J[:, i] = (p0 - _as_vector(P(x_step), dim)) / delta
        else:
            J[:, i] = (_as_vector(P(x_step), dim) - p0) / delta

    return J


def newton_shoot(
    P: StrideMap,
    x0: Iterable[float],
    dim: int | None = None,
    tol: float = 1e-6,
    max_iter: int = 30,
    delta: float = 1e-6,
    feasibility_check: FeasibilityCheck | None = None,
    damping: float = 1.0,
) -> ShootingResult:
    """Find a fixed point of ``P`` with Newton shooting.

    The residual is ``F(x) = P(x) - x``.  The Newton system is
    ``(J_P - I) step = F`` and the update is ``x <- x - step``.
    """
    x = _as_vector(x0, dim)
    dim = x.size
    if feasibility_check is not None and not feasibility_check(x):
        raise ValueError("Initial guess is infeasible.")

    history: list[np.ndarray] = [x.copy()]
    residual_history: list[float] = []

    for iteration in range(max_iter + 1):
        p = _as_vector(P(x), dim)
        residual = p - x
        residual_norm = float(np.linalg.norm(residual))
        residual_history.append(residual_norm)

        if residual_norm < tol:
            return ShootingResult(
                x=x,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                history=history,
                residual_history=residual_history,
            )
        if iteration == max_iter:
            break

        Jp = finite_difference_jacobian(
            P,
            x,
            dim=dim,
            delta=delta,
            feasibility_check=feasibility_check,
            p_at_x=p,
        )
        A = Jp - np.eye(dim)
        try:
            step = np.linalg.solve(A, residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(A, residual, rcond=None)[0]

        scale = float(damping)
        while scale > 1e-4:
            candidate = x - scale * step
            if feasibility_check is None or feasibility_check(candidate):
                x = candidate
                break
            scale *= 0.5
        else:
            raise ValueError("Newton step could not find a feasible candidate.")

        history.append(x.copy())

    return ShootingResult(
        x=x,
        converged=False,
        iterations=max_iter,
        residual_norm=residual_history[-1],
        history=history,
        residual_history=residual_history,
    )


def continue_fixed_point_branch(
    P_factory: ParameterizedStrideMapFactory,
    parameters: Sequence[float],
    x0: Iterable[float],
    dim: int | None = None,
    feasibility_factory: ParameterizedFeasibilityFactory | None = None,
    tol: float = 1e-7,
    max_iter: int = 5,
    delta: float = 1e-6,
    damping: float = 1.0,
    compute_stability: bool = True,
    stop_on_failure: bool = True,
) -> ContinuationResult:
    """Warm-start Newton continuation over a scalar parameter.

    ``P_factory(parameter)`` builds the stride map at that parameter value.
    The fixed point found at one parameter is used as the initial guess for the
    next.  This is intended for cheap branch tracing, e.g. sweeping slope
    angle once a nearby fixed point is known.
    """
    params = [float(value) for value in parameters]
    if not params:
        raise ValueError("parameters must not be empty.")

    x_current = _as_vector(x0, dim)
    dim = x_current.size
    points: list[ContinuationPoint] = []

    for parameter in params:
        P = P_factory(parameter)
        feasibility_check = (
            None
            if feasibility_factory is None
            else feasibility_factory(parameter)
        )
        try:
            result = newton_shoot(
                P,
                x_current,
                dim=dim,
                tol=tol,
                max_iter=max_iter,
                delta=delta,
                feasibility_check=feasibility_check,
                damping=damping,
            )
            p_of_x = _as_vector(P(result.x), dim)
            residual = p_of_x - result.x
            residual_norm = float(np.linalg.norm(residual))

            jacobian: np.ndarray | None = None
            eigenvalues: np.ndarray | None = None
            spectral_radius: float | None = None
            if compute_stability and result.converged:
                jacobian = finite_difference_jacobian(
                    P,
                    result.x,
                    dim=dim,
                    delta=delta,
                    feasibility_check=feasibility_check,
                    p_at_x=p_of_x,
                )
                eigenvalues = np.linalg.eigvals(jacobian)
                spectral_radius = float(np.max(np.abs(eigenvalues)))

            points.append(
                ContinuationPoint(
                    parameter=parameter,
                    x=result.x.copy(),
                    p_of_x=p_of_x,
                    residual=residual,
                    residual_norm=residual_norm,
                    converged=result.converged and residual_norm <= tol,
                    iterations=result.iterations,
                    spectral_radius=spectral_radius,
                    eigenvalues=eigenvalues,
                    jacobian=jacobian,
                )
            )
            if not result.converged or residual_norm > tol:
                reason = (
                    f"Newton did not converge at parameter={parameter:.9g}; "
                    f"residual={residual_norm:.3e}"
                )
                if stop_on_failure:
                    return ContinuationResult(
                        points=points,
                        stopped_early=True,
                        stop_reason=reason,
                    )
            else:
                x_current = result.x.copy()

        except Exception as exc:
            points.append(
                ContinuationPoint(
                    parameter=parameter,
                    x=x_current.copy(),
                    p_of_x=None,
                    residual=None,
                    residual_norm=float("inf"),
                    converged=False,
                    iterations=0,
                    spectral_radius=None,
                    eigenvalues=None,
                    jacobian=None,
                    failure_reason=str(exc),
                )
            )
            if stop_on_failure:
                return ContinuationResult(
                    points=points,
                    stopped_early=True,
                    stop_reason=f"{type(exc).__name__} at parameter={parameter:.9g}: {exc}",
                )

    return ContinuationResult(points=points, stopped_early=False)


def grid_search_initial_guesses(
    P: StrideMap,
    bounds: Sequence[tuple[float, float]],
    grid_sizes: Sequence[int] | int,
    top_k: int = 10,
    feasibility_check: FeasibilityCheck | None = None,
) -> list[GridSearchResult]:
    """Evaluate ``P`` on a rectangular grid and return the best candidates."""
    dim = len(bounds)
    if isinstance(grid_sizes, int):
        sizes = [grid_sizes] * dim
    else:
        sizes = list(grid_sizes)
    if len(sizes) != dim:
        raise ValueError("grid_sizes must be an int or match bounds dimension.")

    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, sizes)]
    results: list[GridSearchResult] = []

    for values in product(*axes):
        x = np.asarray(values, dtype=float)
        if feasibility_check is not None and not feasibility_check(x):
            continue
        try:
            p = _as_vector(P(x), dim)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(p)):
            continue
        residual_norm = float(np.linalg.norm(p - x))
        results.append(GridSearchResult(x=x, residual_norm=residual_norm, p_of_x=p))

    results.sort(key=lambda item: item.residual_norm)
    return results[:top_k]


def _call_scalar_stride_map(P: StrideMap, value: float) -> float:
    """Evaluate a scalar stride map, accepting vector- or scalar-style callables."""
    try:
        result = P(np.array([float(value)], dtype=float))
    except Exception:
        result = P(float(value))  # type: ignore[arg-type]
    return float(_as_vector(result)[0])


def scalar_fixed_point_residual(P: StrideMap, value: float) -> float:
    """Return ``F(x) = P(x) - x`` for a scalar fixed-point map."""
    return _call_scalar_stride_map(P, value) - float(value)


def make_iterated_stride_map(P: StrideMap, period: int) -> StrideMap:
    """Return the iterated map ``P`` applied ``period`` times."""
    if period < 1:
        raise ValueError("period must be at least 1.")

    def P_iterated(x: np.ndarray) -> np.ndarray:
        value = _as_vector(x)
        for _ in range(period):
            value = _as_vector(P(value), value.size)
        return value

    return P_iterated


def residual_sign_change_scan(
    P: StrideMap,
    bounds: tuple[float, float],
    num_points: int = 30,
    feasibility_check: FeasibilityCheck | None = None,
    zero_tol: float = 0.0,
) -> ResidualScanResult:
    """Scan a scalar fixed-point residual and collect sign-change brackets.

    This is the cheap global pass before applying a local 1-D root solver such
    as Brent's method.  Invalid or infeasible grid points are recorded as
    ``nan`` and skipped when forming brackets.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    lo, hi = bounds
    if not lo < hi:
        raise ValueError("bounds must satisfy lower < upper.")

    grid = np.linspace(float(lo), float(hi), int(num_points))
    residuals = np.full(grid.shape, np.nan, dtype=float)
    exact_roots: list[float] = []

    for index, value in enumerate(grid):
        x = np.array([value], dtype=float)
        if feasibility_check is not None and not feasibility_check(x):
            continue
        try:
            residual = scalar_fixed_point_residual(P, float(value))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if np.isfinite(residual):
            residuals[index] = residual
            if abs(residual) <= zero_tol:
                exact_roots.append(float(value))

    brackets: list[FixedPointBracket] = []
    for index in range(len(grid) - 1):
        f_left = residuals[index]
        f_right = residuals[index + 1]
        if not (np.isfinite(f_left) and np.isfinite(f_right)):
            continue
        if f_left == 0.0 or f_right == 0.0:
            continue
        if f_left * f_right < 0.0:
            brackets.append(
                FixedPointBracket(
                    lower=float(grid[index]),
                    upper=float(grid[index + 1]),
                    residual_lower=float(f_left),
                    residual_upper=float(f_right),
                )
            )

    return ResidualScanResult(
        grid=grid,
        residuals=residuals,
        brackets=brackets,
        exact_roots=exact_roots,
    )


def solve_fixed_point_bracket_1d(
    P: StrideMap,
    bracket: FixedPointBracket | tuple[float, float],
    method: str = "brentq",
    xtol: float = 1e-10,
    rtol: float = 1e-10,
    maxiter: int = 50,
) -> FixedPoint1DResult:
    """Solve one bracketed scalar fixed-point equation ``P(x) - x = 0``."""
    if isinstance(bracket, FixedPointBracket):
        lo, hi = bracket.lower, bracket.upper
    else:
        lo, hi = float(bracket[0]), float(bracket[1])

    if method != "brentq":
        raise ValueError("Only method='brentq' is currently implemented.")

    from scipy.optimize import brentq

    root, info = brentq(
        lambda value: scalar_fixed_point_residual(P, value),
        lo,
        hi,
        xtol=xtol,
        rtol=rtol,
        maxiter=maxiter,
        full_output=True,
        disp=False,
    )
    p_of_x = _call_scalar_stride_map(P, root)
    residual = p_of_x - root
    return FixedPoint1DResult(
        x=float(root),
        p_of_x=float(p_of_x),
        residual=float(residual),
        residual_norm=abs(float(residual)),
        method=method,
        converged=bool(info.converged),
        bracket=(float(lo), float(hi)),
        iterations=int(info.iterations),
        function_calls=int(info.function_calls),
    )


def find_fixed_points_1d(
    P: StrideMap,
    bounds: tuple[float, float],
    num_scan_points: int = 30,
    method: str = "brentq",
    feasibility_check: FeasibilityCheck | None = None,
    zero_tol: float = 0.0,
    dedupe_tol: float = 1e-7,
    residual_tol: float | None = 1e-7,
    xtol: float = 1e-10,
    rtol: float = 1e-10,
    maxiter: int = 50,
) -> tuple[list[FixedPoint1DResult], ResidualScanResult]:
    """Find scalar fixed points by residual sign-change scan plus local solve."""
    scan = residual_sign_change_scan(
        P,
        bounds=bounds,
        num_points=num_scan_points,
        feasibility_check=feasibility_check,
        zero_tol=zero_tol,
    )

    results: list[FixedPoint1DResult] = []

    for root in scan.exact_roots:
        p_of_x = _call_scalar_stride_map(P, root)
        residual = p_of_x - root
        results.append(
            FixedPoint1DResult(
                x=float(root),
                p_of_x=float(p_of_x),
                residual=float(residual),
                residual_norm=abs(float(residual)),
                method="scan_exact",
                converged=True,
            )
        )

    for bracket in scan.brackets:
        try:
            result = solve_fixed_point_bracket_1d(
                P,
                bracket,
                method=method,
                xtol=xtol,
                rtol=rtol,
                maxiter=maxiter,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        residual_ok = residual_tol is None or result.residual_norm <= residual_tol
        if result.converged and residual_ok:
            results.append(result)

    deduped: list[FixedPoint1DResult] = []
    for result in sorted(results, key=lambda item: item.x):
        if all(abs(result.x - existing.x) > dedupe_tol for existing in deduped):
            deduped.append(result)

    return deduped, scan


def poincare_jacobian_eigenvalues_1d(
    P: StrideMap,
    x: float,
    delta: float = 1e-5,
    feasibility_check: FeasibilityCheck | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return Jacobian, eigenvalues, and spectral radius for a scalar map."""
    point = np.array([float(x)], dtype=float)
    jacobian = finite_difference_jacobian(
        P,
        point,
        dim=1,
        delta=delta,
        feasibility_check=feasibility_check,
    )
    eigenvalues = np.linalg.eigvals(jacobian)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    return jacobian, eigenvalues, spectral_radius


def evaluate_elbow_below_slope_section(
    samples: Sequence[SimulationSample],
    slope: Slope,
    tolerance: float = 1e-9,
) -> ElbowSectionLegality:
    """Check that every elbow sample remains below the slope section.

    The slope convention treats negative signed distance as the robot-side free
    space below the ceiling-like section.  A sample is therefore legal when
    ``signed_distance(elbow) <= tolerance``.
    """
    if not samples:
        raise ValueError("samples must not be empty.")

    distances = np.array([slope.signed_distance(sample.elbow) for sample in samples])
    violations = distances > tolerance
    return ElbowSectionLegality(
        legal=not bool(np.any(violations)),
        max_signed_distance=float(np.max(distances)),
        min_signed_distance=float(np.min(distances)),
        violation_count=int(np.count_nonzero(violations)),
    )


def slope_direction(slope: Slope, direction: float = 1.0) -> np.ndarray:
    """Return the unit tangent direction used for stride distances."""
    sign = 1.0 if direction >= 0.0 else -1.0
    return sign * np.array([np.cos(slope.gamma), -np.sin(slope.gamma)], dtype=float)


def stride_point_from_distance(
    stride_distance: float,
    slope: Slope,
    support_point: Iterable[float] | None = None,
    direction: float = 1.0,
) -> np.ndarray:
    """Return a point on the slope at arclength ``stride_distance``."""
    support = (
        np.zeros(2, dtype=float)
        if support_point is None
        else np.asarray(support_point, dtype=float)
    )
    return support + float(stride_distance) * slope_direction(slope, direction)


def stride_distance_from_point(
    point: Iterable[float],
    slope: Slope,
    support_point: Iterable[float] | None = None,
    direction: float = 1.0,
) -> float:
    """Project a point displacement onto the selected slope tangent."""
    support = (
        np.zeros(2, dtype=float)
        if support_point is None
        else np.asarray(support_point, dtype=float)
    )
    return float((np.asarray(point, dtype=float) - support) @ slope_direction(slope, direction))


def ik_from_stride_distance(
    stride_distance: float,
    slope: Slope,
    parameters: BrachiationParameters,
    direction: float = 1.0,
    branch: str = "positive",
) -> np.ndarray:
    """Return ``[q1, q2]`` whose free endpoint lies at the stride point.

    ``branch="positive"`` mirrors the IK branch used by
    ``passive-brachiator-main/stride_function.py``.  ``branch="negative"``
    selects the other planar two-link solution.
    """
    d = float(stride_distance)
    if d <= 0.0:
        raise ValueError("stride_distance must be positive.")

    l1 = parameters.l1
    l2 = parameters.l2
    cos_q2 = (d**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    if not -1.0 <= cos_q2 <= 1.0:
        raise ValueError("stride_distance is outside the two-link workspace.")

    target = stride_point_from_distance(d, slope, direction=direction)
    target_angle = float(np.arctan2(target[0], -target[1]))
    alpha = float(np.arccos(np.clip((l1**2 + d**2 - l2**2) / (2.0 * l1 * d), -1.0, 1.0)))
    q2_mag = float(np.arccos(np.clip(cos_q2, -1.0, 1.0)))

    if branch == "positive":
        q1 = target_angle - alpha
        q2 = q2_mag
    elif branch == "negative":
        q1 = target_angle + alpha
        q2 = -q2_mag
    else:
        raise ValueError("branch must be 'positive' or 'negative'.")

    return np.array([q1, q2], dtype=float)


def stride_distance_from_q2(
    q2: float,
    parameters: BrachiationParameters,
) -> float:
    """Return endpoint separation implied by a relative angle ``q2``."""
    l1 = parameters.l1
    l2 = parameters.l2
    distance_sq = l1**2 + l2**2 + 2.0 * l1 * l2 * np.cos(float(q2))
    if distance_sq <= 0.0:
        raise ValueError("q2 gives a degenerate endpoint separation.")
    return float(np.sqrt(distance_sq))


def ik_from_q2_on_slope(
    q2: float,
    slope: Slope,
    parameters: BrachiationParameters,
    direction: float = -1.0,
) -> np.ndarray:
    """Return ``[q1, q2]`` with the free endpoint on the slope.

    The relative angle ``q2`` fixes the endpoint separation.  ``direction``
    selects which side of the current support the free endpoint lies on.
    This is the natural release-section parameterization for full-grab
    two-link brachiation: after switching, the old support/free end is on the
    upslope side, so the default caller should use ``direction=-1``.
    """
    q2 = float(q2)
    d = stride_distance_from_q2(q2, parameters)
    l1 = parameters.l1
    l2 = parameters.l2

    target = stride_point_from_distance(d, slope, direction=direction)
    target_angle = float(np.arctan2(target[0], -target[1]))
    alpha = float(np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2)))
    q1 = target_angle - alpha
    return np.array([q1, q2], dtype=float)


def q2_from_release_state(state: BrachiationState) -> float:
    """Return the release-section q2 coordinate."""
    return float(state.q[1])


def validate_release_section_geometry(
    q: Iterable[float],
    slope: Slope,
    parameters: BrachiationParameters,
    support_point: Iterable[float] | None = None,
    tolerance: float = 1e-6,
) -> None:
    """Raise if a release-section IK pose intersects the slope.

    Both the free endpoint and elbow must lie on or below the ceiling-like
    slope.  Catching this before simulation prevents physically impossible IK
    branches from becoming numerical fixed-point candidates.
    """
    support = (
        np.zeros(2, dtype=float)
        if support_point is None
        else np.asarray(support_point, dtype=float)
    )
    points = forward_kinematics(q, support, parameters.l1, parameters.l2)
    free_distance = slope.signed_distance(points.free)
    elbow_distance = slope.signed_distance(points.elbow)
    if free_distance > tolerance:
        raise ValueError(
            "Release free end penetrates slope: "
            f"signed_dist = {free_distance:.3e} m"
        )
    if elbow_distance > tolerance:
        raise ValueError(
            "Release elbow penetrates slope: "
            f"signed_dist = {elbow_distance:.3e} m"
        )


def make_passive_brachiation_feasibility_check(
    parameters: BrachiationParameters,
    min_stride: float = 1e-6,
    max_stride_margin: float = 1e-6,
    dim: int | None = None,
) -> FeasibilityCheck:
    """Return a feasibility check for stride-map section coordinates."""
    max_stride = parameters.l1 + parameters.l2 - max_stride_margin

    def check(x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=float).reshape(-1)
        if dim is not None and x.size != dim:
            return False
        if x.size not in (1, 3):
            return False
        if not np.all(np.isfinite(x)):
            return False
        return min_stride < x[0] < max_stride

    return check


def make_q2_feasibility_check(
    min_abs_sin: float = 1e-4,
    bounds: tuple[float, float] = (-np.pi + 1e-4, np.pi - 1e-4),
) -> FeasibilityCheck:
    """Return a basic feasibility check for q2 shooting."""

    def check(x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != 1 or not np.all(np.isfinite(x)):
            return False
        q2 = float(x[0])
        return bounds[0] < q2 < bounds[1] and abs(np.sin(q2)) > min_abs_sin

    return check


def make_passive_brachiation_q2_stride_map(
    model: TwoLinkBrachiationModel,
    slope: Slope,
    dt: float = 0.005,
    t_max: float = 10.0,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
    collision_model: CollisionModel | None = None,
    initial_direction: float = -1.0,
    switch_policy: Callable[[float, BrachiationState, np.ndarray, np.ndarray, Slope], SwitchDecision] | None = None,
    return_mode: str = "post_switch",
) -> StrideMap:
    """Return a 1-D stride map whose section coordinate is release ``q2``.

    The input ``x = [q2_release]`` defines a zero-velocity release state with
    the free end on the upslope side of the support.  The simulation advances
    to the next impact and switch.

    ``return_mode='post_switch'`` returns the next release ``q2`` after
    coordinate switching.  ``return_mode='preimpact_reflected'`` returns
    ``-q2_preimpact``; these are equivalent for a pure support relabeling.
    """
    policy = switch_policy or (lambda *_args: SwitchDecision.SWITCH)

    def P(x: np.ndarray) -> np.ndarray:
        x = _as_vector(x, dim=1)
        q = ik_from_q2_on_slope(
            q2=x[0],
            slope=slope,
            parameters=model.p,
            direction=initial_direction,
        )
        validate_release_section_geometry(q, slope, model.p)
        state0 = BrachiationState(q=q, qd=np.zeros(2, dtype=float), support_index=0)
        samples = simulate(
            model=model,
            slope=slope,
            initial_state=state0,
            initial_support_point=np.zeros(2, dtype=float),
            duration=t_max,
            dt=dt,
            switch_policy=policy,
            collision_mode=collision_mode,
            collision_model=collision_model,
        )
        release_samples = [sample for sample in samples if sample.phase.value == "release"]
        if not release_samples:
            raise ValueError("q2 stride map did not reach a release event.")

        if return_mode == "post_switch":
            q2_next = q2_from_release_state(release_samples[0].state)
        elif return_mode == "preimpact_reflected":
            impact_samples = [sample for sample in samples if sample.phase.value == "impact"]
            if not impact_samples:
                raise ValueError("q2 stride map did not reach an impact event.")
            q2_next = -float(impact_samples[0].state.q[1])
        else:
            raise ValueError("return_mode must be 'post_switch' or 'preimpact_reflected'.")

        return np.array([q2_next], dtype=float)

    return P


def evaluate_passive_brachiation_q2_stride(
    model: TwoLinkBrachiationModel,
    slope: Slope,
    q2: float,
    dt: float = 0.005,
    t_max: float = 10.0,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
    collision_model: CollisionModel | None = None,
    initial_direction: float = -1.0,
) -> StrideMapEvaluation:
    """Evaluate one q2-parameterized stride and keep the trajectory."""
    q = ik_from_q2_on_slope(q2, slope, model.p, direction=initial_direction)
    validate_release_section_geometry(q, slope, model.p)
    state0 = BrachiationState(q=q, qd=np.zeros(2, dtype=float), support_index=0)
    samples = simulate(
        model=model,
        slope=slope,
        initial_state=state0,
        initial_support_point=np.zeros(2, dtype=float),
        duration=t_max,
        dt=dt,
        switch_policy=lambda *_args: SwitchDecision.SWITCH,
        collision_mode=collision_mode,
        collision_model=collision_model,
    )
    release_samples = [sample for sample in samples if sample.phase.value == "release"]
    if not release_samples:
        raise ValueError("q2 stride evaluation did not reach a release event.")
    p_of_x = np.array([q2_from_release_state(release_samples[0].state)], dtype=float)
    return StrideMapEvaluation(x=np.array([q2], dtype=float), p_of_x=p_of_x, samples=samples)


def make_passive_brachiation_stride_map(
    model: TwoLinkBrachiationModel,
    slope: Slope,
    dt: float = 0.005,
    t_max: float = 10.0,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
    collision_model: CollisionModel | None = None,
    initial_direction: float = -1.0,
    impact_direction: float = 1.0,
    branch: str = "positive",
    support_point: Iterable[float] | None = None,
    switch_policy: Callable[[float, BrachiationState, np.ndarray, np.ndarray, Slope], SwitchDecision] | None = None,
    direction: float | None = None,
) -> StrideMap:
    """Return a passive-brachiation stride map ``P``.

    Section coordinates:

    - ``dim=1``: ``x = [stride_distance]`` with zero initial velocity.
    - ``dim=3``: ``x = [stride_distance, q1_dot, q2_dot]``.

    The output has the same dimension as the input.  The default section is a
    release-to-impact map: the old support/free end starts on the upslope side
    of the current support (``initial_direction=-1``), and the next grabbed
    support is measured downslope (``impact_direction=+1``).

    ``direction`` is kept only for old callers.  Passing it sets both
    directions and therefore reproduces the earlier target-to-target map.
    """
    if direction is not None:
        initial_direction = direction
        impact_direction = direction

    base_support = (
        np.zeros(2, dtype=float)
        if support_point is None
        else np.asarray(support_point, dtype=float)
    )
    policy = switch_policy or (lambda *_args: SwitchDecision.SWITCH)

    def P(x: np.ndarray) -> np.ndarray:
        x = _as_vector(x)
        if x.size not in (1, 3):
            raise ValueError("Passive brachiation stride map supports dim=1 or dim=3.")

        q = ik_from_stride_distance(
            stride_distance=x[0],
            slope=slope,
            parameters=model.p,
            direction=initial_direction,
            branch=branch,
        )
        validate_release_section_geometry(q, slope, model.p, support_point=base_support)
        qd = np.zeros(2, dtype=float) if x.size == 1 else x[1:3].copy()
        initial_state = BrachiationState(q=q, qd=qd, support_index=0)

        samples = simulate(
            model=model,
            slope=slope,
            initial_state=initial_state,
            initial_support_point=base_support,
            duration=t_max,
            dt=dt,
            switch_policy=policy,
            collision_mode=collision_mode,
            collision_model=collision_model,
        )
        release_samples = [sample for sample in samples if sample.phase.value == "release"]
        if not release_samples:
            raise ValueError("Stride map did not reach a release event.")

        release = release_samples[0]
        d_next = stride_distance_from_point(
            release.support_point,
            slope=slope,
            support_point=base_support,
            direction=impact_direction,
        )
        if x.size == 1:
            return np.array([d_next], dtype=float)
        return np.array([d_next, release.state.qd[0], release.state.qd[1]], dtype=float)

    return P


def evaluate_passive_brachiation_stride(
    model: TwoLinkBrachiationModel,
    slope: Slope,
    x: Iterable[float],
    dt: float = 0.005,
    t_max: float = 10.0,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
    collision_model: CollisionModel | None = None,
    initial_direction: float = -1.0,
    impact_direction: float = 1.0,
    branch: str = "positive",
    support_point: Iterable[float] | None = None,
    switch_policy: Callable[
        [float, BrachiationState, np.ndarray, np.ndarray, Slope],
        SwitchDecision,
    ]
    | None = None,
    direction: float | None = None,
) -> StrideMapEvaluation:
    """Evaluate one passive-brachiation stride and keep the trajectory."""
    if direction is not None:
        initial_direction = direction
        impact_direction = direction

    x = _as_vector(x)
    base_support = (
        np.zeros(2, dtype=float)
        if support_point is None
        else np.asarray(support_point, dtype=float)
    )
    q = ik_from_stride_distance(
        x[0],
        slope,
        model.p,
        direction=initial_direction,
        branch=branch,
    )
    validate_release_section_geometry(q, slope, model.p, support_point=base_support)
    qd = np.zeros(2, dtype=float) if x.size == 1 else x[1:3].copy()
    initial_state = BrachiationState(q=q, qd=qd, support_index=0)
    samples = simulate(
        model=model,
        slope=slope,
        initial_state=initial_state,
        initial_support_point=base_support,
        duration=t_max,
        dt=dt,
        switch_policy=switch_policy or (lambda *_args: SwitchDecision.SWITCH),
        collision_mode=collision_mode,
        collision_model=collision_model,
    )
    release_samples = [sample for sample in samples if sample.phase.value == "release"]
    if not release_samples:
        raise ValueError("Stride evaluation did not reach a release event.")
    release = release_samples[0]
    d_next = stride_distance_from_point(
        release.support_point,
        slope=slope,
        support_point=base_support,
        direction=impact_direction,
    )
    p_of_x = np.array([d_next], dtype=float) if x.size == 1 else np.array(
        [d_next, release.state.qd[0], release.state.qd[1]],
        dtype=float,
    )
    return StrideMapEvaluation(x=x, p_of_x=p_of_x, samples=samples)
