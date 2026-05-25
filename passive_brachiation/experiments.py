"""Reusable experiment workflows for passive-brachiation studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .analysis import (
    release_indices,
    release_q2_values,
    release_stride_distances,
    samples_to_arrays,
    tail_half_range,
    tail_std,
)
from .kinematics import Slope
from .model import BrachiationParameters, BrachiationState, TwoLinkBrachiationModel
from .policies import (
    always_switch_policy,
    zero_endpoint_force_policy,
    zero_generalized_force_policy,
    zero_torque_policy,
)
from .shooting import (
    ElbowSectionLegality,
    FixedPoint1DResult,
    ResidualScanResult,
    StrideMap,
    evaluate_elbow_below_slope_section,
    find_fixed_points_1d,
    ik_from_q2_on_slope,
    ik_from_stride_distance,
    make_iterated_stride_map,
    make_passive_brachiation_feasibility_check,
    make_passive_brachiation_stride_map,
    poincare_jacobian_eigenvalues_1d,
)
from .simulation import SimulationSample, simulate
from .switching import CollisionMode, SwitchDecision


SwitchPolicy = Callable[[float, BrachiationState, np.ndarray, np.ndarray, Slope], SwitchDecision]


@dataclass(frozen=True)
class PassiveSetup:
    """Default physical setup used by the notebook experiments."""

    params: BrachiationParameters
    model: TwoLinkBrachiationModel
    slope: Slope
    initial_support: np.ndarray


@dataclass(frozen=True)
class StrideFixedPointTrial:
    """One validated stride fixed-point candidate."""

    root: FixedPoint1DResult
    d: float
    branch: str
    period: int
    q0: np.ndarray
    validation_error: float
    validation_std: float
    validation_release_q2: np.ndarray
    validation_release_stride: np.ndarray
    legality: ElbowSectionLegality
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    spectral_radius: float
    stable: bool
    P: StrideMap
    P_base: StrideMap
    samples: list[SimulationSample]


@dataclass(frozen=True)
class StrideFixedPointSearchResult:
    """Result of the notebook-style stride fixed-point search."""

    scan_results: dict[tuple[str, int], dict[str, object]]
    trials: list[StrideFixedPointTrial]
    legal_trials: list[StrideFixedPointTrial]
    selected_trial: StrideFixedPointTrial | None


@dataclass(frozen=True)
class FixedGaitFreeRun:
    """Free-run simulation initialized from a selected fixed gait."""

    initial_state: BrachiationState
    samples: list[SimulationSample]
    history: dict[str, np.ndarray]
    release_stride: np.ndarray
    release_q2: np.ndarray


def make_default_uniform_setup(
    gamma_degrees: float = 45.0,
    m1: float = 1.041,
    m2: float = 1.041,
    l1: float = 0.314,
    l2: float = 0.314,
    damping1: float = 0.0,
    damping2: float = 0.0,
    gravity: float = 9.81,
    initial_support: np.ndarray | None = None,
) -> PassiveSetup:
    """Create the notebook's default uniform-link setup."""

    params = BrachiationParameters.uniform_links(
        m1=m1,
        m2=m2,
        l1=l1,
        l2=l2,
        damping1=damping1,
        damping2=damping2,
        gravity=gravity,
    )
    return PassiveSetup(
        params=params,
        model=TwoLinkBrachiationModel(params),
        slope=Slope(gamma=np.deg2rad(gamma_degrees)),
        initial_support=np.zeros(2, dtype=float)
        if initial_support is None
        else np.asarray(initial_support, dtype=float),
    )


def make_default_rod_point_setup(
    gamma_degrees: float = 45.0,
    m1: float = 1.041,
    m2: float = 1.041,
    l1: float = 0.314,
    l2: float = 0.314,
    weight_position1: float = 0.5,
    weight_position2: float = 0.5,
    rod_mass_fraction: float = 0.2,
    damping1: float = 0.0,
    damping2: float = 0.0,
    gravity: float = 9.81,
    initial_support: np.ndarray | None = None,
) -> PassiveSetup:
    """Create the project's rod + movable point-mass default setup."""

    params = BrachiationParameters.rod_point_mass(
        weight_position1=weight_position1,
        weight_position2=weight_position2,
        m1=m1,
        m2=m2,
        l1=l1,
        l2=l2,
        rod_mass_fraction=rod_mass_fraction,
        damping1=damping1,
        damping2=damping2,
        gravity=gravity,
    )
    return PassiveSetup(
        params=params,
        model=TwoLinkBrachiationModel(params),
        slope=Slope(gamma=np.deg2rad(gamma_degrees)),
        initial_support=np.zeros(2, dtype=float)
        if initial_support is None
        else np.asarray(initial_support, dtype=float),
    )


def run_passive_simulation(
    model: TwoLinkBrachiationModel,
    slope: Slope,
    initial_state: BrachiationState,
    initial_support_point: np.ndarray | None = None,
    duration: float = 5.0,
    dt: float = 0.005,
    switch_policy: SwitchPolicy | None = None,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
) -> FixedGaitFreeRun:
    """Run a passive simulation using zero-force policies."""

    support = (
        np.zeros(2, dtype=float)
        if initial_support_point is None
        else np.asarray(initial_support_point, dtype=float)
    )
    samples = simulate(
        model=model,
        slope=slope,
        initial_state=initial_state,
        initial_support_point=support,
        duration=duration,
        dt=dt,
        torque_policy=zero_torque_policy,
        endpoint_force_policy=zero_endpoint_force_policy,
        generalized_force_policy=zero_generalized_force_policy,
        switch_policy=switch_policy or always_switch_policy,
        collision_mode=collision_mode,
    )
    return FixedGaitFreeRun(
        initial_state=initial_state,
        samples=samples,
        history=samples_to_arrays(samples, slope=slope),
        release_stride=release_stride_distances(samples, slope=slope),
        release_q2=release_q2_values(samples),
    )


def scan_stride_fixed_points(
    model: TwoLinkBrachiationModel,
    slope: Slope,
    initial_support_point: np.ndarray | None = None,
    dt: float = 0.005,
    t_max: float = 8.0,
    d_bounds: tuple[float, float] | None = None,
    d_scan_points: int = 30,
    branches: tuple[str, ...] = ("positive", "negative"),
    periods: tuple[int, ...] = (1, 2),
    initial_direction: float = -1.0,
    impact_direction: float = 1.0,
    switch_policy: SwitchPolicy | None = None,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
    residual_tol: float = 1e-7,
    root_xtol: float = 1e-10,
    root_rtol: float = 1e-10,
    root_maxiter: int = 60,
    jacobian_delta: float = 1e-5,
    validation_duration: float | None = None,
    validation_tail_count: int = 8,
    legality_tolerance: float = 1e-9,
    period_duplicate_tol: float = 1e-6,
    raise_on_empty: bool = True,
) -> StrideFixedPointSearchResult:
    """Run the notebook's stride-distance residual scan and validation."""

    support = (
        np.zeros(2, dtype=float)
        if initial_support_point is None
        else np.asarray(initial_support_point, dtype=float)
    )
    bounds = d_bounds or (0.05, 0.95 * (model.p.l1 + model.p.l2))
    d_feasible = make_passive_brachiation_feasibility_check(model.p, dim=1)
    scan_results: dict[tuple[str, int], dict[str, object]] = {}
    trials: list[StrideFixedPointTrial] = []
    policy = switch_policy or always_switch_policy
    validate_duration = t_max if validation_duration is None else validation_duration

    for branch in branches:
        P_base = make_passive_brachiation_stride_map(
            model=model,
            slope=slope,
            dt=dt,
            t_max=t_max,
            collision_mode=collision_mode,
            initial_direction=initial_direction,
            impact_direction=impact_direction,
            branch=branch,
            support_point=support,
            switch_policy=policy,
        )

        for period in periods:
            P_search = P_base if period == 1 else make_iterated_stride_map(P_base, period)
            roots, scan = find_fixed_points_1d(
                P_search,
                bounds=bounds,
                num_scan_points=d_scan_points,
                method="brentq",
                feasibility_check=d_feasible,
                residual_tol=residual_tol,
                xtol=root_xtol,
                rtol=root_rtol,
                maxiter=root_maxiter,
            )
            if period > 1:
                roots = [
                    root
                    for root in roots
                    if abs(float(P_base(np.array([root.x], dtype=float))[0]) - root.x)
                    > period_duplicate_tol
                ]

            scan_results[(branch, period)] = {
                "P": P_search,
                "P_base": P_base,
                "scan": scan,
                "roots": roots,
            }
            trials.extend(
                _validate_stride_roots(
                    roots=roots,
                    branch=branch,
                    period=period,
                    P_search=P_search,
                    P_base=P_base,
                    model=model,
                    slope=slope,
                    support=support,
                    dt=dt,
                    validation_duration=validate_duration,
                    initial_direction=initial_direction,
                    impact_direction=impact_direction,
                    switch_policy=policy,
                    collision_mode=collision_mode,
                    jacobian_delta=jacobian_delta,
                    d_feasible=d_feasible,
                    validation_tail_count=validation_tail_count,
                    legality_tolerance=legality_tolerance,
                )
            )

    legal_trials = [trial for trial in trials if trial.legality.legal]
    legal_trials.sort(
        key=lambda trial: (
            trial.spectral_radius,
            trial.period,
            trial.validation_error,
            trial.root.residual_norm,
        )
    )
    selected = legal_trials[0] if legal_trials else None
    if selected is None and raise_on_empty:
        raise RuntimeError("No legal period-1/period-2 stride fixed point found.")

    return StrideFixedPointSearchResult(
        scan_results=scan_results,
        trials=trials,
        legal_trials=legal_trials,
        selected_trial=selected,
    )


def _validate_stride_roots(
    roots: list[FixedPoint1DResult],
    branch: str,
    period: int,
    P_search: StrideMap,
    P_base: StrideMap,
    model: TwoLinkBrachiationModel,
    slope: Slope,
    support: np.ndarray,
    dt: float,
    validation_duration: float,
    initial_direction: float,
    impact_direction: float,
    switch_policy: SwitchPolicy,
    collision_mode: CollisionMode | str,
    jacobian_delta: float,
    d_feasible: Callable[[np.ndarray], bool],
    validation_tail_count: int,
    legality_tolerance: float,
) -> list[StrideFixedPointTrial]:
    trials: list[StrideFixedPointTrial] = []

    for root in roots:
        d_candidate = float(root.x)
        if not d_feasible(np.array([d_candidate], dtype=float)):
            continue

        q0 = ik_from_stride_distance(
            d_candidate,
            slope=slope,
            parameters=model.p,
            direction=initial_direction,
            branch=branch,
        )
        samples = simulate(
            model=model,
            slope=slope,
            initial_state=BrachiationState(q=q0, qd=np.zeros(2), support_index=0),
            initial_support_point=support,
            duration=validation_duration,
            dt=dt,
            switch_policy=switch_policy,
            collision_mode=collision_mode,
        )
        rel_indices = release_indices(samples)
        rel_stride = release_stride_distances(
            samples,
            slope=slope,
            support_origin=support,
            direction=impact_direction,
        )
        if len(rel_stride) < max(5, period) or len(rel_indices) < period:
            continue

        period_samples = samples[: rel_indices[period - 1] + 1]
        legality = evaluate_elbow_below_slope_section(
            period_samples,
            slope=slope,
            tolerance=legality_tolerance,
        )
        jacobian, eigenvalues, spectral_radius = poincare_jacobian_eigenvalues_1d(
            P_search,
            d_candidate,
            delta=jacobian_delta,
            feasibility_check=d_feasible,
        )

        trials.append(
            StrideFixedPointTrial(
                root=root,
                d=d_candidate,
                branch=branch,
                period=period,
                q0=q0,
                validation_error=tail_half_range(rel_stride, validation_tail_count),
                validation_std=tail_std(rel_stride, validation_tail_count),
                validation_release_q2=release_q2_values(samples),
                validation_release_stride=rel_stride,
                legality=legality,
                jacobian=jacobian,
                eigenvalues=eigenvalues,
                spectral_radius=spectral_radius,
                stable=spectral_radius < 1.0,
                P=P_search,
                P_base=P_base,
                samples=samples,
            )
        )

    return trials


def run_fixed_gait_free_run(
    model: TwoLinkBrachiationModel,
    slope: Slope,
    q2_fixed: float | None = None,
    d_fixed: float | None = None,
    branch: str = "positive",
    initial_direction: float = -1.0,
    initial_support_point: np.ndarray | None = None,
    duration: float = 8.0,
    dt: float = 0.005,
    switch_policy: SwitchPolicy | None = None,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
) -> FixedGaitFreeRun:
    """Run a longer free simulation from a selected fixed-gait coordinate."""

    if q2_fixed is None and d_fixed is None:
        raise ValueError("Either q2_fixed or d_fixed must be provided.")

    if q2_fixed is not None:
        q0 = ik_from_q2_on_slope(
            q2_fixed,
            slope=slope,
            parameters=model.p,
            direction=initial_direction,
        )
    else:
        q0 = ik_from_stride_distance(
            float(d_fixed),
            slope=slope,
            parameters=model.p,
            direction=initial_direction,
            branch=branch,
        )

    return run_passive_simulation(
        model=model,
        slope=slope,
        initial_state=BrachiationState(q=q0, qd=np.zeros(2, dtype=float), support_index=0),
        initial_support_point=initial_support_point,
        duration=duration,
        dt=dt,
        switch_policy=switch_policy,
        collision_mode=collision_mode,
    )


def selected_stride_summary(trial: StrideFixedPointTrial) -> dict[str, object]:
    """Return notebook-friendly scalar fields for one selected trial."""

    return {
        "branch": trial.branch,
        "period": trial.period,
        "d_fixed": trial.d,
        "q2_fixed": float(trial.q0[1]),
        "residual_norm": trial.root.residual_norm,
        "validation_error": trial.validation_error,
        "validation_std": trial.validation_std,
        "elbow_legal": trial.legality.legal,
        "max_elbow_distance": trial.legality.max_signed_distance,
        "jacobian": trial.jacobian,
        "eigenvalues": trial.eigenvalues,
        "spectral_radius": trial.spectral_radius,
        "stable": trial.stable,
    }
