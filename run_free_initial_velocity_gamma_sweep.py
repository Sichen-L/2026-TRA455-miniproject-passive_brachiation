"""Part 7: minimum-energy free initial velocity for a fixed target stride.

For each slope angle gamma the script scans endpoint initial-velocity
directions that satisfy the non-penetration constraint and solves
``P_gamma(d_target; v_t, v_n) - d_target = 0``, then keeps the minimum-energy
legal (and, when possible, stable) root.  The velocity is parameterized as
``v_ee = speed (cos(theta) tangent + sin(theta) away_normal)`` so theta in
[0, 180] deg enforces ``v_n >= 0``.

``FreeVelocityProblem``, ``scan_gamma`` and ``select_candidate`` are reused by
the part-8 control scripts.

Importable interface::

    from run_free_initial_velocity_gamma_sweep import run
    res = run(); res.figure("main")
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import brentq

from passive_brachiation import (
    BrachiationParameters,
    BrachiationState,
    CollisionMode,
    Slope,
    SwitchDecision,
    TwoLinkBrachiationModel,
    evaluate_elbow_below_slope_section,
    ik_from_stride_distance,
    make_passive_brachiation_feasibility_check,
    parameters_with_symmetric_com_offset,
    poincare_jacobian_eigenvalues_1d,
    samples_to_arrays,
    simulate,
    stride_distance_from_point,
)
from report_cache import PartResult, cached_run
from report_setup import load_best_com_setup

PART = "free_initial_velocity_gamma_sweep"
ALGO_VERSION = "v1"

DEFAULTS: dict[str, Any] = {
    "gamma_start": 44.0,
    "gamma_end": 0.0,
    "gamma_step": 1.0,
    "max_speed": 5.0,
    "speed_points": 41,
    "theta_points": 37,
    "theta_min_deg": 0.0,
    "theta_max_deg": 180.0,
    "min_normal_speed": 0.0,
    "allow_unstable": False,
    "allow_illegal": False,
    "dt": 0.005,
    "t_max": 8.0,
    "xtol": 1e-10,
    "rtol": 1e-10,
    "maxiter": 60,
}


@dataclass(frozen=True)
class VelocityCandidate:
    gamma_deg: float
    theta: float
    theta_deg: float
    speed: float
    v_t: float
    v_n: float
    d_target: float
    d_next: float
    residual: float
    energy: float
    spectral_radius: float | None
    stable: bool
    legal: bool
    max_elbow_distance: float | None
    min_elbow_distance: float | None
    max_free_distance: float | None
    min_free_distance: float | None
    q: list[float]
    qd: list[float]
    bracket_lower: float | None
    bracket_upper: float | None
    status: str = "ok"
    failure_reason: str | None = None


def slope_tangent(slope: Slope) -> np.ndarray:
    return np.array([np.cos(slope.gamma), -np.sin(slope.gamma)], dtype=float)


def slope_away_normal(slope: Slope) -> np.ndarray:
    return -np.array([np.sin(slope.gamma), np.cos(slope.gamma)], dtype=float)


def qdot_for_endpoint_velocity(q: np.ndarray, endpoint_velocity: np.ndarray, model: TwoLinkBrachiationModel) -> np.ndarray:
    jacobian = model.free_endpoint_jacobian(q)
    try:
        return np.linalg.solve(jacobian, endpoint_velocity)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(jacobian, endpoint_velocity, rcond=None)[0]


def endpoint_velocity_from_theta_speed(slope: Slope, theta: float, speed: float) -> tuple[np.ndarray, float, float]:
    tangent = slope_tangent(slope)
    normal = slope_away_normal(slope)
    v_t = float(speed) * float(np.cos(theta))
    v_n = float(speed) * float(np.sin(theta))
    return v_t * tangent + v_n * normal, v_t, v_n


class FreeVelocityProblem:
    def __init__(self, gamma_deg, params, d_target, branch, dt, t_max, min_normal_speed=0.0):
        self.gamma_deg = float(gamma_deg)
        self.params = params
        self.model = TwoLinkBrachiationModel(params)
        self.slope = Slope(gamma=np.deg2rad(gamma_deg))
        self.d_target = float(d_target)
        self.branch = branch
        self.dt = float(dt)
        self.t_max = float(t_max)
        self.min_normal_speed = float(min_normal_speed)
        self.support0 = np.zeros(2, dtype=float)
        self.feasibility = make_passive_brachiation_feasibility_check(params, dim=1)

    def release_q(self, d: float) -> np.ndarray:
        return ik_from_stride_distance(stride_distance=float(d), slope=self.slope, parameters=self.params, direction=-1.0, branch=self.branch)

    def release_state(self, d: float, theta: float, speed: float) -> BrachiationState:
        q = self.release_q(d)
        endpoint_velocity, _v_t, v_n = endpoint_velocity_from_theta_speed(self.slope, theta, speed)
        if v_n < self.min_normal_speed:
            raise ValueError(f"Normal speed {v_n:.6g} is below min_normal_speed={self.min_normal_speed:.6g}.")
        qd = qdot_for_endpoint_velocity(q, endpoint_velocity, self.model)
        return BrachiationState(q=q, qd=qd, support_index=0)

    def evaluate_stride(self, d: float, theta: float, speed: float, stop_after_releases: int | None = 1) -> tuple[float, list]:
        samples = simulate(
            model=self.model, slope=self.slope, initial_state=self.release_state(d, theta, speed),
            initial_support_point=self.support0, duration=self.t_max, dt=self.dt,
            switch_policy=lambda *_args: SwitchDecision.SWITCH, collision_mode=CollisionMode.FULL_GRAB_1D,
            stop_after_releases=stop_after_releases,
        )
        release_samples = [sample for sample in samples if sample.phase.value == "release"]
        if not release_samples:
            raise ValueError("No release event reached.")
        d_next = stride_distance_from_point(release_samples[0].support_point, slope=self.slope, support_point=self.support0, direction=1.0)
        return float(d_next), samples

    def residual(self, theta: float, speed: float) -> float:
        d_next, _ = self.evaluate_stride(self.d_target, theta, speed)
        return d_next - self.d_target

    def stride_map_for_fixed_velocity(self, theta: float, speed: float) -> Callable[[np.ndarray], np.ndarray]:
        def p_of_d(x: np.ndarray) -> np.ndarray:
            d = float(np.asarray(x, dtype=float).reshape(-1)[0])
            d_next, _ = self.evaluate_stride(d, theta, speed)
            return np.array([d_next], dtype=float)
        return p_of_d

    def evaluate_candidate(self, theta: float, speed: float, bracket: tuple[float, float] | None) -> VelocityCandidate:
        d_next, samples = self.evaluate_stride(self.d_target, theta, speed)
        state = self.release_state(self.d_target, theta, speed)
        _endpoint_velocity, v_t, v_n = endpoint_velocity_from_theta_speed(self.slope, theta, speed)
        energy = float(self.model.kinetic_energy(state.q, state.qd))
        legality = evaluate_elbow_below_slope_section(samples, self.slope)
        history = samples_to_arrays(samples, slope=self.slope)
        free_dist = np.asarray(history["free_dist"], dtype=float)
        try:
            _jac, _eig, rho = poincare_jacobian_eigenvalues_1d(self.stride_map_for_fixed_velocity(theta, speed), self.d_target, delta=1e-5, feasibility_check=self.feasibility)
            spectral_radius: float | None = float(rho)
        except Exception:
            spectral_radius = None
        stable = spectral_radius is not None and spectral_radius < 1.0
        return VelocityCandidate(
            gamma_deg=self.gamma_deg, theta=float(theta), theta_deg=float(np.rad2deg(theta)), speed=float(speed),
            v_t=float(v_t), v_n=float(v_n), d_target=self.d_target, d_next=float(d_next), residual=float(d_next - self.d_target),
            energy=energy, spectral_radius=spectral_radius, stable=bool(stable), legal=bool(legality.legal),
            max_elbow_distance=float(legality.max_signed_distance), min_elbow_distance=float(legality.min_signed_distance),
            max_free_distance=float(np.max(free_dist)), min_free_distance=float(np.min(free_dist)),
            q=[float(v) for v in state.q], qd=[float(v) for v in state.qd],
            bracket_lower=None if bracket is None else float(bracket[0]), bracket_upper=None if bracket is None else float(bracket[1]),
        )


def find_speed_roots_for_theta(problem, theta, speed_bounds, speed_points, xtol, rtol, maxiter, zero_tol=1e-10, dedupe_tol=1e-6):
    lo, hi = speed_bounds
    grid = np.linspace(float(lo), float(hi), int(speed_points))
    values = np.full(grid.shape, np.nan, dtype=float)
    for index, speed in enumerate(grid):
        try:
            values[index] = problem.residual(theta, float(speed))
        except Exception:
            continue
    roots: list[tuple[float, tuple[float, float] | None]] = []
    for index, value in enumerate(values):
        if np.isfinite(value) and abs(value) <= zero_tol:
            roots.append((float(grid[index]), None))
    for index in range(len(grid) - 1):
        left, right = values[index], values[index + 1]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if left == 0.0 or right == 0.0:
            continue
        if left * right < 0.0:
            bracket = (float(grid[index]), float(grid[index + 1]))
            try:
                root = float(brentq(lambda speed: problem.residual(theta, speed), bracket[0], bracket[1], xtol=xtol, rtol=rtol, maxiter=maxiter))
            except Exception:
                continue
            roots.append((root, bracket))
    deduped: list[tuple[float, tuple[float, float] | None]] = []
    for root, bracket in sorted(roots, key=lambda item: item[0]):
        if all(abs(root - existing[0]) > dedupe_tol for existing in deduped):
            deduped.append((root, bracket))
    candidates: list[VelocityCandidate] = []
    for root, bracket in deduped:
        try:
            candidates.append(problem.evaluate_candidate(theta, root, bracket))
        except Exception:
            continue
    return candidates


def scan_gamma(problem, theta_values, speed_bounds, speed_points, xtol, rtol, maxiter) -> list[VelocityCandidate]:
    candidates: list[VelocityCandidate] = []
    for theta in theta_values:
        candidates.extend(find_speed_roots_for_theta(problem, float(theta), speed_bounds, speed_points, xtol, rtol, maxiter))
    candidates.sort(key=lambda row: (row.energy, row.theta, row.speed))
    return candidates


def select_candidate(candidates, allow_unstable, allow_illegal) -> VelocityCandidate | None:
    pool = candidates
    if not allow_illegal:
        pool = [row for row in pool if row.legal]
    if not allow_unstable:
        stable_pool = [row for row in pool if row.stable]
        if stable_pool:
            pool = stable_pool
    if not pool:
        return None
    return min(pool, key=lambda row: row.energy)


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base = BrachiationParameters.rod_point_mass(
        m1=cfg["m1"], m2=cfg["m2"], l1=cfg["l1"], l2=cfg["l2"],
        rod_mass_fraction=0.2, damping1=cfg["damping1"], damping2=cfg["damping2"], gravity=cfg["gravity"],
    )
    params = parameters_with_symmetric_com_offset(cfg["com_offset"], base)
    branch = str(cfg["branch"])

    if cfg["gamma_step"] <= 0.0:
        raise ValueError("gamma_step must be positive.")
    direction = -1.0 if cfg["gamma_end"] < cfg["gamma_start"] else 1.0
    gamma_values = np.arange(cfg["gamma_start"], cfg["gamma_end"] + direction * 0.5 * cfg["gamma_step"], direction * cfg["gamma_step"], dtype=float)
    gamma_values[-1] = cfg["gamma_end"]
    theta_values = np.linspace(np.deg2rad(cfg["theta_min_deg"]), np.deg2rad(cfg["theta_max_deg"]), cfg["theta_points"])

    rows: list[dict[str, Any]] = []
    for gamma_deg in gamma_values:
        problem = FreeVelocityProblem(float(gamma_deg), params, cfg["d_target"], branch, cfg["dt"], cfg["t_max"], cfg["min_normal_speed"])
        candidates = scan_gamma(problem, theta_values, (0.0, cfg["max_speed"]), cfg["speed_points"], cfg["xtol"], cfg["rtol"], cfg["maxiter"])
        selected = select_candidate(candidates, cfg["allow_unstable"], cfg["allow_illegal"])
        legal = [c for c in candidates if c.legal]
        rows.append(
            {
                "gamma_deg": float(gamma_deg),
                "selected": selected is not None,
                "energy": np.nan if selected is None else selected.energy,
                "speed": np.nan if selected is None else selected.speed,
                "theta_deg": np.nan if selected is None else selected.theta_deg,
                "v_t": np.nan if selected is None else selected.v_t,
                "v_n": np.nan if selected is None else selected.v_n,
                "spectral_radius": np.nan if (selected is None or selected.spectral_radius is None) else selected.spectral_radius,
                "num_candidates": len(candidates),
                "num_legal": len(legal),
                "num_stable_legal": len([c for c in legal if c.stable]),
            }
        )

    arrays = {
        "gamma_deg": np.array([r["gamma_deg"] for r in rows], dtype=float),
        "selected": np.array([r["selected"] for r in rows], dtype=bool),
        "energy": np.array([r["energy"] for r in rows], dtype=float),
        "speed": np.array([r["speed"] for r in rows], dtype=float),
        "theta_deg": np.array([r["theta_deg"] for r in rows], dtype=float),
        "v_t": np.array([r["v_t"] for r in rows], dtype=float),
        "v_n": np.array([r["v_n"] for r in rows], dtype=float),
        "spectral_radius": np.array([r["spectral_radius"] for r in rows], dtype=float),
        "num_candidates": np.array([r["num_candidates"] for r in rows], dtype=int),
        "num_legal": np.array([r["num_legal"] for r in rows], dtype=int),
        "num_stable_legal": np.array([r["num_stable_legal"] for r in rows], dtype=int),
    }
    summary = {
        "gamma_start": cfg["gamma_start"],
        "gamma_end": cfg["gamma_end"],
        "d_target": cfg["d_target"],
        "com_offset": cfg["com_offset"],
        "branch": branch,
        "selected_count": int(np.count_nonzero(arrays["selected"])),
        "gamma_count": len(rows),
        "source_com": cfg.get("source_com", ""),
    }
    return arrays, summary


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    sel = arrays["selected"]
    gamma = arrays["gamma_deg"][sel]
    order = np.argsort(gamma)
    gamma = gamma[order]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes[0, 0].plot(gamma, arrays["energy"][sel][order], marker="o")
    axes[0, 1].plot(gamma, arrays["speed"][sel][order], marker="o")
    axes[0, 2].plot(gamma, arrays["theta_deg"][sel][order], marker="o")
    axes[1, 0].plot(gamma, arrays["v_t"][sel][order], marker="o", label="v_t")
    axes[1, 0].plot(gamma, arrays["v_n"][sel][order], marker="o", label="v_n")
    axes[1, 1].plot(gamma, arrays["spectral_radius"][sel][order], marker="o")
    axes[1, 2].plot(gamma, arrays["num_candidates"][sel][order], marker="o")

    axes[0, 0].set_ylabel("minimum initial energy [J]")
    axes[0, 1].set_ylabel("endpoint speed [m/s]")
    axes[0, 2].set_ylabel("theta [deg]")
    axes[1, 0].set_ylabel("velocity components [m/s]")
    axes[1, 1].set_ylabel("spectral radius")
    axes[1, 2].set_ylabel("candidate count")
    for ax in axes[1, :]:
        ax.set_xlabel("gamma [deg]")
    axes[1, 0].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1, 1].axhline(1.0, color="k", linestyle="--", linewidth=1)
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.invert_xaxis()
    axes[1, 0].legend()
    fig.suptitle(f"Minimum-energy free initial velocity, target d={summary['d_target']:.4f} m, COM offset {summary['com_offset']:+.4f}")
    fig.tight_layout()
    return {"main": fig}


def run(params=None, *, force=False, results_dir: Path | str = Path("results"), verbose=True) -> PartResult:
    cfg = {**DEFAULTS, **(params or {})}
    setup = load_best_com_setup(results_dir=results_dir, branch_override=cfg.get("branch"))
    bp = setup.base_params
    cfg.update(
        {
            "_algo": ALGO_VERSION,
            "com_offset": setup.com_offset,
            "d_target": setup.d_target,
            "branch": setup.branch,
            "m1": bp.m1, "m2": bp.m2, "l1": bp.l1, "l2": bp.l2,
            "damping1": bp.damping1, "damping2": bp.damping2, "gravity": bp.gravity,
            "source_com": setup.source,
        }
    )
    return cached_run(part=PART, config=cfg, compute=_compute, plot=_plot, results_dir=results_dir, force=force, verbose=verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan non-penetrating free initial velocities and select minimum-energy roots.")
    parser.add_argument("--gamma-start", type=float, default=DEFAULTS["gamma_start"])
    parser.add_argument("--gamma-end", type=float, default=DEFAULTS["gamma_end"])
    parser.add_argument("--gamma-step", type=float, default=DEFAULTS["gamma_step"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--t-max", type=float, default=DEFAULTS["t_max"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    params = {"gamma_start": args.gamma_start, "gamma_end": args.gamma_end, "gamma_step": args.gamma_step, "dt": args.dt, "t_max": args.t_max}
    result = run(params, force=args.force, results_dir=args.results_dir)
    print(f"Data: {result.data_path}")
    print(f"Figure: {result.figure('main')}")
    print(f"Selected gammas: {result.summary['selected_count']}/{result.summary['gamma_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
