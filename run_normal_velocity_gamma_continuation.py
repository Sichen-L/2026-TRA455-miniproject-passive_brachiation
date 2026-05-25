"""Part 5: normal-velocity compensation continued from 44 deg down to 0 deg.

The target stride is the most stable passive gait (from the COM sweep).  For
each slope angle gamma the release point receives an endpoint velocity along the
slope normal and the script solves ``P_gamma(d_target; v_n) - d_target = 0`` on
two energy branches.

Importable interface::

    from run_normal_velocity_gamma_continuation import run
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

PART = "normal_velocity_gamma_continuation"
ALGO_VERSION = "v1"

DEFAULTS: dict[str, Any] = {
    "gamma_start": 44.0,
    "gamma_end": 0.0,
    "gamma_step": 1.0,
    "dt": 0.005,
    "t_max": 8.0,
    "max_abs_velocity": 5.0,
    "initial_bracket_width": 0.04,
    "max_bracket_width": 1.5,
    "fallback_scan_points": 401,
    "xtol": 1e-10,
    "rtol": 1e-10,
    "maxiter": 60,
}


@dataclass(frozen=True)
class ContinuationRow:
    branch_name: str
    gamma_deg: float
    v_n: float
    d_target: float
    d_next: float
    residual: float
    spectral_radius: float | None
    stable: bool
    legal: bool
    injected_energy: float
    max_elbow_distance: float | None
    status: str
    failure_reason: str | None = None


def free_endpoint_jacobian(q: np.ndarray, params: BrachiationParameters) -> np.ndarray:
    q1, q2 = np.asarray(q, dtype=float)
    q12 = q1 + q2
    return np.array(
        [
            [params.l1 * np.cos(q1) + params.l2 * np.cos(q12), params.l2 * np.cos(q12)],
            [params.l1 * np.sin(q1) + params.l2 * np.sin(q12), params.l2 * np.sin(q12)],
        ],
        dtype=float,
    )


def slope_away_normal(slope: Slope) -> np.ndarray:
    return -np.array([np.sin(slope.gamma), np.cos(slope.gamma)], dtype=float)


def qdot_for_free_endpoint_velocity(q: np.ndarray, v_ee: np.ndarray, params: BrachiationParameters) -> np.ndarray:
    jacobian = free_endpoint_jacobian(q, params)
    try:
        return np.linalg.solve(jacobian, v_ee)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(jacobian, v_ee, rcond=None)[0]


class GammaProblem:
    def __init__(self, gamma_deg, params, d_target, branch, dt, t_max):
        self.gamma_deg = float(gamma_deg)
        self.params = params
        self.model = TwoLinkBrachiationModel(params)
        self.slope = Slope(gamma=np.deg2rad(gamma_deg))
        self.d_target = float(d_target)
        self.branch = branch
        self.dt = float(dt)
        self.t_max = float(t_max)
        self.support0 = np.zeros(2, dtype=float)
        self.feasibility = make_passive_brachiation_feasibility_check(params, dim=1)

    def release_state(self, d: float, v_n: float) -> BrachiationState:
        q = ik_from_stride_distance(stride_distance=float(d), slope=self.slope, parameters=self.params, direction=-1.0, branch=self.branch)
        v_ee = float(v_n) * slope_away_normal(self.slope)
        qd = qdot_for_free_endpoint_velocity(q, v_ee, self.params)
        return BrachiationState(q=q, qd=qd, support_index=0)

    def evaluate_stride(self, d: float, v_n: float, stop_after_releases: int | None = 1) -> tuple[float, list]:
        samples = simulate(
            model=self.model, slope=self.slope, initial_state=self.release_state(d, v_n),
            initial_support_point=self.support0, duration=self.t_max, dt=self.dt,
            switch_policy=lambda *_args: SwitchDecision.SWITCH, collision_mode=CollisionMode.FULL_GRAB_1D,
            stop_after_releases=stop_after_releases,
        )
        release_samples = [sample for sample in samples if sample.phase.value == "release"]
        if not release_samples:
            raise ValueError("No release event reached.")
        d_next = stride_distance_from_point(release_samples[0].support_point, slope=self.slope, support_point=self.support0, direction=1.0)
        return float(d_next), samples

    def residual(self, v_n: float) -> float:
        d_next, _ = self.evaluate_stride(self.d_target, v_n)
        return d_next - self.d_target

    def stride_map_for_fixed_velocity(self, v_n: float) -> Callable[[np.ndarray], np.ndarray]:
        def p_of_d(x: np.ndarray) -> np.ndarray:
            d = float(np.asarray(x, dtype=float).reshape(-1)[0])
            d_next, _ = self.evaluate_stride(d, v_n)
            return np.array([d_next], dtype=float)
        return p_of_d

    def evaluate_root(self, branch_name: str, v_n: float) -> ContinuationRow:
        d_next, samples = self.evaluate_stride(self.d_target, v_n)
        state = self.release_state(self.d_target, v_n)
        energy = float(self.model.kinetic_energy(state.q, state.qd))
        legality = evaluate_elbow_below_slope_section(samples, self.slope)
        try:
            _jac, _eig, rho = poincare_jacobian_eigenvalues_1d(self.stride_map_for_fixed_velocity(v_n), self.d_target, delta=1e-5, feasibility_check=self.feasibility)
            spectral_radius: float | None = float(rho)
        except Exception:
            spectral_radius = None
        stable = spectral_radius is not None and spectral_radius < 1.0
        return ContinuationRow(
            branch_name=branch_name, gamma_deg=self.gamma_deg, v_n=float(v_n), d_target=self.d_target,
            d_next=float(d_next), residual=float(d_next - self.d_target), spectral_radius=spectral_radius,
            stable=bool(stable), legal=bool(legality.legal), injected_energy=energy,
            max_elbow_distance=float(legality.max_signed_distance), status="ok",
        )


def find_local_bracket(residual, center, initial_half_width, max_half_width, bounds, growth=1.7):
    lo_bound, hi_bound = bounds
    half_width = float(initial_half_width)
    for _ in range(30):
        lo = max(lo_bound, center - half_width)
        hi = min(hi_bound, center + half_width)
        if lo >= hi:
            return None
        try:
            f_lo, f_center, f_hi = residual(lo), residual(center), residual(hi)
        except Exception:
            f_lo = f_center = f_hi = np.nan
        candidates = []
        if np.isfinite(f_lo) and np.isfinite(f_center) and f_lo * f_center <= 0.0:
            candidates.append((lo, center))
        if np.isfinite(f_center) and np.isfinite(f_hi) and f_center * f_hi <= 0.0:
            candidates.append((center, hi))
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0.0:
            candidates.append((lo, hi))
        if candidates:
            candidates.sort(key=lambda item: abs(0.5 * (item[0] + item[1]) - center))
            return candidates[0]
        if half_width >= max_half_width:
            return None
        half_width = min(max_half_width, half_width * growth)
    return None


def find_scan_bracket(residual, center, bounds, points):
    grid = np.linspace(bounds[0], bounds[1], points)
    values = np.full(grid.shape, np.nan, dtype=float)
    for index, value in enumerate(grid):
        try:
            values[index] = residual(float(value))
        except Exception:
            continue
    brackets = []
    for idx in range(len(grid) - 1):
        left, right = values[idx], values[idx + 1]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if left == 0.0 or left * right < 0.0:
            brackets.append((float(grid[idx]), float(grid[idx + 1])))
    if not brackets:
        return None
    brackets.sort(key=lambda item: abs(0.5 * (item[0] + item[1]) - center))
    return brackets[0]


def find_velocity_roots(problem, bounds, points):
    grid = np.linspace(bounds[0], bounds[1], points)
    values = np.full(grid.shape, np.nan, dtype=float)
    for index, value in enumerate(grid):
        try:
            values[index] = problem.residual(float(value))
        except Exception:
            continue
    roots = []
    for idx in range(len(grid) - 1):
        left, right = values[idx], values[idx + 1]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if abs(left) <= 1e-10:
            roots.append(float(grid[idx]))
            continue
        if left * right < 0.0:
            try:
                roots.append(float(brentq(problem.residual, float(grid[idx]), float(grid[idx + 1]), xtol=1e-10, rtol=1e-10, maxiter=60)))
            except Exception:
                continue
    deduped = []
    for root in sorted(roots):
        if all(abs(root - existing) > 1e-6 for existing in deduped):
            deduped.append(root)
    return deduped


def continue_branch(branch_name, seed_v, gamma_values, params, branch, cfg):
    rows: list[ContinuationRow] = []
    previous_v = float(seed_v)
    bounds = (-cfg["max_abs_velocity"], cfg["max_abs_velocity"])
    for gamma_deg in gamma_values:
        problem = GammaProblem(float(gamma_deg), params, cfg["d_target"], branch, cfg["dt"], cfg["t_max"])
        bracket = find_local_bracket(problem.residual, previous_v, cfg["initial_bracket_width"], cfg["max_bracket_width"], bounds)
        if bracket is None:
            bracket = find_scan_bracket(problem.residual, previous_v, bounds, cfg["fallback_scan_points"])
        if bracket is None:
            rows.append(ContinuationRow(branch_name, float(gamma_deg), float(previous_v), cfg["d_target"], np.nan, np.nan, None, False, False, np.nan, None, "failed", "Could not bracket a root."))
            break
        try:
            root_v = brentq(problem.residual, bracket[0], bracket[1], xtol=cfg["xtol"], rtol=cfg["rtol"], maxiter=cfg["maxiter"])
            row = problem.evaluate_root(branch_name, root_v)
        except Exception as exc:
            rows.append(ContinuationRow(branch_name, float(gamma_deg), float(previous_v), cfg["d_target"], np.nan, np.nan, None, False, False, np.nan, None, "failed", str(exc)))
            break
        rows.append(row)
        previous_v = row.v_n
    return rows


def _rows_to_arrays(rows: list[ContinuationRow]) -> dict[str, np.ndarray]:
    return {
        "branch_name": np.array([r.branch_name for r in rows], dtype="U16"),
        "gamma_deg": np.array([r.gamma_deg for r in rows], dtype=float),
        "v_n": np.array([r.v_n for r in rows], dtype=float),
        "d_next": np.array([r.d_next for r in rows], dtype=float),
        "residual": np.array([r.residual for r in rows], dtype=float),
        "spectral_radius": np.array([np.nan if r.spectral_radius is None else r.spectral_radius for r in rows], dtype=float),
        "stable": np.array([r.stable for r in rows], dtype=bool),
        "legal": np.array([r.legal for r in rows], dtype=bool),
        "injected_energy": np.array([r.injected_energy for r in rows], dtype=float),
        "max_elbow_distance": np.array([np.nan if r.max_elbow_distance is None else r.max_elbow_distance for r in rows], dtype=float),
        "status": np.array([r.status for r in rows], dtype="U16"),
    }


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

    seed_problem = GammaProblem(cfg["gamma_start"], params, cfg["d_target"], branch, cfg["dt"], cfg["t_max"])
    seed_roots = find_velocity_roots(seed_problem, (-cfg["max_abs_velocity"], cfg["max_abs_velocity"]), cfg["fallback_scan_points"])
    if len(seed_roots) < 2:
        raise ValueError(f"Expected at least two velocity roots at gamma={cfg['gamma_start']}, found {len(seed_roots)}: {seed_roots}")
    seed_rows = [seed_problem.evaluate_root(f"seed_{i}", root) for i, root in enumerate(seed_roots)]
    seed_rows.sort(key=lambda row: row.injected_energy)
    low_seed = seed_rows[0].v_n
    high_seed = seed_rows[1].v_n

    low_rows = continue_branch("low_energy", low_seed, gamma_values, params, branch, cfg)
    high_rows = continue_branch("high_energy", high_seed, gamma_values, params, branch, cfg)
    rows = low_rows + high_rows

    arrays = _rows_to_arrays(rows)
    ok = arrays["status"] == "ok"
    summary = {
        "gamma_start": cfg["gamma_start"],
        "gamma_end": cfg["gamma_end"],
        "d_target": cfg["d_target"],
        "com_offset": cfg["com_offset"],
        "branch": branch,
        "low_seed": float(low_seed),
        "high_seed": float(high_seed),
        "ok_rows": int(np.count_nonzero(ok)),
        "row_count": len(rows),
        "source_com": cfg.get("source_com", ""),
    }
    return arrays, summary


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    ok = arrays["status"] == "ok"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for branch_name in sorted(set(arrays["branch_name"][ok])):
        sel = ok & (arrays["branch_name"] == branch_name)
        gamma = arrays["gamma_deg"][sel]
        order = np.argsort(gamma)
        gamma = gamma[order]
        axes[0, 0].plot(gamma, arrays["v_n"][sel][order], marker="o", label=branch_name)
        axes[0, 1].plot(gamma, arrays["injected_energy"][sel][order], marker="o", label=branch_name)
        axes[1, 0].plot(gamma, arrays["spectral_radius"][sel][order], marker="o", label=branch_name)
        axes[1, 1].plot(gamma, arrays["max_elbow_distance"][sel][order], marker="o", label=branch_name)

    axes[0, 0].set_ylabel("normal speed v_n [m/s]")
    axes[0, 1].set_ylabel("release kinetic energy [J]")
    axes[1, 0].set_ylabel("spectral radius")
    axes[1, 1].set_ylabel("max elbow signed distance [m]")
    for ax in axes[1, :]:
        ax.set_xlabel("gamma [deg]")
    axes[1, 0].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0.0, color="k", linestyle="--", linewidth=1)
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.invert_xaxis()
        ax.legend()
    fig.suptitle(f"Normal-velocity compensation, target d={summary['d_target']:.4f} m, COM offset {summary['com_offset']:+.4f}")
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
    parser = argparse.ArgumentParser(description="Continue the two normal-velocity roots from gamma_start down to gamma_end.")
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
    print(f"OK rows: {result.summary['ok_rows']}/{result.summary['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
