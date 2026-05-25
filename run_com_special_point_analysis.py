"""Part 3b: COM special-point (fold) analysis (compute + plot + cache).

Emits two figures:

* ``fine_scan`` - fine COM scan near the suspected special point (eigenvalue,
  spectral radius, fixed stride vs symmetric COM offset).
* ``fold_arms`` - global multi-root scan resolving the target gait's stable and
  unstable branches against unrelated roots.

Importable interface::

    from run_com_special_point_analysis import run
    res = run()
    res.figure("fine_scan"); res.figure("fold_arms")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from passive_brachiation import (
    BrachiationParameters,
    CollisionMode,
    Slope,
    SwitchDecision,
    TwoLinkBrachiationModel,
    continue_fixed_point_branch,
    find_fixed_points_1d,
    make_passive_brachiation_feasibility_check,
    make_passive_brachiation_stride_map,
    parameters_with_symmetric_com_offset,
    poincare_jacobian_eigenvalues_1d,
    scan_stride_fixed_points,
)
from report_cache import PartResult, cached_run

PART = "com_special_point"
ALGO_VERSION = "v2"

DEFAULTS: dict[str, Any] = {
    "gamma_deg": 45.0,
    "dt": 0.005,
    "t_max": 8.0,
    "d_scan_points": 30,
    "d_lower": 0.05,
    "d_upper_fraction": 0.995,
    "local_start": -0.19,
    "local_end": -0.205,
    "local_step": -0.00025,
    "fold_tol": 7.5e-2,
    "arm_offset_start": -0.16,
    "arm_offset_end": -0.205,
    "arm_offset_step": -0.0025,
    "arm_d_scan_points": 180,
    "root_residual_tol": 1e-6,
    "degenerate_rho": 50.0,
    "jacobian_delta": 1e-5,
    "target_branch_half_width": 0.045,
    # physical model
    "m1": 1.041,
    "m2": 1.041,
    "l1": 0.314,
    "l2": 0.314,
    "rod_mass_fraction": 0.2,
    "weight_position1": 0.5,
    "weight_position2": 0.5,
    "damping1": 0.0,
    "damping2": 0.0,
    "gravity": 9.81,
}


def _base_params(cfg: dict[str, Any]) -> BrachiationParameters:
    return BrachiationParameters.rod_point_mass(
        weight_position1=cfg["weight_position1"],
        weight_position2=cfg["weight_position2"],
        m1=cfg["m1"],
        m2=cfg["m2"],
        l1=cfg["l1"],
        l2=cfg["l2"],
        rod_mass_fraction=cfg["rod_mass_fraction"],
        damping1=cfg["damping1"],
        damping2=cfg["damping2"],
        gravity=cfg["gravity"],
    )


def _switch_policy(_t, _state, _support_point, _impact_point, _slope):
    return SwitchDecision.SWITCH


def _nonconverged_runs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, row in enumerate(rows):
        if not row["converged"] and start is None:
            start = idx
        if start is not None and (row["converged"] or idx == len(rows) - 1):
            end = idx - 1 if row["converged"] else idx
            runs.append((start, end))
            start = None
    return runs


def _skipped_nonconverged_between(rows: list[dict[str, Any]], left_offset: float, right_offset: float) -> int:
    left_idx = next(i for i, row in enumerate(rows) if row["offset"] == left_offset)
    right_idx = next(i for i, row in enumerate(rows) if row["offset"] == right_offset)
    lo, hi = sorted((left_idx, right_idx))
    return sum(not row["converged"] for row in rows[lo + 1 : hi])


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    params = _base_params(cfg)
    model = TwoLinkBrachiationModel(params)
    slope = Slope(gamma=np.deg2rad(cfg["gamma_deg"]))
    support = np.zeros(2, dtype=float)
    initial_direction = -1.0
    impact_direction = 1.0

    stride_search = scan_stride_fixed_points(
        model=model,
        slope=slope,
        initial_support_point=support,
        dt=cfg["dt"],
        t_max=cfg["t_max"],
        d_bounds=(cfg["d_lower"], cfg["d_upper_fraction"] * (params.l1 + params.l2)),
        d_scan_points=cfg["d_scan_points"],
        branches=("positive", "negative"),
        periods=(1, 2),
        initial_direction=initial_direction,
        impact_direction=impact_direction,
        switch_policy=_switch_policy,
        collision_mode=CollisionMode.FULL_GRAB_1D,
    )
    selected = stride_search.selected_trial
    if selected is None:
        raise RuntimeError("No legal stride fixed point found.")
    d_fixed = float(selected.d)
    selected_branch = selected.branch
    selected_period = int(selected.period)

    def model_with_com_offset(com_offset: float) -> TwoLinkBrachiationModel:
        return TwoLinkBrachiationModel(parameters_with_symmetric_com_offset(com_offset, params))

    def make_stride_map_for_offset(com_offset: float):
        return make_passive_brachiation_stride_map(
            model=model_with_com_offset(com_offset),
            slope=slope,
            dt=cfg["dt"],
            t_max=cfg["t_max"],
            collision_mode=CollisionMode.FULL_GRAB_1D,
            initial_direction=initial_direction,
            impact_direction=impact_direction,
            branch=selected_branch,
            support_point=support,
            switch_policy=_switch_policy,
        )

    def make_feasibility(_parameter: float):
        return make_passive_brachiation_feasibility_check(params, dim=1)

    approach_offsets = np.linspace(0.0, cfg["local_start"], 41)
    approach_result = continue_fixed_point_branch(
        P_factory=make_stride_map_for_offset,
        parameters=approach_offsets,
        x0=np.array([d_fixed], dtype=float),
        dim=1,
        feasibility_factory=make_feasibility,
        tol=1e-6,
        max_iter=10,
        delta=cfg["jacobian_delta"],
        damping=0.8,
        compute_stability=True,
        stop_on_failure=True,
    )
    seed_point = next(point for point in reversed(approach_result.points) if point.converged)

    local_offsets = np.arange(cfg["local_start"], cfg["local_end"] + 0.5 * cfg["local_step"], cfg["local_step"])
    local_result = continue_fixed_point_branch(
        P_factory=make_stride_map_for_offset,
        parameters=local_offsets,
        x0=seed_point.x,
        dim=1,
        feasibility_factory=make_feasibility,
        tol=1e-7,
        max_iter=12,
        delta=cfg["jacobian_delta"],
        damping=0.8,
        compute_stability=True,
        stop_on_failure=False,
    )

    local_rows: list[dict[str, Any]] = []
    for point in local_result.points:
        eig_real = np.nan if point.eigenvalues is None else float(np.real(point.eigenvalues[0]))
        fold_indicator = np.nan if point.fold_indicator is None else float(point.fold_indicator)
        local_rows.append(
            {
                "offset": float(point.parameter),
                "d": float(point.x[0]),
                "converged": point.converged,
                "eig_real": eig_real,
                "rho": np.nan if point.spectral_radius is None else float(point.spectral_radius),
                "fold_indicator": fold_indicator,
                "failure_reason": point.failure_reason,
            }
        )
    valid_rows = [row for row in local_rows if row["converged"]]
    if not valid_rows:
        raise RuntimeError("Local scan found no converged points.")

    fine_offset = np.array([row["offset"] for row in valid_rows], dtype=float)
    fine_eig_real = np.array([row["eig_real"] for row in valid_rows], dtype=float)
    fine_rho = np.array([row["rho"] for row in valid_rows], dtype=float)
    fine_fold_indicator = np.array([row["fold_indicator"] for row in valid_rows], dtype=float)
    fine_d = np.array([row["d"] for row in valid_rows], dtype=float)

    summary: dict[str, Any] = {
        "gamma_deg": cfg["gamma_deg"],
        "d_fixed": d_fixed,
        "branch": selected_branch,
        "period": selected_period,
        "local_points": len(local_rows),
        "local_converged": len(valid_rows),
    }
    if len(fine_offset) >= 2:
        eig_jump = np.abs(np.diff(fine_eig_real))
        eig_jump_idx = int(np.nanargmax(eig_jump))
        d_jump = np.abs(np.diff(fine_d))
        d_jump_idx = int(np.nanargmax(d_jump))
        skipped = _skipped_nonconverged_between(local_rows, fine_offset[d_jump_idx], fine_offset[d_jump_idx + 1])
        summary.update(
            {
                "largest_eig_jump_left": float(fine_offset[eig_jump_idx]),
                "largest_eig_jump_right": float(fine_offset[eig_jump_idx + 1]),
                "largest_eig_jump": float(eig_jump[eig_jump_idx]),
                "largest_d_jump_left": float(fine_offset[d_jump_idx]),
                "largest_d_jump_right": float(fine_offset[d_jump_idx + 1]),
                "largest_d_jump": float(d_jump[d_jump_idx]),
                "largest_d_jump_skipped_nonconverged": int(skipped),
            }
        )

    fail_runs = _nonconverged_runs(local_rows)
    summary["nonconverged_gaps"] = [
        {
            "start_offset": float(local_rows[start]["offset"]),
            "end_offset": float(local_rows[end]["offset"]),
            "points": int(end - start + 1),
            "first_reason": local_rows[start]["failure_reason"],
        }
        for start, end in fail_runs
    ]

    arm_offsets = np.arange(
        cfg["arm_offset_start"], cfg["arm_offset_end"] + 0.5 * cfg["arm_offset_step"], cfg["arm_offset_step"]
    )
    arm_records: list[dict[str, Any]] = []
    for offset in arm_offsets:
        p_map = make_stride_map_for_offset(float(offset))
        feasibility = make_feasibility(float(offset))
        roots, _ = find_fixed_points_1d(
            p_map,
            bounds=(cfg["d_lower"], cfg["d_upper_fraction"] * (params.l1 + params.l2)),
            num_scan_points=cfg["arm_d_scan_points"],
            feasibility_check=feasibility,
            residual_tol=cfg["root_residual_tol"],
        )
        for root in roots:
            try:
                _, _, root_rho = poincare_jacobian_eigenvalues_1d(
                    p_map, root.x, delta=cfg["jacobian_delta"], feasibility_check=feasibility
                )
            except Exception:
                root_rho = np.nan
            arm_records.append(
                {
                    "offset": float(offset),
                    "d": float(root.x),
                    "rho": float(root_rho),
                    "stable": bool(np.isfinite(root_rho) and root_rho < 1.0),
                    "degenerate": bool(np.isfinite(root_rho) and root_rho >= cfg["degenerate_rho"]),
                }
            )

    arm_offset = np.array([row["offset"] for row in arm_records], dtype=float)
    arm_d = np.array([row["d"] for row in arm_records], dtype=float)
    arm_rho = np.array([row["rho"] for row in arm_records], dtype=float)
    arm_stable = np.array([row["stable"] for row in arm_records], dtype=bool)
    arm_degenerate = np.array([row["degenerate"] for row in arm_records], dtype=bool)

    physical = [row for row in arm_records if not row["degenerate"]]
    target_branch_center = float(np.nanmedian(fine_d))
    target_half_width = float(cfg["target_branch_half_width"])
    target_family = [row for row in physical if abs(row["d"] - target_branch_center) <= target_half_width]
    target_stable = [row for row in target_family if row["stable"]]
    target_unstable = [row for row in target_family if not row["stable"]]
    other_family = [row for row in physical if abs(row["d"] - target_branch_center) > target_half_width]
    common = sorted(
        set(np.round([r["offset"] for r in target_stable], 6))
        & set(np.round([r["offset"] for r in target_unstable], 6))
    )
    summary.update(
        {
            "arm_offsets_scanned": len(arm_offsets),
            "arm_non_degenerate_roots": len(physical),
            "target_branch_center": target_branch_center,
            "target_branch_half_width": target_half_width,
            "target_stable_roots": len(target_stable),
            "target_unstable_roots": len(target_unstable),
            "other_family_roots": len(other_family),
            "fold_coexistence_min_offset": None if not common else float(min(common)),
        }
    )

    arrays = {
        "fine_offset": fine_offset,
        "fine_eig_real": fine_eig_real,
        "fine_rho": fine_rho,
        "fine_fold_indicator": fine_fold_indicator,
        "fine_d": fine_d,
        "arm_offset": arm_offset,
        "arm_d": arm_d,
        "arm_rho": arm_rho,
        "arm_stable": arm_stable,
        "arm_degenerate": arm_degenerate,
    }
    return arrays, summary


def _sorted_xy(offsets: np.ndarray, d: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(offsets)
    return offsets[order], d[order], rho[order]


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    offsets = arrays["fine_offset"]
    eig_real = arrays["fine_eig_real"]
    rho = arrays["fine_rho"]
    fold_indicator = arrays["fine_fold_indicator"]
    d_values = arrays["fine_d"]

    fig1, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(offsets, eig_real, "o-", markersize=3, linewidth=1.4)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, label="+1 fold condition")
    axes[0].axhline(-1.0, color="tab:purple", linestyle="--", linewidth=1, label="-1 period-doubling")
    axes[0].set_ylabel("real eigenvalue")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(offsets, rho, "o-", markersize=3, linewidth=1.4, color="tab:orange")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("spectral radius")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(offsets, d_values, "o-", markersize=3, linewidth=1.4, color="tab:green")
    axes[2].set_xlabel("symmetric COM offset")
    axes[2].set_ylabel("fixed stride d [m]")
    axes[2].grid(True, alpha=0.3)
    candidate_mask = np.isfinite(fold_indicator) & (fold_indicator <= cfg["fold_tol"])
    for ax in axes:
        for x_value in offsets[candidate_mask]:
            ax.axvline(x_value, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
    fig1.suptitle("Fine COM scan near suspected special point")
    fig1.tight_layout()

    keep = ~arrays["arm_degenerate"]
    a_off = arrays["arm_offset"][keep]
    a_d = arrays["arm_d"][keep]
    a_rho = arrays["arm_rho"][keep]
    a_stable = arrays["arm_stable"][keep]
    target_center = float(summary["target_branch_center"])
    target_half_width = float(summary["target_branch_half_width"])
    target = np.abs(a_d - target_center) <= target_half_width
    xs_s, ds_s, rho_s = _sorted_xy(a_off[target & a_stable], a_d[target & a_stable], a_rho[target & a_stable])
    xs_u, ds_u, _ = _sorted_xy(a_off[target & ~a_stable], a_d[target & ~a_stable], a_rho[target & ~a_stable])
    xs_o, ds_o, _ = _sorted_xy(a_off[~target], a_d[~target], a_rho[~target])

    fig2, ax = plt.subplots(figsize=(9.5, 6))
    colorbar_source = None
    if len(xs_s):
        ax.plot(xs_s, ds_s, "-", color="0.6", linewidth=1.2, zorder=1)
        colorbar_source = ax.scatter(
            xs_s, ds_s, c=rho_s, cmap="viridis_r", s=70, edgecolors="black", linewidths=0.5, zorder=3,
            label="target stable arm (|lambda|<1)",
        )
    if len(xs_u):
        ax.plot(xs_u, ds_u, "--", color="tab:red", linewidth=1.2, zorder=1)
        ax.scatter(xs_u, ds_u, marker="x", s=80, color="tab:red", linewidths=2, zorder=3, label="target unstable arm (|lambda|>1)")
    if len(xs_o):
        ax.scatter(xs_o, ds_o, marker="s", s=30, facecolors="none", edgecolors="0.5", zorder=2, label="other roots")

    nose = summary.get("fold_coexistence_min_offset")
    if nose is not None and len(xs_s):
        nose_d = float(ds_s[np.argmin(np.abs(xs_s - nose))])
        ax.axvline(nose, color="black", linestyle=":", linewidth=1.1)
        ax.annotate(
            f"target arms coexist down to\noffset={nose:+.4f}\n(fold just beyond)",
            xy=(nose, nose_d),
            xytext=(12, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

    ax.set_xlabel("symmetric COM offset")
    ax.set_ylabel("fixed stride distance d [m]")
    ax.set_title("Fold near lower COM limit: target stable + unstable arms vs other roots")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if colorbar_source is not None:
        cbar = fig2.colorbar(colorbar_source, ax=ax)
        cbar.set_label("Poincare spectral radius |lambda| on the target stable arm")
    fig2.tight_layout()

    return {"fine_scan": fig1, "fold_arms": fig2}


def run(
    params: dict[str, Any] | None = None,
    *,
    force: bool = False,
    results_dir: Path | str = Path("results"),
    verbose: bool = True,
) -> PartResult:
    cfg = {**DEFAULTS, **(params or {})}
    cfg["_algo"] = ALGO_VERSION
    return cached_run(
        part=PART,
        config=cfg,
        compute=_compute,
        plot=_plot,
        results_dir=results_dir,
        force=force,
        verbose=verbose,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the COM special-point (fold) analysis and cache two figures.")
    parser.add_argument("--gamma-deg", type=float, default=DEFAULTS["gamma_deg"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--t-max", type=float, default=DEFAULTS["t_max"])
    parser.add_argument("--target-branch-half-width", type=float, default=DEFAULTS["target_branch_half_width"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true", help="Recompute even if the cache exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    params = {
        "gamma_deg": args.gamma_deg,
        "dt": args.dt,
        "t_max": args.t_max,
        "target_branch_half_width": args.target_branch_half_width,
    }
    result = run(params, force=args.force, results_dir=args.results_dir)
    print(f"Data: {result.data_path}")
    for name, path in result.figures.items():
        print(f"Figure[{name}]: {path}")
    print(
        f"Baseline gait d*={result.summary['d_fixed']:.6f}, "
        f"branch={result.summary['branch']}, period={result.summary['period']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
