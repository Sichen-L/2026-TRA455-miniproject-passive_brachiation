"""Run the COM special-point analysis and save only plots 1 and 3.

This is the script version of ``com_sweep_special_point_analysis.ipynb``.
It keeps the fine local scan plot and the global two-arm fold plot, while
omitting the full free-run comparison plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _base_params() -> BrachiationParameters:
    return BrachiationParameters.uniform_links(
        m1=1.041,
        m2=1.041,
        l1=0.314,
        l2=0.314,
        damping1=0.0,
        damping2=0.0,
        gravity=9.81,
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


def _skipped_nonconverged_between(
    rows: list[dict[str, Any]],
    left_offset: float,
    right_offset: float,
) -> int:
    left_idx = next(i for i, row in enumerate(rows) if row["offset"] == left_offset)
    right_idx = next(i for i, row in enumerate(rows) if row["offset"] == right_offset)
    lo, hi = sorted((left_idx, right_idx))
    return sum(not row["converged"] for row in rows[lo + 1 : hi])


def _sorted_xy(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec = sorted(records, key=lambda item: item["offset"])
    return (
        np.array([item["offset"] for item in rec], dtype=float),
        np.array([item["d"] for item in rec], dtype=float),
        np.array([item["rho"] for item in rec], dtype=float),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the COM special-point analysis and save plots 1 and 3.",
    )
    parser.add_argument("--gamma-deg", type=float, default=45.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--t-max", type=float, default=8.0)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", type=str, default="com_special_point")
    parser.add_argument("--d-scan-points", type=int, default=30)
    parser.add_argument("--d-lower", type=float, default=0.05)
    parser.add_argument("--d-upper-fraction", type=float, default=0.95)
    parser.add_argument("--local-start", type=float, default=-0.34)
    parser.add_argument("--local-end", type=float, default=-0.36)
    parser.add_argument("--local-step", type=float, default=-0.00025)
    parser.add_argument("--fold-tol", type=float, default=7.5e-2)
    parser.add_argument("--arm-offset-start", type=float, default=-0.30)
    parser.add_argument("--arm-offset-end", type=float, default=-0.3525)
    parser.add_argument("--arm-offset-step", type=float, default=-0.0025)
    parser.add_argument("--arm-d-scan-points", type=int, default=70)
    parser.add_argument("--root-residual-tol", type=float, default=1e-6)
    parser.add_argument("--degenerate-rho", type=float, default=50.0)
    parser.add_argument("--jacobian-delta", type=float, default=1e-5)
    parser.add_argument("--d-split", type=float, default=0.45)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    params = _base_params()
    model = TwoLinkBrachiationModel(params)
    slope = Slope(gamma=np.deg2rad(args.gamma_deg))
    support = np.zeros(2, dtype=float)
    initial_direction = -1.0
    impact_direction = 1.0

    print("Base uniform-link parameters before applying any COM offset:")
    print(params)
    print(f"gamma = {args.gamma_deg:.1f} deg")

    stride_search = scan_stride_fixed_points(
        model=model,
        slope=slope,
        initial_support_point=support,
        dt=args.dt,
        t_max=args.t_max,
        d_bounds=(args.d_lower, args.d_upper_fraction * (params.l1 + params.l2)),
        d_scan_points=args.d_scan_points,
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
    print(
        "Selected baseline gait: "
        f"d*={d_fixed:.9f}, branch={selected_branch}, period={selected_period}"
    )

    def model_with_com_offset(com_offset: float) -> TwoLinkBrachiationModel:
        return TwoLinkBrachiationModel(
            parameters_with_symmetric_com_offset(com_offset, params)
        )

    def make_stride_map_for_offset(com_offset: float):
        return make_passive_brachiation_stride_map(
            model=model_with_com_offset(com_offset),
            slope=slope,
            dt=args.dt,
            t_max=args.t_max,
            collision_mode=CollisionMode.FULL_GRAB_1D,
            initial_direction=initial_direction,
            impact_direction=impact_direction,
            branch=selected_branch,
            support_point=support,
            switch_policy=_switch_policy,
        )

    def make_feasibility(_parameter: float):
        return make_passive_brachiation_feasibility_check(params, dim=1)

    approach_offsets = np.linspace(0.0, args.local_start, 41)
    approach_result = continue_fixed_point_branch(
        P_factory=make_stride_map_for_offset,
        parameters=approach_offsets,
        x0=np.array([d_fixed], dtype=float),
        dim=1,
        feasibility_factory=make_feasibility,
        tol=1e-6,
        max_iter=10,
        delta=args.jacobian_delta,
        damping=0.8,
        compute_stability=True,
        stop_on_failure=True,
    )
    if approach_result.stopped_early:
        print("Approach stopped early:", approach_result.stop_reason)

    seed_point = next(point for point in reversed(approach_result.points) if point.converged)
    local_offsets = np.arange(
        args.local_start,
        args.local_end + 0.5 * args.local_step,
        args.local_step,
    )
    local_result = continue_fixed_point_branch(
        P_factory=make_stride_map_for_offset,
        parameters=local_offsets,
        x0=seed_point.x,
        dim=1,
        feasibility_factory=make_feasibility,
        tol=1e-7,
        max_iter=12,
        delta=args.jacobian_delta,
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
                "residual": point.residual_norm,
                "eig_real": eig_real,
                "rho": np.nan
                if point.spectral_radius is None
                else float(point.spectral_radius),
                "fold_indicator": fold_indicator,
                "fold_candidate": bool(
                    np.isfinite(fold_indicator) and fold_indicator <= args.fold_tol
                ),
                "failure_reason": point.failure_reason,
            }
        )

    valid_rows = [row for row in local_rows if row["converged"]]
    print(f"local scan points = {len(local_rows)}, converged = {len(valid_rows)}")
    if not valid_rows:
        raise RuntimeError("Local scan found no converged points.")

    offsets = np.array([row["offset"] for row in valid_rows], dtype=float)
    eig_real = np.array([row["eig_real"] for row in valid_rows], dtype=float)
    rho = np.array([row["rho"] for row in valid_rows], dtype=float)
    fold_indicator = np.array([row["fold_indicator"] for row in valid_rows], dtype=float)
    d_values = np.array([row["d"] for row in valid_rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
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
    candidate_mask = np.isfinite(fold_indicator) & (fold_indicator <= args.fold_tol)
    for ax in axes:
        for x_value in offsets[candidate_mask]:
            ax.axvline(x_value, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
    fig.suptitle("Fine COM scan near suspected special point")
    fig.tight_layout()
    fine_plot_path = args.results_dir / f"{args.output_prefix}_plot1_fine_scan.png"
    fig.savefig(fine_plot_path, dpi=180)
    plt.close(fig)
    print(f"Wrote: {fine_plot_path}")

    summary: dict[str, Any] = {
        "gamma_deg": args.gamma_deg,
        "d_fixed": d_fixed,
        "branch": selected_branch,
        "period": selected_period,
        "local_points": len(local_rows),
        "local_converged": len(valid_rows),
        "fine_plot_path": str(fine_plot_path),
    }
    if len(offsets) >= 2:
        eig_jump = np.abs(np.diff(eig_real))
        eig_jump_idx = int(np.nanargmax(eig_jump))
        d_jump = np.abs(np.diff(d_values))
        d_jump_idx = int(np.nanargmax(d_jump))
        skipped = _skipped_nonconverged_between(
            local_rows,
            offsets[d_jump_idx],
            offsets[d_jump_idx + 1],
        )
        print("largest eig jump:")
        print(f"  between {offsets[eig_jump_idx]:+.6f} and {offsets[eig_jump_idx + 1]:+.6f}")
        print(f"  eig {eig_real[eig_jump_idx]:+.6f} -> {eig_real[eig_jump_idx + 1]:+.6f}, jump={eig_jump[eig_jump_idx]:.6f}")
        print("largest d jump:")
        print(f"  between {offsets[d_jump_idx]:+.6f} and {offsets[d_jump_idx + 1]:+.6f}")
        print(f"  d {d_values[d_jump_idx]:.9f} -> {d_values[d_jump_idx + 1]:.9f}, jump={d_jump[d_jump_idx]:.9f} m")
        print(f"  skipped nonconverged points={skipped}")
        summary.update(
            {
                "largest_eig_jump_left": float(offsets[eig_jump_idx]),
                "largest_eig_jump_right": float(offsets[eig_jump_idx + 1]),
                "largest_eig_jump": float(eig_jump[eig_jump_idx]),
                "largest_d_jump_left": float(offsets[d_jump_idx]),
                "largest_d_jump_right": float(offsets[d_jump_idx + 1]),
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
    if fail_runs:
        longest = max(fail_runs, key=lambda item: item[1] - item[0] + 1)
        print("nonconverged gaps:")
        print(f"  total gaps={len(fail_runs)}, longest consecutive nonconverged points={longest[1] - longest[0] + 1}")
        for start, end in fail_runs:
            print(
                f"  {local_rows[start]['offset']:+.6f} to {local_rows[end]['offset']:+.6f}: "
                f"{end - start + 1} points, first reason={local_rows[start]['failure_reason']}"
            )
    else:
        print("nonconverged gaps: none")

    arm_offsets = np.arange(
        args.arm_offset_start,
        args.arm_offset_end + 0.5 * args.arm_offset_step,
        args.arm_offset_step,
    )
    arm_records: list[dict[str, Any]] = []
    for offset in arm_offsets:
        p_map = make_stride_map_for_offset(float(offset))
        feasibility = make_feasibility(float(offset))
        roots, _ = find_fixed_points_1d(
            p_map,
            bounds=(args.d_lower, args.d_upper_fraction * (params.l1 + params.l2)),
            num_scan_points=args.arm_d_scan_points,
            feasibility_check=feasibility,
            residual_tol=args.root_residual_tol,
        )
        for root in roots:
            try:
                _, _, root_rho = poincare_jacobian_eigenvalues_1d(
                    p_map,
                    root.x,
                    delta=args.jacobian_delta,
                    feasibility_check=feasibility,
                )
            except Exception:
                root_rho = np.nan
            arm_records.append(
                {
                    "offset": float(offset),
                    "d": float(root.x),
                    "rho": float(root_rho),
                    "stable": bool(np.isfinite(root_rho) and root_rho < 1.0),
                    "degenerate": bool(
                        np.isfinite(root_rho) and root_rho >= args.degenerate_rho
                    ),
                }
            )

    physical_records = [row for row in arm_records if not row["degenerate"]]
    print(
        f"offsets scanned = {len(arm_offsets)}, "
        f"non-degenerate roots = {len(physical_records)}"
    )
    for offset in arm_offsets:
        here = sorted(
            (row for row in physical_records if abs(row["offset"] - offset) < 1e-9),
            key=lambda row: row["d"],
        )
        listing = ", ".join(
            f"d={row['d']:.4f}(rho={row['rho']:.3f},{'S' if row['stable'] else 'U'})"
            for row in here
        )
        print(f"offset {offset:+.4f}: {len(here)} root(s) -> {listing}")

    stable_short = [
        row for row in physical_records if row["d"] < args.d_split and row["stable"]
    ]
    unstable_short = [
        row for row in physical_records if row["d"] < args.d_split and not row["stable"]
    ]
    long_family = [row for row in physical_records if row["d"] >= args.d_split]
    xs_s, ds_s, rho_s = _sorted_xy(stable_short)
    xs_u, ds_u, _rho_u = _sorted_xy(unstable_short)
    xs_l, ds_l, _ = _sorted_xy(long_family)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    colorbar_source = None
    if len(xs_s):
        ax.plot(xs_s, ds_s, "-", color="0.6", linewidth=1.2, zorder=1)
        colorbar_source = ax.scatter(
            xs_s,
            ds_s,
            c=rho_s,
            cmap="viridis_r",
            s=70,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
            label="short stable arm (|lambda|<1)",
        )
    if len(xs_u):
        ax.plot(xs_u, ds_u, "--", color="tab:red", linewidth=1.2, zorder=1)
        ax.scatter(
            xs_u,
            ds_u,
            marker="x",
            s=80,
            color="tab:red",
            linewidths=2,
            zorder=3,
            label="short unstable arm (|lambda|>1)",
        )
    if len(xs_l):
        ax.scatter(
            xs_l,
            ds_l,
            marker="s",
            s=30,
            facecolors="none",
            edgecolors="0.5",
            zorder=2,
            label="separate long-stride family",
        )

    common_offsets = sorted(set(np.round(xs_s, 6)) & set(np.round(xs_u, 6)))
    if common_offsets:
        nose_offset = min(common_offsets)
        nose_d = float(ds_s[np.argmin(np.abs(xs_s - nose_offset))])
        ax.axvline(nose_offset, color="black", linestyle=":", linewidth=1.1)
        ax.annotate(
            f"both arms coexist down to\noffset={nose_offset:+.4f}\n(fold just beyond)",
            xy=(nose_offset, nose_d),
            xytext=(12, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )
        print(f"Two short-stride arms coexist for offsets >= {nose_offset:+.4f} (fold just beyond).")
    else:
        print("No coexisting short-stride arm pair found in the scanned window.")

    ax.set_xlabel("symmetric COM offset")
    ax.set_ylabel("fixed stride distance d [m]")
    ax.set_title("Fold near -0.35: short stable + unstable arms vs separate long-stride family")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if colorbar_source is not None:
        cbar = fig.colorbar(colorbar_source, ax=ax)
        cbar.set_label("Poincare spectral radius |lambda| on the stable arm")
    fig.tight_layout()
    fold_plot_path = args.results_dir / f"{args.output_prefix}_plot3_fold_arms.png"
    fig.savefig(fold_plot_path, dpi=180)
    plt.close(fig)
    print(f"Wrote: {fold_plot_path}")

    summary.update(
        {
            "arm_offsets_scanned": len(arm_offsets),
            "arm_non_degenerate_roots": len(physical_records),
            "short_stable_roots": len(stable_short),
            "short_unstable_roots": len(unstable_short),
            "long_family_roots": len(long_family),
            "fold_coexistence_min_offset": None
            if not common_offsets
            else float(min(common_offsets)),
            "fold_plot_path": str(fold_plot_path),
        }
    )
    summary_path = args.results_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
