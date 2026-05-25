"""Part 3a: adaptive symmetric-COM continuation sweep (compute + plot + cache).

Importable interface::

    from run_com_sweep import run
    res = run({"gamma_deg": 45.0}, force=False)
    res.figure("main")          # -> Path to the cached PNG

A run is cached by a hash of the full config dict (including ``_algo``), so
calling :func:`run` again with unchanged parameters reuses the cached figure
instead of recomputing the continuation.  The CLI entry point wraps the same
``run`` function.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from passive_brachiation import (
    BrachiationParameters,
    CollisionMode,
    Slope,
    SwitchDecision,
    TwoLinkBrachiationModel,
    run_symmetric_com_continuation,
    scan_stride_fixed_points,
)
from report_cache import PartResult, cached_run

PART = "com_sweep"
ALGO_VERSION = "v1"

# Reachable COM range for the rod+point model is lc/L in [0.1, 0.9],
# i.e. offset in [-0.4, 0.4]; values beyond that are infeasible.
DEFAULTS: dict[str, Any] = {
    "gamma_deg": 45.0,
    "offset_low": -0.4,
    "offset_high": 0.4,
    "steps_per_side": 31,
    "dt": 0.005,
    "t_max": 8.0,
    "continuation_tol": 1e-6,
    "continuation_max_iter": 10,
    "continuation_delta": 1e-5,
    "continuation_damping": 0.8,
    "adaptive_min_step": 5e-4,
    "adaptive_max_step": 0.04,
    "adaptive_growth": 1.35,
    "adaptive_shrink": 0.5,
    "fold_tolerance": 7.5e-2,
    "d_scan_points": 30,
    "d_lower": 0.05,
    "d_upper_fraction": 0.95,
    "branches": ["positive", "negative"],
    "periods": [1, 2],
    "d_fixed": None,
    "branch": None,
    "period": None,
    # physical model (rod + centred movable point mass; 80% point, 20% rod)
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


def build_params(cfg: dict[str, Any]) -> BrachiationParameters:
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


def _rows_to_arrays(rows) -> dict[str, np.ndarray]:
    return {
        "direction": np.array([row.direction for row in rows], dtype="U32"),
        "com_offset": np.array([row.com_offset for row in rows], dtype=float),
        "lc1_fraction": np.array([row.lc1_fraction for row in rows], dtype=float),
        "lc2_fraction": np.array([row.lc2_fraction for row in rows], dtype=float),
        "d_primary": np.array([row.d_primary for row in rows], dtype=float),
        "d_next": np.array([row.d_next for row in rows], dtype=float),
        "stride_plot": np.array([row.stride_plot for row in rows], dtype=float),
        "period": np.array([row.period for row in rows], dtype=int),
        "converged": np.array([row.converged for row in rows], dtype=bool),
        "residual_norm": np.array([row.residual_norm for row in rows], dtype=float),
        "spectral_radius": np.array([row.spectral_radius for row in rows], dtype=float),
        "eigen_real": np.array([row.eigen_real for row in rows], dtype=float),
        "stable": np.array([row.stable for row in rows], dtype=bool),
        "legal": np.array([row.legal for row in rows], dtype=bool),
        "max_elbow_distance": np.array([row.max_elbow_distance for row in rows], dtype=float),
        "min_elbow_distance": np.array([row.min_elbow_distance for row in rows], dtype=float),
        "parameter_step": np.array(
            [np.nan if row.parameter_step is None else row.parameter_step for row in rows],
            dtype=float,
        ),
        "fold_indicator": np.array(
            [np.nan if row.fold_indicator is None else row.fold_indicator for row in rows],
            dtype=float,
        ),
        "fold_candidate": np.array([row.fold_candidate for row in rows], dtype=bool),
        "failure_reason": np.array(
            ["" if row.failure_reason is None else row.failure_reason for row in rows],
            dtype="U256",
        ),
    }


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    params = build_params(cfg)
    model = TwoLinkBrachiationModel(params)
    slope = Slope(gamma=np.deg2rad(cfg["gamma_deg"]))
    support = np.zeros(2, dtype=float)
    branches = tuple(cfg["branches"])
    periods = tuple(cfg["periods"])

    if cfg["d_fixed"] is None:
        search = scan_stride_fixed_points(
            model=model,
            slope=slope,
            initial_support_point=support,
            dt=cfg["dt"],
            t_max=cfg["t_max"],
            d_bounds=(cfg["d_lower"], cfg["d_upper_fraction"] * (params.l1 + params.l2)),
            d_scan_points=cfg["d_scan_points"],
            branches=branches,
            periods=periods,
            initial_direction=-1.0,
            impact_direction=1.0,
            switch_policy=_switch_policy,
            collision_mode=CollisionMode.FULL_GRAB_1D,
        )
        selected = search.selected_trial
        if selected is None:
            raise RuntimeError("No legal stride fixed point found.")
        d_fixed = float(selected.d)
        branch = selected.branch
        period = int(selected.period)
        q2_fixed = float(selected.q0[1])
    else:
        if cfg["branch"] is None or cfg["period"] is None:
            raise ValueError("branch and period are required when d_fixed is supplied.")
        d_fixed = float(cfg["d_fixed"])
        branch = str(cfg["branch"])
        period = int(cfg["period"])
        q2_fixed = float("nan")

    initial_step = abs(cfg["offset_high"]) / max(cfg["steps_per_side"] - 1, 1)
    continuation = run_symmetric_com_continuation(
        base_params=params,
        slope=slope,
        d_fixed=d_fixed,
        branch=branch,
        period=period,
        offset_low=cfg["offset_low"],
        offset_high=cfg["offset_high"],
        n_steps_per_side=cfg["steps_per_side"],
        dt=cfg["dt"],
        t_max=cfg["t_max"],
        initial_support_point=support,
        initial_direction=-1.0,
        impact_direction=1.0,
        switch_policy=_switch_policy,
        collision_mode=CollisionMode.FULL_GRAB_1D,
        continuation_tol=cfg["continuation_tol"],
        continuation_max_iter=cfg["continuation_max_iter"],
        continuation_delta=cfg["continuation_delta"],
        continuation_damping=cfg["continuation_damping"],
        adaptive_steps=True,
        adaptive_initial_step=initial_step,
        adaptive_min_step=cfg["adaptive_min_step"],
        adaptive_max_step=cfg["adaptive_max_step"],
        adaptive_step_growth=cfg["adaptive_growth"],
        adaptive_step_shrink=cfg["adaptive_shrink"],
        fold_tolerance=cfg["fold_tolerance"],
    )

    arrays = _rows_to_arrays(continuation.rows)
    finite = arrays["converged"] & arrays["legal"] & np.isfinite(arrays["spectral_radius"])
    best_index = int(np.where(finite)[0][np.argmin(arrays["spectral_radius"][finite])])
    summary = {
        "d_fixed": d_fixed,
        "q2_fixed": q2_fixed,
        "branch": branch,
        "period": period,
        "row_count": len(continuation.rows),
        "contact_stopped_early": continuation.contact_result.stopped_early,
        "contact_stop_reason": continuation.contact_result.stop_reason,
        "contact_fold_detected": continuation.contact_result.fold_detected,
        "contact_fold_parameter": continuation.contact_result.fold_parameter,
        "elbow_stopped_early": continuation.elbow_result.stopped_early,
        "elbow_stop_reason": continuation.elbow_result.stop_reason,
        "elbow_fold_detected": continuation.elbow_result.fold_detected,
        "elbow_fold_parameter": continuation.elbow_result.fold_parameter,
        "best_lambda_offset": float(arrays["com_offset"][best_index]),
        "best_lambda": float(arrays["spectral_radius"][best_index]),
    }
    return arrays, summary


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    x = arrays["com_offset"].astype(float)
    y = arrays["stride_plot"].astype(float)
    rho = arrays["spectral_radius"].astype(float)
    converged = arrays["converged"].astype(bool)
    legal = arrays["legal"].astype(bool)
    stable = arrays["stable"].astype(bool)
    eig_real = arrays["eigen_real"].astype(float)
    fold_candidate = arrays["fold_candidate"].astype(bool)
    fold_indicator = arrays["fold_indicator"].astype(float)

    mask = converged & np.isfinite(y)
    x, y, rho = x[mask], y[mask], rho[mask]
    legal, stable, eig_real = legal[mask], stable[mask], eig_real[mask]
    fold_candidate, fold_indicator = fold_candidate[mask], fold_indicator[mask]
    order = np.argsort(x)
    x, y, rho = x[order], y[order], rho[order]
    legal, stable, eig_real = legal[order], stable[order], eig_real[order]
    fold_candidate, fold_indicator = fold_candidate[order], fold_indicator[order]

    pd_mask = legal & np.isfinite(eig_real) & (eig_real <= -1.0)
    unstable_mask = legal & ~stable & ~pd_mask
    illegal_mask = ~legal

    states = np.array(
        [
            "illegal"
            if illegal_mask[i]
            else "period-doubling"
            if pd_mask[i]
            else "unstable"
            if unstable_mask[i]
            else "stable"
            for i in range(len(x))
        ],
        dtype=object,
    )
    segments = (
        np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
        if len(x) > 1
        else np.empty((0, 2, 2), dtype=float)
    )
    segment_states = []
    segment_rho = []
    for i in range(max(0, len(x) - 1)):
        pair = {states[i], states[i + 1]}
        if "illegal" in pair:
            state = "illegal"
        elif "period-doubling" in pair:
            state = "period-doubling"
        elif "unstable" in pair:
            state = "unstable"
        else:
            state = "stable"
        segment_states.append(state)
        segment_rho.append(np.nanmean(rho[i : i + 2]))
    segment_states = np.array(segment_states, dtype=object)
    segment_rho = np.array(segment_rho, dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    finite_rho = np.isfinite(rho) & legal
    norm = Normalize(vmin=float(np.nanmin(rho[finite_rho])), vmax=float(np.nanmax(rho[finite_rho])))
    cmap = plt.get_cmap("viridis_r")
    stable_segments = segments[segment_states == "stable"]
    stable_segment_rho = segment_rho[segment_states == "stable"]
    if len(stable_segments):
        lines = LineCollection(
            stable_segments,
            cmap=cmap,
            norm=norm,
            linewidths=2.6,
            linestyles="solid",
            label="stable, colored by |lambda|max",
        )
        lines.set_array(stable_segment_rho)
        ax.add_collection(lines)
        colorbar_source = lines
    else:
        colorbar_source = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    styles = {
        "period-doubling": ("tab:purple", "--", "period-doubling / eig <= -1"),
        "unstable": ("tab:red", "-.", "unstable |lambda| >= 1"),
        "illegal": ("0.35", ":", "illegal geometry"),
    }
    for state, (color, linestyle, label) in styles.items():
        first = True
        for segment in segments[segment_states == state]:
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label if first else None,
            )
            first = False

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    y_span = max(y_max - y_min, 1e-6)
    switch_x = [0.5 * (x[i] + x[i + 1]) for i in range(len(states) - 1) if states[i] != states[i + 1]]
    for value in switch_x:
        ax.axvline(value, color="0.45", linestyle=":", linewidth=1.0)
        ax.annotate(
            f"x={value:+.3f}",
            xy=(value, y_max),
            xytext=(6, -12),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color="0.35",
        )

    fold_idx = np.where(fold_candidate & np.isfinite(fold_indicator))[0]
    if len(fold_idx):
        idx = fold_idx[np.argmin(fold_indicator[fold_idx])]
        ax.axvline(x[idx], color="black", linestyle="--", linewidth=1.2, label="fold candidate lambda -> +1")

    best_x = float(summary["best_lambda_offset"])
    best_lambda = float(summary["best_lambda"])
    best_idx = int(np.argmin(np.abs(x - best_x)))
    ax.axvline(best_x, color="tab:green", linewidth=1.2, label="minimum |lambda|max")
    ax.annotate(
        f"min |lambda|={best_lambda:.4f}\nx={best_x:+.4f}",
        xy=(x[best_idx], y[best_idx]),
        xytext=(10, 18),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="tab:green"),
        color="tab:green",
        fontsize=9,
    )

    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, label="center COM")
    ax.set_xlim(float(np.nanmin(x)) - 0.02, float(np.nanmax(x)) + 0.02)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.10 * y_span)
    ax.set_xlabel("symmetric COM offset: + toward elbow, - toward contact endpoints")
    ax.set_ylabel("steady stride distance [m]" if int(summary["period"]) == 1 else "mean period-2 stride distance [m]")
    ax.set_title(f"Adaptive COM continuation: branch={summary['branch']}, period={summary['period']}, gamma={cfg['gamma_deg']:.1f} deg")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    cbar = fig.colorbar(colorbar_source, ax=ax)
    cbar.set_label("Poincare spectral radius |lambda|max on stable segments")
    fig.tight_layout()
    return {"main": fig}


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


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an adaptive symmetric COM-offset continuation and cache the result.",
    )
    parser.add_argument("--gamma-deg", type=float, default=DEFAULTS["gamma_deg"])
    parser.add_argument("--offset-low", type=float, default=DEFAULTS["offset_low"])
    parser.add_argument("--offset-high", type=float, default=DEFAULTS["offset_high"])
    parser.add_argument("--steps-per-side", type=int, default=DEFAULTS["steps_per_side"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--t-max", type=float, default=DEFAULTS["t_max"])
    parser.add_argument("--d-scan-points", type=int, default=DEFAULTS["d_scan_points"])
    parser.add_argument("--branches", type=_parse_csv_strings, default=None)
    parser.add_argument("--periods", type=_parse_csv_ints, default=None)
    parser.add_argument("--d-fixed", type=float, default=None)
    parser.add_argument("--branch", type=str, default=None)
    parser.add_argument("--period", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true", help="Recompute even if the cache exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    params: dict[str, Any] = {
        "gamma_deg": args.gamma_deg,
        "offset_low": args.offset_low,
        "offset_high": args.offset_high,
        "steps_per_side": args.steps_per_side,
        "dt": args.dt,
        "t_max": args.t_max,
        "d_scan_points": args.d_scan_points,
        "d_fixed": args.d_fixed,
        "branch": args.branch,
        "period": args.period,
    }
    if args.branches is not None:
        params["branches"] = args.branches
    if args.periods is not None:
        params["periods"] = args.periods

    result = run(params, force=args.force, results_dir=args.results_dir)
    print(f"Data: {result.data_path}")
    print(f"Figure: {result.figure('main')}")
    print(
        "Most stable COM offset: "
        f"x={result.summary['best_lambda_offset']:+.6f}, "
        f"|lambda|max={result.summary['best_lambda']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
