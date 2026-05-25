"""Part 3c: slope-angle sweep at the most stable COM offset (compute + plot + cache).

The operating point (COM offset, anchor stride, branch, period, base slope) is
read from the cached COM sweep (:mod:`report_setup`), holding the COM offset
fixed while the slope angle gamma is continued in both directions.

Importable interface::

    from run_gamma_sweep import run
    res = run({"gamma_low": 25.0, "gamma_high": 65.0})
    res.figure("main")
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
    ContinuationResult,
    Slope,
    SwitchDecision,
    TwoLinkBrachiationModel,
    continue_fixed_point_branch_adaptive,
    evaluate_elbow_below_slope_section,
    evaluate_passive_brachiation_stride,
    make_iterated_stride_map,
    make_passive_brachiation_feasibility_check,
    make_passive_brachiation_stride_map,
    parameters_with_symmetric_com_offset,
    release_indices,
)
from report_cache import PartResult, cached_run
from report_setup import load_best_com_setup

PART = "gamma_sweep"
ALGO_VERSION = "v1"

DEFAULTS: dict[str, Any] = {
    "com_offset": None,  # None -> use the most stable COM-sweep offset
    "gamma_low": 25.0,
    "gamma_high": 65.0,
    "steps_per_side": 31,
    "dt": 0.005,
    "t_max": 8.0,
    "continuation_tol": 1e-6,
    "continuation_max_iter": 10,
    "continuation_delta": 1e-5,
    "continuation_damping": 0.8,
    "adaptive_min_step": 0.05,
    "adaptive_max_step": 2.0,
    "adaptive_growth": 1.35,
    "adaptive_shrink": 0.5,
    "fold_tolerance": 7.5e-2,
}


def _switch_policy(_t, _state, _support_point, _impact_point, _slope):
    return SwitchDecision.SWITCH


def _model_at_com(com_offset: float, base_params: BrachiationParameters) -> TwoLinkBrachiationModel:
    return TwoLinkBrachiationModel(parameters_with_symmetric_com_offset(com_offset, base_params))


def _row_arrays(
    result: ContinuationResult,
    com_offset: float,
    branch: str,
    period: int,
    dt: float,
    t_max: float,
    base_params: BrachiationParameters,
) -> dict[str, list]:
    support = np.zeros(2, dtype=float)
    rows: list[dict[str, Any]] = []
    for point in result.points:
        gamma_deg = float(point.parameter)
        slope = Slope(gamma=np.deg2rad(gamma_deg))
        d_value = float(point.x[0])
        row: dict[str, Any] = {
            "gamma_deg": gamma_deg,
            "d_primary": d_value,
            "d_next": np.nan,
            "stride_plot": np.nan,
            "converged": point.converged,
            "residual_norm": point.residual_norm,
            "spectral_radius": np.nan if point.spectral_radius is None else float(point.spectral_radius),
            "eigen_real": np.nan if point.eigenvalues is None else float(np.real(point.eigenvalues[0])),
            "stable": bool(
                point.converged
                and point.spectral_radius is not None
                and np.isfinite(point.spectral_radius)
                and point.spectral_radius < 1.0
            ),
            "legal": False,
            "max_elbow_distance": np.nan,
            "min_elbow_distance": np.nan,
            "parameter_step": np.nan if point.parameter_step is None else point.parameter_step,
            "fold_indicator": np.nan if point.fold_indicator is None else point.fold_indicator,
            "fold_candidate": point.fold_candidate,
            "failure_reason": "" if point.failure_reason is None else point.failure_reason,
        }
        if point.converged:
            evaluation = evaluate_passive_brachiation_stride(
                model=_model_at_com(com_offset, base_params),
                slope=slope,
                x=np.array([d_value], dtype=float),
                dt=dt,
                t_max=t_max,
                collision_mode=CollisionMode.FULL_GRAB_1D,
                initial_direction=-1.0,
                impact_direction=1.0,
                branch=branch,
                support_point=support,
                switch_policy=_switch_policy,
                stop_after_releases=max(1, period),
            )
            row["d_next"] = float(evaluation.p_of_x[0])
            row["stride_plot"] = d_value if period == 1 else 0.5 * (d_value + row["d_next"])
            rel_indices = release_indices(evaluation.samples)
            if len(rel_indices) >= period:
                legality = evaluate_elbow_below_slope_section(
                    evaluation.samples[: rel_indices[period - 1] + 1],
                    slope=slope,
                    tolerance=1e-9,
                )
                row["legal"] = legality.legal
                row["max_elbow_distance"] = legality.max_signed_distance
                row["min_elbow_distance"] = legality.min_signed_distance
        rows.append(row)

    if not rows:
        return {}
    return {key: [row[key] for row in rows] for key in rows[0].keys()}


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    com_offset = float(cfg["com_offset"])
    d_fixed = float(cfg["d_fixed"])
    branch = str(cfg["branch"])
    period = int(cfg["period"])
    base_gamma_deg = float(cfg["base_gamma_deg"])
    base_params = BrachiationParameters.rod_point_mass(
        m1=cfg["m1"],
        m2=cfg["m2"],
        l1=cfg["l1"],
        l2=cfg["l2"],
        rod_mass_fraction=0.2,
        damping1=cfg["damping1"],
        damping2=cfg["damping2"],
        gravity=cfg["gravity"],
    )

    if not cfg["gamma_low"] < base_gamma_deg < cfg["gamma_high"]:
        raise ValueError(
            f"Gamma range must contain the source gamma {base_gamma_deg}; "
            f"got [{cfg['gamma_low']}, {cfg['gamma_high']}]."
        )

    support = np.zeros(2, dtype=float)

    def P_factory(gamma_deg: float):
        slope = Slope(gamma=np.deg2rad(gamma_deg))
        P_base = make_passive_brachiation_stride_map(
            model=_model_at_com(com_offset, base_params),
            slope=slope,
            dt=cfg["dt"],
            t_max=cfg["t_max"],
            collision_mode=CollisionMode.FULL_GRAB_1D,
            initial_direction=-1.0,
            impact_direction=1.0,
            branch=branch,
            support_point=support,
            switch_policy=_switch_policy,
        )
        return P_base if period == 1 else make_iterated_stride_map(P_base, period)

    def feasibility_factory(_gamma_deg: float):
        return make_passive_brachiation_feasibility_check(base_params, dim=1)

    initial_step = max(
        abs(cfg["gamma_high"] - base_gamma_deg), abs(base_gamma_deg - cfg["gamma_low"])
    ) / max(cfg["steps_per_side"] - 1, 1)

    common = dict(
        P_factory=P_factory,
        start_parameter=base_gamma_deg,
        x0=np.array([d_fixed], dtype=float),
        dim=1,
        feasibility_factory=feasibility_factory,
        tol=cfg["continuation_tol"],
        max_iter=cfg["continuation_max_iter"],
        delta=cfg["continuation_delta"],
        damping=cfg["continuation_damping"],
        compute_stability=True,
        initial_step=initial_step,
        min_step=cfg["adaptive_min_step"],
        max_step=cfg["adaptive_max_step"],
        step_growth=cfg["adaptive_growth"],
        step_shrink=cfg["adaptive_shrink"],
        fold_tolerance=cfg["fold_tolerance"],
    )
    low_result = continue_fixed_point_branch_adaptive(target_parameter=cfg["gamma_low"], **common)
    high_result = continue_fixed_point_branch_adaptive(target_parameter=cfg["gamma_high"], **common)

    low_arrays = _row_arrays(low_result, com_offset, branch, period, cfg["dt"], cfg["t_max"], base_params)
    high_arrays = _row_arrays(high_result, com_offset, branch, period, cfg["dt"], cfg["t_max"], base_params)
    rows: list[dict[str, Any]] = []
    for arrays in (low_arrays, high_arrays):
        for i in range(len(arrays["gamma_deg"])):
            if arrays is high_arrays and abs(float(arrays["gamma_deg"][i]) - base_gamma_deg) < 1e-12:
                continue
            rows.append({key: arrays[key][i] for key in arrays})
    rows.sort(key=lambda row: float(row["gamma_deg"]))

    arrays: dict[str, np.ndarray] = {}
    for key in ("gamma_deg", "d_primary", "d_next", "stride_plot", "residual_norm", "spectral_radius", "eigen_real", "max_elbow_distance", "min_elbow_distance", "parameter_step", "fold_indicator"):
        arrays[key] = np.array([row[key] for row in rows], dtype=float)
    for key in ("converged", "stable", "legal", "fold_candidate"):
        arrays[key] = np.array([row[key] for row in rows], dtype=bool)
    arrays["failure_reason"] = np.array([row["failure_reason"] for row in rows], dtype="U256")

    finite = arrays["converged"] & arrays["legal"] & np.isfinite(arrays["spectral_radius"])
    best_index = int(np.where(finite)[0][np.argmin(arrays["spectral_radius"][finite])])
    summary = {
        "source_com_result": cfg.get("source_com", ""),
        "com_offset": com_offset,
        "d_fixed": d_fixed,
        "branch": branch,
        "period": period,
        "base_gamma_deg": base_gamma_deg,
        "row_count": len(rows),
        "low_stopped_early": low_result.stopped_early,
        "low_stop_reason": low_result.stop_reason,
        "high_stopped_early": high_result.stopped_early,
        "high_stop_reason": high_result.stop_reason,
        "best_lambda_gamma_deg": float(arrays["gamma_deg"][best_index]),
        "best_lambda": float(arrays["spectral_radius"][best_index]),
    }
    return arrays, summary


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    x = arrays["gamma_deg"].astype(float)
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
    if len(stable_segments):
        lines = LineCollection(stable_segments, cmap=cmap, norm=norm, linewidths=2.6, label="stable, colored by |lambda|max")
        lines.set_array(segment_rho[segment_states == "stable"])
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
            ax.plot(segment[:, 0], segment[:, 1], color=color, linestyle=linestyle, linewidth=2.2, label=label if first else None)
            first = False

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    y_span = max(y_max - y_min, 1e-6)
    switch_x = [0.5 * (x[i] + x[i + 1]) for i in range(len(states) - 1) if states[i] != states[i + 1]]
    for value in switch_x:
        ax.axvline(value, color="0.45", linestyle=":", linewidth=1.0)
        ax.text(value, y_max + 0.02 * y_span, f"gamma={value:.2f}", rotation=90, ha="center", va="bottom", fontsize=8)

    fold_idx = np.where(fold_candidate & np.isfinite(fold_indicator))[0]
    if len(fold_idx):
        idx = fold_idx[np.argmin(fold_indicator[fold_idx])]
        ax.axvline(x[idx], color="black", linestyle="--", linewidth=1.2, label="fold candidate lambda -> +1")

    best_gamma = float(summary["best_lambda_gamma_deg"])
    best_lambda = float(summary["best_lambda"])
    best_idx = int(np.argmin(np.abs(x - best_gamma)))
    ax.axvline(best_gamma, color="tab:green", linewidth=1.2, label="minimum |lambda|max")
    ax.annotate(
        f"min |lambda|={best_lambda:.4f}\ngamma={best_gamma:.2f} deg",
        xy=(x[best_idx], y[best_idx]),
        xytext=(10, 18),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="tab:green"),
        color="tab:green",
        fontsize=9,
    )

    ax.set_xlim(float(np.nanmin(x)) - 1.0, float(np.nanmax(x)) + 1.0)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.10 * y_span)
    ax.set_xlabel("slope angle gamma [deg]")
    ax.set_ylabel("steady stride distance [m]" if int(summary["period"]) == 1 else "mean period-2 stride distance [m]")
    ax.set_title(
        "Slope-angle sweep at fixed COM offset "
        f"{summary['com_offset']:+.4f}, branch={summary['branch']}, period={summary['period']}"
    )
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
    setup = load_best_com_setup(results_dir=results_dir)
    com_offset = setup.com_offset if cfg["com_offset"] is None else float(cfg["com_offset"])
    bp = setup.base_params
    cfg.update(
        {
            "_algo": ALGO_VERSION,
            "com_offset": com_offset,
            "d_fixed": setup.d_target,
            "branch": setup.branch,
            "period": setup.period,
            "base_gamma_deg": setup.gamma_deg,
            "m1": bp.m1,
            "m2": bp.m2,
            "l1": bp.l1,
            "l2": bp.l2,
            "damping1": bp.damping1,
            "damping2": bp.damping2,
            "gravity": bp.gravity,
            "source_com": setup.source,
        }
    )
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
    parser = argparse.ArgumentParser(
        description="Scan slope angle while holding the COM offset at the most stable COM-sweep point.",
    )
    parser.add_argument("--com-offset", type=float, default=None)
    parser.add_argument("--gamma-low", type=float, default=DEFAULTS["gamma_low"])
    parser.add_argument("--gamma-high", type=float, default=DEFAULTS["gamma_high"])
    parser.add_argument("--steps-per-side", type=int, default=DEFAULTS["steps_per_side"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--t-max", type=float, default=DEFAULTS["t_max"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true", help="Recompute even if the cache exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    params = {
        "com_offset": args.com_offset,
        "gamma_low": args.gamma_low,
        "gamma_high": args.gamma_high,
        "steps_per_side": args.steps_per_side,
        "dt": args.dt,
        "t_max": args.t_max,
    }
    result = run(params, force=args.force, results_dir=args.results_dir)
    print(f"Data: {result.data_path}")
    print(f"Figure: {result.figure('main')}")
    print(f"Fixed COM offset: {result.summary['com_offset']:+.6f}")
    print(
        "Most stable slope angle: "
        f"gamma={result.summary['best_lambda_gamma_deg']:.6f} deg, "
        f"|lambda|max={result.summary['best_lambda']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
