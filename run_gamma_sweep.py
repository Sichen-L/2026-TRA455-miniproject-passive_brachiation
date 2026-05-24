"""Run and cache a slope-angle sweep at the most stable COM offset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

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


def _latest_com_result(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("com_sweep_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No com_sweep_*.npz files found in {results_dir}.")
    return candidates[-1]


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _write_latest_alias(output_path: Path, metadata_path: Path) -> None:
    latest_output = output_path.with_name("gamma_sweep_latest.npz")
    latest_metadata = output_path.with_name("gamma_sweep_latest.json")
    shutil.copy2(output_path, latest_output)
    shutil.copy2(metadata_path, latest_metadata)
    print(f"Wrote fixed alias: {latest_output}")
    print(f"Wrote fixed alias: {latest_metadata}")


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


def _model_at_com(com_offset: float) -> TwoLinkBrachiationModel:
    return TwoLinkBrachiationModel(
        parameters_with_symmetric_com_offset(com_offset, _base_params())
    )


def _row_arrays(
    result: ContinuationResult,
    com_offset: float,
    branch: str,
    period: int,
    dt: float,
    t_max: float,
) -> dict[str, np.ndarray]:
    support = np.zeros(2, dtype=float)

    def switch_policy(_t, _state, _support_point, _impact_point, _slope):
        return SwitchDecision.SWITCH

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
            "eigen_real": np.nan
            if point.eigenvalues is None
            else float(np.real(point.eigenvalues[0])),
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
                model=_model_at_com(com_offset),
                slope=slope,
                x=np.array([d_value], dtype=float),
                dt=dt,
                t_max=t_max,
                collision_mode=CollisionMode.FULL_GRAB_1D,
                initial_direction=-1.0,
                impact_direction=1.0,
                branch=branch,
                support_point=support,
                switch_policy=switch_policy,
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
    return {
        key: np.array([row[key] for row in rows], dtype=object)
        for key in rows[0].keys()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan slope angle while holding the COM offset at the most stable COM-sweep point.",
    )
    parser.add_argument("--com-result", type=Path, default=None, help="Path to com_sweep_*.npz.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--com-offset",
        type=float,
        default=None,
        help="Override the COM offset. By default, use the most stable COM-sweep offset.",
    )
    parser.add_argument("--gamma-low", type=float, default=25.0)
    parser.add_argument("--gamma-high", type=float, default=65.0)
    parser.add_argument("--steps-per-side", type=int, default=31)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--t-max", type=float, default=8.0)
    parser.add_argument("--continuation-tol", type=float, default=1e-6)
    parser.add_argument("--continuation-max-iter", type=int, default=10)
    parser.add_argument("--continuation-delta", type=float, default=1e-5)
    parser.add_argument("--continuation-damping", type=float, default=0.8)
    parser.add_argument("--adaptive-min-step", type=float, default=0.05)
    parser.add_argument("--adaptive-max-step", type=float, default=2.0)
    parser.add_argument("--adaptive-growth", type=float, default=1.35)
    parser.add_argument("--adaptive-shrink", type=float, default=0.5)
    parser.add_argument("--fold-tolerance", type=float, default=7.5e-2)
    parser.add_argument("--force", action="store_true", help="Recompute even if the cache exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    com_result_path = args.com_result or _latest_com_result(args.results_dir)
    com_data = np.load(com_result_path)
    com_summary = json.loads(str(com_data["summary_json"]))
    com_config = json.loads(str(com_data["config_json"]))

    com_offset = (
        float(com_summary["best_lambda_offset"])
        if args.com_offset is None
        else float(args.com_offset)
    )
    d_fixed = float(com_summary["d_fixed"])
    branch = str(com_summary["branch"])
    period = int(com_summary["period"])
    base_gamma_deg = float(com_config["gamma_deg"])

    if not args.gamma_low < base_gamma_deg < args.gamma_high:
        raise ValueError(
            f"Gamma range must contain the source gamma {base_gamma_deg}; "
            f"got [{args.gamma_low}, {args.gamma_high}]."
        )

    config = {
        "impact_event": "subdt_bisection_v1",
        "source_com_result": str(com_result_path),
        "source_com_hash": com_summary.get("hash", ""),
        "com_offset": com_offset,
        "d_fixed": d_fixed,
        "branch": branch,
        "period": period,
        "base_gamma_deg": base_gamma_deg,
        "gamma_low": args.gamma_low,
        "gamma_high": args.gamma_high,
        "steps_per_side": args.steps_per_side,
        "dt": args.dt,
        "t_max": args.t_max,
        "continuation_tol": args.continuation_tol,
        "continuation_max_iter": args.continuation_max_iter,
        "continuation_delta": args.continuation_delta,
        "continuation_damping": args.continuation_damping,
        "adaptive_min_step": args.adaptive_min_step,
        "adaptive_max_step": args.adaptive_max_step,
        "adaptive_growth": args.adaptive_growth,
        "adaptive_shrink": args.adaptive_shrink,
        "fold_tolerance": args.fold_tolerance,
    }
    run_hash = _config_hash(config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / f"gamma_sweep_{run_hash}.npz"
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and not args.force:
        print(f"Cache hit: {output_path}")
        if metadata_path.exists():
            _write_latest_alias(output_path, metadata_path)
        print("Use --force to recompute.")
        return 0

    support = np.zeros(2, dtype=float)

    def switch_policy(_t, _state, _support_point, _impact_point, _slope):
        return SwitchDecision.SWITCH

    def P_factory(gamma_deg: float):
        slope = Slope(gamma=np.deg2rad(gamma_deg))
        P_base = make_passive_brachiation_stride_map(
            model=_model_at_com(com_offset),
            slope=slope,
            dt=args.dt,
            t_max=args.t_max,
            collision_mode=CollisionMode.FULL_GRAB_1D,
            initial_direction=-1.0,
            impact_direction=1.0,
            branch=branch,
            support_point=support,
            switch_policy=switch_policy,
        )
        return P_base if period == 1 else make_iterated_stride_map(P_base, period)

    def feasibility_factory(_gamma_deg: float):
        return make_passive_brachiation_feasibility_check(_base_params(), dim=1)

    initial_step = max(abs(args.gamma_high - base_gamma_deg), abs(base_gamma_deg - args.gamma_low)) / max(args.steps_per_side - 1, 1)
    low_result = continue_fixed_point_branch_adaptive(
        P_factory=P_factory,
        start_parameter=base_gamma_deg,
        target_parameter=args.gamma_low,
        x0=np.array([d_fixed], dtype=float),
        dim=1,
        feasibility_factory=feasibility_factory,
        tol=args.continuation_tol,
        max_iter=args.continuation_max_iter,
        delta=args.continuation_delta,
        damping=args.continuation_damping,
        compute_stability=True,
        initial_step=initial_step,
        min_step=args.adaptive_min_step,
        max_step=args.adaptive_max_step,
        step_growth=args.adaptive_growth,
        step_shrink=args.adaptive_shrink,
        fold_tolerance=args.fold_tolerance,
    )
    high_result = continue_fixed_point_branch_adaptive(
        P_factory=P_factory,
        start_parameter=base_gamma_deg,
        target_parameter=args.gamma_high,
        x0=np.array([d_fixed], dtype=float),
        dim=1,
        feasibility_factory=feasibility_factory,
        tol=args.continuation_tol,
        max_iter=args.continuation_max_iter,
        delta=args.continuation_delta,
        damping=args.continuation_damping,
        compute_stability=True,
        initial_step=initial_step,
        min_step=args.adaptive_min_step,
        max_step=args.adaptive_max_step,
        step_growth=args.adaptive_growth,
        step_shrink=args.adaptive_shrink,
        fold_tolerance=args.fold_tolerance,
    )

    low_arrays = _row_arrays(low_result, com_offset, branch, period, args.dt, args.t_max)
    high_arrays = _row_arrays(high_result, com_offset, branch, period, args.dt, args.t_max)
    rows = []
    for arrays in (low_arrays, high_arrays):
        for i in range(len(arrays["gamma_deg"])):
            if arrays is high_arrays and abs(float(arrays["gamma_deg"][i]) - base_gamma_deg) < 1e-12:
                continue
            rows.append({key: arrays[key][i] for key in arrays})
    rows.sort(key=lambda row: float(row["gamma_deg"]))
    arrays = {key: np.array([row[key] for row in rows]) for key in rows[0].keys()}
    for key in ("gamma_deg", "d_primary", "d_next", "stride_plot", "residual_norm", "spectral_radius", "eigen_real", "max_elbow_distance", "min_elbow_distance", "parameter_step", "fold_indicator"):
        arrays[key] = arrays[key].astype(float)
    for key in ("converged", "stable", "legal", "fold_candidate"):
        arrays[key] = arrays[key].astype(bool)
    arrays["failure_reason"] = arrays["failure_reason"].astype("U256")

    finite = arrays["converged"] & arrays["legal"] & np.isfinite(arrays["spectral_radius"])
    best_index = int(np.where(finite)[0][np.argmin(arrays["spectral_radius"][finite])])
    summary = {
        "output_path": str(output_path),
        "hash": run_hash,
        "source_com_result": str(com_result_path),
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

    np.savez_compressed(
        output_path,
        config_json=np.array(json.dumps(config, sort_keys=True), dtype="U"),
        summary_json=np.array(json.dumps(summary, sort_keys=True), dtype="U"),
        **arrays,
    )
    metadata_path.write_text(
        json.dumps({"config": config, "summary": summary}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote: {output_path}")
    print(f"Wrote: {metadata_path}")
    _write_latest_alias(output_path, metadata_path)
    print(f"Fixed COM offset: {com_offset:+.6f}")
    print(
        "Most stable slope angle: "
        f"gamma={summary['best_lambda_gamma_deg']:.6f} deg, "
        f"|lambda|max={summary['best_lambda']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
