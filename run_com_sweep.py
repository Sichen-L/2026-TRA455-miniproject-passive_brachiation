"""Run and cache an adaptive symmetric-COM continuation sweep."""

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
    Slope,
    SwitchDecision,
    TwoLinkBrachiationModel,
    run_symmetric_com_continuation,
    scan_stride_fixed_points,
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _write_latest_alias(output_path: Path, metadata_path: Path) -> None:
    latest_output = output_path.with_name("com_sweep_latest.npz")
    latest_metadata = output_path.with_name("com_sweep_latest.json")
    shutil.copy2(output_path, latest_output)
    shutil.copy2(metadata_path, latest_metadata)
    print(f"Wrote fixed alias: {latest_output}")
    print(f"Wrote fixed alias: {latest_metadata}")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an adaptive symmetric COM-offset continuation and cache the result.",
    )
    parser.add_argument("--gamma-deg", type=float, default=45.0)
    parser.add_argument("--offset-low", type=float, default=-0.5)
    parser.add_argument("--offset-high", type=float, default=0.5)
    parser.add_argument("--steps-per-side", type=int, default=31)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--t-max", type=float, default=8.0)
    parser.add_argument("--continuation-tol", type=float, default=1e-6)
    parser.add_argument("--continuation-max-iter", type=int, default=10)
    parser.add_argument("--continuation-delta", type=float, default=1e-5)
    parser.add_argument("--continuation-damping", type=float, default=0.8)
    parser.add_argument("--adaptive-min-step", type=float, default=5e-4)
    parser.add_argument("--adaptive-max-step", type=float, default=0.04)
    parser.add_argument("--adaptive-growth", type=float, default=1.35)
    parser.add_argument("--adaptive-shrink", type=float, default=0.5)
    parser.add_argument("--fold-tolerance", type=float, default=7.5e-2)
    parser.add_argument("--d-scan-points", type=int, default=30)
    parser.add_argument("--d-lower", type=float, default=0.05)
    parser.add_argument("--d-upper-fraction", type=float, default=0.95)
    parser.add_argument("--branches", type=_parse_csv_strings, default=("positive", "negative"))
    parser.add_argument("--periods", type=_parse_csv_ints, default=(1, 2))
    parser.add_argument("--d-fixed", type=float, default=None)
    parser.add_argument("--branch", type=str, default=None)
    parser.add_argument("--period", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true", help="Recompute even if the cache exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    params = BrachiationParameters.uniform_links(
        m1=1.041,
        m2=1.041,
        l1=0.314,
        l2=0.314,
        damping1=0.0,
        damping2=0.0,
        gravity=9.81,
    )
    config = {
        "impact_event": "subdt_bisection_v1",
        "gamma_deg": args.gamma_deg,
        "offset_low": args.offset_low,
        "offset_high": args.offset_high,
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
        "d_scan_points": args.d_scan_points,
        "d_lower": args.d_lower,
        "d_upper_fraction": args.d_upper_fraction,
        "branches": list(args.branches),
        "periods": list(args.periods),
        "d_fixed": args.d_fixed,
        "branch": args.branch,
        "period": args.period,
        "params": {
            "m1": params.m1,
            "m2": params.m2,
            "l1": params.l1,
            "l2": params.l2,
            "lc1": params.lc1,
            "lc2": params.lc2,
            "I1": params.I1,
            "I2": params.I2,
            "gravity": params.gravity,
        },
    }
    run_hash = _config_hash(config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / f"com_sweep_{run_hash}.npz"
    metadata_path = output_path.with_suffix(".json")

    if output_path.exists() and not args.force:
        print(f"Cache hit: {output_path}")
        if metadata_path.exists():
            _write_latest_alias(output_path, metadata_path)
        print("Use --force to recompute.")
        return 0

    model = TwoLinkBrachiationModel(params)
    slope = Slope(gamma=np.deg2rad(args.gamma_deg))
    support = np.zeros(2, dtype=float)

    def switch_policy(_t, _state, _support_point, _impact_point, _slope):
        return SwitchDecision.SWITCH

    if args.d_fixed is None:
        search = scan_stride_fixed_points(
            model=model,
            slope=slope,
            initial_support_point=support,
            dt=args.dt,
            t_max=args.t_max,
            d_bounds=(args.d_lower, args.d_upper_fraction * (params.l1 + params.l2)),
            d_scan_points=args.d_scan_points,
            branches=args.branches,
            periods=args.periods,
            initial_direction=-1.0,
            impact_direction=1.0,
            switch_policy=switch_policy,
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
        if args.branch is None or args.period is None:
            raise ValueError("--branch and --period are required when --d-fixed is supplied.")
        d_fixed = float(args.d_fixed)
        branch = str(args.branch)
        period = int(args.period)
        q2_fixed = float("nan")

    initial_step = abs(args.offset_high) / max(args.steps_per_side - 1, 1)
    continuation = run_symmetric_com_continuation(
        base_params=params,
        slope=slope,
        d_fixed=d_fixed,
        branch=branch,
        period=period,
        offset_low=args.offset_low,
        offset_high=args.offset_high,
        n_steps_per_side=args.steps_per_side,
        dt=args.dt,
        t_max=args.t_max,
        initial_support_point=support,
        initial_direction=-1.0,
        impact_direction=1.0,
        switch_policy=switch_policy,
        collision_mode=CollisionMode.FULL_GRAB_1D,
        continuation_tol=args.continuation_tol,
        continuation_max_iter=args.continuation_max_iter,
        continuation_delta=args.continuation_delta,
        continuation_damping=args.continuation_damping,
        adaptive_steps=True,
        adaptive_initial_step=initial_step,
        adaptive_min_step=args.adaptive_min_step,
        adaptive_max_step=args.adaptive_max_step,
        adaptive_step_growth=args.adaptive_growth,
        adaptive_step_shrink=args.adaptive_shrink,
        fold_tolerance=args.fold_tolerance,
    )

    rows = continuation.rows
    arrays = _rows_to_arrays(rows)
    finite = arrays["converged"] & arrays["legal"] & np.isfinite(arrays["spectral_radius"])
    best_index = int(np.where(finite)[0][np.argmin(arrays["spectral_radius"][finite])])
    summary = {
        "output_path": str(output_path),
        "hash": run_hash,
        "d_fixed": d_fixed,
        "q2_fixed": q2_fixed,
        "branch": branch,
        "period": period,
        "row_count": len(rows),
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
    print(
        "Most stable COM offset: "
        f"x={summary['best_lambda_offset']:+.6f}, "
        f"|lambda|max={summary['best_lambda']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
