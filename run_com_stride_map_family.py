"""Compare one-step stride error maps while varying only COM offset."""

from __future__ import annotations

import argparse
import json
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
    evaluate_passive_brachiation_stride,
    find_fixed_points_1d,
    make_passive_brachiation_feasibility_check,
    make_passive_brachiation_stride_map,
    parameters_with_symmetric_com_offset,
    poincare_jacobian_eigenvalues_1d,
)


def _load_npz_json(path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    data = np.load(path)
    config = json.loads(str(data["config_json"]))
    summary = json.loads(str(data["summary_json"]))
    return config, summary, data


def _parse_offsets(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _base_params_with_mass(target: str, value: float) -> BrachiationParameters:
    m1 = 1.041
    m2 = 1.041
    if target == "m1":
        m1 = value
    elif target == "m2":
        m2 = value
    elif target == "both":
        m1 = value
        m2 = value
    else:
        raise ValueError("mass_target must be 'm1', 'm2', or 'both'.")
    return BrachiationParameters.uniform_links(
        m1=m1,
        m2=m2,
        l1=0.314,
        l2=0.314,
        damping1=0.0,
        damping2=0.0,
        gravity=9.81,
    )


def _switch_policy(_t, _state, _support_point, _impact_point, _slope):
    return SwitchDecision.SWITCH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw multiple one-step stride error maps in one figure while "
            "holding mass/slope fixed and changing only symmetric COM offset."
        ),
    )
    parser.add_argument("--mass-result", type=Path, default=Path("results/mass_sweep_latest.npz"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--offsets",
        type=_parse_offsets,
        default=(0.3333333333333333, -0.3475, -0.3525),
        help="Comma-separated COM offsets to compare.",
    )
    parser.add_argument(
        "--representative-roots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For each offset, plot one representative root instead of every root. "
            "This keeps the default figure to stable / near-critical / unstable examples."
        ),
    )
    parser.add_argument("--points", type=int, default=81)
    parser.add_argument("--error-half-width", type=float, default=0.035)
    parser.add_argument("--d-lower", type=float, default=0.05)
    parser.add_argument("--d-upper-fraction", type=float, default=0.995)
    parser.add_argument("--root-scan-points", type=int, default=90)
    parser.add_argument("--root-residual-tol", type=float, default=1e-6)
    parser.add_argument("--jacobian-delta", type=float, default=1e-5)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--output-prefix", type=str, default="com_stride_map_family")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    mass_config, mass_summary, _mass_data = _load_npz_json(args.mass_result)
    mass_target = str(mass_summary["mass_target"])
    mass_value = float(mass_summary["best_lambda_mass"])
    gamma_deg = float(mass_summary["gamma_deg"])
    branch = str(mass_summary["branch"])
    dt = float(args.dt if args.dt is not None else mass_config["dt"])
    t_max = float(args.t_max if args.t_max is not None else mass_config["t_max"])

    base_params = _base_params_with_mass(mass_target, mass_value)
    slope = Slope(gamma=np.deg2rad(gamma_deg))
    support = np.zeros(2, dtype=float)
    total_length = base_params.l1 + base_params.l2
    d_bounds = (
        float(args.d_lower),
        float(args.d_upper_fraction * total_length),
    )

    records: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for offset in args.offsets:
        params = parameters_with_symmetric_com_offset(float(offset), base_params)
        model = TwoLinkBrachiationModel(params)
        p_map = make_passive_brachiation_stride_map(
            model=model,
            slope=slope,
            dt=dt,
            t_max=t_max,
            collision_mode=CollisionMode.FULL_GRAB_1D,
            initial_direction=-1.0,
            impact_direction=1.0,
            branch=branch,
            support_point=support,
            switch_policy=_switch_policy,
        )
        feasibility = make_passive_brachiation_feasibility_check(params, dim=1)
        roots, _ = find_fixed_points_1d(
            p_map,
            bounds=d_bounds,
            num_scan_points=args.root_scan_points,
            feasibility_check=feasibility,
            residual_tol=args.root_residual_tol,
        )
        print(f"offset {offset:+.5f}: {len(roots)} root(s)")
        for root_idx, root in enumerate(roots):
            try:
                eig, _jac, rho = poincare_jacobian_eigenvalues_1d(
                    p_map,
                    root.x,
                    delta=args.jacobian_delta,
                    feasibility_check=feasibility,
                )
                eig_real = float(np.real(np.ravel(eig)[0]))
                rho = float(rho)
            except Exception:
                eig_real = float("nan")
                rho = float("nan")
            stable = bool(np.isfinite(rho) and rho < 1.0)
            d_star = float(root.x)
            error_grid = np.linspace(
                -args.error_half_width,
                args.error_half_width,
                args.points,
            )
            d_input = np.clip(d_star + error_grid, d_bounds[0], d_bounds[1])
            d_input = np.unique(np.r_[d_input, d_star])
            d_input.sort()
            input_error = d_input - d_star
            output_error = np.full_like(input_error, np.nan, dtype=float)
            valid = np.zeros_like(input_error, dtype=bool)
            for i, d_value in enumerate(d_input):
                try:
                    evaluation = evaluate_passive_brachiation_stride(
                        model=model,
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
                        stop_after_releases=1,
                    )
                    output_error[i] = float(evaluation.p_of_x[0]) - d_star
                    valid[i] = np.isfinite(output_error[i])
                except Exception:
                    pass

            record = {
                "offset": float(offset),
                "root_index": int(root_idx),
                "d_star": d_star,
                "rho": rho,
                "eigen_real": eig_real,
                "stable": stable,
                "valid_points": int(np.count_nonzero(valid)),
            }
            records.append(record)
            curves.append(
                {
                    **record,
                    "input_error": input_error,
                    "output_error": output_error,
                    "valid": valid,
                }
            )
            label = "stable" if stable else "unstable"
            print(
                f"  root {root_idx}: d*={d_star:.6f}, "
                f"rho={rho:.4f}, eig={eig_real:+.4f}, {label}"
            )

    if not curves:
        raise RuntimeError("No fixed-point roots found for the requested COM offsets.")

    if args.representative_roots:
        selected_curves: list[dict[str, Any]] = []
        for offset in args.offsets:
            here = [curve for curve in curves if abs(curve["offset"] - float(offset)) < 1e-12]
            if not here:
                continue
            offset_value = float(offset)
            if offset_value > 0.0:
                # Clearly stable reference: strongest contraction.
                chosen = min(here, key=lambda curve: curve["rho"])
            elif offset_value <= -0.35:
                # Clearly unstable reference: closest unstable root to |lambda|=1.
                unstable = [curve for curve in here if not curve["stable"]]
                chosen = min(unstable or here, key=lambda curve: abs(curve["rho"] - 1.0))
            else:
                # Near-critical reference: closest root to |lambda|=1 on either side.
                chosen = min(here, key=lambda curve: abs(curve["rho"] - 1.0))
            selected_curves.append(chosen)
        curves = selected_curves
        records = [
            {
                key: curve[key]
                for key in (
                    "offset",
                    "root_index",
                    "d_star",
                    "rho",
                    "eigen_real",
                    "stable",
                    "valid_points",
                )
            }
            for curve in curves
        ]

    fig, ax = plt.subplots(figsize=(9.5, 7))
    all_errors = []
    colors = plt.get_cmap("tab10")
    for idx, curve in enumerate(curves):
        valid = curve["valid"]
        if not np.any(valid):
            continue
        x = curve["input_error"][valid]
        y = curve["output_error"][valid]
        all_errors.extend(np.abs(x))
        all_errors.extend(np.abs(y))
        linestyle = "-" if curve["stable"] else "--"
        color = colors(idx % 10)
        ax.plot(
            x,
            y,
            linestyle=linestyle,
            linewidth=2.0,
            color=color,
            label=(
                f"x={curve['offset']:+.4f}, d*={curve['d_star']:.3f}, "
                f"rho={curve['rho']:.2f}, {'S' if curve['stable'] else 'U'}"
            ),
        )

    limit = 1.05 * max(max(all_errors), 1e-6)
    ax.plot(
        [-limit, limit],
        [-limit, limit],
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="no correction: e_{n+1}=e_n",
    )
    ax.axhline(0.0, color="0.45", linewidth=1.0)
    ax.axvline(0.0, color="0.45", linewidth=1.0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("input error e_n = d_n - d* [m]")
    ax.set_ylabel("output error e_{n+1} = P(d_n) - d* [m]")
    ax.set_title(
        "One-step stride error maps for different COM offsets\n"
        f"mass fixed: {mass_target}={mass_value:.4f} kg, gamma={gamma_deg:.1f} deg"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()

    output_npz = args.results_dir / f"{args.output_prefix}_latest.npz"
    output_json = args.results_dir / f"{args.output_prefix}_latest.json"
    output_png = args.results_dir / f"{args.output_prefix}_latest.png"
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["offsets"] = [float(value) for value in args.offsets]
    summary = {
        "mass_result": str(args.mass_result),
        "mass_target": mass_target,
        "mass_value": mass_value,
        "gamma_deg": gamma_deg,
        "branch": branch,
        "offsets": [float(value) for value in args.offsets],
        "root_count": len(records),
        "stable_root_count": int(sum(row["stable"] for row in records)),
        "unstable_root_count": int(sum(not row["stable"] for row in records)),
        "plot_path": str(output_png),
    }
    np.savez_compressed(
        output_npz,
        config_json=np.array(json.dumps(config, sort_keys=True), dtype="U"),
        summary_json=np.array(json.dumps(summary, sort_keys=True), dtype="U"),
        offset=np.array([row["offset"] for row in records], dtype=float),
        root_index=np.array([row["root_index"] for row in records], dtype=int),
        d_star=np.array([row["d_star"] for row in records], dtype=float),
        spectral_radius=np.array([row["rho"] for row in records], dtype=float),
        eigen_real=np.array([row["eigen_real"] for row in records], dtype=float),
        stable=np.array([row["stable"] for row in records], dtype=bool),
        valid_points=np.array([row["valid_points"] for row in records], dtype=int),
    )
    output_json.write_text(
        json.dumps({"config": config, "summary": summary, "roots": records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote: {output_npz}")
    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_png}")
    print(
        f"Roots: {summary['root_count']} total, "
        f"{summary['stable_root_count']} stable, "
        f"{summary['unstable_root_count']} unstable"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
