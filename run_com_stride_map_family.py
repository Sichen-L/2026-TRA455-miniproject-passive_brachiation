"""Part 4: one-step stride error maps (basin probe) for three COM offsets.

For a fixed slope and mass, this varies only the symmetric COM offset and draws
the one-step return-error map ``e_{n+1} = P(d* + e_n) - d*`` around each fixed
point.  By default three offsets are chosen from the cached COM sweep so the
figure shows a most-stable, a near-unstable-but-still-stable, and an unstable
example:

* most_stable            - stable row with the smallest spectral radius
* near_unstable_stable   - stable companion root at the mildest unstable offset
* unstable               - mildest unstable row (spectral radius just above 1)

Importable interface::

    from run_com_stride_map_family import run
    res = run()
    res.figure("main")
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
    evaluate_passive_brachiation_stride,
    find_fixed_points_1d,
    make_passive_brachiation_feasibility_check,
    make_passive_brachiation_stride_map,
    parameters_with_symmetric_com_offset,
    poincare_jacobian_eigenvalues_1d,
)
from report_cache import PartResult, cached_run
from report_setup import load_best_com_setup

PART = "com_stride_map_family"
ALGO_VERSION = "v2"

DEFAULTS: dict[str, Any] = {
    "offsets": None,  # None -> auto-pick stable / critical / unstable from the COM sweep
    "dt": 0.005,
    "t_max": 8.0,
    "points": 81,
    "error_half_width": 0.035,
    "d_lower": 0.05,
    "d_upper_fraction": 0.995,
    "root_scan_points": 90,
    "root_residual_tol": 1e-6,
    "jacobian_delta": 1e-5,
}


def _switch_policy(_t, _state, _support_point, _impact_point, _slope):
    return SwitchDecision.SWITCH


def _select_offsets(com_path: Path) -> list[dict[str, Any]]:
    data = np.load(com_path, allow_pickle=True)
    off = np.asarray(data["com_offset"], dtype=float)
    rho = np.asarray(data["spectral_radius"], dtype=float)
    base = (
        np.asarray(data["converged"], dtype=bool)
        & np.asarray(data["legal"], dtype=bool)
        & np.isfinite(rho)
    )
    stable = base & np.asarray(data["stable"], dtype=bool) & (rho < 1.0)
    specs: list[dict[str, Any]] = []
    if stable.any():
        stable_indices = np.where(stable)[0]
        specs.append({
            "regime": "most_stable",
            "offset": float(off[stable_indices[np.argmin(rho[stable_indices])]]),
        })
    unstable = base & (rho > 1.0)
    if unstable.any():
        unstable_indices = np.where(unstable)[0]
        mild_unstable_offset = float(off[unstable_indices[np.argmin(rho[unstable_indices])]])
        specs.append({"regime": "near_unstable_stable", "offset": mild_unstable_offset})
        specs.append({
            "regime": "unstable",
            "offset": mild_unstable_offset,
        })
    elif base.any():
        base_indices = np.where(base)[0]
        if stable.any():
            stable_indices = np.where(stable)[0]
            specs.append({
                "regime": "near_unstable_stable",
                "offset": float(off[stable_indices[np.argmax(rho[stable_indices])]]),
            })
        specs.append({
            "regime": "unstable",
            "offset": float(off[base_indices[np.argmax(rho[base_indices])]]),
        })
    return specs


def _pick_root(roots, rho_of, regime: str):
    scored = []
    for root in roots:
        try:
            rho = float(rho_of(root.x))
        except Exception:
            rho = float("nan")
        scored.append((root, rho))
    finite = [(root, rho) for root, rho in scored if np.isfinite(rho)]
    if not finite:
        return None
    stable = [item for item in finite if item[1] < 1.0]
    if regime in {"stable", "most_stable"}:
        return min(stable or finite, key=lambda item: item[1])
    if regime == "near_unstable_stable":
        return max(stable, key=lambda item: item[1]) if stable else min(finite, key=lambda item: abs(item[1] - 1.0))
    if regime == "unstable":
        unstable = [item for item in finite if item[1] > 1.0]
        return min(unstable or finite, key=lambda item: abs(item[1] - 1.0))
    return min(finite, key=lambda item: abs(item[1] - 1.0))


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base_params = BrachiationParameters.rod_point_mass(
        m1=cfg["m1"], m2=cfg["m2"], l1=cfg["l1"], l2=cfg["l2"],
        rod_mass_fraction=0.2, damping1=cfg["damping1"], damping2=cfg["damping2"], gravity=cfg["gravity"],
    )
    slope = Slope(gamma=np.deg2rad(cfg["gamma_deg"]))
    branch = str(cfg["branch"])
    support = np.zeros(2, dtype=float)
    total_length = base_params.l1 + base_params.l2
    d_bounds = (float(cfg["d_lower"]), float(cfg["d_upper_fraction"] * total_length))
    error_grid = np.linspace(-cfg["error_half_width"], cfg["error_half_width"], int(cfg["points"]))

    regimes: list[str] = []
    offsets: list[float] = []
    d_stars: list[float] = []
    rhos: list[float] = []
    stables: list[bool] = []
    output_errors: list[np.ndarray] = []

    for spec in cfg["offset_specs"]:
        offset = float(spec["offset"])
        params = parameters_with_symmetric_com_offset(offset, base_params)
        model = TwoLinkBrachiationModel(params)
        p_map = make_passive_brachiation_stride_map(
            model=model, slope=slope, dt=cfg["dt"], t_max=cfg["t_max"],
            collision_mode=CollisionMode.FULL_GRAB_1D, initial_direction=-1.0, impact_direction=1.0,
            branch=branch, support_point=support, switch_policy=_switch_policy,
        )
        feasibility = make_passive_brachiation_feasibility_check(params, dim=1)
        roots, _ = find_fixed_points_1d(
            p_map, bounds=d_bounds, num_scan_points=cfg["root_scan_points"],
            feasibility_check=feasibility, residual_tol=cfg["root_residual_tol"],
        )

        def rho_of(x):
            _, _, rho = poincare_jacobian_eigenvalues_1d(
                p_map, x, delta=cfg["jacobian_delta"], feasibility_check=feasibility
            )
            return rho

        picked = _pick_root(roots, rho_of, spec["regime"])
        if picked is None:
            continue
        root, rho = picked
        d_star = float(root.x)
        d_input = np.clip(d_star + error_grid, d_bounds[0], d_bounds[1])
        out_err = np.full(error_grid.shape, np.nan, dtype=float)
        for i, d_value in enumerate(d_input):
            try:
                evaluation = evaluate_passive_brachiation_stride(
                    model=model, slope=slope, x=np.array([d_value], dtype=float), dt=cfg["dt"], t_max=cfg["t_max"],
                    collision_mode=CollisionMode.FULL_GRAB_1D, initial_direction=-1.0, impact_direction=1.0,
                    branch=branch, support_point=support, switch_policy=_switch_policy, stop_after_releases=1,
                )
                out_err[i] = float(evaluation.p_of_x[0]) - d_star
            except Exception:
                pass

        regimes.append(spec["regime"])
        offsets.append(offset)
        d_stars.append(d_star)
        rhos.append(float(rho))
        stables.append(bool(np.isfinite(rho) and rho < 1.0))
        output_errors.append(out_err)

    if not output_errors:
        raise RuntimeError("No fixed-point roots found for the requested COM offsets.")

    arrays = {
        "regime": np.array(regimes, dtype="U16"),
        "offset": np.array(offsets, dtype=float),
        "d_star": np.array(d_stars, dtype=float),
        "spectral_radius": np.array(rhos, dtype=float),
        "stable": np.array(stables, dtype=bool),
        "input_error": error_grid,
        "output_error": np.vstack(output_errors),
    }
    summary = {
        "gamma_deg": cfg["gamma_deg"],
        "branch": branch,
        "curve_count": len(regimes),
        "regimes": regimes,
        "offsets": offsets,
        "d_stars": d_stars,
        "spectral_radii": rhos,
    }
    return arrays, summary


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    x = arrays["input_error"].astype(float)
    output_error = arrays["output_error"]
    regimes = arrays["regime"]
    offsets = arrays["offset"].astype(float)
    d_stars = arrays["d_star"].astype(float)
    rhos = arrays["spectral_radius"].astype(float)
    stables = arrays["stable"].astype(bool)

    colors = {
        "most_stable": "tab:green",
        "near_unstable_stable": "tab:orange",
        "unstable": "tab:red",
        "custom": "tab:blue",
    }
    fig, ax = plt.subplots(figsize=(9.5, 7))
    all_errors = [cfg["error_half_width"]]
    for i in range(len(regimes)):
        y = output_error[i]
        valid = np.isfinite(y)
        if not np.any(valid):
            continue
        all_errors.extend(np.abs(y[valid]))
        regime = str(regimes[i])
        ax.plot(
            x[valid], y[valid],
            linestyle="-" if stables[i] else "--",
            linewidth=2.0,
            color=colors.get(regime, "tab:blue"),
            label=f"{regime}: x={offsets[i]:+.4f}, d*={d_stars[i]:.3f}, rho={rhos[i]:.2f}",
        )

    limit = 1.05 * max(max(all_errors), 1e-6)
    ax.plot([-limit, limit], [-limit, limit], color="black", linestyle=":", linewidth=1.2, label="no correction: e_{n+1}=e_n")
    ax.axhline(0.0, color="0.45", linewidth=1.0)
    ax.axvline(0.0, color="0.45", linewidth=1.0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("input error e_n = d_n - d* [m]")
    ax.set_ylabel("output error e_{n+1} = P(d_n) - d* [m]")
    ax.set_title(
        "One-step stride error maps for most-stable / near-unstable-stable / unstable COM offsets\n"
        f"gamma={summary['gamma_deg']:.1f} deg, branch={summary['branch']}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
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
    bp = setup.base_params

    if cfg["offsets"] is None:
        offset_specs = _select_offsets(Path(results_dir) / "com_sweep_latest.npz")
    else:
        offset_specs = [{"regime": "custom", "offset": float(value)} for value in cfg["offsets"]]

    cfg.update(
        {
            "_algo": ALGO_VERSION,
            "offset_specs": offset_specs,
            "gamma_deg": setup.gamma_deg,
            "branch": setup.branch,
            "m1": bp.m1, "m2": bp.m2, "l1": bp.l1, "l2": bp.l2,
            "damping1": bp.damping1, "damping2": bp.damping2, "gravity": bp.gravity,
            "source_com": setup.source,
        }
    )
    cfg.pop("offsets", None)
    return cached_run(
        part=PART, config=cfg, compute=_compute, plot=_plot,
        results_dir=results_dir, force=force, verbose=verbose,
    )


def _parse_offsets(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw one-step stride error maps for most-stable/near-unstable-stable/unstable COM offsets.")
    parser.add_argument("--offsets", type=_parse_offsets, default=None, help="Comma-separated COM offsets; default auto-picks 3 regimes.")
    parser.add_argument("--points", type=int, default=DEFAULTS["points"])
    parser.add_argument("--error-half-width", type=float, default=DEFAULTS["error_half_width"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--t-max", type=float, default=DEFAULTS["t_max"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    params = {
        "offsets": args.offsets,
        "points": args.points,
        "error_half_width": args.error_half_width,
        "dt": args.dt,
        "t_max": args.t_max,
    }
    result = run(params, force=args.force, results_dir=args.results_dir)
    print(f"Data: {result.data_path}")
    print(f"Figure: {result.figure('main')}")
    print(f"Curves: {result.summary['curve_count']} ({', '.join(result.summary['regimes'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
