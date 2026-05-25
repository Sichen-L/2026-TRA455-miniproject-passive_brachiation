"""Part 2: find and visualize one passive steady gait.

For a fixed parameter set, this script searches for a one-step stride fixed
point, simulates the resulting gait for several releases, and plots the state
portrait and energy history.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from passive_brachiation import (
    BrachiationParameters,
    BrachiationState,
    CollisionMode,
    Slope,
    SwitchDecision,
    TwoLinkBrachiationModel,
    ik_from_stride_distance,
    release_stride_distances,
    samples_to_arrays,
    scan_stride_fixed_points,
    simulate,
)
from report_cache import PartResult, cached_run


PART = "part2_steady_state"
ALGO_VERSION = "v1"

DEFAULTS: dict[str, Any] = {
    "gamma_deg": 45.0,
    "dt": 0.005,
    "t_max": 8.0,
    "stop_after_releases": 10,
    "d_scan_points": 30,
    "d_lower": 0.05,
    "d_upper_fraction": 0.95,
    "branches": ["positive", "negative"],
    "periods": [1, 2],
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


def _params(cfg: dict[str, Any]) -> BrachiationParameters:
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


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    params = _params(cfg)
    model = TwoLinkBrachiationModel(params)
    slope = Slope(gamma=np.deg2rad(cfg["gamma_deg"]))
    support0 = np.zeros(2, dtype=float)
    search = scan_stride_fixed_points(
        model=model,
        slope=slope,
        initial_support_point=support0,
        dt=cfg["dt"],
        t_max=cfg["t_max"],
        d_bounds=(cfg["d_lower"], cfg["d_upper_fraction"] * (params.l1 + params.l2)),
        d_scan_points=cfg["d_scan_points"],
        branches=tuple(cfg["branches"]),
        periods=tuple(cfg["periods"]),
        initial_direction=-1.0,
        impact_direction=1.0,
        switch_policy=lambda *_args: SwitchDecision.SWITCH,
        collision_mode=CollisionMode.FULL_GRAB_1D,
    )
    trial = search.selected_trial
    if trial is None:
        raise RuntimeError("No steady fixed gait found.")

    q0 = ik_from_stride_distance(
        trial.d,
        slope=slope,
        parameters=params,
        direction=-1.0,
        branch=trial.branch,
    )
    samples = simulate(
        model=model,
        slope=slope,
        initial_state=BrachiationState(q=q0, qd=np.zeros(2), support_index=0),
        initial_support_point=support0,
        duration=cfg["t_max"],
        dt=cfg["dt"],
        switch_policy=lambda *_args: SwitchDecision.SWITCH,
        collision_mode=CollisionMode.FULL_GRAB_1D,
        stop_after_releases=cfg["stop_after_releases"],
    )
    history = samples_to_arrays(samples, slope=slope)
    release_stride = release_stride_distances(samples, slope=slope, support_origin=support0, direction=1.0)
    arrays = {
        "time": history["times"],
        "q": history["q"],
        "qd": history["qd"],
        "free_yz": history["frees"],
        "elbow_yz": history["elbows"],
        "support_yz": history["supports"],
        "kinetic": history["kinetic_energy"],
        "potential": history["potential_energy"],
        "total": history["total_energy"],
        "release_stride": release_stride,
        "d_fixed": np.array([trial.d], dtype=float),
        "q0": q0.astype(float),
        "eigenvalues": np.asarray(trial.eigenvalues, dtype=complex),
    }
    summary = {
        "d_fixed": float(trial.d),
        "branch": trial.branch,
        "period": int(trial.period),
        "spectral_radius": float(trial.spectral_radius),
        "stable": bool(trial.stable),
        "legal": bool(trial.legality.legal),
        "validation_error": float(trial.validation_error),
        "release_stride_mean": float(np.mean(release_stride)) if release_stride.size else float("nan"),
    }
    return arrays, summary


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    t = arrays["time"]
    q = arrays["q"]
    qd = arrays["qd"]
    free = arrays["free_yz"]
    elbow = arrays["elbow_yz"]
    support = arrays["support_yz"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(free[:, 0], free[:, 1], color="tab:blue", label="free end")
    axes[0, 0].plot(elbow[:, 0], elbow[:, 1], color="tab:orange", alpha=0.75, label="elbow")
    axes[0, 0].scatter(support[:, 0], support[:, 1], s=8, color="black", label="support")
    y_grid = np.linspace(np.min(free[:, 0]) - 0.1, np.max(free[:, 0]) + 0.1, 200)
    axes[0, 0].plot(y_grid, -y_grid * np.tan(np.deg2rad(cfg["gamma_deg"])), "k--", linewidth=1, label="slope")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_title("steady gait geometry")
    axes[0, 0].set_xlabel("y [m]")
    axes[0, 0].set_ylabel("z [m]")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(q[:, 0], qd[:, 0], label="q1")
    axes[0, 1].plot(q[:, 1], qd[:, 1], label="q2")
    axes[0, 1].set_xlabel("angle [rad]")
    axes[0, 1].set_ylabel("angular velocity [rad/s]")
    axes[0, 1].set_title("state portrait")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(t, arrays["kinetic"], label="kinetic")
    axes[1, 0].plot(t, arrays["potential"], label="potential")
    axes[1, 0].plot(t, arrays["total"], label="total")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("energy [J]")
    axes[1, 0].set_title("energy over repeated steps")
    axes[1, 0].legend(loc="best")

    stride = arrays["release_stride"]
    stride_error = stride - summary["d_fixed"]
    axes[1, 1].plot(np.arange(1, len(stride_error) + 1), stride_error, "o-", label="release stride - fixed d")
    axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=1, label="fixed point")
    axes[1, 1].set_xlabel("release index")
    axes[1, 1].set_ylabel("stride error [m]")
    axes[1, 1].set_title("fixed-point stride repeatability error")
    axes[1, 1].legend(loc="best")

    fig.suptitle(
        f"Part 2: passive steady gait, d={summary['d_fixed']:.4f} m, "
        f"rho={summary['spectral_radius']:.3f}"
    )
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return {"main": fig}


def run(params=None, *, force=False, results_dir: Path | str = Path("results"), verbose=True) -> PartResult:
    cfg = {**DEFAULTS, **(params or {}), "_algo": ALGO_VERSION}
    return cached_run(part=PART, config=cfg, compute=_compute, plot=_plot, results_dir=results_dir, force=force, verbose=verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run part 2 steady-state gait search.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma_deg"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run({"gamma_deg": args.gamma}, force=args.force)
    print(f"Figure: {result.figure('main')}")
    print(f"d={result.summary['d_fixed']:.6f}, rho={result.summary['spectral_radius']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
