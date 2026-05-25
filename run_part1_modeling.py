"""Part 1: physical model overview and generic passive state portrait.

This part documents the ordinary underactuated two-link brachiation model by
rolling out one representative passive trajectory.  The figure shows the
workspace path, phase portraits, and energy conservation/loss across switching
events.

Importable interface::

    from run_part1_modeling import run
    res = run(); res.figure("main")
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
    samples_to_arrays,
    simulate,
)
from report_cache import PartResult, cached_run


PART = "part1_modeling"
ALGO_VERSION = "v1"

DEFAULTS: dict[str, Any] = {
    "gamma_deg": 45.0,
    "duration": 3.0,
    "dt": 0.005,
    "q0": [-0.8, -0.8],
    "qd0": [0.1, 0.5],
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
    initial_state = BrachiationState(
        q=np.asarray(cfg["q0"], dtype=float),
        qd=np.asarray(cfg["qd0"], dtype=float),
        support_index=0,
    )
    samples = simulate(
        model=model,
        slope=slope,
        initial_state=initial_state,
        initial_support_point=np.zeros(2, dtype=float),
        duration=cfg["duration"],
        dt=cfg["dt"],
        switch_policy=lambda *_args: SwitchDecision.SWITCH,
        collision_mode=CollisionMode.FULL_GRAB_1D,
    )
    history = samples_to_arrays(samples, slope=slope)
    phase = history["phase"]
    arrays = {
        "time": history["times"],
        "q": history["q"],
        "qd": history["qd"],
        "free_yz": history["frees"],
        "elbow_yz": history["elbows"],
        "support_yz": history["supports"],
        "free_dist": history["free_dist"],
        "elbow_dist": history["elbow_dist"],
        "kinetic": history["kinetic_energy"],
        "potential": history["potential_energy"],
        "total": history["total_energy"],
        "is_impact": (phase == "impact"),
        "is_release": (phase == "release"),
    }
    summary = {
        "num_samples": int(len(samples)),
        "num_impacts": int(np.count_nonzero(arrays["is_impact"])),
        "num_releases": int(np.count_nonzero(arrays["is_release"])),
        "initial_total_energy": float(arrays["total"][0]),
        "final_total_energy": float(arrays["total"][-1]),
        "model": "rod_point_mass two-link underactuated brachiation",
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

    ax = axes[0, 0]
    ax.plot(free[:, 0], free[:, 1], color="tab:blue", label="free end")
    ax.plot(elbow[:, 0], elbow[:, 1], color="tab:orange", alpha=0.8, label="elbow")
    ax.scatter(support[:, 0], support[:, 1], s=8, color="black", label="support")
    y_grid = np.linspace(np.min(free[:, 0]) - 0.1, np.max(free[:, 0]) + 0.1, 200)
    ax.plot(y_grid, -y_grid * np.tan(np.deg2rad(cfg["gamma_deg"])), "k--", linewidth=1, label="slope")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("workspace trajectory")
    ax.legend(loc="best")

    axes[0, 1].plot(q[:, 0], qd[:, 0], color="tab:green", label="q1")
    axes[0, 1].plot(q[:, 1], qd[:, 1], color="tab:purple", label="q2")
    axes[0, 1].set_xlabel("angle [rad]")
    axes[0, 1].set_ylabel("angular velocity [rad/s]")
    axes[0, 1].set_title("phase portrait")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(t, q[:, 0], label="q1")
    axes[1, 0].plot(t, q[:, 1], label="q2")
    axes[1, 0].plot(t, qd[:, 0], "--", label="q1_dot")
    axes[1, 0].plot(t, qd[:, 1], "--", label="q2_dot")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("state")
    axes[1, 0].set_title("state time history")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(t, arrays["kinetic"], label="kinetic")
    axes[1, 1].plot(t, arrays["potential"], label="potential")
    axes[1, 1].plot(t, arrays["total"], label="total")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("energy [J]")
    axes[1, 1].set_title("energy")
    axes[1, 1].legend(loc="best")

    fig.suptitle("Part 1: underactuated passive brachiation model")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return {"main": fig}


def run(params=None, *, force=False, results_dir: Path | str = Path("results"), verbose=True) -> PartResult:
    cfg = {**DEFAULTS, **(params or {}), "_algo": ALGO_VERSION}
    return cached_run(part=PART, config=cfg, compute=_compute, plot=_plot, results_dir=results_dir, force=force, verbose=verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run part 1 model overview figure.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma_deg"])
    parser.add_argument("--duration", type=float, default=DEFAULTS["duration"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run({"gamma_deg": args.gamma, "duration": args.duration}, force=args.force)
    print(f"Figure: {result.figure('main')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
