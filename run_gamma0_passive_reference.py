"""Part 8 prelude: passive gamma=0 reference rollout before control.

The part-8 controllers track the gamma=0 minimum-energy free-initial-velocity
solution from part 7.  This script first visualizes that passive reference by
rolling it out from release to the pre-impact instant with zero control.

Outputs:

* a PNG with state histories, kinetic/total energy, endpoint distances, and
  workspace geometry;
* a GIF animation of the same horizontal-plane swing.

Importable interface::

    from run_gamma0_passive_reference import run, animation_path
    res = run(); res.figure("main"); animation_path(res)
"""

from __future__ import annotations

import argparse
import shutil
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
    parameters_with_symmetric_com_offset,
    samples_to_arrays,
    simulate,
)
from passive_brachiation.simulation import SimPhase
from report_cache import PartResult, cached_run
from report_setup import load_best_com_setup
from run_free_initial_velocity_gamma_sweep import FreeVelocityProblem, scan_gamma, select_candidate

PART = "gamma0_passive_reference"
ALGO_VERSION = "v1"

DEFAULTS: dict[str, Any] = {
    "gamma": 0.0,
    "dt": 0.005,
    "t_max": 8.0,
    "max_speed": 5.0,
    "speed_points": 41,
    "theta_points": 37,
    "theta_min_deg": 0.0,
    "theta_max_deg": 180.0,
    "min_normal_speed": 0.0,
    "allow_unstable_reference": False,
    "allow_illegal_reference": False,
    "root_xtol": 1e-10,
    "root_rtol": 1e-10,
    "root_maxiter": 60,
    "gif_fps": 24,
    "gif_frames": 90,
}


def compute_gamma_candidate(cfg: dict[str, Any], params: BrachiationParameters, d_target: float, branch: str):
    theta_values = np.linspace(np.deg2rad(cfg["theta_min_deg"]), np.deg2rad(cfg["theta_max_deg"]), int(cfg["theta_points"]))
    problem = FreeVelocityProblem(
        cfg["gamma"], params, d_target, branch, cfg["dt"], cfg["t_max"], cfg["min_normal_speed"]
    )
    candidates = scan_gamma(
        problem,
        theta_values,
        (0.0, cfg["max_speed"]),
        cfg["speed_points"],
        cfg["root_xtol"],
        cfg["root_rtol"],
        cfg["root_maxiter"],
    )
    selected = select_candidate(candidates, cfg["allow_unstable_reference"], cfg["allow_illegal_reference"])
    if selected is None:
        raise RuntimeError(f"No gamma={cfg['gamma']:.3f} free-initial-velocity candidate from {len(candidates)} roots.")
    return selected, candidates


def passive_preimpact_samples(model, slope, candidate, dt, t_max) -> list:
    initial_state = BrachiationState(
        q=np.asarray(candidate.q, dtype=float),
        qd=np.asarray(candidate.qd, dtype=float),
        support_index=0,
    )
    samples = simulate(
        model=model,
        slope=slope,
        initial_state=initial_state,
        initial_support_point=np.zeros(2, dtype=float),
        duration=t_max,
        dt=dt,
        switch_policy=lambda *_args: SwitchDecision.SWITCH,
        collision_mode=CollisionMode.FULL_GRAB_1D,
        stop_after_releases=1,
    )
    impact_index = next((i for i, sample in enumerate(samples) if sample.phase == SimPhase.IMPACT), None)
    return samples[: impact_index + 1] if impact_index is not None else samples


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base = BrachiationParameters.rod_point_mass(
        m1=cfg["m1"],
        m2=cfg["m2"],
        l1=cfg["l1"],
        l2=cfg["l2"],
        rod_mass_fraction=0.2,
        damping1=cfg["damping1"],
        damping2=cfg["damping2"],
        gravity=cfg["gravity"],
    )
    params = parameters_with_symmetric_com_offset(cfg["com_offset"], base)
    branch = str(cfg["branch"])
    model = TwoLinkBrachiationModel(params)
    slope = Slope(gamma=np.deg2rad(cfg["gamma"]))

    candidate, candidates = compute_gamma_candidate(cfg, params, cfg["d_target"], branch)
    samples = passive_preimpact_samples(model, slope, candidate, cfg["dt"], cfg["t_max"])
    history = samples_to_arrays(samples, slope=slope)

    arrays = {
        "time": np.asarray(history["times"], dtype=float),
        "q": np.asarray(history["q"], dtype=float),
        "qd": np.asarray(history["qd"], dtype=float),
        "support": np.asarray(history["supports"], dtype=float),
        "elbow": np.asarray(history["elbows"], dtype=float),
        "free": np.asarray(history["frees"], dtype=float),
        "kinetic": np.asarray(history["kinetic_energy"], dtype=float),
        "potential": np.asarray(history["potential_energy"], dtype=float),
        "total": np.asarray(history["total_energy"], dtype=float),
        "free_dist": np.asarray(history["free_dist"], dtype=float),
        "elbow_dist": np.asarray(history["elbow_dist"], dtype=float),
        "phase": np.asarray(history["phase"], dtype="U16"),
    }
    summary = {
        "gamma": cfg["gamma"],
        "d_target": cfg["d_target"],
        "com_offset": cfg["com_offset"],
        "branch": branch,
        "candidate_count": len(candidates),
        "reference_energy_J": float(candidate.energy),
        "theta_deg": float(candidate.theta_deg),
        "speed": float(candidate.speed),
        "v_t": float(candidate.v_t),
        "v_n": float(candidate.v_n),
        "spectral_radius": None if candidate.spectral_radius is None else float(candidate.spectral_radius),
        "legal": bool(candidate.legal),
        "stable": bool(candidate.stable),
        "duration_s": float(arrays["time"][-1] - arrays["time"][0]),
        "kinetic_initial_J": float(arrays["kinetic"][0]),
        "kinetic_final_J": float(arrays["kinetic"][-1]),
        "total_initial_J": float(arrays["total"][0]),
        "total_final_J": float(arrays["total"][-1]),
        "source_com": cfg.get("source_com", ""),
    }
    return arrays, summary


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    t = arrays["time"]
    q = arrays["q"]
    qd = arrays["qd"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(t, np.rad2deg(q[:, 0]), label="q1")
    axes[0, 0].plot(t, np.rad2deg(q[:, 1]), label="q2")
    axes[0, 0].set_ylabel("angle [deg]")
    axes[0, 0].set_title("joint angles")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, qd[:, 0], label="qdot1")
    axes[0, 1].plot(t, qd[:, 1], label="qdot2")
    axes[0, 1].set_ylabel("angular speed [rad/s]")
    axes[0, 1].set_title("joint velocities")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, arrays["kinetic"], color="tab:blue", label="kinetic")
    axes[1, 0].plot(t, arrays["total"], color="tab:green", linestyle="--", label="total")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("energy [J]")
    axes[1, 0].set_title("passive energy")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(arrays["support"][:, 0], arrays["support"][:, 1], "o", color="black", markersize=4, label="support")
    axes[1, 1].plot(arrays["elbow"][:, 0], arrays["elbow"][:, 1], color="tab:orange", label="elbow")
    axes[1, 1].plot(arrays["free"][:, 0], arrays["free"][:, 1], color="tab:purple", label="free end")
    y_min = float(np.min([arrays["support"][:, 0].min(), arrays["elbow"][:, 0].min(), arrays["free"][:, 0].min()]))
    y_max = float(np.max([arrays["support"][:, 0].max(), arrays["elbow"][:, 0].max(), arrays["free"][:, 0].max()]))
    axes[1, 1].plot([y_min - 0.05, y_max + 0.05], [0.0, 0.0], color="0.35", linestyle="--", label="gamma=0 plane")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].set_xlabel("y [m]")
    axes[1, 1].set_ylabel("z [m]")
    axes[1, 1].set_title("workspace rollout")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"Gamma=0 passive reference before control: "
        f"K0={summary['kinetic_initial_J']:.3g} J, theta={summary['theta_deg']:.1f} deg, speed={summary['speed']:.3g} m/s"
    )
    fig.tight_layout()
    return {"main": fig}


def animation_path(result: PartResult | None = None, *, results_dir: Path | str = Path("results"), latest: bool = True) -> Path:
    results_dir = Path(results_dir)
    if result is None or latest:
        return results_dir / f"{PART}_latest__motion.gif"
    return results_dir / f"{PART}_{result.hash}__motion.gif"


def _make_animation(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    time = arrays["time"]
    support = arrays["support"]
    elbow = arrays["elbow"]
    free = arrays["free"]
    kinetic = arrays["kinetic"]
    frame_count = min(int(cfg["gif_frames"]), len(time))
    frame_indices = np.unique(np.linspace(0, len(time) - 1, frame_count, dtype=int))

    all_y = np.concatenate((support[:, 0], elbow[:, 0], free[:, 0]))
    all_z = np.concatenate((support[:, 1], elbow[:, 1], free[:, 1], np.array([0.0])))
    pad = 0.08
    y_limits = (float(np.min(all_y) - pad), float(np.max(all_y) + pad))
    z_limits = (float(np.min(all_z) - pad), float(np.max(all_z) + pad))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax_motion, ax_energy = axes
    ax_motion.set_xlim(*y_limits)
    ax_motion.set_ylim(*z_limits)
    ax_motion.set_aspect("equal", adjustable="box")
    ax_motion.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    ax_motion.set_xlabel("y [m]")
    ax_motion.set_ylabel("z [m]")
    ax_motion.set_title("gamma=0 passive swing")
    link_line, = ax_motion.plot([], [], "-o", color="tab:purple", linewidth=2.0, markersize=4)
    trace_line, = ax_motion.plot([], [], color="tab:orange", linewidth=1.0, alpha=0.7)
    time_text = ax_motion.text(0.02, 0.94, "", transform=ax_motion.transAxes, fontsize=8, va="top")

    ax_energy.plot(time, kinetic, color="0.75", linewidth=1.0)
    energy_marker, = ax_energy.plot([], [], "o", color="tab:blue", markersize=4)
    ax_energy.set_xlim(float(time[0]), float(time[-1]))
    y0 = float(np.min(kinetic))
    y1 = float(np.max(kinetic))
    margin = max(1e-6, 0.08 * (y1 - y0))
    ax_energy.set_ylim(y0 - margin, y1 + margin)
    ax_energy.set_xlabel("time [s]")
    ax_energy.set_ylabel("kinetic energy [J]")
    ax_energy.set_title("kinetic energy")
    for ax in axes:
        ax.grid(True, alpha=0.25)

    def update(frame_index: int):
        idx = int(frame_indices[frame_index])
        ys = [support[idx, 0], elbow[idx, 0], free[idx, 0]]
        zs = [support[idx, 1], elbow[idx, 1], free[idx, 1]]
        link_line.set_data(ys, zs)
        trace_line.set_data(free[: idx + 1, 0], free[: idx + 1, 1])
        energy_marker.set_data([time[idx]], [kinetic[idx]])
        time_text.set_text(f"t = {time[idx]:.3f} s")
        return link_line, trace_line, energy_marker, time_text

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=int(cfg["gif_fps"])))
    plt.close(fig)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {name: np.asarray(data[name]) for name in data.files if not name.endswith("_json")}


def run(params=None, *, force=False, results_dir: Path | str = Path("results"), verbose=True) -> PartResult:
    cfg = {**DEFAULTS, **(params or {})}
    setup = load_best_com_setup(results_dir=results_dir, branch_override=cfg.get("branch"))
    bp = setup.base_params
    cfg.update(
        {
            "_algo": ALGO_VERSION,
            "com_offset": setup.com_offset,
            "d_target": setup.d_target,
            "branch": setup.branch,
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
    result = cached_run(part=PART, config=cfg, compute=_compute, plot=_plot, results_dir=results_dir, force=force, verbose=verbose)

    results_dir = Path(results_dir)
    gif_path = animation_path(result, results_dir=results_dir, latest=False)
    gif_alias = animation_path(result, results_dir=results_dir, latest=True)
    if force or not gif_path.exists():
        if verbose:
            print(f"[{PART}] writing animation {gif_path.name}...")
        _make_animation(_load_arrays(result.data_path), result.summary, result.config, gif_path)
    if gif_path.resolve() != gif_alias.resolve():
        shutil.copy2(gif_path, gif_alias)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Roll out and animate the gamma=0 passive reference before control.")
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--t-max", type=float, default=DEFAULTS["t_max"])
    parser.add_argument("--gif-fps", type=int, default=DEFAULTS["gif_fps"])
    parser.add_argument("--gif-frames", type=int, default=DEFAULTS["gif_frames"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    params = {"dt": args.dt, "t_max": args.t_max, "gif_fps": args.gif_fps, "gif_frames": args.gif_frames}
    result = run(params, force=args.force, results_dir=args.results_dir)
    print(f"Data: {result.data_path}")
    print(f"Figure: {result.figure('main')}")
    print(f"Animation: {animation_path(result, results_dir=args.results_dir)}")
    print(f"K0={result.summary['kinetic_initial_J']:.6g} J, duration={result.summary['duration_s']:.6g} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
