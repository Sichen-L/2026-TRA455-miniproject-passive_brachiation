"""Part 8a: direct-collocation elbow-torque tracking for the gamma=0 reference.

Loads the most-stable passive gait (COM sweep), computes the gamma=0
minimum-energy free-initial-velocity reference, rolls it out to a pre-impact
swing, then solves a direct-collocation tracking problem.

The figure reports the four requested quantities versus time: elbow torque,
system energy, state residual, and cumulative external (actuator) work.

Importable interface::

    from run_direct_collocation_gamma0_reference import run
    res = run(); res.figure("main")
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import minimize

from passive_brachiation import (
    BrachiationParameters,
    BrachiationState,
    CollisionMode,
    Slope,
    SwitchDecision,
    TwoLinkBrachiationModel,
    forward_kinematics,
    parameters_with_symmetric_com_offset,
    simulate,
)
from passive_brachiation.simulation import SimPhase
from report_cache import PartResult, cached_run
from report_setup import load_best_com_setup
from run_free_initial_velocity_gamma_sweep import FreeVelocityProblem, scan_gamma, select_candidate

PART = "direct_collocation_gamma0_reference"
ALGO_VERSION = "v2"

DEFAULTS: dict[str, Any] = {
    "gamma": 0.0,
    "nodes": 41,
    "reference_dt": 0.005,
    "reference_t_max": 8.0,
    "initial_speed_scale": 0.0,
    "torque_limit": 20.0,
    "torque_weight": 1.0,
    "tracking_weight": 1.0,
    "state_weights": [50.0, 50.0, 2.0, 2.0],
    "maxiter": 300,
    "ftol": 1e-7,
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
    theta_values = np.linspace(np.deg2rad(cfg["theta_min_deg"]), np.deg2rad(cfg["theta_max_deg"]), cfg["theta_points"])
    problem = FreeVelocityProblem(cfg["gamma"], params, d_target, branch, cfg["reference_dt"], cfg["reference_t_max"], cfg["min_normal_speed"])
    candidates = scan_gamma(problem, theta_values, (0.0, cfg["max_speed"]), cfg["speed_points"], cfg["root_xtol"], cfg["root_rtol"], cfg["root_maxiter"])
    selected = select_candidate(candidates, cfg["allow_unstable_reference"], cfg["allow_illegal_reference"])
    if selected is None:
        raise RuntimeError(f"No gamma={cfg['gamma']:.3f} free-initial-velocity candidate from {len(candidates)} roots.")
    return selected


def state_array(state: BrachiationState) -> np.ndarray:
    return np.concatenate((state.q, state.qd)).astype(float)


def passive_reference_samples(model, slope, candidate, dt, t_max) -> list:
    initial_state = BrachiationState(q=np.asarray(candidate.q, dtype=float), qd=np.asarray(candidate.qd, dtype=float), support_index=0)
    samples = simulate(
        model=model, slope=slope, initial_state=initial_state, initial_support_point=np.zeros(2, dtype=float),
        duration=t_max, dt=dt, switch_policy=lambda *_args: SwitchDecision.SWITCH,
        collision_mode=CollisionMode.FULL_GRAB_1D, stop_after_releases=1,
    )
    impact_index = next((i for i, sample in enumerate(samples) if sample.phase == SimPhase.IMPACT), None)
    return samples[: impact_index + 1] if impact_index is not None else samples


def resample_reference(samples, nodes) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([sample.time for sample in samples], dtype=float)
    states = np.vstack([state_array(sample.state) for sample in samples])
    unique_times, unique_indices = np.unique(times, return_index=True)
    states = states[unique_indices]
    if len(unique_times) < 2:
        raise RuntimeError("Reference trajectory is too short for collocation.")
    grid = np.linspace(float(unique_times[0]), float(unique_times[-1]), int(nodes))
    x_ref = np.column_stack([np.interp(grid, unique_times, states[:, dim]) for dim in range(states.shape[1])])
    return grid, x_ref


def dynamics(model, x, torque) -> np.ndarray:
    state = BrachiationState(q=np.asarray(x[:2]), qd=np.asarray(x[2:]), support_index=0)
    return model.derivative(state, elbow_torque=float(torque))


def pack_decision(x_nodes, u_nodes) -> np.ndarray:
    return np.concatenate((x_nodes.reshape(-1), u_nodes.reshape(-1)))


def unpack_decision(z, nodes) -> tuple[np.ndarray, np.ndarray]:
    state_count = nodes * 4
    return np.asarray(z[:state_count], dtype=float).reshape(nodes, 4), np.asarray(z[state_count:], dtype=float)


def collocation_constraints(z, model, time_grid, x_initial, x_terminal) -> np.ndarray:
    x_nodes, u_nodes = unpack_decision(z, len(time_grid))
    constraints = [x_nodes[0] - x_initial, x_nodes[-1] - x_terminal]
    for index in range(len(time_grid) - 1):
        h = float(time_grid[index + 1] - time_grid[index])
        f0 = dynamics(model, x_nodes[index], u_nodes[index])
        f1 = dynamics(model, x_nodes[index + 1], u_nodes[index + 1])
        constraints.append(x_nodes[index + 1] - x_nodes[index] - 0.5 * h * (f0 + f1))
    return np.concatenate(constraints)


def objective(z, time_grid, x_ref, q_weights, torque_weight, tracking_weight) -> float:
    x_nodes, u_nodes = unpack_decision(z, len(time_grid))
    dt = np.diff(time_grid)
    torque_cost = float(np.sum(0.5 * dt * (u_nodes[:-1] ** 2 + u_nodes[1:] ** 2)))
    error = x_nodes - x_ref
    node_tracking = np.einsum("ij,j,ij->i", error, q_weights, error)
    tracking_cost = float(np.sum(0.5 * dt * (node_tracking[:-1] + node_tracking[1:])))
    return torque_weight * torque_cost + tracking_weight * tracking_cost


def energy_series(model, x_nodes) -> dict[str, np.ndarray]:
    kinetic = np.array([model.kinetic_energy(x[:2], x[2:]) for x in x_nodes], dtype=float)
    potential = np.array([model.potential_energy(x[:2]) for x in x_nodes], dtype=float)
    return {"kinetic": kinetic, "potential": potential, "total": kinetic + potential}


def control_energy(time_grid, x_nodes, u_nodes) -> dict[str, Any]:
    dt = np.diff(time_grid)
    power = u_nodes * x_nodes[:, 3]
    abs_power = np.abs(power)
    effort = float(np.sum(0.5 * dt * (u_nodes[:-1] ** 2 + u_nodes[1:] ** 2)))
    work = float(np.sum(0.5 * dt * (power[:-1] + power[1:])))
    abs_work = float(np.sum(0.5 * dt * (abs_power[:-1] + abs_power[1:])))
    # Net mechanical work on the system (signed; positive and negative cancel).
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * dt * (power[:-1] + power[1:]))))
    # Actuator energy consumed: the motor spends energy whenever it acts, so
    # braking (negative power) counts as expenditure too -> integral of |power|.
    cumulative_abs = np.concatenate(([0.0], np.cumsum(0.5 * dt * (abs_power[:-1] + abs_power[1:]))))
    return {
        "power": power,
        "cumulative_work": cumulative,
        "cumulative_abs_work": cumulative_abs,
        "torque_squared_integral": effort,
        "mechanical_work": work,
        "absolute_mechanical_work": abs_work,
    }


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base = BrachiationParameters.rod_point_mass(
        m1=cfg["m1"], m2=cfg["m2"], l1=cfg["l1"], l2=cfg["l2"],
        rod_mass_fraction=0.2, damping1=cfg["damping1"], damping2=cfg["damping2"], gravity=cfg["gravity"],
    )
    params = parameters_with_symmetric_com_offset(cfg["com_offset"], base)
    branch = str(cfg["branch"])
    model = TwoLinkBrachiationModel(params)
    slope = Slope(gamma=np.deg2rad(cfg["gamma"]))

    candidate = compute_gamma_candidate(cfg, params, cfg["d_target"], branch)
    samples = passive_reference_samples(model, slope, candidate, cfg["reference_dt"], cfg["reference_t_max"])
    time_grid, x_ref = resample_reference(samples, cfg["nodes"])

    x_initial = x_ref[0].copy()
    x_initial[2:] *= cfg["initial_speed_scale"]
    x_terminal = x_ref[-1].copy()

    x_guess = x_ref.copy()
    x_guess[0] = x_initial
    z0 = pack_decision(x_guess, np.zeros(cfg["nodes"], dtype=float))
    bounds = [(None, None)] * (cfg["nodes"] * 4) + [(-cfg["torque_limit"], cfg["torque_limit"])] * cfg["nodes"]
    q_weights = np.asarray(cfg["state_weights"], dtype=float)
    constraints = {"type": "eq", "fun": lambda z: collocation_constraints(z, model, time_grid, x_initial, x_terminal)}

    result = minimize(
        lambda z: objective(z, time_grid, x_ref, q_weights, cfg["torque_weight"], cfg["tracking_weight"]),
        z0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": cfg["maxiter"], "ftol": cfg["ftol"]},
    )
    x_opt, u_opt = unpack_decision(result.x, cfg["nodes"])
    ref_energy = energy_series(model, x_ref)
    opt_energy = energy_series(model, x_opt)
    ce = control_energy(time_grid, x_opt, u_opt)
    constraint_inf = float(np.linalg.norm(collocation_constraints(result.x, model, time_grid, x_initial, x_terminal), ord=np.inf))
    residual_norm = np.linalg.norm(x_opt - x_ref, axis=1)
    tracking_rmse = float(np.sqrt(np.mean(np.sum((x_opt - x_ref) ** 2, axis=1))))

    arrays = {
        "time": time_grid,
        "u_elbow": u_opt,
        "x_ref": x_ref,
        "x_opt": x_opt,
        "ref_kinetic": ref_energy["kinetic"],
        "ref_total": ref_energy["total"],
        "opt_kinetic": opt_energy["kinetic"],
        "opt_total": opt_energy["total"],
        "power": ce["power"],
        "cumulative_work": ce["cumulative_work"],
        "cumulative_abs_work": ce["cumulative_abs_work"],
        "residual_norm": residual_norm,
    }
    summary = {
        "gamma": cfg["gamma"],
        "d_target": cfg["d_target"],
        "com_offset": cfg["com_offset"],
        "branch": branch,
        "success": bool(result.success),
        "objective": float(result.fun),
        "constraint_inf_norm": constraint_inf,
        "tracking_rmse": tracking_rmse,
        "torque_squared_integral": ce["torque_squared_integral"],
        "mechanical_work": ce["mechanical_work"],
        "absolute_mechanical_work": ce["absolute_mechanical_work"],
        "reference_energy_J": float(candidate.energy),
        "source_com": cfg.get("source_com", ""),
    }
    return arrays, summary


def _plot_tracking(arrays, summary, cfg, method_label: str) -> Figure:
    import matplotlib.pyplot as plt

    t = arrays["time"].astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    axes[0, 0].plot(t, arrays["u_elbow"], color="tab:red", linewidth=1.8)
    axes[0, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("elbow torque [N m]")
    axes[0, 0].set_title("torque")

    axes[0, 1].plot(t, arrays["ref_kinetic"], "--", color="tab:blue", label="ref kinetic")
    axes[0, 1].plot(t, arrays["opt_kinetic"], color="tab:blue", label=f"{method_label} kinetic")
    axes[0, 1].plot(t, arrays["ref_total"], "--", color="tab:green", label="ref total")
    axes[0, 1].plot(t, arrays["opt_total"], color="tab:green", label=f"{method_label} total")
    axes[0, 1].set_ylabel("system energy [J]")
    axes[0, 1].set_title("system energy")
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=7)

    axes[1, 0].plot(t, arrays["residual_norm"], color="tab:purple", linewidth=1.8)
    axes[1, 0].set_ylabel("state residual ||x - x_ref||")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_title("state residual")

    axes[1, 1].plot(t, arrays["cumulative_abs_work"], color="tab:orange", linewidth=1.8, label="actuator energy (int |tau*qd|)")
    axes[1, 1].plot(t, arrays["cumulative_work"], color="0.6", linewidth=1.2, linestyle="--", label="net work on system (int tau*qd)")
    axes[1, 1].set_ylabel("cumulative energy [J]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_title("actuator energy consumed")
    axes[1, 1].legend(loc="best", fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"{method_label}: gamma={summary['gamma']:.1f} deg, "
        f"int tau^2={summary['torque_squared_integral']:.4g}, |W|={summary['absolute_mechanical_work']:.4g} J"
    )
    fig.tight_layout()
    return fig


def _plot(arrays, summary, cfg) -> dict[str, Figure]:
    return {"main": _plot_tracking(arrays, summary, cfg, "direct collocation")}


def animation_path(
    result: PartResult | None = None,
    *,
    results_dir: Path | str = Path("results"),
    latest: bool = True,
    part: str = PART,
) -> Path:
    results_dir = Path(results_dir)
    if result is None or latest:
        return results_dir / f"{part}_latest__motion.gif"
    return results_dir / f"{part}_{result.hash}__motion.gif"


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {name: np.asarray(data[name]) for name in data.files if not name.endswith("_json")}


def _trajectory_points(x_nodes: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.zeros((len(x_nodes), 2), dtype=float)
    elbow = np.zeros_like(support)
    free = np.zeros_like(support)
    for index, x in enumerate(x_nodes):
        pts = forward_kinematics(x[:2], np.zeros(2, dtype=float), cfg["l1"], cfg["l2"])
        support[index] = pts.support
        elbow[index] = pts.elbow
        free[index] = pts.free
    return support, elbow, free


def make_control_animation(arrays: dict[str, np.ndarray], cfg: dict[str, Any], path: Path, method_label: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    time = np.asarray(arrays["time"], dtype=float)
    x_opt = np.asarray(arrays["x_opt"], dtype=float)
    x_ref = np.asarray(arrays["x_ref"], dtype=float)
    torque = np.asarray(arrays["u_elbow"], dtype=float)
    support, elbow, free = _trajectory_points(x_opt, cfg)
    _ref_support, ref_elbow, ref_free = _trajectory_points(x_ref, cfg)
    frame_count = min(int(cfg.get("gif_frames", 90)), len(time))
    frame_indices = np.unique(np.linspace(0, len(time) - 1, frame_count, dtype=int))

    all_y = np.concatenate((support[:, 0], elbow[:, 0], free[:, 0], ref_elbow[:, 0], ref_free[:, 0]))
    all_z = np.concatenate((support[:, 1], elbow[:, 1], free[:, 1], ref_elbow[:, 1], ref_free[:, 1], np.array([0.0])))
    pad = 0.08
    y_limits = (float(np.min(all_y) - pad), float(np.max(all_y) + pad))
    z_limits = (float(np.min(all_z) - pad), float(np.max(all_z) + pad))
    torque_span = float(np.max(torque) - np.min(torque))
    torque_margin = max(1e-6, 0.08 * torque_span)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    ax_motion, ax_torque = axes
    ax_motion.set_xlim(*y_limits)
    ax_motion.set_ylim(*z_limits)
    ax_motion.set_aspect("equal", adjustable="box")
    ax_motion.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    ax_motion.plot(ref_free[:, 0], ref_free[:, 1], color="0.65", linestyle="--", linewidth=1.0, label="reference free")
    ax_motion.set_xlabel("y [m]")
    ax_motion.set_ylabel("z [m]")
    ax_motion.set_title(f"{method_label} motion")
    link_line, = ax_motion.plot([], [], "-o", color="tab:purple", linewidth=2.0, markersize=4)
    trace_line, = ax_motion.plot([], [], color="tab:orange", linewidth=1.0, alpha=0.8, label="optimized free")
    time_text = ax_motion.text(0.02, 0.94, "", transform=ax_motion.transAxes, fontsize=8, va="top")
    ax_motion.legend(loc="lower left", fontsize=7)

    ax_torque.plot(time, torque, color="0.75", linewidth=1.0)
    ax_torque.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    torque_marker, = ax_torque.plot([], [], "o", color="tab:red", markersize=4)
    ax_torque.set_xlim(float(time[0]), float(time[-1]))
    ax_torque.set_ylim(float(np.min(torque) - torque_margin), float(np.max(torque) + torque_margin))
    ax_torque.set_xlabel("time [s]")
    ax_torque.set_ylabel("elbow torque [N m]")
    ax_torque.set_title("torque")
    for ax in axes:
        ax.grid(True, alpha=0.25)

    def update(frame_index: int):
        idx = int(frame_indices[frame_index])
        link_line.set_data([support[idx, 0], elbow[idx, 0], free[idx, 0]], [support[idx, 1], elbow[idx, 1], free[idx, 1]])
        trace_line.set_data(free[: idx + 1, 0], free[: idx + 1, 1])
        torque_marker.set_data([time[idx]], [torque[idx]])
        time_text.set_text(f"t = {time[idx]:.3f} s")
        return link_line, trace_line, torque_marker, time_text

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=int(cfg.get("gif_fps", 24))))
    plt.close(fig)


def ensure_animation(result: PartResult, *, results_dir: Path | str, force: bool, verbose: bool, method_label: str, part: str = PART) -> Path:
    results_dir = Path(results_dir)
    gif_path = animation_path(result, results_dir=results_dir, latest=False, part=part)
    gif_alias = animation_path(result, results_dir=results_dir, latest=True, part=part)
    if force or not gif_path.exists():
        if verbose:
            print(f"[{part}] writing animation {gif_path.name}...")
        make_control_animation(_load_arrays(result.data_path), result.config, gif_path, method_label)
    if gif_path.resolve() != gif_alias.resolve():
        shutil.copy2(gif_path, gif_alias)
    return gif_alias


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
            "m1": bp.m1, "m2": bp.m2, "l1": bp.l1, "l2": bp.l2,
            "damping1": bp.damping1, "damping2": bp.damping2, "gravity": bp.gravity,
            "source_com": setup.source,
        }
    )
    result = cached_run(part=PART, config=cfg, compute=_compute, plot=_plot, results_dir=results_dir, force=force, verbose=verbose)
    ensure_animation(result, results_dir=results_dir, force=force, verbose=verbose, method_label="direct collocation")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct-collocation controller around the gamma=0 minimum-energy trajectory.")
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--nodes", type=int, default=DEFAULTS["nodes"])
    parser.add_argument("--initial-speed-scale", type=float, default=DEFAULTS["initial_speed_scale"])
    parser.add_argument("--torque-limit", type=float, default=DEFAULTS["torque_limit"])
    parser.add_argument("--gif-fps", type=int, default=DEFAULTS["gif_fps"])
    parser.add_argument("--gif-frames", type=int, default=DEFAULTS["gif_frames"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    params = {
        "gamma": args.gamma,
        "nodes": args.nodes,
        "initial_speed_scale": args.initial_speed_scale,
        "torque_limit": args.torque_limit,
        "gif_fps": args.gif_fps,
        "gif_frames": args.gif_frames,
    }
    result = run(params, force=args.force, results_dir=args.results_dir)
    print(f"Data: {result.data_path}")
    print(f"Figure: {result.figure('main')}")
    print(f"Animation: {animation_path(result, results_dir=args.results_dir)}")
    print(f"success={result.summary['success']} objective={result.summary['objective']:.6g} tracking_rmse={result.summary['tracking_rmse']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
