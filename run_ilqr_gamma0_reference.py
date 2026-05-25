"""Part 8b: iLQR elbow-torque tracking for the gamma=0 reference swing.

Uses the same gamma=0 minimum-energy free-initial-velocity reference as the
direct-collocation script, but optimizes the continuous swing with iterative
LQR.  The figure reports the four requested quantities versus time: elbow
torque, system energy, state residual, and cumulative external work.

Importable interface::

    from run_ilqr_gamma0_reference import run
    res = run(); res.figure("main")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from passive_brachiation import (
    BrachiationParameters,
    BrachiationState,
    Slope,
    TwoLinkBrachiationModel,
    parameters_with_symmetric_com_offset,
)
from report_cache import PartResult, cached_run
from report_setup import load_best_com_setup
from run_direct_collocation_gamma0_reference import (
    _plot_tracking,
    animation_path,
    compute_gamma_candidate,
    control_energy,
    energy_series,
    ensure_animation,
    passive_reference_samples,
    resample_reference,
)

PART = "ilqr_gamma0_reference"
ALGO_VERSION = "v2"

DEFAULTS: dict[str, Any] = {
    "gamma": 0.0,
    "nodes": 101,
    "reference_dt": 0.005,
    "reference_t_max": 8.0,
    "initial_speed_scale": 0.0,
    "torque_limit": 20.0,
    "control_weight": 1e-3,
    "state_weights": [20.0, 20.0, 1.0, 1.0],
    "terminal_weights": [300.0, 300.0, 30.0, 30.0],
    "terminal_weight_scale": 1.0,
    "max_iter": 80,
    "tolerance": 1e-7,
    "regularization": 1e-6,
    "x_eps": 1e-5,
    "u_eps": 1e-5,
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


def discrete_step(model, x, u, dt) -> np.ndarray:
    state = BrachiationState(q=np.asarray(x[:2]), qd=np.asarray(x[2:]), support_index=0)
    next_state = model.step_rk4(state, dt=dt, elbow_torque=float(u))
    return np.concatenate((next_state.q, next_state.qd))


def rollout(model, x0, controls, dt, torque_limit=None) -> tuple[np.ndarray, np.ndarray]:
    controls = np.asarray(controls, dtype=float).copy()
    if torque_limit is not None:
        controls = np.clip(controls, -torque_limit, torque_limit)
    states = np.zeros((len(controls) + 1, 4), dtype=float)
    states[0] = np.asarray(x0, dtype=float)
    for index, torque in enumerate(controls):
        states[index + 1] = discrete_step(model, states[index], float(torque), dt)
    return states, controls


def finite_difference_linearization(model, x, u, dt, x_eps, u_eps) -> tuple[np.ndarray, np.ndarray]:
    a_matrix = np.zeros((4, 4), dtype=float)
    for dim in range(4):
        step = np.zeros(4, dtype=float)
        step[dim] = x_eps
        a_matrix[:, dim] = (discrete_step(model, x + step, u, dt) - discrete_step(model, x - step, u, dt)) / (2.0 * x_eps)
    b_matrix = ((discrete_step(model, x, u + u_eps, dt) - discrete_step(model, x, u - u_eps, dt)) / (2.0 * u_eps)).reshape(4, 1)
    if not (np.all(np.isfinite(a_matrix)) and np.all(np.isfinite(b_matrix))):
        raise FloatingPointError("Non-finite finite-difference linearization.")
    return a_matrix, b_matrix


def trajectory_cost(states, controls, x_ref, dt, q_weights, qf_weights, control_weight) -> float:
    total = 0.0
    for index, torque in enumerate(controls):
        error = states[index] - x_ref[index]
        total += 0.5 * dt * (float(error @ (q_weights * error)) + control_weight * float(torque) ** 2)
    terminal_error = states[-1] - x_ref[-1]
    total += 0.5 * float(terminal_error @ (qf_weights * terminal_error))
    return float(total)


def ilqr_solve(model, x0, x_ref, dt, q_weights, qf_weights, control_weight, torque_limit, max_iter, tolerance, regularization, x_eps, u_eps) -> dict:
    horizon = len(x_ref)
    controls = np.zeros(horizon - 1, dtype=float)
    states, controls = rollout(model, x0, controls, dt, torque_limit=torque_limit)
    cost = trajectory_cost(states, controls, x_ref, dt, q_weights, qf_weights, control_weight)
    alphas = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    reg = float(regularization)
    history: list[dict[str, Any]] = []

    for iteration in range(max_iter):
        linearizations = [finite_difference_linearization(model, states[i], float(controls[i]), dt, x_eps, u_eps) for i in range(horizon - 1)]
        terminal_error = states[-1] - x_ref[-1]
        vx = qf_weights * terminal_error
        vxx = np.diag(qf_weights)
        k_ff = np.zeros(horizon - 1, dtype=float)
        k_fb = np.zeros((horizon - 1, 1, 4), dtype=float)
        backward_ok = True
        for index in reversed(range(horizon - 1)):
            a_matrix, b_matrix = linearizations[index]
            error = states[index] - x_ref[index]
            lx = dt * q_weights * error
            lu = np.array([dt * control_weight * controls[index]], dtype=float)
            lxx = dt * np.diag(q_weights)
            luu = np.array([[dt * control_weight]], dtype=float)
            qx = lx + a_matrix.T @ vx
            qu = lu + b_matrix.T @ vx
            qxx = lxx + a_matrix.T @ vxx @ a_matrix
            quu = luu + b_matrix.T @ vxx @ b_matrix + reg * np.eye(1)
            qux = b_matrix.T @ vxx @ a_matrix
            if quu[0, 0] <= 0.0 or not np.isfinite(quu[0, 0]):
                backward_ok = False
                break
            inv_quu = np.linalg.inv(quu)
            k = -inv_quu @ qu
            k_matrix = -inv_quu @ qux
            k_ff[index] = float(k[0])
            k_fb[index] = k_matrix
            vx = qx + k_matrix.T @ quu @ k + k_matrix.T @ qu + qux.T @ k
            vxx = qxx + k_matrix.T @ quu @ k_matrix + k_matrix.T @ qux + qux.T @ k_matrix
            vxx = 0.5 * (vxx + vxx.T)

        if not backward_ok:
            reg *= 10.0
            history.append({"iteration": iteration, "cost": cost, "accepted": False})
            if reg > 1e8:
                break
            continue

        accepted = False
        best_trial = None
        for alpha in alphas:
            trial_controls = np.zeros_like(controls)
            trial_states = np.zeros_like(states)
            trial_states[0] = x0
            for index in range(horizon - 1):
                feedback = k_fb[index, 0] @ (trial_states[index] - states[index])
                trial_controls[index] = np.clip(controls[index] + alpha * k_ff[index] + feedback, -torque_limit, torque_limit)
                trial_states[index + 1] = discrete_step(model, trial_states[index], trial_controls[index], dt)
            trial_cost = trajectory_cost(trial_states, trial_controls, x_ref, dt, q_weights, qf_weights, control_weight)
            if np.isfinite(trial_cost) and trial_cost < cost:
                best_trial = (trial_states, trial_controls, trial_cost, alpha)
                accepted = True
                break

        if not accepted:
            reg *= 10.0
            history.append({"iteration": iteration, "cost": cost, "accepted": False})
            if reg > 1e8:
                break
            continue

        new_states, new_controls, new_cost, alpha = best_trial
        improvement = cost - new_cost
        states, controls, cost = new_states, new_controls, float(new_cost)
        reg = max(reg * 0.5, 1e-9)
        history.append({"iteration": iteration, "cost": cost, "accepted": True, "alpha": float(alpha)})
        if improvement < tolerance:
            break

    return {"states": states, "controls": controls, "cost": cost, "history": history, "iterations": len(history)}


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
    dt = float(time_grid[1] - time_grid[0])
    if not np.allclose(np.diff(time_grid), dt):
        raise RuntimeError("iLQR requires a uniform reference grid.")

    x0 = x_ref[0].copy()
    x0[2:] *= cfg["initial_speed_scale"]
    q_weights = np.asarray(cfg["state_weights"], dtype=float)
    qf_weights = cfg["terminal_weight_scale"] * np.asarray(cfg["terminal_weights"], dtype=float)
    solution = ilqr_solve(
        model, x0, x_ref, dt, q_weights, qf_weights, cfg["control_weight"], cfg["torque_limit"],
        cfg["max_iter"], cfg["tolerance"], cfg["regularization"], cfg["x_eps"], cfg["u_eps"],
    )
    x_opt = solution["states"]
    controls = solution["controls"]
    node_controls = np.concatenate((controls, controls[-1:]))
    ref_energy = energy_series(model, x_ref)
    opt_energy = energy_series(model, x_opt)
    ce = control_energy(time_grid, x_opt, node_controls)
    residual_norm = np.linalg.norm(x_opt - x_ref, axis=1)
    tracking_rmse = float(np.sqrt(np.mean(np.sum((x_opt - x_ref) ** 2, axis=1))))
    terminal_error_norm = float(np.linalg.norm(x_opt[-1] - x_ref[-1]))

    arrays = {
        "time": time_grid,
        "u_elbow": node_controls,
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
        "cost": float(solution["cost"]),
        "iterations": int(solution["iterations"]),
        "tracking_rmse": tracking_rmse,
        "terminal_error_norm": terminal_error_norm,
        "torque_squared_integral": ce["torque_squared_integral"],
        "mechanical_work": ce["mechanical_work"],
        "absolute_mechanical_work": ce["absolute_mechanical_work"],
        "reference_energy_J": float(candidate.energy),
        "source_com": cfg.get("source_com", ""),
    }
    return arrays, summary


def _plot(arrays, summary, cfg):
    return {"main": _plot_tracking(arrays, summary, cfg, "iLQR")}


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
    ensure_animation(result, results_dir=results_dir, force=force, verbose=verbose, method_label="iLQR", part=PART)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iLQR controller around the gamma=0 minimum-energy pre-impact swing.")
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
    print(f"Animation: {animation_path(result, results_dir=args.results_dir, part=PART)}")
    print(f"cost={result.summary['cost']:.6g} iterations={result.summary['iterations']} tracking_rmse={result.summary['tracking_rmse']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
