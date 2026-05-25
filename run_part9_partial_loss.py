"""Part 9: slope continuation for instantaneous partial-loss release.

This part studies the collision mode where the impact and release happen in
one event, so part of the pre-impact kinetic energy is retained.  The default
case fixes the retention at ``s = 1`` (normal-plastic: remove slope-normal
velocity, retain tangent velocity), finds an s=1 fixed point at both 45 deg and
3 deg, then continues each solution branch across the slope range.  The two
continuations are plotted together to check whether they trace the same gait
family.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from passive_brachiation import (
    BrachiationState,
    CollisionContext,
    CollisionMode,
    Slope,
    SwitchDecision,
    SwitchResult,
    TwoLinkBrachiationModel,
    compute_full_grab_collision_velocity,
    compute_plastic_collision_velocity,
    continue_fixed_point_branch,
    evaluate_elbow_below_slope_section,
    forward_kinematics,
    ik_from_stride_distance,
    make_passive_brachiation_feasibility_check,
    scan_stride_fixed_points,
    simulate,
    stride_distance_from_point,
)
from passive_brachiation.switching import switch_support
from report_cache import PartResult, cached_run
from report_setup import load_best_com_setup


PART = "part9_partial_loss"
ALGO_VERSION = "v2"

DEFAULTS: dict[str, Any] = {
    "retention": 1.0,
    "gamma_low": 3.0,
    "gamma_high": 45.0,
    "gamma_step": 1.0,
    "seed_gammas": [45.0, 3.0],
    "dt": 0.005,
    "t_max": 8.0,
    "d_scan_points": 30,
    "s_seed_points": 31,
    "continuation_tol": 2e-5,
    "continuation_max_iter": 20,
    "continuation_delta": 1e-5,
    "continuation_damping": 0.5,
    "legality_releases": 3,
    "d_upper_fraction": 0.995,
}


def _switch_policy(_t, _state, _support_point, _impact_point, _slope):
    return SwitchDecision.SWITCH


def make_blend_collision(retention: float):
    s = float(np.clip(retention, 0.0, 1.0))

    def collision_model(context: CollisionContext) -> SwitchResult:
        if context.decision != SwitchDecision.SWITCH:
            return SwitchResult(state=context.state, phase=context.decision)
        if context.slope is None or context.mass_matrix is None:
            raise ValueError("Partial-loss collision requires slope and mass matrix.")
        p = context.parameters
        qd_full = compute_full_grab_collision_velocity(
            q=context.state.q,
            qd_before=context.state.qd,
            mass_matrix=context.mass_matrix,
            l1=p.l1,
            l2=p.l2,
        )
        qd_plastic = compute_plastic_collision_velocity(
            q=context.state.q,
            qd_before=context.state.qd,
            slope=context.slope,
            mass_matrix=context.mass_matrix,
            l1=p.l1,
            l2=p.l2,
        )
        qd_after = (1.0 - s) * qd_full + s * qd_plastic
        collided = BrachiationState(
            q=context.state.q.copy(),
            qd=qd_after,
            support_index=context.state.support_index,
        )
        pts = forward_kinematics(context.state.q, context.support_point, p.l1, p.l2)
        new_state = switch_support(
            state=collided,
            old_support=context.support_point,
            elbow=pts.elbow,
            new_support=context.collision_point,
        )
        return SwitchResult(state=new_state, phase=SwitchDecision.SWITCH)

    return collision_model


def make_partial_loss_map(model, slope, branch, dt, t_max, retention):
    collision_model = make_blend_collision(retention)
    support0 = np.zeros(2, dtype=float)

    def p_of_x(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        d = float(x[0])
        q = ik_from_stride_distance(
            d,
            slope=slope,
            parameters=model.p,
            direction=-1.0,
            branch=branch,
        )
        state0 = BrachiationState(q=q, qd=x[1:3].copy(), support_index=0)
        samples = simulate(
            model=model,
            slope=slope,
            initial_state=state0,
            initial_support_point=support0,
            duration=t_max,
            dt=dt,
            switch_policy=_switch_policy,
            collision_mode=CollisionMode.FULL_GRAB_1D,
            collision_model=collision_model,
            stop_after_releases=1,
        )
        releases = [sample for sample in samples if sample.phase.value == "release"]
        if not releases:
            raise ValueError("No release reached.")
        release = releases[0]
        d_next = stride_distance_from_point(
            release.support_point,
            slope=slope,
            support_point=support0,
            direction=1.0,
        )
        return np.array([float(d_next), float(release.state.qd[0]), float(release.state.qd[1])], dtype=float)

    return p_of_x


def _simulate_legality(model, slope, branch, dt, t_max, retention, x, releases: int) -> tuple[bool, float]:
    q0 = ik_from_stride_distance(float(x[0]), slope=slope, parameters=model.p, direction=-1.0, branch=branch)
    samples = simulate(
        model=model,
        slope=slope,
        initial_state=BrachiationState(q=q0, qd=np.asarray(x[1:3], dtype=float), support_index=0),
        initial_support_point=np.zeros(2, dtype=float),
        duration=max(t_max, 4.0 * t_max),
        dt=dt,
        switch_policy=_switch_policy,
        collision_model=make_blend_collision(retention),
        stop_after_releases=releases,
    )
    legality = evaluate_elbow_below_slope_section(samples, slope=slope, tolerance=1e-9)
    return bool(legality.legal), float(legality.max_signed_distance)


def _full_grab_seed(model, slope, branch, cfg):
    params = model.p
    search = scan_stride_fixed_points(
        model=model,
        slope=slope,
        initial_support_point=np.zeros(2, dtype=float),
        dt=cfg["dt"],
        t_max=cfg["t_max"],
        d_bounds=(0.05, cfg["d_upper_fraction"] * (params.l1 + params.l2)),
        d_scan_points=cfg["d_scan_points"],
        branches=(branch,),
        periods=(1,),
        initial_direction=-1.0,
        impact_direction=1.0,
        switch_policy=_switch_policy,
        collision_mode=CollisionMode.FULL_GRAB_1D,
        raise_on_empty=False,
    )
    return search.selected_trial


def _find_retained_root(model, gamma_deg: float, branch: str, cfg: dict[str, Any]):
    slope = Slope(gamma=np.deg2rad(gamma_deg))
    seed = _full_grab_seed(model, slope, branch, cfg)
    if seed is None:
        raise RuntimeError(f"No full-grab seed gait found at gamma={gamma_deg:.3f} deg.")

    feasibility = make_passive_brachiation_feasibility_check(model.p, dim=3)

    def factory(retention: float):
        return make_partial_loss_map(model, slope, branch, cfg["dt"], cfg["t_max"], retention)

    s_values = np.linspace(0.0, float(cfg["retention"]), int(cfg["s_seed_points"]))
    result = continue_fixed_point_branch(
        P_factory=factory,
        parameters=s_values,
        x0=np.array([seed.d, 0.0, 0.0], dtype=float),
        dim=3,
        feasibility_factory=lambda _s: feasibility,
        tol=cfg["continuation_tol"],
        max_iter=cfg["continuation_max_iter"],
        delta=cfg["continuation_delta"],
        damping=cfg["continuation_damping"],
        compute_stability=True,
        stop_on_failure=True,
    )
    last = result.points[-1]
    if not last.converged:
        raise RuntimeError(
            f"s-continuation to retention={cfg['retention']:.3f} failed at "
            f"gamma={gamma_deg:.3f} deg: {result.stop_reason}"
        )
    return last


def _gamma_values(start: float, stop: float, step: float) -> np.ndarray:
    direction = 1.0 if stop >= start else -1.0
    h = abs(float(step)) * direction
    values = np.arange(float(start), float(stop) + 0.5 * h, h)
    if values.size == 0 or abs(values[-1] - stop) > 1e-9:
        values = np.append(values, float(stop))
    return values


def _continue_from_seed(model, x0, seed_gamma: float, target_gamma: float, branch: str, cfg):
    gammas = _gamma_values(seed_gamma, target_gamma, cfg["gamma_step"])
    feasibility = make_passive_brachiation_feasibility_check(model.p, dim=3)

    def factory(gamma_deg: float):
        slope = Slope(gamma=np.deg2rad(gamma_deg))
        return make_partial_loss_map(model, slope, branch, cfg["dt"], cfg["t_max"], cfg["retention"])

    return continue_fixed_point_branch(
        P_factory=factory,
        parameters=gammas,
        x0=x0,
        dim=3,
        feasibility_factory=lambda _g: feasibility,
        tol=cfg["continuation_tol"],
        max_iter=cfg["continuation_max_iter"],
        delta=cfg["continuation_delta"],
        damping=cfg["continuation_damping"],
        compute_stability=True,
        stop_on_failure=False,
    )


def _rows_from_result(label: str, result, model, branch: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in result.points:
        gamma_deg = float(point.parameter)
        x = np.asarray(point.x, dtype=float)
        rho = np.nan if point.spectral_radius is None else float(point.spectral_radius)
        legal = False
        max_elbow = float("nan")
        if point.converged:
            try:
                legal, max_elbow = _simulate_legality(
                    model,
                    Slope(gamma=np.deg2rad(gamma_deg)),
                    branch,
                    cfg["dt"],
                    cfg["t_max"],
                    cfg["retention"],
                    x,
                    int(cfg["legality_releases"]),
                )
            except Exception:
                legal = False
                max_elbow = float("nan")
        rows.append(
            {
                "source": label,
                "gamma_deg": gamma_deg,
                "d": float(x[0]),
                "q1_dot": float(x[1]),
                "q2_dot": float(x[2]),
                "spectral_radius": rho,
                "residual_norm": float(point.residual_norm),
                "converged": bool(point.converged),
                "stable": bool(np.isfinite(rho) and rho < 1.0),
                "legal": bool(legal),
                "max_elbow_distance": max_elbow,
            }
        )
    return rows


def _branch_distance(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    by_a = {round(row["gamma_deg"], 8): row for row in rows_a if row["converged"]}
    by_b = {round(row["gamma_deg"], 8): row for row in rows_b if row["converged"]}
    common = np.array(sorted(set(by_a) & set(by_b)), dtype=float)
    distances = []
    for gamma in common:
        xa = np.array([by_a[gamma]["d"], by_a[gamma]["q1_dot"], by_a[gamma]["q2_dot"]], dtype=float)
        xb = np.array([by_b[gamma]["d"], by_b[gamma]["q1_dot"], by_b[gamma]["q2_dot"]], dtype=float)
        distances.append(float(np.linalg.norm(xa - xb)))
    return common, np.array(distances, dtype=float)


def _compute(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    setup = load_best_com_setup(results_dir=cfg["results_dir"])
    model = TwoLinkBrachiationModel(setup.params)
    branch = setup.branch
    seed_high, seed_low = [float(value) for value in cfg["seed_gammas"]]

    high_root = _find_retained_root(model, seed_high, branch, cfg)
    low_root = _find_retained_root(model, seed_low, branch, cfg)

    high_result = _continue_from_seed(model, high_root.x, seed_high, cfg["gamma_low"], branch, cfg)
    low_result = _continue_from_seed(model, low_root.x, seed_low, cfg["gamma_high"], branch, cfg)

    high_rows = _rows_from_result(f"seed_{seed_high:g}deg", high_result, model, branch, cfg)
    low_rows = _rows_from_result(f"seed_{seed_low:g}deg", low_result, model, branch, cfg)
    rows = high_rows + low_rows

    common_gamma, branch_distance = _branch_distance(high_rows, low_rows)

    arrays = {
        "source": np.array([row["source"] for row in rows], dtype="U32"),
        "gamma_deg": np.array([row["gamma_deg"] for row in rows], dtype=float),
        "d": np.array([row["d"] for row in rows], dtype=float),
        "q1_dot": np.array([row["q1_dot"] for row in rows], dtype=float),
        "q2_dot": np.array([row["q2_dot"] for row in rows], dtype=float),
        "spectral_radius": np.array([row["spectral_radius"] for row in rows], dtype=float),
        "residual_norm": np.array([row["residual_norm"] for row in rows], dtype=float),
        "converged": np.array([row["converged"] for row in rows], dtype=bool),
        "stable": np.array([row["stable"] for row in rows], dtype=bool),
        "legal": np.array([row["legal"] for row in rows], dtype=bool),
        "max_elbow_distance": np.array([row["max_elbow_distance"] for row in rows], dtype=float),
        "common_gamma_deg": common_gamma,
        "branch_distance": branch_distance,
    }
    stable_legal = arrays["converged"] & arrays["stable"] & arrays["legal"]
    summary = {
        "retention": float(cfg["retention"]),
        "branch": branch,
        "com_offset": float(setup.com_offset),
        "d_target_full_grab": float(setup.d_target),
        "gamma_low": float(cfg["gamma_low"]),
        "gamma_high": float(cfg["gamma_high"]),
        "seed_high_gamma": seed_high,
        "seed_low_gamma": seed_low,
        "seed_high_rho": None if high_root.spectral_radius is None else float(high_root.spectral_radius),
        "seed_low_rho": None if low_root.spectral_radius is None else float(low_root.spectral_radius),
        "num_points": int(len(rows)),
        "num_stable_legal": int(np.count_nonzero(stable_legal)),
        "stable_legal_gammas": sorted(set(np.round(arrays["gamma_deg"][stable_legal], 8).tolist())),
        "branch_overlap_points": int(common_gamma.size),
        "branch_distance_max": float(np.nanmax(branch_distance)) if branch_distance.size else float("nan"),
        "branch_distance_rms": float(np.sqrt(np.nanmean(branch_distance**2))) if branch_distance.size else float("nan"),
    }
    return arrays, summary


def _plot(arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Figure]:
    import matplotlib.pyplot as plt

    sources = list(dict.fromkeys(arrays["source"].tolist()))
    colors = {sources[0]: "tab:blue"}
    if len(sources) > 1:
        colors[sources[1]] = "tab:orange"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for source in sources:
        mask = arrays["source"] == source
        order = np.argsort(arrays["gamma_deg"][mask])
        g = arrays["gamma_deg"][mask][order]
        d = arrays["d"][mask][order]
        q1d = arrays["q1_dot"][mask][order]
        q2d = arrays["q2_dot"][mask][order]
        rho = arrays["spectral_radius"][mask][order]
        legal = arrays["legal"][mask][order]
        stable = arrays["stable"][mask][order]
        color = colors.get(source, None)
        axes[0, 0].plot(g, d, "o-", color=color, label=source)
        axes[0, 1].plot(g, q1d, "o-", color=color, label=f"{source} q1_dot")
        axes[0, 1].plot(g, q2d, "s--", color=color, alpha=0.65, label=f"{source} q2_dot")
        axes[1, 0].plot(g, rho, "o-", color=color, label=source)
        good = legal & stable
        bad = ~good
        axes[1, 0].scatter(g[good], rho[good], color="tab:green", s=48, zorder=4)
        axes[1, 0].scatter(g[bad], rho[bad], color="tab:red", marker="x", s=48, zorder=4)

    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].plot(
        arrays["common_gamma_deg"],
        arrays["branch_distance"],
        "o-",
        color="tab:purple",
        label="branch state distance",
    )
    axes[1, 1].set_yscale("log")

    axes[0, 0].set_ylabel("fixed stride d [m]")
    axes[0, 0].set_title("s=1 branch stride")
    axes[0, 1].set_ylabel("release q_dot [rad/s]")
    axes[0, 1].set_title("retained release velocity")
    axes[1, 0].set_xlabel("slope angle gamma [deg]")
    axes[1, 0].set_ylabel("Poincare spectral radius")
    axes[1, 0].set_title("stability and legality")
    axes[1, 1].set_xlabel("slope angle gamma [deg]")
    axes[1, 1].set_ylabel("||x_45(gamma) - x_3(gamma)||")
    axes[1, 1].set_title("branch overlap check")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Part 9: normal-plastic release branch from 45 deg and 3 deg seeds\n"
        f"COM offset={summary['com_offset']:+.4f}, retention s={summary['retention']:.2f}"
    )
    fig.tight_layout()
    return {"main": fig}


def run(params=None, *, force=False, results_dir: Path | str = Path("results"), verbose=True) -> PartResult:
    cfg = {**DEFAULTS, **(params or {}), "_algo": ALGO_VERSION, "results_dir": str(results_dir)}
    return cached_run(part=PART, config=cfg, compute=_compute, plot=_plot, results_dir=results_dir, force=force, verbose=verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run part 9 slope continuation for partial-loss release.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retention", type=float, default=DEFAULTS["retention"])
    parser.add_argument("--gamma-low", type=float, default=DEFAULTS["gamma_low"])
    parser.add_argument("--gamma-high", type=float, default=DEFAULTS["gamma_high"])
    parser.add_argument("--gamma-step", type=float, default=DEFAULTS["gamma_step"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(
        {
            "retention": args.retention,
            "gamma_low": args.gamma_low,
            "gamma_high": args.gamma_high,
            "gamma_step": args.gamma_step,
        },
        force=args.force,
        results_dir=args.results_dir,
    )
    print(f"Figure: {result.figure('main')}")
    print(
        "branch overlap: "
        f"n={result.summary['branch_overlap_points']}, "
        f"rms={result.summary['branch_distance_rms']:.3e}, "
        f"max={result.summary['branch_distance_max']:.3e}"
    )
    print(f"stable/legal gammas: {result.summary['stable_legal_gammas']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
