"""State-machine simulation loop for passive brachiation on a slope.

Phases
------
1. SWINGING - continuous RK4 integration with a slope-impact guard.
2. IMPACT   - the exact interpolated contact sample.
3. RELEASE  - the sample right after the selected collision/switch handler.

Collision handling is pluggable.  The default built-in model is
``CollisionMode.FULL_GRAB_1D``, which matches the reference inverted-compass
grab map from ``passive-brachiator-main``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .kinematics import Slope, forward_kinematics
from .model import BrachiationState, TwoLinkBrachiationModel
from .switching import (
    CollisionContext,
    CollisionMode,
    CollisionModel,
    SwitchDecision,
    apply_switch_decision,
    detect_slope_impact,
)


TorquePolicy = Callable[[float, BrachiationState], float]
EndpointForcePolicy = Callable[[float, BrachiationState], np.ndarray]
GeneralizedForcePolicy = Callable[[float, BrachiationState], np.ndarray]
SwitchPolicy = Callable[
    [float, BrachiationState, np.ndarray, np.ndarray, Slope],
    SwitchDecision,
]


class SimPhase(str, Enum):
    """Labels attached to each recorded sample."""

    SWINGING = "swinging"
    DWELL = "dwelling"
    IMPACT = "impact"
    RELEASE = "release"


@dataclass(frozen=True)
class SimulationSample:
    """One recorded simulation sample."""

    time: float
    state: BrachiationState
    support_point: np.ndarray
    free_end: np.ndarray
    elbow: np.ndarray
    phase: SimPhase
    elbow_torque: float
    external_endpoint_force_yz: np.ndarray
    external_generalized_force: np.ndarray
    kinetic_energy: float
    potential_energy: float
    total_energy: float


def simulate(
    model: TwoLinkBrachiationModel,
    slope: Slope,
    initial_state: BrachiationState,
    initial_support_point: np.ndarray,
    duration: float,
    dt: float,
    torque_policy: TorquePolicy | None = None,
    endpoint_force_policy: EndpointForcePolicy | None = None,
    generalized_force_policy: GeneralizedForcePolicy | None = None,
    switch_policy: SwitchPolicy | None = None,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
    collision_model: CollisionModel | None = None,
    leave_tol: float = 1e-3,
    initial_geometry_tolerance: float = 1e-6,
    require_initial_below_slope: bool = True,
    stop_after_releases: int | None = None,
    impact_root_tolerance: float = 1e-10,
    impact_root_max_iterations: int = 50,
) -> list[SimulationSample]:
    """Run a fixed-step simulation with slope-contact guards.

    ``collision_model`` is the extension point for a future 2-D shooting impact
    map.  When it is omitted, ``collision_mode`` selects one of the built-in
    collision handlers.

    Passing ``stop_after_releases`` returns as soon as that many release
    samples have been recorded.  This is useful for stride maps, which only
    need the first section return instead of a full free-run trajectory.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if duration < 0.0:
        raise ValueError("duration must be non-negative.")
    if stop_after_releases is not None and stop_after_releases < 1:
        raise ValueError("stop_after_releases must be positive when provided.")
    if impact_root_tolerance <= 0.0:
        raise ValueError("impact_root_tolerance must be positive.")
    if impact_root_max_iterations < 1:
        raise ValueError("impact_root_max_iterations must be positive.")

    torque_policy = torque_policy or (lambda _t, _s: 0.0)
    endpoint_force_policy = endpoint_force_policy or (
        lambda _t, _s: np.zeros(2, dtype=float)
    )
    generalized_force_policy = generalized_force_policy or (
        lambda _t, _s: np.zeros(2, dtype=float)
    )
    switch_policy = switch_policy or (
        lambda _t, _s, _sp, _ip, _sl: SwitchDecision.SWITCH
    )

    state = initial_state
    support_point = np.asarray(initial_support_point, dtype=float)
    initial_points = forward_kinematics(state.q, support_point, model.p.l1, model.p.l2)
    initial_free_distance = slope.signed_distance(initial_points.free)
    initial_elbow_distance = slope.signed_distance(initial_points.elbow)
    if require_initial_below_slope:
        if initial_free_distance > initial_geometry_tolerance:
            raise ValueError(
                "Initial free end penetrates slope: "
                f"signed_dist = {initial_free_distance:.3e} m"
            )
        if initial_elbow_distance > initial_geometry_tolerance:
            raise ValueError(
                "Initial elbow penetrates slope: "
                f"signed_dist = {initial_elbow_distance:.3e} m"
            )

    samples: list[SimulationSample] = []
    has_left_slope = initial_free_distance < -leave_tol
    release_count = 0

    def _record(
        time: float,
        state: BrachiationState,
        support_point: np.ndarray,
        phase: SimPhase,
        elbow_torque: float,
        endpoint_force: np.ndarray,
        generalized_force: np.ndarray,
    ) -> None:
        pts = forward_kinematics(state.q, support_point, model.p.l1, model.p.l2)
        samples.append(
            SimulationSample(
                time=time,
                state=state,
                support_point=support_point.copy(),
                free_end=pts.free.copy(),
                elbow=pts.elbow.copy(),
                phase=phase,
                elbow_torque=elbow_torque,
                external_endpoint_force_yz=endpoint_force.copy(),
                external_generalized_force=generalized_force.copy(),
                kinetic_energy=model.kinetic_energy(state.q, state.qd),
                potential_energy=model.potential_energy(
                    state.q,
                    support_z=float(support_point[1]),
                ),
                total_energy=model.total_energy(
                    state.q,
                    state.qd,
                    support_z=float(support_point[1]),
                ),
            )
        )

    time = 0.0
    _record(
        time,
        state,
        support_point,
        SimPhase.SWINGING,
        0.0,
        np.zeros(2),
        np.zeros(2),
    )

    while time < duration:
        elbow_torque = float(torque_policy(time, state))
        endpoint_force = np.asarray(endpoint_force_policy(time, state), dtype=float)
        generalized_force = np.asarray(
            generalized_force_policy(time, state),
            dtype=float,
        )
        if endpoint_force.shape != (2,):
            raise ValueError("endpoint_force_policy must return [Fy, Fz].")
        if generalized_force.shape != (2,):
            raise ValueError("generalized_force_policy must return [Q1, Q2].")

        state_next = model.step_rk4(
            state,
            dt=dt,
            elbow_torque=elbow_torque,
            external_endpoint_force_yz=endpoint_force,
            external_generalized_force=generalized_force,
        )
        time_next = time + dt

        def _state_at_fraction(alpha: float) -> BrachiationState:
            alpha = float(np.clip(alpha, 0.0, 1.0))
            if alpha <= 0.0:
                return state
            if alpha >= 1.0:
                return state_next
            return model.step_rk4(
                state,
                dt=alpha * dt,
                elbow_torque=elbow_torque,
                external_endpoint_force_yz=endpoint_force,
                external_generalized_force=generalized_force,
            )

        impact, has_left_slope = detect_slope_impact(
            state_prev=state,
            state_curr=state_next,
            support_point=support_point,
            slope=slope,
            parameters=model.p,
            t_prev=time,
            t_curr=time_next,
            has_left_slope=has_left_slope,
            leave_tol=leave_tol,
            state_at_fraction=_state_at_fraction,
            root_tolerance=impact_root_tolerance,
            root_max_iterations=impact_root_max_iterations,
        )

        if impact.contact_occurred:
            qd_impact = (
                impact.impact_velocity.copy()
                if impact.impact_velocity is not None
                else state.qd
                + ((impact.impact_time - time) / (time_next - time))
                * (state_next.qd - state.qd)
            )
            impact_state = BrachiationState(
                q=impact.impact_state.copy(),
                qd=qd_impact,
                support_index=state.support_index,
            )
            _record(
                impact.impact_time,
                impact_state,
                support_point,
                SimPhase.IMPACT,
                elbow_torque,
                endpoint_force,
                generalized_force,
            )

            state = impact_state
            time = impact.impact_time

            decision = switch_policy(
                time,
                state,
                support_point,
                impact.impact_point,
                slope,
            )
            mass_matrix = model.mass_matrix(impact.impact_state)
            context = CollisionContext(
                state=state,
                support_point=support_point,
                collision_point=impact.impact_point,
                parameters=model.p,
                decision=decision,
                slope=slope,
                mass_matrix=mass_matrix,
            )
            if collision_model is not None:
                result = collision_model(context)
            else:
                result = apply_switch_decision(
                    state=state,
                    support_point=support_point,
                    collision_point=impact.impact_point,
                    parameters=model.p,
                    decision=decision,
                    slope=slope,
                    mass_matrix=mass_matrix,
                    collision_mode=collision_mode,
                )

            state = result.state
            if result.phase == SwitchDecision.SWITCH:
                support_point = impact.impact_point.copy()
                state = BrachiationState(
                    q=state.q.copy(),
                    qd=state.qd.copy(),
                    support_index=state.support_index + 1,
                )

            record_phase = (
                SimPhase.DWELL
                if result.phase == SwitchDecision.DWELL
                else SimPhase.RELEASE
            )
            _record(
                time,
                state,
                support_point,
                record_phase,
                0.0,
                np.zeros(2),
                np.zeros(2),
            )
            if record_phase == SimPhase.RELEASE:
                release_count += 1
                if (
                    stop_after_releases is not None
                    and release_count >= stop_after_releases
                ):
                    break
            has_left_slope = False
            continue

        state = state_next
        time = time_next
        _record(
            time,
            state,
            support_point,
            SimPhase.SWINGING,
            elbow_torque,
            endpoint_force,
            generalized_force,
        )

    return samples
