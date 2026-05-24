"""Support-switching and collision logic for passive brachiation on a slope.

The free endpoint contacts the slope when it crosses the surface.  Collision
handling is deliberately pluggable: the default model is the full-grab impact
map used by the ``passive-brachiator-main`` reference project, and callers may
provide their own collision function for future shooting models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .kinematics import Slope, angle_from_vertical_down, forward_kinematics, normalize_angle
from .model import BrachiationParameters, BrachiationState


class SwitchDecision(Enum):
    """Decision made after the free endpoint contacts the slope."""

    SWITCH = "switch"
    NO_SWITCH = "no_switch"
    DWELL = "dwell"


class CollisionMode(Enum):
    """Built-in collision models available to the simulator.

    ``FULL_GRAB_1D`` is the reference inverted-compass grab: the collision
    point becomes a fixed support, and a full endpoint velocity constraint is
    imposed through a KKT impact solve.

    ``NORMAL_PLASTIC`` keeps the legacy sliding-contact model that only removes
    the slope-normal endpoint velocity.  It is useful for diagnostics, but it
    is not a fixed-support grab.
    """

    FULL_GRAB_1D = "full_grab_1d"
    NORMAL_PLASTIC = "normal_plastic"


@dataclass(frozen=True)
class ImpactResult:
    """Result returned by ``detect_slope_impact``."""

    contact_occurred: bool
    impact_time: float | None
    impact_state: np.ndarray | None
    impact_point: np.ndarray | None
    impact_velocity: np.ndarray | None = None


@dataclass(frozen=True)
class SwitchResult:
    """Result returned by a collision or switch handler."""

    state: BrachiationState
    phase: SwitchDecision


@dataclass(frozen=True)
class CollisionContext:
    """Inputs passed to a collision model.

    A future 2-D shooting collision map can implement the same call signature
    and return a ``SwitchResult`` without changing the simulation loop.
    """

    state: BrachiationState
    support_point: np.ndarray
    collision_point: np.ndarray
    parameters: BrachiationParameters
    decision: SwitchDecision
    slope: Slope | None = None
    mass_matrix: np.ndarray | None = None


SwitchPolicy = Callable[
    [float, BrachiationState, np.ndarray, np.ndarray, Slope],
    SwitchDecision,
]
CollisionModel = Callable[[CollisionContext], SwitchResult]


def _free_endpoint_jacobian(q: np.ndarray, l1: float, l2: float) -> np.ndarray:
    """Return the 2x2 Jacobian mapping q_dot to free-end [y_dot, z_dot]."""
    q1, q2 = q
    q12 = q1 + q2
    return np.array(
        [
            [l1 * np.cos(q1) + l2 * np.cos(q12), l2 * np.cos(q12)],
            [l1 * np.sin(q1) + l2 * np.sin(q12), l2 * np.sin(q12)],
        ],
        dtype=float,
    )


def compute_plastic_collision_velocity(
    q: np.ndarray,
    qd_before: np.ndarray,
    slope: Slope,
    mass_matrix: np.ndarray,
    l1: float,
    l2: float,
) -> np.ndarray:
    """Project velocity so the free-end slope-normal velocity is removed.

    This is the legacy sliding-contact collision model.  It preserves the
    tangential velocity component and therefore should not be treated as a
    fixed-support grab in a two-DOF chain.
    """
    J = _free_endpoint_jacobian(np.asarray(q, dtype=float), l1, l2)
    n = np.array([np.sin(slope.gamma), np.cos(slope.gamma)], dtype=float)
    Jn = n @ J
    vn_before = float(Jn @ qd_before)

    if vn_before <= 0.0:
        return np.asarray(qd_before, dtype=float).copy()

    D = np.asarray(mass_matrix, dtype=float)
    D_inv_JnT = np.linalg.solve(D, Jn.reshape(-1, 1))
    lambda_n_inv = float(Jn @ D_inv_JnT)

    if lambda_n_inv < 1e-14:
        return np.zeros(2, dtype=float)

    lambda_n = 1.0 / lambda_n_inv
    impulse = np.linalg.solve(D, Jn) * (lambda_n * vn_before)
    return np.asarray(qd_before, dtype=float) - impulse


def compute_full_grab_collision_velocity(
    q: np.ndarray,
    qd_before: np.ndarray,
    mass_matrix: np.ndarray,
    l1: float,
    l2: float,
) -> np.ndarray:
    """Return post-impact velocity for a fixed-support grab.

    This mirrors ``grab_transition_new`` in ``passive-brachiator-main``.  The
    impact impulse is chosen so that the just-contacted endpoint has zero world
    velocity after impact:

    ``J(q) qd_after = 0``.

    In KKT form:

    ``[M -J.T; J 0] [qd_after; impulse] = [M qd_before; 0]``.
    """
    q = np.asarray(q, dtype=float)
    qd_before = np.asarray(qd_before, dtype=float)
    M = np.asarray(mass_matrix, dtype=float)
    J = _free_endpoint_jacobian(q, l1, l2)

    system = np.block(
        [
            [M, -J.T],
            [J, np.zeros((J.shape[0], J.shape[0]), dtype=float)],
        ]
    )
    rhs = np.concatenate((M @ qd_before, np.zeros(J.shape[0], dtype=float)))

    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, rhs, rcond=None)[0]

    return np.asarray(solution[:2], dtype=float)


def switch_support(
    state: BrachiationState,
    old_support: np.ndarray,
    elbow: np.ndarray,
    new_support: np.ndarray,
) -> BrachiationState:
    """Rewrite state after the free endpoint becomes the support endpoint.

    Before switching: ``old_support -> elbow -> new_support/free_end``.
    After switching:  ``new_support -> elbow -> old_support/free_end``.

    The velocity map is the pure relabeling transform
    ``qd_new = [[1, 1], [0, -1]] @ qd_old``.  Any collision impulse must be
    applied before this coordinate swap.
    """
    qd1_old, qd2_old = state.qd

    new_first_link = np.asarray(elbow, dtype=float) - np.asarray(new_support, dtype=float)
    new_second_link = np.asarray(old_support, dtype=float) - np.asarray(elbow, dtype=float)

    new_q1 = angle_from_vertical_down(new_first_link)
    new_abs_q2 = angle_from_vertical_down(new_second_link)
    new_q2 = normalize_angle(new_abs_q2 - new_q1)
    qd_new = np.array([qd1_old + qd2_old, -qd2_old], dtype=float)

    return BrachiationState(
        q=np.array([normalize_angle(new_q1), new_q2], dtype=float),
        qd=qd_new,
        support_index=state.support_index,
    )


def _coerce_collision_mode(mode: CollisionMode | str | None) -> CollisionMode:
    if mode is None:
        return CollisionMode.FULL_GRAB_1D
    if isinstance(mode, CollisionMode):
        return mode
    return CollisionMode(str(mode))


def apply_switch_decision(
    state: BrachiationState,
    support_point: np.ndarray,
    collision_point: np.ndarray,
    parameters: BrachiationParameters,
    decision: SwitchDecision,
    slope: Slope | None = None,
    mass_matrix: np.ndarray | None = None,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
) -> SwitchResult:
    """Apply a built-in collision model and optional support switch."""
    context = CollisionContext(
        state=state,
        support_point=support_point,
        collision_point=collision_point,
        parameters=parameters,
        decision=decision,
        slope=slope,
        mass_matrix=mass_matrix,
    )
    return apply_collision_model(context, collision_mode=collision_mode)


def apply_collision_model(
    context: CollisionContext,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
) -> SwitchResult:
    """Apply one of the built-in collision models."""
    mode = _coerce_collision_mode(collision_mode)

    if context.decision == SwitchDecision.DWELL:
        return SwitchResult(
            state=BrachiationState(
                q=context.state.q.copy(),
                qd=np.zeros(2, dtype=float),
                support_index=context.state.support_index,
            ),
            phase=SwitchDecision.DWELL,
        )

    if context.decision == SwitchDecision.SWITCH:
        return _apply_switch_collision(context, mode)

    if context.decision == SwitchDecision.NO_SWITCH:
        return _apply_no_switch_collision(context, mode)

    raise ValueError(f"Unknown SwitchDecision: {context.decision}")


def _require_mass_matrix(context: CollisionContext) -> np.ndarray:
    if context.mass_matrix is None:
        raise ValueError("mass_matrix is required for collision handling.")
    return context.mass_matrix


def _apply_switch_collision(
    context: CollisionContext,
    mode: CollisionMode,
) -> SwitchResult:
    pts = forward_kinematics(
        context.state.q,
        context.support_point,
        context.parameters.l1,
        context.parameters.l2,
    )

    if mode == CollisionMode.FULL_GRAB_1D:
        qd_after = compute_full_grab_collision_velocity(
            q=context.state.q,
            qd_before=context.state.qd,
            mass_matrix=_require_mass_matrix(context),
            l1=context.parameters.l1,
            l2=context.parameters.l2,
        )
    elif mode == CollisionMode.NORMAL_PLASTIC:
        if context.slope is None:
            raise ValueError("slope is required for NORMAL_PLASTIC collision.")
        qd_after = compute_plastic_collision_velocity(
            q=context.state.q,
            qd_before=context.state.qd,
            slope=context.slope,
            mass_matrix=_require_mass_matrix(context),
            l1=context.parameters.l1,
            l2=context.parameters.l2,
        )
    else:
        raise ValueError(f"Unsupported collision mode: {mode}")

    collided = BrachiationState(
        q=context.state.q.copy(),
        qd=qd_after,
        support_index=context.state.support_index,
    )
    new_state = switch_support(
        state=collided,
        old_support=context.support_point,
        elbow=pts.elbow,
        new_support=context.collision_point,
    )
    return SwitchResult(state=new_state, phase=SwitchDecision.SWITCH)


def _apply_no_switch_collision(
    context: CollisionContext,
    mode: CollisionMode,
) -> SwitchResult:
    if mode == CollisionMode.FULL_GRAB_1D:
        # No grab means no support transfer.  The reference full-grab impact map
        # is only meaningful when SWITCH is selected, so leave the state as-is.
        return SwitchResult(state=context.state, phase=SwitchDecision.NO_SWITCH)

    if mode == CollisionMode.NORMAL_PLASTIC:
        if context.slope is None:
            raise ValueError("slope is required for NORMAL_PLASTIC collision.")
        qd_after = compute_plastic_collision_velocity(
            q=context.state.q,
            qd_before=context.state.qd,
            slope=context.slope,
            mass_matrix=_require_mass_matrix(context),
            l1=context.parameters.l1,
            l2=context.parameters.l2,
        )
        return SwitchResult(
            state=BrachiationState(
                q=context.state.q.copy(),
                qd=qd_after,
                support_index=context.state.support_index,
            ),
            phase=SwitchDecision.NO_SWITCH,
        )

    raise ValueError(f"Unsupported collision mode: {mode}")


def detect_slope_impact(
    state_prev: BrachiationState,
    state_curr: BrachiationState,
    support_point: np.ndarray,
    slope: Slope,
    parameters: BrachiationParameters,
    t_prev: float,
    t_curr: float,
    has_left_slope: bool = True,
    leave_tol: float = 1e-3,
    state_at_fraction: Callable[[float], BrachiationState] | None = None,
    root_tolerance: float = 1e-10,
    root_max_iterations: int = 50,
) -> tuple[ImpactResult, bool]:
    """Detect whether the free endpoint crossed the slope surface.

    The free end must first move below the slope by ``leave_tol`` before a
    zero-crossing can count as an impact.  This keeps release-section starts
    and small numerical chatter from being recorded as immediate re-impacts.

    When ``state_at_fraction`` is supplied, the impact time is refined by
    bisection on the true signed distance along the RK substep trajectory.  The
    legacy linear interpolation path is kept as a fallback for direct callers.
    """
    pts_prev = forward_kinematics(state_prev.q, support_point, parameters.l1, parameters.l2)
    pts_curr = forward_kinematics(state_curr.q, support_point, parameters.l1, parameters.l2)

    d_prev = slope.signed_distance(pts_prev.free)
    d_curr = slope.signed_distance(pts_curr.free)
    no_impact = ImpactResult(
        contact_occurred=False,
        impact_time=None,
        impact_state=None,
        impact_point=None,
        impact_velocity=None,
    )

    if not has_left_slope:
        return no_impact, d_curr < -leave_tol

    if d_prev < 0.0 and d_curr >= 0.0:
        if state_at_fraction is None:
            alpha = d_prev / (d_prev - d_curr)
            q_impact = state_prev.q + alpha * (state_curr.q - state_prev.q)
            qd_impact = state_prev.qd + alpha * (state_curr.qd - state_prev.qd)
            impact_point = pts_prev.free + alpha * (pts_curr.free - pts_prev.free)
        else:
            lo = 0.0
            hi = 1.0
            alpha = d_prev / (d_prev - d_curr)

            for _ in range(root_max_iterations):
                mid = 0.5 * (lo + hi)
                state_mid = state_at_fraction(mid)
                pts_mid = forward_kinematics(
                    state_mid.q,
                    support_point,
                    parameters.l1,
                    parameters.l2,
                )
                d_mid = slope.signed_distance(pts_mid.free)
                if abs(d_mid) <= root_tolerance or (hi - lo) <= root_tolerance:
                    alpha = mid
                    break
                if d_mid < 0.0:
                    lo = mid
                else:
                    hi = mid
                alpha = 0.5 * (lo + hi)

            state_impact = state_at_fraction(alpha)
            pts_impact = forward_kinematics(
                state_impact.q,
                support_point,
                parameters.l1,
                parameters.l2,
            )
            q_impact = state_impact.q.copy()
            qd_impact = state_impact.qd.copy()
            impact_point = pts_impact.free.copy()

        t_impact = t_prev + alpha * (t_curr - t_prev)
        return (
            ImpactResult(
                contact_occurred=True,
                impact_time=t_impact,
                impact_state=q_impact,
                impact_point=impact_point,
                impact_velocity=qd_impact,
            ),
            has_left_slope,
        )

    return no_impact, has_left_slope
