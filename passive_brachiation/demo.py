"""Minimal runnable demo for the simplified brachiation framework.

Run from the project root:

    python -m passive_brachiation.demo

The demo intentionally prints numbers instead of plotting.  That keeps it
usable in a plain terminal and makes the model behavior easy to inspect.
"""

from __future__ import annotations

import numpy as np

from .kinematics import Slope
from .model import BrachiationParameters, BrachiationState, TwoLinkBrachiationModel
from .simulation import simulate
from .switching import SwitchDecision


def main() -> None:
    # Project-default rod + movable point-mass parameters.
    params = BrachiationParameters.rod_point_mass()
    model = TwoLinkBrachiationModel(params)

    # An 8-degree inclined slope (ceiling-like surface).
    slope = Slope(gamma=np.deg2rad(8.0))

    # Initial support point at the origin.
    initial_support_point = np.array([0.0, 0.0], dtype=float)

    # Start with both links hanging roughly straight down, a small
    # initial displacement so the robot starts swinging.
    initial_state = BrachiationState.from_values(
        q1=0.25,
        q2=-0.35,
        q1_dot=0.0,
        q2_dot=0.0,
        support_index=0,
    )

    def switch_policy(_time, _state, _support_point, _impact_point, _slope):
        # Always switch hands on contact (passive brachiation).
        return SwitchDecision.SWITCH

    def endpoint_force_policy(_time, _state):
        # Wind/vibration force at the free endpoint, [Fy, Fz] in Newtons.
        # Default demo value is zero disturbance.
        return np.zeros(2)

    samples = simulate(
        model=model,
        slope=slope,
        initial_state=initial_state,
        initial_support_point=initial_support_point,
        duration=1.0,
        dt=0.01,
        endpoint_force_policy=endpoint_force_policy,
        switch_policy=switch_policy,
    )

    final = samples[-1]
    print("Final time:", final.time)
    print("Final support point:", np.round(final.support_point, 6))
    print("Final q:", np.round(final.state.q, 6))
    print("Final qdot:", np.round(final.state.qd, 6))
    print("Final free end y-z:", np.round(final.free_end, 6))
    print("Final elbow torque:", final.elbow_torque)
    print("Final total energy:", round(final.total_energy, 6))
    print("Phases seen:", sorted({s.phase.value for s in samples}))


if __name__ == "__main__":
    main()
