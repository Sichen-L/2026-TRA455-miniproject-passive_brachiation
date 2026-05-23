"""Reusable simulation policies for passive-brachiation experiments."""

from __future__ import annotations

import numpy as np

from .switching import SwitchDecision


def zero_torque_policy(_time, _state) -> float:
    """Return zero elbow torque."""

    return 0.0


def zero_endpoint_force_policy(_time, _state) -> np.ndarray:
    """Return zero external free-end force ``[Fy, Fz]``."""

    return np.zeros(2, dtype=float)


def zero_generalized_force_policy(_time, _state) -> np.ndarray:
    """Return zero direct generalized force ``[Q1, Q2]``."""

    return np.zeros(2, dtype=float)


def always_switch_policy(_time, _state, _support_point, _impact_point, _slope) -> SwitchDecision:
    """Switch support at every detected impact."""

    return SwitchDecision.SWITCH


def never_switch_policy(_time, _state, _support_point, _impact_point, _slope) -> SwitchDecision:
    """Keep the current support after impact."""

    return SwitchDecision.NO_SWITCH


def dwell_policy(_time, _state, _support_point, _impact_point, _slope) -> SwitchDecision:
    """Enter double-support dwell at impact."""

    return SwitchDecision.DWELL
