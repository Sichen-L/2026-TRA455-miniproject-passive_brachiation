"""Kinematics helpers for the simplified brachiation model.

Coordinate convention
---------------------
The motion is restricted to the y-z plane:

* y is the horizontal direction along the slope.
* z is vertical, positive upward.
* the inclined slope surface is defined by ``z = -y * tan(gamma)``.
* the attached endpoint is called the support point.

Angle convention
----------------
``q1`` is the absolute angle of the first link measured from vertical down.
With ``q1 = 0`` the first link points straight downward from the support rung.

``q2`` is the relative angle at the middle joint.  The second link absolute
angle is therefore ``q1 + q2``.

The forward kinematics are intentionally the same as the notebook formulas:

``ee_y = l1 * sin(q1) + l2 * sin(q1 + q2)``
``ee_z = -l1 * cos(q1) - l2 * cos(q1 + q2)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LinkPoints:
    """World-frame points of the two-link model.

    ``support`` is the attached endpoint.
    ``elbow`` is the middle revolute joint.
    ``free`` is the endpoint that may contact the slope surface.
    """

    support: np.ndarray
    elbow: np.ndarray
    free: np.ndarray


@dataclass(frozen=True)
class Slope:
    """An infinitely extending inclined plane.

    The slope is defined by the angle ``gamma`` (in radians) measured
    downward from the horizontal.  A point ``[y, z]`` lies on the surface
    when ``z = -y * tan(gamma)``.

    Positive ``gamma`` means the slope descends as y increases (surface
    normal points up-left).

    Parameters
    ----------
    gamma:
        Slope inclination angle in radians (>= 0).
    """

    gamma: float

    def signed_distance(self, point_yz: Iterable[float]) -> float:
        """Return the signed distance from ``point_yz`` to the slope.

        The distance is measured normal to the surface.

        In this framework the robot hangs *below* a ceiling-like slope, so
        the **free space** corresponds to **negative** signed distances.
        A positive value means the point has crossed or penetrated the
        surface from below.
        """
        y, z = np.asarray(point_yz, dtype=float)
        # Surface normal pointing upward-left: [sin(gamma), cos(gamma)]
        # Signed distance of point [y, z] above the plane z = -y*tan(gamma):
        #   d = y*sin(gamma) + z*cos(gamma)
        return float(y * np.sin(self.gamma) + z * np.cos(self.gamma))

    def is_penetrated(self, point_yz: Iterable[float], tolerance: float = 1e-6) -> bool:
        """Return ``True`` if ``point_yz`` has crossed above the slope surface.

        A small positive *tolerance* avoids noise‑triggered false positives.
        """
        return self.signed_distance(point_yz) > tolerance

    def project_to_slope(self, y: float) -> np.ndarray:
        """Return the point ``[y, -y*tan(gamma)]`` on the slope surface."""
        return np.array([y, -y * np.tan(self.gamma)], dtype=float)


def forward_kinematics(
    q: Iterable[float],
    support: Iterable[float],
    l1: float,
    l2: float,
) -> LinkPoints:
    """Compute support, elbow, and free-end points.

    The function is deliberately stateless.  The current support point is
    passed in explicitly, so support switching can be handled outside the
    kinematics without hiding state changes.
    """

    q1, q2 = np.asarray(q, dtype=float)
    support_point = np.asarray(support, dtype=float)

    first_link = np.array([l1 * np.sin(q1), -l1 * np.cos(q1)])
    second_link = np.array([l2 * np.sin(q1 + q2), -l2 * np.cos(q1 + q2)])

    elbow = support_point + first_link
    free = elbow + second_link

    return LinkPoints(support=support_point, elbow=elbow, free=free)


def angle_from_vertical_down(vector_yz: Iterable[float]) -> float:
    """Return the absolute link angle for a y-z vector.

    A link vector is represented as ``[l*sin(theta), -l*cos(theta)]``.
    Solving that relation gives ``theta = atan2(y, -z)``.
    """

    y, z = np.asarray(vector_yz, dtype=float)
    return float(np.arctan2(y, -z))


def normalize_angle(angle: float) -> float:
    """Wrap an angle to the interval ``[-pi, pi)`` for easier inspection."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)