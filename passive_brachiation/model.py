"""Dynamic model for a simplified two-link passive-brachiation system.

The model is close to a standard Acrobot:

* the first joint is the currently attached hand / hook on a rung,
* the second joint is the middle elbow joint,
* the other endpoint is a free hook that may later become the support point.

The default model treats each link as a light rod with an embedded movable
point mass.  For a more conventional course-theory model, use the explicit
``uniform_links`` constructor:

``lc = l / 2`` and ``I = (1/12) * m * l^2``.

Important simplification
------------------------
The free endpoint is allowed to rotate as a hook/contact point, but that hook
rotation is not an independent generalized coordinate here.  This framework
tracks the two shape coordinates needed for planar brachiation.  If later you
need hook orientation, it can be added as a separate passive coordinate without
changing the support-switching idea.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .kinematics import LinkPoints, forward_kinematics


@dataclass(frozen=True)
class BrachiationParameters:
    """Physical parameters for the two-link model.

    ``m1`` and ``m2`` are the masses assigned to link 1 and link 2.
    ``l1`` and ``l2`` are joint-to-joint lengths.
    ``lc1`` and ``lc2`` are distances from each link's proximal joint to its
    center of mass.
    ``I1`` and ``I2`` are moments of inertia about each link COM, around the
    axis perpendicular to the y-z motion plane.
    ``damping1`` and ``damping2`` are viscous joint-friction coefficients.
    They are generalized torques proportional to ``-q_dot`` and default to
    zero so the original passive model is unchanged unless you opt in.

    The defaults match the rod + centred movable point-mass model: both links
    have the same mass and length, 20% of each link mass is a uniform light
    rod, and 80% is a point mass initially centred on the rod.
    """

    m1: float = 1.041
    m2: float = 1.041
    l1: float = 0.314
    l2: float = 0.314
    lc1: float = 0.157
    lc2: float = 0.157
    I1: float = 0.0017106406
    I2: float = 0.0017106406
    damping1: float = 0.0
    damping2: float = 0.0
    gravity: float = 9.81

    @classmethod
    def uniform_links(
        cls,
        m1: float = 1.041,
        m2: float = 1.041,
        l1: float = 0.314,
        l2: float = 0.314,
        damping1: float = 0.0,
        damping2: float = 0.0,
        gravity: float = 9.81,
    ) -> "BrachiationParameters":
        """Build parameters using the uniform-slender-link assumption."""

        return cls(
            m1=m1,
            m2=m2,
            l1=l1,
            l2=l2,
            lc1=0.5 * l1,
            lc2=0.5 * l2,
            I1=(1.0 / 12.0) * m1 * l1**2,
            I2=(1.0 / 12.0) * m2 * l2**2,
            damping1=damping1,
            damping2=damping2,
            gravity=gravity,
        )

    @classmethod
    def light_rods_with_joint_masses(
        cls,
        middle_mass: float = 1.041,
        free_end_mass: float = 1.041,
        l1: float = 0.314,
        l2: float = 0.314,
        damping1: float = 0.0,
        damping2: float = 0.0,
        gravity: float = 9.81,
    ) -> "BrachiationParameters":
        """Build a massless-rod / point-mass style model.

        In this approximation each rod has negligible rotational inertia.
        ``middle_mass`` is placed at the end of link 1, so ``lc1 = l1``.
        ``free_end_mass`` is placed at the end of link 2, so ``lc2 = l2``.

        This is useful when the desired abstraction is literally "light rods
        connected by revolute joints".  The inertia values are set to zero, and
        the point masses contribute to the system inertia through their COM
        distances in the mass matrix.
        """

        return cls(
            m1=middle_mass,
            m2=free_end_mass,
            l1=l1,
            l2=l2,
            lc1=l1,
            lc2=l2,
            I1=0.0,
            I2=0.0,
            damping1=damping1,
            damping2=damping2,
            gravity=gravity,
        )

    @staticmethod
    def _rod_point_link(
        mass: float,
        length: float,
        weight_position_fraction: float,
        rod_mass_fraction: float = 0.2,
    ) -> tuple[float, float]:
        """Return ``(lc, I_centroidal)`` for one rod + movable point-mass link.

        The link mass is split into a uniform light rod (``rod_mass_fraction`` of
        the mass, spread over the whole length) and a point mass (the rest) at
        ``weight_position_fraction`` of the length from the proximal joint.
        Moving the point mass changes ``lc`` and the centroidal inertia ``I``
        together, so the body stays physically realizable.
        """

        a = float(weight_position_fraction) * length
        rod_mass = rod_mass_fraction * mass
        point_mass = (1.0 - rod_mass_fraction) * mass
        lc = rod_mass_fraction * (0.5 * length) + (1.0 - rod_mass_fraction) * a
        inertia = (
            (1.0 / 12.0) * rod_mass * length**2
            + rod_mass * (0.5 * length - lc) ** 2
            + point_mass * (a - lc) ** 2
        )
        return lc, inertia

    @classmethod
    def rod_point_mass(
        cls,
        weight_position1: float = 0.5,
        weight_position2: float = 0.5,
        m1: float = 1.041,
        m2: float = 1.041,
        l1: float = 0.314,
        l2: float = 0.314,
        rod_mass_fraction: float = 0.2,
        damping1: float = 0.0,
        damping2: float = 0.0,
        gravity: float = 9.81,
    ) -> "BrachiationParameters":
        """Build parameters for the rod + movable point-mass model.

        Each link keeps its total mass ``mN`` but splits it into a uniform light
        rod (``rod_mass_fraction``) and a point mass at ``weight_positionN`` (a
        fraction in ``[0, 1]`` of the link length from the proximal joint).
        ``lc`` and ``I`` are derived consistently, so sweeping the weight
        position only visits realizable bodies.  With ``rod_mass_fraction=0.2``
        and the weight centred (``0.5``), ``lc`` equals the uniform-link value
        while ``I`` is smaller (a centred point mass adds no centroidal inertia).
        The reachable COM fraction is ``lc/L in [rod_mass_fraction/2,
        1 - rod_mass_fraction/2]`` (e.g. ``[0.1, 0.9]`` for the 0.2 default).
        """

        if not 0.0 < rod_mass_fraction < 1.0:
            raise ValueError("rod_mass_fraction must be in (0, 1).")
        lc1, inertia1 = cls._rod_point_link(m1, l1, weight_position1, rod_mass_fraction)
        lc2, inertia2 = cls._rod_point_link(m2, l2, weight_position2, rod_mass_fraction)
        return cls(
            m1=m1,
            m2=m2,
            l1=l1,
            l2=l2,
            lc1=lc1,
            lc2=lc2,
            I1=inertia1,
            I2=inertia2,
            damping1=damping1,
            damping2=damping2,
            gravity=gravity,
        )


@dataclass(frozen=True)
class BrachiationState:
    """State of the currently attached two-link system.

    ``q`` stores ``[q1, q2]``:

    * ``q1``: absolute angle of the support link from vertical down.
    * ``q2``: relative angle from link 1 to link 2.

    ``qd`` stores the corresponding angular velocities.

    ``support_index`` says which ladder rung is currently the attached endpoint.
    The support point itself is obtained from the ``Ladder`` object, so a state
    remains compact and independent of a particular ladder geometry.
    """

    q: np.ndarray
    qd: np.ndarray
    support_index: int

    @classmethod
    def from_values(
        cls,
        q1: float,
        q2: float,
        q1_dot: float = 0.0,
        q2_dot: float = 0.0,
        support_index: int = 0,
    ) -> "BrachiationState":
        """Convenience constructor that accepts scalar values."""

        return cls(
            q=np.array([q1, q2], dtype=float),
            qd=np.array([q1_dot, q2_dot], dtype=float),
            support_index=support_index,
        )


class TwoLinkBrachiationModel:
    """Two-link planar dynamics plus kinematics.

    The class does not own a ladder.  It only knows the physics of the current
    two-link chain.  Ladder contact and hand switching live in ``switching.py``.
    Keeping those responsibilities separate makes it easier to test and modify
    contact logic without touching the equations of motion.
    """

    def __init__(self, parameters: BrachiationParameters | None = None) -> None:
        self.p = parameters or BrachiationParameters.rod_point_mass()

    def points(self, state: BrachiationState, support: Iterable[float]) -> LinkPoints:
        """Return support, elbow, and free-end positions for ``state``."""

        return forward_kinematics(state.q, support, self.p.l1, self.p.l2)

    def mass_matrix(self, q: Iterable[float]) -> np.ndarray:
        """Return the 2x2 manipulator mass matrix ``D(q)``.

        This is the standard planar two-link mass matrix.  It is valid for both
        uniform links and the light-rod point-mass approximation, because those
        choices are expressed through ``lc`` and ``I``.
        """

        _, q2 = np.asarray(q, dtype=float)
        p = self.p

        d11 = (
            p.I1
            + p.I2
            + p.m1 * p.lc1**2
            + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * np.cos(q2))
        )
        d12 = p.I2 + p.m2 * (p.lc2**2 + p.l1 * p.lc2 * np.cos(q2))
        d22 = p.I2 + p.m2 * p.lc2**2

        return np.array([[d11, d12], [d12, d22]], dtype=float)

    def coriolis_centrifugal(self, q: Iterable[float], qd: Iterable[float]) -> np.ndarray:
        """Return velocity-dependent terms ``C(q, qd) * qd``.

        The returned vector is the term that appears in:

        ``D(q) * qdd + C(q, qd) * qd + G(q) = B * tau``

        It is written as a vector because the framework does not need the full
        Coriolis matrix explicitly.
        """

        _, q2 = np.asarray(q, dtype=float)
        q1_dot, q2_dot = np.asarray(qd, dtype=float)
        p = self.p

        h = p.m2 * p.l1 * p.lc2 * np.sin(q2)

        c1 = -h * (2.0 * q1_dot * q2_dot + q2_dot**2)
        c2 = h * q1_dot**2

        return np.array([c1, c2], dtype=float)

    def gravity_terms(self, q: Iterable[float]) -> np.ndarray:
        """Return gravity generalized forces ``G(q)``.

        With the angle convention used here, ``q1 = q2 = 0`` means both links
        hang straight down.  At that configuration the gravity torque is zero,
        which is a useful sanity check for this function.
        """

        q1, q2 = np.asarray(q, dtype=float)
        p = self.p

        g1 = (
            p.m1 * p.gravity * p.lc1 * np.sin(q1)
            + p.m2
            * p.gravity
            * (p.l1 * np.sin(q1) + p.lc2 * np.sin(q1 + q2))
        )
        g2 = p.m2 * p.gravity * p.lc2 * np.sin(q1 + q2)

        return np.array([g1, g2], dtype=float)

    def damping_terms(self, qd: Iterable[float]) -> np.ndarray:
        """Return viscous joint-friction torques ``[b1*q1_dot, b2*q2_dot]``.

        The equations of motion subtract this vector.  In other words, a
        positive velocity creates a negative torque, and a negative velocity
        creates a positive torque.  With the default damping values of zero,
        this method returns ``[0, 0]``.
        """

        qd = np.asarray(qd, dtype=float)
        return np.array([self.p.damping1, self.p.damping2], dtype=float) * qd

    def free_endpoint_jacobian(self, q: Iterable[float]) -> np.ndarray:
        """Return the 2x2 Jacobian from ``q_dot`` to free-end ``[y_dot, z_dot]``.

        This is used for external forces.  If a wind or vibration force
        ``F = [Fy, Fz]`` acts at the free endpoint, the matching generalized
        force is ``J(q).T @ F``.
        """

        q1, q2 = np.asarray(q, dtype=float)
        p = self.p
        q12 = q1 + q2

        return np.array(
            [
                [p.l1 * np.cos(q1) + p.l2 * np.cos(q12), p.l2 * np.cos(q12)],
                [p.l1 * np.sin(q1) + p.l2 * np.sin(q12), p.l2 * np.sin(q12)],
            ],
            dtype=float,
        )

    def generalized_force_from_endpoint_force(
        self,
        q: Iterable[float],
        force_yz: Iterable[float] | None = None,
    ) -> np.ndarray:
        """Map an external free-end force to generalized coordinates.

        ``force_yz`` is a two-element vector ``[Fy, Fz]`` in Newtons.  Passing
        ``None`` means no external force.  This gives wind/vibration code a
        clean interface while keeping the core dynamics compact.
        """

        if force_yz is None:
            return np.zeros(2, dtype=float)

        force = np.asarray(force_yz, dtype=float)
        if force.shape != (2,):
            raise ValueError("force_yz must be a two-element vector [Fy, Fz].")

        return self.free_endpoint_jacobian(q).T @ force

    def kinetic_energy(self, q: Iterable[float], qd: Iterable[float]) -> float:
        """Return kinetic energy in Joules."""

        qd = np.asarray(qd, dtype=float)
        return float(0.5 * qd @ self.mass_matrix(q) @ qd)

    def potential_energy(self, q: Iterable[float], support_z: float = 0.0) -> float:
        """Return gravitational potential energy in Joules.

        The zero level is the global y-z origin.  ``support_z`` is the
        world-frame z-coordinate of the current support point, so that the
        potential energy is consistent across hand‑switches (the support
        point moves, but the global origin stays fixed).

        Parameters
        ----------
        q:
            Generalized coordinates ``[q1, q2]``.
        support_z:
            World-frame z-coordinate of the current support point.  Defaults
            to 0.0 so that the old one‑argument signature still works.
        """

        q1, q2 = np.asarray(q, dtype=float)
        p = self.p

        z1 = support_z - p.lc1 * np.cos(q1)
        z2 = support_z - p.l1 * np.cos(q1) - p.lc2 * np.cos(q1 + q2)

        return float(p.m1 * p.gravity * z1 + p.m2 * p.gravity * z2)

    def total_energy(
        self,
        q: Iterable[float],
        qd: Iterable[float],
        support_z: float = 0.0,
    ) -> float:
        """Return mechanical energy ``kinetic + potential`` in Joules.

        Parameters
        ----------
        q, qd:
            Generalized coordinates and velocities.
        support_z:
            Passed through to ``potential_energy``.
        """

        return self.kinetic_energy(q, qd) + self.potential_energy(q, support_z)

    def acceleration(
        self,
        q: Iterable[float],
        qd: Iterable[float],
        elbow_torque: float = 0.0,
        external_endpoint_force_yz: Iterable[float] | None = None,
        external_generalized_force: Iterable[float] | None = None,
    ) -> np.ndarray:
        """Compute ``qdd`` for the current state and elbow torque.

        The first joint is passive because it is the hook/support contact.
        ``elbow_torque`` acts only at the middle joint, so ``B*tau = [0, tau]``.
        For fully passive brachiation, call this with ``elbow_torque = 0``.

        ``external_endpoint_force_yz`` is a physical force applied at the free
        endpoint, expressed as ``[Fy, Fz]`` in Newtons.  ``external_generalized_force``
        is an optional direct generalized force ``[Q1, Q2]``.  Both default to
        zero and can be used together if needed.
        """

        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)

        generalized_torque = np.array([0.0, elbow_torque], dtype=float)
        endpoint_force = self.generalized_force_from_endpoint_force(
            q,
            external_endpoint_force_yz,
        )
        direct_force = (
            np.zeros(2, dtype=float)
            if external_generalized_force is None
            else np.asarray(external_generalized_force, dtype=float)
        )
        if direct_force.shape != (2,):
            raise ValueError("external_generalized_force must be [Q1, Q2].")

        rhs = (
            generalized_torque
            + endpoint_force
            + direct_force
            - self.damping_terms(qd)
            - self.coriolis_centrifugal(q, qd)
            - self.gravity_terms(q)
        )

        return np.linalg.solve(self.mass_matrix(q), rhs)

    def derivative(
        self,
        state: BrachiationState,
        elbow_torque: float = 0.0,
        external_endpoint_force_yz: Iterable[float] | None = None,
        external_generalized_force: Iterable[float] | None = None,
    ) -> np.ndarray:
        """Return continuous-time derivative ``[q_dot, q_ddot]``."""

        qdd = self.acceleration(
            state.q,
            state.qd,
            elbow_torque,
            external_endpoint_force_yz=external_endpoint_force_yz,
            external_generalized_force=external_generalized_force,
        )
        return np.concatenate((state.qd, qdd))

    def step_rk4(
        self,
        state: BrachiationState,
        dt: float,
        elbow_torque: float = 0.0,
        external_endpoint_force_yz: Iterable[float] | None = None,
        external_generalized_force: Iterable[float] | None = None,
    ) -> BrachiationState:
        """Advance the state by one fixed RK4 step.

        This is intentionally dependency-free.  It is accurate enough for a
        clean framework demo, while still being easy to replace later with
        Drake, scipy, or a variational integrator.
        """

        x0 = np.concatenate((state.q, state.qd))

        def f(x: np.ndarray) -> np.ndarray:
            local_state = BrachiationState(
                q=x[:2],
                qd=x[2:],
                support_index=state.support_index,
            )
            return self.derivative(
                local_state,
                elbow_torque,
                external_endpoint_force_yz=external_endpoint_force_yz,
                external_generalized_force=external_generalized_force,
            )

        k1 = f(x0)
        k2 = f(x0 + 0.5 * dt * k1)
        k3 = f(x0 + 0.5 * dt * k2)
        k4 = f(x0 + dt * k3)
        x_next = x0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return BrachiationState(
            q=x_next[:2],
            qd=x_next[2:],
            support_index=state.support_index,
        )
