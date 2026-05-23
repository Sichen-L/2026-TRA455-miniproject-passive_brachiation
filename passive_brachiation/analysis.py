"""Analysis helpers for simulation samples.

These utilities are intentionally independent of plotting so they can be used
from notebooks, scripts, or tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .kinematics import Slope
from .simulation import SimulationSample
from .shooting import stride_distance_from_point


def samples_to_arrays(
    samples: Sequence[SimulationSample],
    slope: Slope | None = None,
) -> dict[str, np.ndarray]:
    """Convert ``SimulationSample`` objects to numpy arrays.

    Passing ``slope`` adds ``free_dist`` and ``elbow_dist`` signed-distance
    arrays.  Without a slope those keys are omitted.
    """

    if not samples:
        raise ValueError("samples must not be empty.")

    history: dict[str, np.ndarray] = {
        "times": np.array([sample.time for sample in samples], dtype=float),
        "q": np.vstack([sample.state.q for sample in samples]),
        "qd": np.vstack([sample.state.qd for sample in samples]),
        "phase": np.array([sample.phase.value for sample in samples]),
        "elbows": np.vstack([sample.elbow for sample in samples]),
        "frees": np.vstack([sample.free_end for sample in samples]),
        "supports": np.vstack([sample.support_point for sample in samples]),
        "torques": np.array([sample.elbow_torque for sample in samples], dtype=float),
        "endpoint_forces": np.vstack([sample.external_endpoint_force_yz for sample in samples]),
        "generalized_forces": np.vstack([sample.external_generalized_force for sample in samples]),
        "total_energy": np.array([sample.total_energy for sample in samples], dtype=float),
        "kinetic_energy": np.array([sample.kinetic_energy for sample in samples], dtype=float),
        "potential_energy": np.array([sample.potential_energy for sample in samples], dtype=float),
    }

    if slope is not None:
        history["free_dist"] = np.array(
            [slope.signed_distance(sample.free_end) for sample in samples],
            dtype=float,
        )
        history["elbow_dist"] = np.array(
            [slope.signed_distance(sample.elbow) for sample in samples],
            dtype=float,
        )

    return history


def phase_mask(history: dict[str, Any], phase: str) -> np.ndarray:
    """Return a boolean mask for one phase value in a sample-history dict."""

    return np.asarray(history["phase"]) == phase


def release_indices(samples: Sequence[SimulationSample]) -> np.ndarray:
    """Return sample indices whose phase is ``release``."""

    return np.flatnonzero([sample.phase.value == "release" for sample in samples])


def impact_indices(samples: Sequence[SimulationSample]) -> np.ndarray:
    """Return sample indices whose phase is ``impact``."""

    return np.flatnonzero([sample.phase.value == "impact" for sample in samples])


def support_arclengths(
    samples: Sequence[SimulationSample],
    slope: Slope,
    support_origin: np.ndarray | None = None,
    direction: float = 1.0,
) -> np.ndarray:
    """Project every support point onto the slope tangent."""

    origin = np.zeros(2, dtype=float) if support_origin is None else np.asarray(support_origin, dtype=float)
    return np.array(
        [
            stride_distance_from_point(
                sample.support_point,
                slope=slope,
                support_point=origin,
                direction=direction,
            )
            for sample in samples
        ],
        dtype=float,
    )


def release_stride_distances(
    samples: Sequence[SimulationSample],
    slope: Slope,
    support_origin: np.ndarray | None = None,
    direction: float = 1.0,
    initial_arclength: float = 0.0,
) -> np.ndarray:
    """Return stride increments between release support points."""

    indices = release_indices(samples)
    if len(indices) == 0:
        return np.array([], dtype=float)

    support_s = support_arclengths(
        samples,
        slope=slope,
        support_origin=support_origin,
        direction=direction,
    )
    return np.diff(np.r_[float(initial_arclength), support_s[indices]])


def release_q2_values(samples: Sequence[SimulationSample]) -> np.ndarray:
    """Return ``q2`` values at release samples."""

    indices = release_indices(samples)
    if len(indices) == 0:
        return np.array([], dtype=float)
    return np.array([samples[index].state.q[1] for index in indices], dtype=float)


def tail_half_range(values: Sequence[float], max_count: int = 8) -> float:
    """Return max absolute deviation from the tail mean."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    tail = array[-min(int(max_count), array.size):]
    return float(np.max(np.abs(tail - np.mean(tail))))


def tail_std(values: Sequence[float], max_count: int = 8) -> float:
    """Return standard deviation of the last ``max_count`` values."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    tail = array[-min(int(max_count), array.size):]
    return float(np.std(tail))
