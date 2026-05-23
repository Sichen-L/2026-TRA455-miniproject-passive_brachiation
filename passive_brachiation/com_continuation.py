"""COM-offset continuation workflows for passive-brachiation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .analysis import release_indices, release_stride_distances
from .kinematics import Slope
from .model import BrachiationParameters, BrachiationState, TwoLinkBrachiationModel
from .policies import always_switch_policy
from .shooting import (
    ContinuationResult,
    StrideMap,
    continue_fixed_point_branch,
    evaluate_elbow_below_slope_section,
    evaluate_passive_brachiation_stride,
    make_iterated_stride_map,
    make_passive_brachiation_feasibility_check,
    make_passive_brachiation_stride_map,
)
from .simulation import SimulationSample
from .switching import CollisionMode, SwitchDecision


SwitchPolicy = Callable[[float, BrachiationState, np.ndarray, np.ndarray, Slope], SwitchDecision]


@dataclass(frozen=True)
class ComValidation:
    """Physical validation for one COM continuation point."""

    legal: bool
    max_elbow_distance: float
    min_elbow_distance: float
    release_stride: np.ndarray


@dataclass(frozen=True)
class ComContinuationRow:
    """One row in a symmetric COM-offset continuation table."""

    direction: str
    com_offset: float
    lc1_fraction: float
    lc2_fraction: float
    d_primary: float
    d_next: float
    stride_plot: float
    period: int
    converged: bool
    residual_norm: float
    spectral_radius: float
    eigen_real: float
    stable: bool
    legal: bool
    max_elbow_distance: float
    min_elbow_distance: float
    release_stride: np.ndarray
    failure_reason: str | None


@dataclass(frozen=True)
class SymmetricComContinuation:
    """Full continuation result for offsets in both directions."""

    contact_result: ContinuationResult
    elbow_result: ContinuationResult
    contact_rows: list[ComContinuationRow]
    elbow_rows: list[ComContinuationRow]
    rows: list[ComContinuationRow]


def com_fractions_from_offset(
    com_offset: float,
    center_fraction: float = 0.5,
) -> tuple[float, float]:
    """Return ``lc1/L1`` and ``lc2/L2`` for a symmetric COM offset."""

    offset = float(com_offset)
    return center_fraction + offset, center_fraction - offset


def parameters_with_symmetric_com_offset(
    com_offset: float,
    base: BrachiationParameters,
    center_fraction: float = 0.5,
) -> BrachiationParameters:
    """Move both link COMs symmetrically while keeping masses and inertias fixed."""

    lc1_fraction, lc2_fraction = com_fractions_from_offset(com_offset, center_fraction)
    if not (0.0 < lc1_fraction < 1.0 and 0.0 < lc2_fraction < 1.0):
        raise ValueError(
            "COM fractions must stay inside (0, 1); "
            f"got lc1/L1={lc1_fraction:.3f}, lc2/L2={lc2_fraction:.3f}"
        )

    return BrachiationParameters(
        m1=base.m1,
        m2=base.m2,
        l1=base.l1,
        l2=base.l2,
        lc1=lc1_fraction * base.l1,
        lc2=lc2_fraction * base.l2,
        I1=base.I1,
        I2=base.I2,
        damping1=base.damping1,
        damping2=base.damping2,
        gravity=base.gravity,
    )


def validate_com_point(
    com_offset: float,
    d_value: float,
    base_params: BrachiationParameters,
    slope: Slope,
    branch: str,
    period: int,
    dt: float = 0.005,
    t_max: float = 8.0,
    initial_support_point: np.ndarray | None = None,
    initial_direction: float = -1.0,
    impact_direction: float = 1.0,
    switch_policy: SwitchPolicy | None = None,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
    legality_tolerance: float = 1e-9,
) -> ComValidation:
    """Validate one COM-offset fixed-point candidate with a physical simulation."""

    support = (
        np.zeros(2, dtype=float)
        if initial_support_point is None
        else np.asarray(initial_support_point, dtype=float)
    )
    local_model = TwoLinkBrachiationModel(
        parameters_with_symmetric_com_offset(com_offset, base_params)
    )
    evaluation = evaluate_passive_brachiation_stride(
        model=local_model,
        slope=slope,
        x=np.array([d_value], dtype=float),
        dt=dt,
        t_max=t_max,
        collision_mode=collision_mode,
        initial_direction=initial_direction,
        impact_direction=impact_direction,
        branch=branch,
        support_point=support,
        switch_policy=switch_policy or always_switch_policy,
    )
    return _validate_com_samples(
        evaluation.samples,
        slope=slope,
        period=period,
        support=support,
        impact_direction=impact_direction,
        legality_tolerance=legality_tolerance,
    )


def _validate_com_samples(
    samples: list[SimulationSample],
    slope: Slope,
    period: int,
    support: np.ndarray,
    impact_direction: float,
    legality_tolerance: float,
) -> ComValidation:
    """Validate a COM continuation point from an already-run trajectory."""

    rel_indices = release_indices(samples)
    if len(rel_indices) < period:
        return ComValidation(False, np.nan, np.nan, np.array([], dtype=float))

    period_samples = samples[: rel_indices[period - 1] + 1]
    legality = evaluate_elbow_below_slope_section(
        period_samples,
        slope=slope,
        tolerance=legality_tolerance,
    )
    return ComValidation(
        legal=legality.legal,
        max_elbow_distance=legality.max_signed_distance,
        min_elbow_distance=legality.min_signed_distance,
        release_stride=release_stride_distances(
            samples,
            slope=slope,
            support_origin=support,
            direction=impact_direction,
        ),
    )


def run_symmetric_com_continuation(
    base_params: BrachiationParameters,
    slope: Slope,
    d_fixed: float,
    branch: str,
    period: int,
    offset_low: float = -0.3,
    offset_high: float = 0.3,
    n_steps_per_side: int = 31,
    dt: float = 0.005,
    t_max: float = 8.0,
    initial_support_point: np.ndarray | None = None,
    initial_direction: float = -1.0,
    impact_direction: float = 1.0,
    switch_policy: SwitchPolicy | None = None,
    collision_mode: CollisionMode | str = CollisionMode.FULL_GRAB_1D,
    continuation_tol: float = 1e-6,
    continuation_max_iter: int = 10,
    continuation_delta: float = 1e-5,
    continuation_damping: float = 0.8,
    center_fraction: float = 0.5,
) -> SymmetricComContinuation:
    """Trace fixed points as COMs move toward contact endpoints and elbow."""

    support = (
        np.zeros(2, dtype=float)
        if initial_support_point is None
        else np.asarray(initial_support_point, dtype=float)
    )
    policy = switch_policy or always_switch_policy

    def make_model(com_offset: float) -> TwoLinkBrachiationModel:
        return TwoLinkBrachiationModel(
            parameters_with_symmetric_com_offset(
                com_offset,
                base=base_params,
                center_fraction=center_fraction,
            )
        )

    def make_base_stride_map(com_offset: float) -> StrideMap:
        return make_passive_brachiation_stride_map(
            model=make_model(com_offset),
            slope=slope,
            dt=dt,
            t_max=t_max,
            collision_mode=collision_mode,
            initial_direction=initial_direction,
            impact_direction=impact_direction,
            branch=branch,
            support_point=support,
            switch_policy=policy,
        )

    def make_stride_map(com_offset: float) -> StrideMap:
        P_base = make_base_stride_map(com_offset)
        return P_base if period == 1 else make_iterated_stride_map(P_base, period)

    def make_feasibility(_com_offset: float):
        return make_passive_brachiation_feasibility_check(base_params, dim=1)

    def run_one(offsets: np.ndarray, label: str) -> tuple[ContinuationResult, list[ComContinuationRow]]:
        result = continue_fixed_point_branch(
            P_factory=make_stride_map,
            parameters=offsets,
            x0=np.array([d_fixed], dtype=float),
            dim=1,
            feasibility_factory=make_feasibility,
            tol=continuation_tol,
            max_iter=continuation_max_iter,
            delta=continuation_delta,
            damping=continuation_damping,
            compute_stability=True,
            stop_on_failure=True,
        )
        rows: list[ComContinuationRow] = []
        for point in result.points:
            d_value = float(point.x[0])
            lc1_fraction, lc2_fraction = com_fractions_from_offset(
                point.parameter,
                center_fraction=center_fraction,
            )

            if point.converged:
                evaluation = evaluate_passive_brachiation_stride(
                    model=make_model(point.parameter),
                    slope=slope,
                    x=np.array([d_value], dtype=float),
                    dt=dt,
                    t_max=t_max,
                    collision_mode=collision_mode,
                    initial_direction=initial_direction,
                    impact_direction=impact_direction,
                    branch=branch,
                    support_point=support,
                    switch_policy=policy,
                )
                d_next = float(evaluation.p_of_x[0])
                stride_plot = d_value if period == 1 else 0.5 * (d_value + d_next)
                eigen_real = (
                    float(np.real(point.eigenvalues[0]))
                    if point.eigenvalues is not None
                    else np.nan
                )
                validation = _validate_com_samples(
                    evaluation.samples,
                    slope=slope,
                    period=period,
                    support=support,
                    impact_direction=impact_direction,
                    legality_tolerance=1e-9,
                )
            else:
                d_next = np.nan
                stride_plot = np.nan
                eigen_real = np.nan
                validation = ComValidation(False, np.nan, np.nan, np.array([], dtype=float))

            spectral_radius = (
                np.nan if point.spectral_radius is None else float(point.spectral_radius)
            )
            rows.append(
                ComContinuationRow(
                    direction=label,
                    com_offset=float(point.parameter),
                    lc1_fraction=lc1_fraction,
                    lc2_fraction=lc2_fraction,
                    d_primary=d_value,
                    d_next=d_next,
                    stride_plot=stride_plot,
                    period=period,
                    converged=point.converged,
                    residual_norm=point.residual_norm,
                    spectral_radius=spectral_radius,
                    eigen_real=eigen_real,
                    stable=bool(point.converged and np.isfinite(spectral_radius) and spectral_radius < 1.0),
                    legal=validation.legal,
                    max_elbow_distance=validation.max_elbow_distance,
                    min_elbow_distance=validation.min_elbow_distance,
                    release_stride=validation.release_stride,
                    failure_reason=point.failure_reason,
                )
            )

        return result, rows

    offsets_contact = np.linspace(0.0, offset_low, n_steps_per_side)
    offsets_elbow = np.linspace(0.0, offset_high, n_steps_per_side)
    contact_result, contact_rows = run_one(offsets_contact, "toward_contact_endpoints")
    elbow_result, elbow_rows = run_one(offsets_elbow, "toward_elbow")

    rows = contact_rows + [row for row in elbow_rows if abs(row.com_offset) > 1e-12]
    rows.sort(key=lambda row: row.com_offset)
    return SymmetricComContinuation(
        contact_result=contact_result,
        elbow_result=elbow_result,
        contact_rows=contact_rows,
        elbow_rows=elbow_rows,
        rows=rows,
    )
