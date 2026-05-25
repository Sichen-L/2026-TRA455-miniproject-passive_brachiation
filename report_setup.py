"""Shared operating-point loader for the downstream report parts.

The most-stable passive gait used by parts 4-8 comes from the COM-offset
continuation (part 3a, :mod:`run_com_sweep`).  Previously this point was read
from a separate mass sweep; the mass sweep made the two links asymmetric and is
no longer part of the report, so the operating point is taken directly from the
most-stable legal COM-sweep row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from passive_brachiation import BrachiationParameters, parameters_with_symmetric_com_offset


@dataclass(frozen=True)
class BestComSetup:
    params: BrachiationParameters          # centered base with com_offset applied
    base_params: BrachiationParameters     # centered base, offset NOT applied
    d_target: float
    branch: str
    period: int
    com_offset: float
    gamma_deg: float
    spectral_radius: float
    source: str


def _base_params(config: dict[str, Any]) -> BrachiationParameters:
    return BrachiationParameters.rod_point_mass(
        weight_position1=float(config.get("weight_position1", 0.5)),
        weight_position2=float(config.get("weight_position2", 0.5)),
        m1=float(config.get("m1", 1.041)),
        m2=float(config.get("m2", 1.041)),
        l1=float(config.get("l1", 0.314)),
        l2=float(config.get("l2", 0.314)),
        rod_mass_fraction=float(config.get("rod_mass_fraction", 0.2)),
        damping1=float(config.get("damping1", 0.0)),
        damping2=float(config.get("damping2", 0.0)),
        gravity=float(config.get("gravity", 9.81)),
    )


def load_best_com_setup(
    results_dir: Path | str = Path("results"),
    com_result: Path | str | None = None,
    branch_override: str | None = None,
) -> BestComSetup:
    """Return the most-stable legal gait from the cached COM sweep.

    Reads ``com_sweep_latest.npz`` (or an explicit ``com_result``) and selects
    the converged, legal, stable row with the smallest spectral radius.
    """
    path = Path(com_result) if com_result is not None else Path(results_dir) / "com_sweep_latest.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No COM sweep result at {path}. Run run_com_sweep.run(...) first."
        )
    data = np.load(path, allow_pickle=True)
    summary = json.loads(str(data["summary_json"]))
    config = json.loads(str(data["config_json"]))

    spectral_radius = np.asarray(data["spectral_radius"], dtype=float)
    mask = np.asarray(data["converged"], dtype=bool) & np.isfinite(spectral_radius)
    if "legal" in data.files:
        mask &= np.asarray(data["legal"], dtype=bool)
    if "stable" in data.files:
        mask &= np.asarray(data["stable"], dtype=bool)
    if not np.any(mask):
        raise ValueError(f"No stable/legal COM-sweep rows found in {path}.")

    best_index = int(np.where(mask)[0][np.argmin(spectral_radius[mask])])
    com_offset = float(np.asarray(data["com_offset"], dtype=float)[best_index])
    d_target = float(np.asarray(data["d_primary"], dtype=float)[best_index])
    branch = branch_override or str(summary.get("branch", "negative"))
    period = int(summary.get("period", 1))
    gamma_deg = float(config.get("gamma_deg", 45.0))

    base_params = _base_params(config)
    params = parameters_with_symmetric_com_offset(com_offset, base_params)
    return BestComSetup(
        params=params,
        base_params=base_params,
        d_target=d_target,
        branch=branch,
        period=period,
        com_offset=com_offset,
        gamma_deg=gamma_deg,
        spectral_radius=float(spectral_radius[best_index]),
        source=str(path),
    )
