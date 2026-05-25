"""Shared compute-or-load cache for the report parts.

Every report part exposes a ``run(params, force)`` function that builds a
config dict and hands compute/plot callables to :func:`cached_run`.  The cache
key is a hash of the *full* config dict, which must include a manual
``_algo`` version tag so that changing a part's logic invalidates its cache
(parameters alone cannot detect a code change).

File layout per part ``<part>`` and hash ``<h>`` under ``results/``:

* ``<part>_<h>.npz``  - compute arrays + ``config_json`` + ``summary_json``
* ``<part>_<h>.json`` - {part, hash, config, summary, figures: {name: filename}}
* ``<part>_<h>__<name>.png`` - one file per named figure
* ``<part>_latest.json`` and ``<part>_latest__<name>.png`` - fixed aliases

A run is a cache hit only when the npz, the json and every figure file named
in the json all exist.  On a hit nothing is recomputed; the aliases are
refreshed and the cached paths are returned.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # report figures are always saved to PNG, never shown interactively

import numpy as np
from matplotlib.figure import Figure

# (arrays, summary)
ComputeResult = tuple[dict[str, np.ndarray], dict[str, Any]]
ComputeFn = Callable[[dict[str, Any]], ComputeResult]
# (arrays, summary, config) -> {figure_name: Figure}
PlotFn = Callable[[dict[str, np.ndarray], dict[str, Any], dict[str, Any]], dict[str, Figure]]


@dataclass(frozen=True)
class PartResult:
    part: str
    hash: str
    cache_hit: bool
    data_path: Path
    meta_path: Path
    figures: dict[str, Path]
    config: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)

    def figure(self, name: str = "main") -> Path:
        return self.figures[name]


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r} into a cache config.")


def _figure_filename(part: str, stem: str, name: str) -> str:
    return f"{part}_{stem}__{name}.png"


def _alias_filename(part: str, name: str) -> str:
    return f"{part}_latest__{name}.png"


def _refresh_aliases(part: str, figures: dict[str, Path], results_dir: Path) -> None:
    for name, path in figures.items():
        alias = results_dir / _alias_filename(part, name)
        if path.resolve() != alias.resolve():
            shutil.copy2(path, alias)
        if name == "main":
            legacy_alias = results_dir / f"{part}_latest.png"
            if path.resolve() != legacy_alias.resolve():
                shutil.copy2(path, legacy_alias)


def cached_run(
    *,
    part: str,
    config: dict[str, Any],
    compute: ComputeFn,
    plot: PlotFn,
    results_dir: Path | str = Path("results"),
    force: bool = False,
    verbose: bool = True,
) -> PartResult:
    """Return cached figures for ``part`` or recompute them if stale.

    ``config`` must already contain everything that affects the result,
    including the part's ``_algo`` version tag.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    digest = config_hash(config)
    data_path = results_dir / f"{part}_{digest}.npz"
    meta_path = results_dir / f"{part}_{digest}.json"

    if not force and data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        figures = {name: results_dir / fname for name, fname in meta.get("figures", {}).items()}
        if figures and all(path.exists() for path in figures.values()):
            _write_latest_aliases(part, meta, data_path, figures, results_dir)
            if verbose:
                print(f"[{part}] cache hit ({digest}); reused {len(figures)} figure(s).")
            return PartResult(
                part=part,
                hash=digest,
                cache_hit=True,
                data_path=data_path,
                meta_path=meta_path,
                figures=figures,
                config=meta.get("config", config),
                summary=meta.get("summary", {}),
            )

    if verbose:
        print(f"[{part}] cache miss ({digest}); computing...")
    arrays, summary = compute(config)

    np.savez_compressed(
        data_path,
        config_json=np.array(json.dumps(config, sort_keys=True, default=_json_default), dtype="U"),
        summary_json=np.array(json.dumps(summary, sort_keys=True, default=_json_default), dtype="U"),
        **arrays,
    )

    import matplotlib.pyplot as plt

    figs = plot(arrays, summary, config)
    figures: dict[str, Path] = {}
    for name, fig in figs.items():
        fig_path = results_dir / _figure_filename(part, digest, name)
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        figures[name] = fig_path

    meta = {
        "part": part,
        "hash": digest,
        "config": config,
        "summary": summary,
        "figures": {name: path.name for name, path in figures.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    _write_latest_aliases(part, meta, data_path, figures, results_dir)

    if verbose:
        print(f"[{part}] wrote {data_path.name} + {len(figures)} figure(s).")
    return PartResult(
        part=part,
        hash=digest,
        cache_hit=False,
        data_path=data_path,
        meta_path=meta_path,
        figures=figures,
        config=config,
        summary=summary,
    )


def _write_latest_aliases(
    part: str,
    meta: dict[str, Any],
    data_path: Path,
    figures: dict[str, Path],
    results_dir: Path,
) -> None:
    latest_meta = results_dir / f"{part}_latest.json"
    latest_meta.write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    latest_data = results_dir / f"{part}_latest.npz"
    if data_path.resolve() != latest_data.resolve():
        shutil.copy2(data_path, latest_data)
    _refresh_aliases(part, figures, results_dir)
