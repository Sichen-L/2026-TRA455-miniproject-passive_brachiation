"""Plot a cached slope-angle sweep without rerunning the continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def _latest_result(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("gamma_sweep_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No gamma_sweep_*.npz files found in {results_dir}.")
    return candidates[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot cached slope-angle sweep results.")
    parser.add_argument("result", nargs="?", type=Path, help="Path to a gamma_sweep_*.npz file.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result_path = args.result or _latest_result(args.results_dir)
    data = np.load(result_path)
    summary = json.loads(str(data["summary_json"]))

    x = data["gamma_deg"].astype(float)
    y = data["stride_plot"].astype(float)
    rho = data["spectral_radius"].astype(float)
    converged = data["converged"].astype(bool)
    legal = data["legal"].astype(bool)
    stable = data["stable"].astype(bool)
    eig_real = data["eigen_real"].astype(float)
    fold_candidate = data["fold_candidate"].astype(bool)
    fold_indicator = data["fold_indicator"].astype(float)

    mask = converged & np.isfinite(y)
    x, y, rho = x[mask], y[mask], rho[mask]
    legal, stable, eig_real = legal[mask], stable[mask], eig_real[mask]
    fold_candidate, fold_indicator = fold_candidate[mask], fold_indicator[mask]
    order = np.argsort(x)
    x, y, rho = x[order], y[order], rho[order]
    legal, stable, eig_real = legal[order], stable[order], eig_real[order]
    fold_candidate, fold_indicator = fold_candidate[order], fold_indicator[order]

    pd_mask = legal & np.isfinite(eig_real) & (eig_real <= -1.0)
    unstable_mask = legal & ~stable & ~pd_mask
    illegal_mask = ~legal
    states = np.array(
        [
            "illegal"
            if illegal_mask[i]
            else "period-doubling"
            if pd_mask[i]
            else "unstable"
            if unstable_mask[i]
            else "stable"
            for i in range(len(x))
        ],
        dtype=object,
    )
    segments = (
        np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
        if len(x) > 1
        else np.empty((0, 2, 2), dtype=float)
    )
    segment_states = []
    segment_rho = []
    for i in range(max(0, len(x) - 1)):
        pair = {states[i], states[i + 1]}
        if "illegal" in pair:
            state = "illegal"
        elif "period-doubling" in pair:
            state = "period-doubling"
        elif "unstable" in pair:
            state = "unstable"
        else:
            state = "stable"
        segment_states.append(state)
        segment_rho.append(np.nanmean(rho[i : i + 2]))
    segment_states = np.array(segment_states, dtype=object)
    segment_rho = np.array(segment_rho, dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    finite_rho = np.isfinite(rho) & legal
    norm = Normalize(vmin=float(np.nanmin(rho[finite_rho])), vmax=float(np.nanmax(rho[finite_rho])))
    cmap = plt.get_cmap("viridis_r")
    stable_segments = segments[segment_states == "stable"]
    if len(stable_segments):
        lines = LineCollection(
            stable_segments,
            cmap=cmap,
            norm=norm,
            linewidths=2.6,
            label="stable, colored by |lambda|max",
        )
        lines.set_array(segment_rho[segment_states == "stable"])
        ax.add_collection(lines)
        colorbar_source = lines
    else:
        colorbar_source = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    styles = {
        "period-doubling": ("tab:purple", "--", "period-doubling / eig <= -1"),
        "unstable": ("tab:red", "-.", "unstable |lambda| >= 1"),
        "illegal": ("0.35", ":", "illegal geometry"),
    }
    for state, (color, linestyle, label) in styles.items():
        first = True
        for segment in segments[segment_states == state]:
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label if first else None,
            )
            first = False

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    y_span = max(y_max - y_min, 1e-6)
    switch_x = [0.5 * (x[i] + x[i + 1]) for i in range(len(states) - 1) if states[i] != states[i + 1]]
    for value in switch_x:
        ax.axvline(value, color="0.45", linestyle=":", linewidth=1.0)
        ax.text(value, y_max + 0.02 * y_span, f"gamma={value:.2f}", rotation=90, ha="center", va="bottom", fontsize=8)

    fold_idx = np.where(fold_candidate & np.isfinite(fold_indicator))[0]
    if len(fold_idx):
        idx = fold_idx[np.argmin(fold_indicator[fold_idx])]
        ax.axvline(x[idx], color="black", linestyle="--", linewidth=1.2, label="fold candidate lambda -> +1")

    best_gamma = float(summary["best_lambda_gamma_deg"])
    best_lambda = float(summary["best_lambda"])
    best_idx = int(np.argmin(np.abs(x - best_gamma)))
    ax.axvline(best_gamma, color="tab:green", linewidth=1.2, label="minimum |lambda|max")
    ax.annotate(
        f"min |lambda|={best_lambda:.4f}\ngamma={best_gamma:.2f} deg",
        xy=(x[best_idx], y[best_idx]),
        xytext=(10, 18),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="tab:green"),
        color="tab:green",
        fontsize=9,
    )

    ax.set_xlim(float(np.nanmin(x)) - 1.0, float(np.nanmax(x)) + 1.0)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.10 * y_span)
    ax.set_xlabel("slope angle gamma [deg]")
    ax.set_ylabel("steady stride distance [m]" if int(summary["period"]) == 1 else "mean period-2 stride distance [m]")
    ax.set_title(
        "Slope-angle sweep at fixed COM offset "
        f"{summary['com_offset']:+.4f}, branch={summary['branch']}, period={summary['period']}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    cbar = fig.colorbar(colorbar_source, ax=ax)
    cbar.set_label("Poincare spectral radius |lambda|max on stable segments")
    plt.tight_layout()

    output = args.output or result_path.with_name("gamma_sweep_latest.png")
    fig.savefig(output, dpi=180)
    print(f"Loaded: {result_path}")
    print(f"Wrote: {output}")
    print(f"Most stable slope angle: gamma={best_gamma:.6f} deg, |lambda|max={best_lambda:.6f}")
    if switch_x:
        print("State-switch vertical lines at gamma =", ", ".join(f"{value:.6f}" for value in switch_x))
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
