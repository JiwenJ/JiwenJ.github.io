#!/usr/bin/env python3
"""Plot directly optimized ELR schedules under common constant bounds.

For a convex terminal-loss bound with deterministic weights w_t >= 0,
    B(w) = (A + a sum_t w_t**2)/(2 sum_t w_t) + b,
constant positive A, a and constant residual bound b give equal optimal
weights. The terminal-conversion ELR is proportional to the remaining
weight, hence has a linear tail. The optimizer determines the normalized
metric, energy, residual and constants, not a different normalized shape
under these same constant-bound assumptions.

The plotted 5% ELR warmup is prescribed. Its length and peak are not jointly
optimized. These are conditional bound-optimum illustrations, not measured
training runs or a proof of globally optimal neural-network training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/optimal-elr-plot-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/optimal-elr-plot-cache")

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import PercentFormatter


STATIC = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#383832", "#74746c", "#e7e8e2"
WARMUP = 0.05
OPTIMIZERS = {
    "sgd": {"name": "SGD", "color": INK, "ls": "-", "residual": False},
    "adafactor": {"name": "Adafactor", "color": "#b56b30", "ls": (0, (5, 2)), "residual": False},
    "adam": {"name": "Adam", "color": "#517b8b", "ls": (0, (5, 2, 1, 2)), "residual": True},
    "muon": {"name": "Momentum Muon", "color": "#94826d", "ls": (0, (1, 2)), "residual": True},
}


def normalized_elr(progress):
    progress = np.asarray(progress, dtype=float)
    return np.minimum(progress / WARMUP, (1.0 - progress) / (1.0 - WARMUP))


def verify_constant_bound():
    """Check the plotted tail against the exact KKT solution and its gap."""
    rng = np.random.default_rng(20261101)
    max_shape_error, max_gap_error = 0.0, 0.0
    for n in (7, 95, 301):
        for distance, energy, residual in ((1.0, 1.0, 0.0), (2.3, .4, .7), (.05, 3.0, .2)):
            scale = np.sqrt(distance / (energy * n))
            weights = np.full(n, scale)
            total = weights.sum()
            remaining = np.concatenate((np.cumsum(weights[::-1])[::-1][1:], [0.0]))
            elr = weights * remaining / total
            progress = WARMUP + (1.0 - WARMUP) * np.arange(n) / (n - 1)
            error = np.max(np.abs(elr / elr[0] - normalized_elr(progress)))
            max_shape_error = max(max_shape_error, float(error))
            optimum = residual + np.sqrt(distance * energy / n)
            # Completed-square certificate, independent of the plotting formula.
            for candidate in rng.lognormal(mean=np.log(scale), sigma=.7, size=(20, n)):
                s = candidate.sum()
                bound = (distance + energy * np.dot(candidate, candidate)) / (2.0 * s) + residual
                gap = energy * np.sum((candidate - weights)**2) / (2.0 * s)
                max_gap_error = max(max_gap_error, abs(bound - optimum - gap))
                assert bound >= optimum - 1e-13
    grid = np.linspace(0.0, 1.0, 1001)
    curve = normalized_elr(grid)
    peak = int(WARMUP * 1000)
    np.testing.assert_allclose(curve[[0, peak, -1]], [0.0, 1.0, 0.0], atol=1e-14)
    assert np.all(np.diff(curve[:peak + 1]) > 0)
    assert np.all(np.diff(curve[peak:]) < 0)
    assert max_shape_error < 1e-13 and max_gap_error < 1e-13
    return {"constant_KKT_shape_max_error": max_shape_error,
            "completed_square_gap_max_error": max_gap_error}


def configure():
    font = STATIC / "fonts/source-serif-4/SourceSerif4-Regular.ttf"
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "mathtext.fontset": "stix",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.edgecolor": "#c1c4bb", "axes.linewidth": .8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "path",
        "svg.hashsalt": "direct-optimal-elr-20261101",
    })


def draw(output: Path, key: str, mobile: bool, preview_dir: Path | None):
    spec = OPTIMIZERS[key]
    fig, ax = plt.subplots(figsize=(5.4, 6.4) if mobile else (11.2, 5.4))
    fig.subplots_adjust(left=.17 if mobile else .12, right=.965,
                        bottom=.25 if mobile else .19, top=.68 if mobile else .69)
    title = f"{spec['name']}: optimal ELR under fixed bounds"
    display_title = f"{spec['name']}\nOptimal ELR under fixed bounds" if mobile else title
    fig.suptitle(display_title, x=.5, y=.975, fontsize=16 if mobile else 17, linespacing=1.25)
    subtitle = ("Constant energy and residual bounds" if spec["residual"] else
                "Constant energy bound")
    subtitle += "\n5% prescribed warmup" if mobile else " · 5% prescribed warmup"
    fig.text(.5, .85 if mobile else .875, subtitle, ha="center", va="top",
             fontsize=11, color=MUTED, linespacing=1.5)

    progress = np.linspace(0.0, 1.0, 1001)
    ax.axvspan(0.0, WARMUP, color="#f0f0ec", zorder=0)
    ax.axvline(WARMUP, color="#c9cbc1", ls=(0, (3, 3)), lw=1, zorder=1)
    ax.plot(progress, normalized_elr(progress), color=spec["color"], ls=spec["ls"],
            lw=2.25 if key == "sgd" else 2.0, label="ELR (conditional bound optimum)", zorder=3)
    ax.scatter([WARMUP, 1.0], [1.0, 0.0], color=spec["color"], s=15, zorder=4, clip_on=False)
    ax.annotate("Warmup ends", xy=(WARMUP, 1.0), xytext=(.20, 1.055),
                ha="left", va="center", fontsize=10, color=MUTED,
                arrowprops={"arrowstyle": "-", "color": "#92988d", "lw": .85})
    ax.text(.61, .79, r"$\dfrac{\varepsilon_t}{\varepsilon_{\rm peak}}=\dfrac{1-t/T}{1-0.05}$",
            ha="center", va="center", fontsize=15, color=spec["color"])
    ax.annotate(r"$\varepsilon(T)=0$", xy=(1.0, 0.0), xytext=(.77, .075),
                ha="center", va="center", fontsize=11, color=INK,
                arrowprops={"arrowstyle": "-", "color": "#92988d", "lw": .85})
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.13)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Training progress, $t/T$", labelpad=7)
    ax.set_ylabel(r"Normalized ELR $\varepsilon_t/\varepsilon_{\rm peak}$", labelpad=8)
    ax.tick_params(length=3, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="center", frameon=False,
               bbox_to_anchor=(.52, .735 if mobile else .80), fontsize=10.5, handlelength=2.6)

    footer = ("Optimal tail for the stated convex terminal-loss bound with common fixed constants.\n"
              "Warmup is prescribed; its length and peak are not jointly optimized.")
    if mobile:
        footer = ("Conditional convex terminal-bound optimum.\n"
                  "Common fixed bounds; warmup is prescribed.\n"
                  "Warmup length and peak are not jointly optimized.")
    fig.text(.5, .035, footer, ha="center", va="bottom", fontsize=10,
             color=MUTED, linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None, "Title": title,
        "Description": f"Direct ELR conditional terminal-bound optimum for {spec['name']}, with common constant energy bounds" +
        (" and residual bounds" if spec["residual"] else "") +
        ". Fixed illustrative 5% warmup, normalized peak one, linear tail to zero. All axes linear. Warmup is not jointly optimized; no measured training results or universal optimality claim."})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{output.stem}.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    checks = verify_constant_bound()
    configure()
    output = STATIC / "images/lr-schedules"
    for key in OPTIMIZERS:
        for mobile in (False, True):
            variant = "mobile" if mobile else "desktop"
            draw(output / f"{key}-optimal-elr-{variant}.svg", key, mobile, args.preview_dir)
    print("Generated all eight direct ELR plots; KKT, boundary, and monotonicity checks PASS.")
    print(checks)


if __name__ == "__main__":
    main()
