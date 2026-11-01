#!/usr/bin/env python3
"""Plot an SGD warmup and linear-decay schedule on linear axes.

The 5% warmup prefix is prescribed, not optimized. The linear tail illustrates
the article's conditional endpoint-bound solution for convex objectives,
unbiased stochastic gradients, and a common constant gradient-energy bound.
The peak is normalized to one and the final learning rate is zero.
This is an analytical illustration, not a neural-network training result.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sgd-warmup-plot-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/sgd-warmup-plot-cache")

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import PercentFormatter


STATIC = Path(__file__).resolve().parents[1]
INK, MUTED, ACCENT = "#383832", "#74746c", "#b56b30"
WARMUP = 0.05


def schedule(progress):
    progress = np.asarray(progress, dtype=float)
    return np.minimum(progress / WARMUP, (1.0 - progress) / (1.0 - WARMUP))


def configure():
    latin = STATIC / "fonts/source-serif-4/SourceSerif4-Regular.ttf"
    font_manager.fontManager.addfont(str(latin))
    family = font_manager.FontProperties(fname=str(latin)).get_name()
    plt.rcParams.update({
        "font.family": family,
        "font.size": 11,
        "mathtext.fontset": "stix",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#c1c4bb",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
        "svg.hashsalt": "sgd-warmup-linear-20261031",
    })


def draw(output: Path, mobile: bool, preview_dir: Path | None):
    fig, ax = plt.subplots(figsize=(5.4, 6.4) if mobile else (11.2, 5.4))
    fig.subplots_adjust(left=0.15 if mobile else 0.12, right=0.965,
                        bottom=0.25 if mobile else 0.19, top=0.68 if mobile else 0.69)
    title = "SGD: warmup and linear decay"
    fig.suptitle(title, x=0.5, y=0.975, fontsize=16 if mobile else 17)
    fig.text(0.5, 0.90 if mobile else 0.875,
             "5% warmup · normalized peak 1 · minimum LR 0" if not mobile else
             "5% warmup\nNormalized peak 1 · minimum LR 0",
             ha="center", va="top", fontsize=11, color=MUTED, linespacing=1.6)

    warm = np.linspace(0.0, WARMUP, 101)
    decay = np.linspace(WARMUP, 1.0, 501)
    ax.axvspan(0, WARMUP, color="#f0f0ec", zorder=0)
    ax.axvline(WARMUP, color="#c9cbc1", ls=(0, (3, 3)), lw=1, zorder=1)
    ax.plot(warm, schedule(warm), color=INK, lw=2.25,
            label="Fixed warmup", zorder=3)
    ax.plot(decay, schedule(decay), color=ACCENT, lw=2.0, ls=(0, (5, 2)),
            label="Linear (conditional bound)", zorder=3)
    ax.scatter([WARMUP, 1.0], [1.0, 0.0], color=ACCENT, s=15, zorder=4, clip_on=False)
    ax.annotate("Warmup ends", xy=(WARMUP, 1), xytext=(0.20, 1.055),
                ha="left", va="center", fontsize=10, color=MUTED,
                arrowprops={"arrowstyle": "-", "color": "#92988d", "lw": .85})
    ax.text(0.60, 0.79, r"$\dfrac{\eta_t}{\eta_{\rm peak}}=\dfrac{1-t/T}{1-0.05}$",
            ha="center", va="center", fontsize=15, color=ACCENT)
    ax.annotate(r"$\eta(T)=0$", xy=(1, 0), xytext=(0.80, 0.075),
                ha="center", va="center", fontsize=11, color=INK,
                arrowprops={"arrowstyle": "-", "color": "#92988d", "lw": .85})

    ax.set_xlim(0, 1)
    ax.set_yscale("linear")
    ax.set_ylim(0, 1.13)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Training progress, $t/T$", labelpad=7)
    ax.set_ylabel(r"Normalized learning rate $\eta_t/\eta_{\rm peak}$", labelpad=8)
    ax.tick_params(length=3, labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7e8e2", linewidth=.7)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="center", frameon=False,
               bbox_to_anchor=(.52, .775 if mobile else .80), ncol=1 if mobile else 2,
               fontsize=10.5, handlelength=2.6, columnspacing=1.7)

    footer = ("Convex objective, unbiased gradients, and a common constant energy bound.\n"
              "Warmup is prescribed; its length and peak are not jointly optimized.")
    if mobile:
        footer = ("Convex objective, unbiased gradients,\n"
                  "and a common constant energy bound.\n"
                  "Warmup length and peak are prescribed.")
    fig.text(.5, .035, footer, ha="center", va="bottom", fontsize=10,
             color=MUTED, linespacing=1.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None, "Title": title,
        "Description": "SGD schedule with a prescribed illustrative 5% warmup, normalized peak 1, and linear decay to zero. Both axes are linear. The decay-bound assumptions are convex objectives, unbiased gradients, and a common constant gradient-energy bound; warmup length and peak are not jointly optimized."})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / (output.stem + ".png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    configure()
    # Check the specified shape and boundary values before rendering.
    np.testing.assert_allclose(schedule([0, WARMUP, 1]), [0, 1, 0], atol=1e-14)
    tail = np.linspace(WARMUP, 1, 501)
    assert np.all(np.diff(schedule(tail)) < 0)
    np.testing.assert_allclose(np.diff(schedule(tail), n=2), 0, atol=1e-14)
    folder = STATIC / "images/lr-schedules"
    for mobile in (False, True):
        variant = "mobile" if mobile else "desktop"
        draw(folder / f"sgd-warmup-linear-{variant}.svg", mobile, args.preview_dir)
    print("Generated desktop and mobile plots; endpoints and linear tail verified.")


if __name__ == "__main__":
    main()
