#!/usr/bin/env python3
"""Compare external LR and relative-update ELR including Muon warmup.

Linear external LR is prescribed. ELR follows from a heuristic exponential
weight-norm growth model and approximately constant update RMS, as discussed
in Scientific Spaces article 11985. Momentum effects enter these statistics;
the curve does not solve the optimal-control problem for practical Muon.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/muon-bound-plot-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/muon-bound-plot-cache")

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import PercentFormatter


STATIC = Path(__file__).resolve().parents[1]
INK, MUTED, ACCENT, BLUE = "#383832", "#74746c", "#b56b30", "#517b8b"
C0, KAPPA = 0.9, 5.0
WARMUP = 0.05


def schedules(progress):
    progress = np.asarray(progress, dtype=float)
    lr = np.minimum(progress / WARMUP, (1.0 - progress) / (1.0 - WARMUP))
    # Apply one norm-growth model over the whole run, including warmup.
    # Normalize both curves at warmup completion, where both attain their peak.
    norm_at_peak = 1.0 - C0 * np.exp(-KAPPA * WARMUP)
    elr = lr * np.sqrt(norm_at_peak / (1.0 - C0 * np.exp(-KAPPA * progress)))
    return lr, elr


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
        "svg.hashsalt": "muon-lr-elr-20261031",
    })


def draw(output: Path, mobile: bool, preview_dir: Path | None):
    fig, ax = plt.subplots(figsize=(5.4, 6.4) if mobile else (11.2, 5.4))
    fig.subplots_adjust(left=0.15 if mobile else 0.12, right=0.965,
                        bottom=0.25 if mobile else 0.19, top=0.68 if mobile else 0.69)
    title = "Momentum Muon: LR and ELR"
    fig.suptitle(title, x=0.5, y=0.975, fontsize=16 if mobile else 17)
    fig.text(0.5, 0.90 if mobile else 0.875,
             "5% warmup · each curve normalized at the peak" if not mobile else
             "5% warmup\nEach curve normalized at the peak",
             ha="center", va="top", fontsize=11, color=MUTED, linespacing=1.6)

    progress = np.linspace(0, 1, 1001)
    lr, elr = schedules(progress)
    ax.axvspan(0, WARMUP, color="#f0f0ec", zorder=0)
    ax.axvline(WARMUP, color="#c9cbc1", ls=(0, (3, 3)), lw=1, zorder=1)
    ax.plot(progress, lr, color=ACCENT, lw=2.0, ls=(0, (5, 2)),
            label="External LR (prescribed)", zorder=3)
    ax.plot(progress, elr, color=BLUE, lw=2.0, ls=(0, (5, 2, 1, 2)),
            label="ELR (norm-growth model)", zorder=3)
    ax.text(.66, .88, r"$\mathrm{ELR}=\eta\,\dfrac{\mathrm{RMS}(U)}{\mathrm{RMS}(W)}$",
            ha="center", va="center", fontsize=15, color=INK)
    ax.annotate("Warmup ends", xy=(WARMUP, 1), xytext=(.20, 1.055),
                ha="left", va="center", fontsize=10, color=MUTED,
                arrowprops={"arrowstyle": "-", "color": "#92988d", "lw": .85})

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.13)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Training progress, $t/T$", labelpad=7)
    ax.set_ylabel("Value / value at warmup end", labelpad=8)
    ax.tick_params(length=3, labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7e8e2", linewidth=.7)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="center", frameon=False,
               bbox_to_anchor=(.52, .775 if mobile else .80), ncol=1 if mobile else 2,
               fontsize=10.5, handlelength=2.6, columnspacing=1.7)

    footer = ("Constant update RMS and growing weight RMS; illustrative $c_0=0.9$, $\\kappa=5$.\n"
              "A reference LR-to-ELR mapping, not a proof of optimal LR.")
    if mobile:
        footer = ("Constant update RMS; growing weight RMS.\n"
                  "Illustrative $c_0=0.9$, $\\kappa=5$.\n"
                  "A reference mapping, not an optimal-LR proof.")
    fig.text(.5, .035, footer, ha="center", va="bottom", fontsize=10,
             color=MUTED, linespacing=1.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None, "Title": title,
        "Description": "Prescribed 5% linear warmup followed by linear external LR decay and a heuristic ELR curve for momentum Muon. Constant update RMS and exponentially growing weight squared RMS are modeled across the full run; c0=0.9, kappa=5. Each curve is normalized at warmup completion. Not a proof of optimal LR or an empirical fit. All axes linear."})
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
    # The nonlinear ELR comes from the norm ratio, not a change in external LR.
    progress = np.linspace(0, 1, 1001)
    lr, elr = schedules(progress)
    peak = int(round(WARMUP * 1000))
    np.testing.assert_allclose([lr[0], elr[0], lr[peak], elr[peak], lr[-1], elr[-1]], [0, 0, 1, 1, 0, 0])
    np.testing.assert_allclose(np.diff(lr[:peak + 1], n=2), 0, atol=1e-14)
    np.testing.assert_allclose(np.diff(lr[peak:], n=2), 0, atol=1e-14)
    assert np.all(np.diff(elr[:peak + 1]) > 0) and np.all(np.diff(elr[peak:]) < 0)
    assert np.all(elr[peak + 1:-1] < lr[peak + 1:-1])
    folder = STATIC / "images/lr-schedules"
    for mobile in (False, True):
        variant = "mobile" if mobile else "desktop"
        draw(folder / f"muon-lr-elr-{variant}.svg", mobile, args.preview_dir)
    print("Generated LR/ELR plots; normalization, endpoints, and monotonicity verified.")


if __name__ == "__main__":
    main()
