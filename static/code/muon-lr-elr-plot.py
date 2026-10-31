#!/usr/bin/env python3
"""Compare external LR and relative-update ELR after Muon warmup.

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
INK, MUTED, ACCENT = "#383832", "#74746c", "#b56b30"
C0, KAPPA = 0.9, 5.0


def schedules(progress):
    progress = np.asarray(progress, dtype=float)
    lr = 1.0 - progress
    elr = lr * np.sqrt((1.0 - C0) / (1.0 - C0 * np.exp(-KAPPA * progress)))
    return lr, elr


def configure():
    latin = STATIC / "fonts/source-serif-4/SourceSerif4-Regular.ttf"
    font_manager.fontManager.addfont(str(latin))
    family = font_manager.FontProperties(fname=str(latin)).get_name()
    available = {font.name for font in font_manager.fontManager.ttflist}
    cjk = [name for name in ["Songti SC", "Noto Serif CJK SC"] if name in available]
    plt.rcParams.update({
        "font.family": [family, *cjk, "DejaVu Serif"],
        "font.size": 12,
        "mathtext.fontset": "stix",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#bfc1b8",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
        "svg.hashsalt": "muon-lr-elr-20261031",
    })


def draw(output: Path, mobile: bool, preview_dir: Path | None):
    fig, ax = plt.subplots(figsize=(5.4, 6.4) if mobile else (9.2, 6.0))
    fig.subplots_adjust(left=0.15 if mobile else 0.12, right=0.965,
                        bottom=0.25 if mobile else 0.21, top=0.72 if mobile else 0.76)
    title = "带动量 Muon：LR 与 ELR"
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=17 if mobile else 21)
    fig.text(0.5, 0.897, "衰减阶段示意  ·  各自起点归一化为 1",
             ha="center", va="top", fontsize=10.5 if mobile else 12, color=MUTED)

    progress = np.linspace(0, 1, 1001)
    lr, elr = schedules(progress)
    ax.plot(progress, lr, color=ACCENT, lw=2.5, ls=(0, (6, 3)),
            label="外部 LR：给定 Linear Decay", zorder=3)
    ax.plot(progress, elr, color="#497b89", lw=3,
            label="ELR：范数增长近似", zorder=3)
    ax.text(.66, .88, r"$\mathrm{ELR}=\eta\,\dfrac{\mathrm{RMS}(U)}{\mathrm{RMS}(W)}$",
            ha="center", va="center", fontsize=15 if mobile else 19, color=INK)
    ax.annotate("前期下降更快", xy=(.14, schedules(.14)[1]), xytext=(.23, .47),
                ha="left", va="center", fontsize=11, color="#497b89",
                arrowprops={"arrowstyle": "-", "color": "#497b89", "lw": .85})
    ax.annotate("后期接近线性", xy=(.70, schedules(.70)[1]), xytext=(.46, .27),
                ha="left", va="center", fontsize=11, color="#497b89",
                arrowprops={"arrowstyle": "-", "color": "#497b89", "lw": .85})

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.13)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Warmup 后的衰减进度", labelpad=10)
    ax.set_ylabel("相对于各自起点的大小", labelpad=9)
    ax.tick_params(length=3, labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7e8e2", linewidth=.8)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", frameon=False,
               bbox_to_anchor=(.54, .855), ncol=1 if mobile else 2,
               fontsize=10.5 if mobile else 11, handlelength=2.4, columnspacing=2)

    footer = ("假设更新 RMS 近似稳定、权重 RMS 增长；示意参数 c₀ = 0.9，κ = 5。\n"
              "这是 LR 到 ELR 的渐近映射，不是 Muon 最优 LR 的证明。")
    if mobile:
        footer = ("更新 RMS 近似稳定、权重 RMS 增长；\n"
                  "示意参数 c₀ = 0.9，κ = 5。\n"
                  "渐近映射，不是 Muon 最优 LR 的证明。")
    fig.text(.5, .035, footer, ha="center", va="bottom", fontsize=10,
             color=MUTED, linespacing=1.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None, "Title": title,
        "Description": "Prescribed linear external LR and a heuristic ELR curve for momentum Muon, with constant update RMS and exponentially growing weight squared RMS; c0=0.9, kappa=5. Decay phase after warmup, each curve normalized to its own initial value. Not a proof of optimal LR or an empirical fit. All axes linear."})
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
    lr, elr = schedules(np.linspace(0, 1, 1001))
    np.testing.assert_allclose([lr[0], elr[0], lr[-1], elr[-1]], [1, 1, 0, 0])
    np.testing.assert_allclose(np.diff(lr, n=2), 0, atol=1e-14)
    assert np.all(np.diff(elr) < 0) and np.all(elr[1:-1] < lr[1:-1])
    folder = STATIC / "images/lr-schedules"
    for mobile in (False, True):
        variant = "mobile" if mobile else "desktop"
        draw(folder / f"muon-lr-elr-{variant}.svg", mobile, args.preview_dir)
    print("Generated LR/ELR plots; normalization, endpoints, and monotonicity verified.")


if __name__ == "__main__":
    main()
