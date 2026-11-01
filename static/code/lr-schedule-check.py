#!/usr/bin/env python3
"""Reproduce an exactly solvable SGD learning-rate comparison.

Dependencies: NumPy and Matplotlib only (tested with 2.0.2 and 3.9.4).
From the repository root:

    MPLCONFIGDIR=/private/tmp/lr-schedule-mpl \
        python3 static/code/lr-schedule-check.py \
        --preview-dir /private/tmp/lr-schedule-preview

Model: f(x) = x**2 / 2, E[x_0**2] = 1, and martingale gradient noise
with conditional mean zero and conditional variance one. Exactly 100 updates
are applied, indexed t = 0,...,99. No samples or Monte Carlo estimates are used:

    r[t+1] = (1 - eta[t])**2 * r[t] + eta[t]**2
    expected_loss[t] = r[t] / 2.

All schedules share a prescribed five-update linear warmup, eta[t]=peak*t/5
for t=0,...,4. The common peak solves peak=r_5/(1+r_5), with r_5 obtained
from that warmup, so the conditionally optimal tail starts continuously at
eta[5]=peak. This matching condition chooses an illustrative prefix, not an
optimal warmup. The following 95 updates compare a greedy optimal tail with
linear, cosine, and constant tails at the SAME peak. No amplitude is tuned.

The optimal tail has eta[5+j]=r_5/(1+(j+1)*r_5) and
r[5+j]=r_5/(1+j*r_5). Its optimality follows from the monotone map
r -> r/(1+r), conditional on the fixed prefix. The zero endpoint of
linear/cosine at t=100 is not an applied update. All 100 updates, including
warmup, are included in the plotted loss and machine-readable results.
This is not a claim about general convex objectives, AdaGrad, neural networks,
or empirical training performance.

Default CSV/JSON/SVG outputs are under static/images/lr-schedules/.
SVG dates are omitted and the hash seed is fixed for reproducible output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


STATIC = Path(__file__).resolve().parents[1]
OUTPUT = STATIC / "images/lr-schedules"
HORIZON = 100
WARMUP = 5
INK = "#383832"
SECONDARY = "#74746c"
GRID = "#e7e8e2"
STYLES = {
    "exact": {"label": "Optimal tail", "color": INK, "ls": "-", "lw": 2.25},
    "linear": {"label": "Linear", "color": "#b56b30", "ls": (0, (5, 2)), "lw": 2.0},
    "cosine": {"label": "Cosine", "color": "#517b8b", "ls": (0, (5, 2, 1, 2)), "lw": 2.0},
    "constant": {"label": "Constant", "color": "#94826d", "ls": (0, (1, 2)), "lw": 2.0},
}


def schedule_shapes() -> dict[str, np.ndarray]:
    j = np.arange(HORIZON - WARMUP, dtype=np.float64)
    progress = j / (HORIZON - WARMUP)
    return {
        "linear": 1.0 - progress,
        "cosine": 0.5 * (1.0 + np.cos(np.pi * progress)),
        "constant": np.ones_like(progress),
    }


def trajectory(eta: np.ndarray) -> np.ndarray:
    """Expected squared parameter before/after each update, in FP64."""
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 1 or not np.all(np.isfinite(eta)) or np.any(eta < 0.0):
        raise ValueError("Learning rates must be a finite nonnegative vector.")
    r = np.empty(len(eta) + 1, dtype=np.float64)
    r[0] = 1.0
    for t, rate in enumerate(eta):
        r[t + 1] = (1.0 - rate)**2 * r[t] + rate**2
    return r


def shared_warmup():
    """Choose a fixed prefix that joins the conditional optimal tail continuously."""
    shape = np.arange(WARMUP, dtype=np.float64) / WARMUP
    left, right = 0.0, 1.0
    for _ in range(80):
        peak = (left + right) / 2.0
        r = trajectory(peak * shape)[-1]
        if peak < r / (1.0 + r):
            left = peak
        else:
            right = peak
    peak = (left + right) / 2.0
    return peak, peak * shape


def unrolled_final_square(eta: np.ndarray) -> float:
    """Independent product-and-sum expression for the final second moment."""
    future_product, noise_sum = 1.0, 0.0
    for rate in reversed(eta):
        noise_sum += rate**2 * future_product
        future_product *= (1.0 - rate)**2
    return float(future_product + noise_sum)


def calculate_and_verify() -> tuple[dict, dict]:
    peak, prefix = shared_warmup()
    warm_squares = trajectory(prefix)
    r_w = warm_squares[-1]
    j = np.arange(HORIZON - WARMUP, dtype=np.float64)
    exact_rates = np.concatenate((prefix, r_w / (1.0 + (j + 1.0) * r_w)))
    exact_squares = trajectory(exact_rates)
    closed_squares = np.concatenate((warm_squares[:-1],
                                    r_w / (1.0 + np.arange(HORIZON-WARMUP+1) * r_w)))
    records = {"exact": {"alpha": peak, "eta": exact_rates, "r": exact_squares}}
    report = {
        "status": "PASS",
        "scope": "Exact expected loss on one scalar noisy quadratic, with a common fixed warmup and common peak; no sampled data",
        "model": {"objective": "f(x)=x^2/2", "initial_second_moment": 1.0,
                  "noise_conditional_mean": 0.0, "noise_conditional_variance": 1.0,
                  "updates": HORIZON, "update_indices": [0, HORIZON - 1]},
        "warmup": {"updates": WARMUP, "fraction": WARMUP / HORIZON,
                   "formula": "eta_t=peak*t/W for t=0,...,W-1",
                   "common_peak": peak, "post_warmup_second_moment": float(r_w),
                   "peak_selection": "peak=r_W/(1+r_W), to join the optimal tail continuously; not a warmup optimization",
                   "rates": prefix.tolist(), "fixed_for_all_schedules": True},
        "comparison": {"same_peak": True, "amplitude_tuning": False,
                       "objective": "expected loss after all 100 updates, including the fixed warmup",
                       "optimality_scope": "optimal tail conditional on the common fixed warmup"},
        "schedules": {"exact": {"tail_formula": "eta_(W+j)=r_W/(1+(j+1)*r_W)",
                                "final_loss": float(exact_squares[-1] / 2.0),
                                "closed_form_final_loss": float(r_w / (2.0*(1.0+(HORIZON-WARMUP)*r_w))),
                                "ratio_to_exact": 1.0}},
        "checks": {},
    }
    formulas = {"linear": "peak*(1-(t-W)/(T-W))",
                "cosine": "peak*(1+cos(pi*(t-W)/(T-W)))/2", "constant": "peak"}
    for name, shape in schedule_shapes().items():
        eta = np.concatenate((prefix, peak * shape))
        r = trajectory(eta)
        assert np.array_equal(eta[:WARMUP], prefix)
        assert eta[WARMUP] == peak
        assert np.all(r >= exact_squares - 1e-14)
        assert np.all((eta >= 0.0) & (eta <= 1.0))
        records[name] = {"alpha": peak, "eta": eta, "r": r}
        report["schedules"][name] = {
            "tail_formula": formulas[name], "common_peak": peak,
            "final_loss": float(r[-1] / 2.0),
            "ratio_to_exact": float(r[-1] / exact_squares[-1]),
        }
    exact_error = float(np.max(np.abs(exact_squares - closed_squares)))
    product_error = max(abs(unrolled_final_square(v["eta"]) - v["r"][-1])
                        for v in records.values())
    tail_times = np.arange(HORIZON-WARMUP+1)
    power = (1.0 - peak)**(2.0 * tail_times)
    constant_closed = power * r_w + peak / (2.0 - peak) * (1.0 - power)
    constant_error = float(np.max(np.abs(records["constant"]["r"][WARMUP:] - constant_closed)))
    greedy_error = float(np.max(np.abs(exact_rates[WARMUP:] -
        exact_squares[WARMUP:-1] / (1.0 + exact_squares[WARMUP:-1]))))
    join_error = abs(peak - r_w / (1.0 + r_w))
    # Independently retain the unconstrained no-warmup closed-form check.
    unconstrained = trajectory(1.0 / (np.arange(HORIZON) + 2.0))
    unconstrained_error = float(np.max(np.abs(unconstrained - 1.0/(np.arange(HORIZON+1)+1.0))))
    assert max(exact_error, product_error, constant_error, greedy_error, join_error, unconstrained_error) < 5e-14
    assert exact_rates[0] == 0 and np.all(np.diff(exact_rates[:WARMUP+1]) > 0)
    assert np.all(np.diff(exact_rates[WARMUP:]) < 0)
    report["checks"] = {
        "warmup_conditional_optimum_vs_closed_form_max_abs_error": exact_error,
        "recurrence_vs_unrolled_product_max_abs_error": float(product_error),
        "constant_tail_vs_closed_form_max_abs_error": constant_error,
        "optimal_tail_vs_greedy_formula_max_abs_error": greedy_error,
        "warmup_to_tail_join_max_abs_error": float(join_error),
        "unconstrained_model_closed_form_max_abs_error": unconstrained_error,
        "every_baseline_above_optimal_tail_at_every_iterate": True,
        "identical_warmup_for_every_curve": True,
        "all_learning_rates_in_unit_interval": True,
        "all_100_updates_include_warmup": True,
    }
    return records, report


def write_csv(path: Path, records: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["schedule", "alpha", "step", "learning_rate", "expected_square", "expected_loss"]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, record in records.items():
            for t, r in enumerate(record["r"]):
                writer.writerow({"schedule": name, "alpha": record["alpha"], "step": t,
                                 "learning_rate": record["eta"][t] if t < HORIZON else "",
                                 "expected_square": r, "expected_loss": r / 2.0})
                count += 1
    return count


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    font = STATIC / "fonts/source-serif-4/SourceSerif4-Regular.ttf"
    family = "DejaVu Serif"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    matplotlib.rcParams.update({
        "font.family": family, "font.size": 11, "mathtext.fontset": "stix",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": SECONDARY,
        "ytick.color": SECONDARY, "axes.edgecolor": "#c1c4bb", "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "path", "svg.hashsalt": "lr-schedules-quadratic-20261031",
        "savefig.facecolor": "white", "savefig.dpi": 160,
    })


def draw_panel(ax, records: dict, loss: bool):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    curves = {}
    for name, record in records.items():
        values = record["r"] / 2.0 if loss else record["eta"]
        if not loss:
            boundary = (record["r"][-1] / (1.0 + record["r"][-1]) if name == "exact" else
                        record["eta"][-1] if name == "constant" else 0.0)
            values = np.append(values, boundary)
        curves[name] = values
        ax.plot(np.arange(len(values)), values, **STYLES[name])
    ax.axvspan(0, WARMUP, color="#efefeb", zorder=0)
    ax.axvline(WARMUP, color="#c0c3b9", lw=.85, ls=(0, (3, 3)), zorder=1)
    if not loss:
        ax.annotate("Warmup", xy=(WARMUP, records["exact"]["alpha"]),
                    xytext=(18, records["exact"]["alpha"] * 1.09),
                    fontsize=9, color=SECONDARY,
                    arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": .7})
    ax.set_yscale("linear")
    ax.set_xlim(0, HORIZON)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Updates completed, $t$" if loss else "Update index, $t$", labelpad=7)
    if loss:
        ax.set_ylabel(r"Expected loss $\mathbb{E}[f(x_t)]$", labelpad=8)
    else:
        ax.set_ylabel(r"Learning rate $\eta_t$", labelpad=8)
    ax.set_ylim(0, .525 if loss else records["exact"]["alpha"] * 1.22)
    ax.set_yticks(np.arange(0, .51, .1) if loss else [0, .1, .2, .3])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=3, labelsize=10)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # The full LR panel now resolves the warmup and all tails without an inset.
    if loss:
        zoom = ax.inset_axes([.39, .44, .58, .47])
        for name, values in curves.items():
            zoom.plot(np.arange(len(values)), values, **{**STYLES[name], "lw": 1.5})
        zoom.set_yscale("linear")
        zoom.set_xlim(50, HORIZON)
        zoom.set_ylim(0, .1)
        zoom.set_xticks([50, 75, 100])
        zoom.set_yticks([0, .05, .1])
        zoom.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        zoom.tick_params(labelsize=8, length=2)
        zoom.set_title("Late loss (linear zoom)", fontsize=9, pad=5)
        zoom.grid(axis="y", color=GRID, linewidth=.5)
        zoom.set_axisbelow(True)
        zoom.spines[["top", "right"]].set_visible(False)


def make_figure(path: Path, records: dict, mobile: bool, preview_dir: Path | None):
    import matplotlib.pyplot as plt
    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(5.4, 9.5))
        fig.subplots_adjust(left=0.19, right=0.965, top=0.775, bottom=0.13, hspace=0.43)
        title_y, subtitle_y, legend_y = 0.977, 0.925, 0.85
        subtitle = r"$f(x)=x^2/2$ · noise variance $1$" + "\n" + r"$\mathbb{E}[x_0^2]=1$ · $100$ updates"
        footer = "Same 5-step warmup and same peak for every curve.\nOptimal tail is conditional on this fixed prefix."
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
        fig.subplots_adjust(left=0.09, right=0.975, top=0.69, bottom=0.19, wspace=0.28)
        title_y, subtitle_y, legend_y = 0.975, 0.875, 0.80
        subtitle = r"$f(x)=x^2/2$ · noise variance $1$ · $\mathbb{E}[x_0^2]=1$ · $100$ updates"
        footer = "Same 5-step warmup and same peak; the optimal tail is conditional on this fixed prefix."
    fig.suptitle("SGD with a shared warmup", x=0.5, y=title_y,
                 fontsize=16 if mobile else 17)
    fig.text(0.5, subtitle_y, subtitle, ha="center", va="top", color=SECONDARY,
             fontsize=11, linespacing=1.6)
    draw_panel(axes[0], records, loss=False)
    draw_panel(axes[1], records, loss=True)
    axes[0].set_title("(a) Learning-rate schedules", loc="left", fontsize=12, pad=13)
    axes[1].set_title("(b) Expected loss", loc="left", fontsize=12, pad=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.52, legend_y),
               ncol=2 if mobile else 4, frameon=False, fontsize=10.5,
               handlelength=2.6, columnspacing=1.7, labelspacing=0.9)
    fig.text(0.5, 0.04, footer, ha="center", color=SECONDARY, fontsize=10.5, linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None,
        "Title": "SGD schedules with a shared five-update warmup",
        "Description": "Scalar quadratic with unit noise variance and unit initial second moment. All four schedules share five warmup updates and a common peak; total budget is 100 updates. The optimal tail is conditional on the fixed warmup, compared with linear, cosine, and constant tails at the same peak. All axes are linear. The t=100 boundary is not an applied update. Losses include warmup and are exact recurrences, not sampled training data."})
    path.write_text("".join(line.rstrip() + "\n" for line in path.read_text(encoding="utf-8").splitlines()),
                    encoding="utf-8")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{path.stem}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=OUTPUT / "comparison.csv")
    parser.add_argument("--report", type=Path, default=OUTPUT / "verification.json")
    parser.add_argument("--figure", type=Path, default=OUTPUT / "quadratic.svg")
    parser.add_argument("--mobile-figure", type=Path, default=OUTPUT / "quadratic-mobile.svg")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    records, report = calculate_and_verify()
    report["csv_rows"] = write_csv(args.csv, records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    configure_plotting()
    make_figure(args.figure, records, mobile=False, preview_dir=args.preview_dir)
    make_figure(args.mobile_figure, records, mobile=True, preview_dir=args.preview_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"CSV: {args.csv}\nReport: {args.report}\nDesktop: {args.figure}\nMobile: {args.mobile_figure}")


if __name__ == "__main__":
    main()
