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

The optimal deterministic nonnegative schedule for this instance is
eta[t] = 1/(t+2), giving r[t] = 1/(t+1). Each comparison is independently
tuned on alpha in [0,1] for the SAME final-loss objective:

    linear:   eta[t] = alpha * (1 - t/100)
    cosine:   eta[t] = alpha * (1 + cos(pi*t/100)) / 2
    constant: eta[t] = alpha.

The zero endpoint of linear/cosine at t=100 is not an applied update.
Amplitude tuning scans the derivative on two dense grids, bisects every
bracketed stationary point, and compares these points with both endpoints.
The reported minimizers are numerical results, not a formal global root
isolation certificate. The instance-optimal schedule follows analytically
from the monotone map r -> r/(1+r); this is not a claim about general convex
objectives, AdaGrad, neural networks, or empirical training performance.

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
INK = "#383832"
SECONDARY = "#74746c"
GRID = "#e7e8e2"
STYLES = {
    "exact": {"label": "Exact optimum", "color": INK, "ls": "-", "lw": 2.25},
    "linear": {"label": "Linear (tuned)", "color": "#b56b30", "ls": (0, (5, 2)), "lw": 2.0},
    "cosine": {"label": "Cosine (tuned)", "color": "#517b8b", "ls": (0, (5, 2, 1, 2)), "lw": 2.0},
    "constant": {"label": "Constant (tuned)", "color": "#94826d", "ls": (0, (1, 2)), "lw": 2.0},
}


def schedule_shapes() -> dict[str, np.ndarray]:
    t = np.arange(HORIZON, dtype=np.float64)
    return {
        "linear": 1.0 - t / HORIZON,
        "cosine": 0.5 * (1.0 + np.cos(np.pi * t / HORIZON)),
        "constant": np.ones(HORIZON, dtype=np.float64),
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


def final_loss_and_derivative(alpha, shape: np.ndarray):
    """Forward sensitivity dr/dalpha, including dependence of every iterate."""
    alpha = np.asarray(alpha, dtype=np.float64)
    r = np.ones_like(alpha)
    derivative = np.zeros_like(alpha)
    for c in shape:
        eta = alpha * c
        derivative = ((1.0 - eta)**2 * derivative
                      - 2.0 * c * (1.0 - eta) * r + 2.0 * c * eta)
        r = (1.0 - eta)**2 * r + eta**2
    return r / 2.0, derivative / 2.0


def bisect_derivative(shape: np.ndarray, left: float, right: float) -> float:
    f_left = float(final_loss_and_derivative(left, shape)[1])
    f_right = float(final_loss_and_derivative(right, shape)[1])
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if f_left * f_right > 0.0:
        raise ValueError("Stationary-point bracket must contain a sign change.")
    for _ in range(80):
        middle = (left + right) / 2.0
        f_middle = float(final_loss_and_derivative(middle, shape)[1])
        if f_middle == 0.0 or right - left <= 2e-15:
            return middle
        if (f_middle < 0.0) == (f_left < 0.0):
            left, f_left = middle, f_middle
        else:
            right = middle
    return (left + right) / 2.0


def tune_grid(shape: np.ndarray, grid_points: int) -> dict:
    grid = np.linspace(0.0, 1.0, grid_points)
    losses, derivatives = final_loss_and_derivative(grid, shape)
    crossings = np.flatnonzero(derivatives[:-1] * derivatives[1:] <= 0.0)
    roots = [bisect_derivative(shape, float(grid[i]), float(grid[i + 1]))
             for i in crossings]
    candidates = [0.0, 1.0, *roots]
    values = [float(final_loss_and_derivative(a, shape)[0]) for a in candidates]
    best = int(np.argmin(values))
    return {
        "alpha": candidates[best], "final_loss": values[best],
        "stationary_points": roots, "grid_points": grid_points,
        "best_grid_loss": float(np.min(losses)),
        "endpoint_losses": {"alpha_0": values[0], "alpha_1": values[1]},
    }


def unrolled_final_square(eta: np.ndarray) -> float:
    """Independent product-and-sum expression for the final second moment."""
    future_product, noise_sum = 1.0, 0.0
    for rate in reversed(eta):
        noise_sum += rate**2 * future_product
        future_product *= (1.0 - rate)**2
    return float(future_product + noise_sum)


def calculate_and_verify() -> tuple[dict, dict]:
    times = np.arange(HORIZON + 1, dtype=np.float64)
    exact_rates = 1.0 / (np.arange(HORIZON, dtype=np.float64) + 2.0)
    closed_squares = 1.0 / (times + 1.0)
    exact_squares = trajectory(exact_rates)
    records = {"exact": {"alpha": None, "eta": exact_rates, "r": exact_squares}}
    report = {
        "status": "PASS",
        "scope": "Exact expected-loss recurrence for one scalar noisy quadratic; no sampled data",
        "model": {"objective": "f(x)=x^2/2", "initial_second_moment": 1.0,
                  "noise_conditional_mean": 0.0, "noise_conditional_variance": 1.0,
                  "updates": HORIZON, "update_indices": [0, HORIZON - 1]},
        "tuning": {"alpha_interval": [0.0, 1.0],
                   "objective": "expected loss after all 100 updates",
                   "method": "scan derivative; bisect every sign change; compare all roots and both endpoints",
                   "grids": [32769, 65537], "formal_global_certificate": False},
        "schedules": {"exact": {"formula": "eta_t=1/(t+2)",
                                "final_loss": float(exact_squares[-1] / 2.0),
                                "closed_form_final_loss": 1.0 / (2.0 * (HORIZON + 1)),
                                "ratio_to_exact": 1.0}},
        "checks": {},
    }
    formulas = {"linear": "alpha*(1-t/T)",
                "cosine": "alpha*(1+cos(pi*t/T))/2", "constant": "alpha"}
    for name, shape in schedule_shapes().items():
        coarse = tune_grid(shape, 32769)
        fine = tune_grid(shape, 65537)
        alpha = fine["alpha"]
        eta = alpha * shape
        r = trajectory(eta)
        _, gradient = final_loss_and_derivative(alpha, shape)
        assert abs(alpha - coarse["alpha"]) < 1e-12
        assert fine["final_loss"] <= fine["best_grid_loss"] + 1e-15
        assert abs(float(gradient)) < 1e-11
        assert np.all(r >= exact_squares - 1e-14)
        assert np.all((eta >= 0.0) & (eta <= 1.0))
        records[name] = {"alpha": alpha, "eta": eta, "r": r}
        report["schedules"][name] = {
            "formula": formulas[name], "optimal_alpha": alpha,
            "final_loss": float(r[-1] / 2.0),
            "ratio_to_exact": float(r[-1] / exact_squares[-1]),
            "stationary_points_found": fine["stationary_points"],
            "derivative_at_alpha": float(gradient),
            "coarse_vs_fine_alpha_difference": abs(alpha - coarse["alpha"]),
            "best_dense_grid_loss": fine["best_grid_loss"],
            "endpoint_losses": fine["endpoint_losses"],
        }
    exact_error = float(np.max(np.abs(exact_squares - closed_squares)))
    product_error = max(abs(unrolled_final_square(v["eta"]) - v["r"][-1])
                        for v in records.values())
    a = records["constant"]["alpha"]
    power = (1.0 - a)**(2.0 * times)
    constant_closed = power + a / (2.0 - a) * (1.0 - power)
    constant_error = float(np.max(np.abs(records["constant"]["r"] - constant_closed)))
    # Greedy state map gives the global optimum because r/(1+r) is increasing.
    greedy_error = float(np.max(np.abs(exact_rates - exact_squares[:-1] / (1.0 + exact_squares[:-1]))))
    assert max(exact_error, product_error, constant_error, greedy_error) < 5e-14
    report["checks"] = {
        "exact_schedule_vs_closed_form_max_abs_error": exact_error,
        "recurrence_vs_unrolled_product_max_abs_error": float(product_error),
        "constant_schedule_vs_closed_form_max_abs_error": constant_error,
        "exact_schedule_vs_greedy_formula_max_abs_error": greedy_error,
        "every_baseline_above_exact_at_every_iterate": True,
        "all_learning_rates_in_unit_interval": True,
        "refined_tuning_matches_on_both_grids": True,
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
    for name, record in records.items():
        values = record["r"] / 2.0 if loss else record["eta"]
        ax.plot(np.arange(len(values)), values, **STYLES[name])
    ax.set_yscale("log")
    ax.set_xlim(0, HORIZON)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Updates completed, $t$" if loss else "Update index, $t$", labelpad=7)
    if loss:
        ax.set_ylim(0.0038, 0.7)
        ax.set_yticks([0.005, 0.01, 0.05, 0.1, 0.5])
        ax.set_ylabel(r"Expected loss $\mathbb{E}[f(x_t)]$ (log scale)", labelpad=8)
    else:
        ax.set_ylim(1e-5, 0.75)
        ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5])
        ax.set_ylabel(r"Learning rate $\eta_t$ (log scale)", labelpad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=3, labelsize=10)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(path: Path, records: dict, mobile: bool, preview_dir: Path | None):
    import matplotlib.pyplot as plt
    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(5.4, 9.5))
        fig.subplots_adjust(left=0.19, right=0.965, top=0.775, bottom=0.13, hspace=0.43)
        title_y, subtitle_y, legend_y = 0.977, 0.925, 0.85
        subtitle = r"$f(x)=x^2/2$ · noise variance $1$" + "\n" + r"$\mathbb{E}[x_0^2]=1$ · $100$ updates"
        footer = "Exact expectation; each baseline amplitude is tuned\nfor the final loss on this same scalar quadratic."
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
        fig.subplots_adjust(left=0.09, right=0.975, top=0.69, bottom=0.19, wspace=0.28)
        title_y, subtitle_y, legend_y = 0.975, 0.875, 0.80
        subtitle = r"$f(x)=x^2/2$ · noise variance $1$ · $\mathbb{E}[x_0^2]=1$ · $100$ updates"
        footer = "Exact expectation; each baseline amplitude is tuned for the final loss on this same scalar quadratic."
    fig.suptitle("An exactly solvable SGD example", x=0.5, y=title_y,
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
        "Title": "Exact expected-loss comparison of SGD learning-rate schedules",
        "Description": "Scalar quadratic with unit noise variance, unit initial second moment, and 100 updates. Exact optimum compared with independently tuned linear, cosine, and constant schedules. Both vertical axes are logarithmic. No sampled training data."})
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
