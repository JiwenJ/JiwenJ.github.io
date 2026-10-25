#!/usr/bin/env python3
"""Reproduce a one-step SGD coordinate check for SP and muP using NumPy.

Dependencies: numpy, matplotlib. No deep-learning framework is needed.

From the repository root:
    python3 static/code/mup-coordinate-check.py \
        --csv static/data/mup/coordinate-check.csv \
        --figure static/images/mup/coordinate-check.svg \
        --mobile-figure static/images/mup/coordinate-check-mobile.svg

The network is bias-free: x -> ReLU(W1 x) -> ReLU(W2 h1) -> w3 h2.
The arrays store the actual forward weights: there are no forward multipliers.
The objective is 0.5 * mean((prediction - target)**2).

                         W1                 W2                w3
    SP init std       1/sqrt(d)          1/sqrt(n)          1/sqrt(n)
    muP init std      1/sqrt(d)          1/sqrt(n)             1/n
    SP SGD LR            eta                eta                eta
    muP SGD LR          eta*n               eta               eta/n

All gradients are evaluated at initialization on one fixed training batch.
Each layer is then updated ALONE; the measurement on a fixed probe batch is
the RMS of its direct PREACTIVATION effect:
    input:   delta_z1 = X_probe @ delta_W1.T
    hidden:  delta_z2 = h1_probe @ delta_W2.T
    readout: delta_f  = h2_probe @ delta_w3
Here h1_probe and h2_probe are the unmodified initial activations. These are
coordinate-update measurements, not loss-transfer or downstream benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Small matrices run faster without a large BLAS thread pool. Set these before
# importing NumPy; explicit caller settings remain respected.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


LAYERS = ("input", "hidden", "readout")
PARAMETERIZATIONS = ("SP", "muP")


def forward(x: np.ndarray, weights: tuple[np.ndarray, ...]):
    """Return predictions and the intermediates needed for manual backprop."""
    w1, w2, w3 = weights
    z1 = x @ w1.T
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ w2.T
    h2 = np.maximum(z2, 0.0)
    return h2 @ w3, (z1, h1, z2, h2)


def loss_and_gradients(x, y, weights):
    prediction, (z1, h1, z2, h2) = forward(x, weights)
    _, w2, w3 = weights
    residual = prediction - y
    df = residual / x.shape[0]
    dz2 = (df[:, None] * w3[None, :]) * (z2 > 0.0)
    dz1 = (dz2 @ w2) * (z1 > 0.0)
    gradients = (dz1.T @ x, dz2.T @ h1, h2.T @ df)
    return float(0.5 * np.mean(residual**2)), gradients


def finite_difference_check():
    """Check every weight in a tiny network, safely away from ReLU kinks."""
    rng = np.random.default_rng(20261025)
    x = rng.normal(size=(4, 3))
    y = rng.normal(size=4)
    for _ in range(1000):
        weights = (
            rng.normal(size=(5, 3)) / np.sqrt(3),
            rng.normal(size=(5, 5)) / np.sqrt(5),
            rng.normal(size=5) / 5,
        )
        _, (z1, _, z2, _) = forward(x, weights)
        margin = min(float(np.min(np.abs(z1))), float(np.min(np.abs(z2))))
        if margin > 1e-3:
            break
    else:
        raise RuntimeError("Could not construct a gradient check away from ReLU zero.")

    _, analytic = loss_and_gradients(x, y, weights)
    numeric = []
    epsilon = 1e-6
    for weight in weights:
        finite = np.zeros_like(weight)
        for index in np.ndindex(weight.shape):
            original = weight[index]
            weight[index] = original + epsilon
            plus, _ = loss_and_gradients(x, y, weights)
            weight[index] = original - epsilon
            minus, _ = loss_and_gradients(x, y, weights)
            weight[index] = original
            finite[index] = (plus - minus) / (2 * epsilon)
        numeric.append(finite)

    exact = np.concatenate([gradient.ravel() for gradient in analytic])
    approximate = np.concatenate([gradient.ravel() for gradient in numeric])
    relative_error = float(
        np.linalg.norm(exact - approximate)
        / max(np.linalg.norm(exact) + np.linalg.norm(approximate), 1e-15)
    )
    max_error = float(np.max(np.abs(exact - approximate)))
    if relative_error > 1e-6 or max_error > 1e-8:
        raise AssertionError(f"Manual gradient check failed: {relative_error=}, {max_error=}")
    return relative_error, max_error, margin


def synthetic_batches(dimension, batch_size, probe_size, data_seed):
    """A fixed nonlinear teacher gives unit-RMS, nonzero-mean targets.

    The 0.5 offset prevents an exactly centered batch from cancelling the
    constant component of the ReLU readout kernel. The target is explicitly
    fixed independently of network width, initialization, and parameterization.
    """
    rng = np.random.default_rng(data_seed)
    x = rng.normal(size=(batch_size, dimension))
    probe = rng.normal(size=(probe_size, dimension))
    teacher = rng.normal(size=dimension) / np.sqrt(dimension)
    y = 0.5 + np.sin(x @ teacher) + 0.2 * (x[:, 0] ** 2 - 1.0)
    y /= np.sqrt(np.mean(y**2))
    return x, y, probe


def coordinate_check(args):
    x, y, probe = synthetic_batches(
        args.dimension, args.batch_size, args.probe_size, args.data_seed
    )
    rows = []
    for width in args.widths:
        for seed in range(args.seeds):
            rng = np.random.default_rng(np.random.SeedSequence([args.weight_seed, width, seed]))
            # SP and muP share the SAME standard-normal draws for a paired comparison.
            w1 = rng.normal(size=(width, args.dimension)) / np.sqrt(args.dimension)
            w2 = rng.normal(size=(width, width)) / np.sqrt(width)
            readout_draw = rng.normal(size=width)
            for parameterization in PARAMETERIZATIONS:
                is_mup = parameterization == "muP"
                w3 = readout_draw / (width if is_mup else np.sqrt(width))
                weights = (w1, w2, w3)
                _, gradients = loss_and_gradients(x, y, weights)
                _, (_, h1_probe, _, h2_probe) = forward(probe, weights)
                rates = (
                    (args.eta * width, args.eta, args.eta / width)
                    if is_mup else (args.eta, args.eta, args.eta)
                )
                delta_w1, delta_w2, delta_w3 = (
                    -rate * gradient for rate, gradient in zip(rates, gradients)
                )
                effects = (
                    probe @ delta_w1.T,
                    h1_probe @ delta_w2.T,
                    h2_probe @ delta_w3,
                )
                for layer, effect in zip(LAYERS, effects):
                    rms = float(np.sqrt(np.mean(effect**2)))
                    if not np.isfinite(rms) or rms <= 0.0:
                        raise AssertionError(f"Nonpositive or nonfinite effect: {width=}, {seed=}")
                    rows.append({
                        "parameterization": parameterization,
                        "width": width,
                        "seed": seed,
                        "layer": layer,
                        "rms_direct_preactivation_update": rms,
                        "eta": args.eta,
                        "dimension": args.dimension,
                        "batch_size": args.batch_size,
                        "probe_size": args.probe_size,
                        "data_seed": args.data_seed,
                        "weight_seed": args.weight_seed,
                    })
    return rows


def aggregate(rows, widths):
    result = {}
    for layer in LAYERS:
        for parameterization in PARAMETERIZATIONS:
            values = [
                [row["rms_direct_preactivation_update"] for row in rows
                 if row["layer"] == layer and row["parameterization"] == parameterization
                 and row["width"] == width]
                for width in widths
            ]
            result[layer, parameterization] = np.quantile(values, [0.25, 0.5, 0.75], axis=1)
    return result


def draw_figure(summary, args, *, mobile=False):
    # Keep this script usable after downloading it outside the website repository.
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mup-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent
    site_font = repo_root / "static/fonts/source-serif-4/SourceSerif4-Regular.ttf"
    font_path = args.font or site_font
    font_family = "DejaVu Serif"
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 14,
        "mathtext.fontset": "cm",
        # Embed glyph outlines so the self-hosted font survives SVG <img> use.
        "svg.fonttype": "path",
        "svg.hashsalt": "mup-coordinate-check-v1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#aaa7a1",
        "axes.labelcolor": "#36332e",
        "text.color": "#36332e",
        "xtick.color": "#69655e",
        "ytick.color": "#69655e",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    titles = (r"Input $W_1$", r"Hidden $W_2$", r"Readout $w_3$")
    labels = (r"RMS of $\Delta z_1$", r"RMS of $\Delta z_2$", r"RMS of $\Delta f$")
    colors = {"SP": "#8d8b87", "muP": "#267962"}
    if mobile:
        fig, axes = plt.subplots(3, 1, figsize=(4.8, 10.3))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.2))
    for ax, layer, title, ylabel in zip(axes, LAYERS, titles, labels):
        for parameterization in PARAMETERIZATIONS:
            q25, median, q75 = summary[layer, parameterization]
            ax.plot(
                args.widths, median,
                color=colors[parameterization],
                linestyle="--" if parameterization == "SP" else "-",
                marker="o", markersize=4.5, linewidth=1.8,
                label="SP" if parameterization == "SP" else r"$\mu$P",
            )
            ax.fill_between(args.widths, q25, q75, color=colors[parameterization], alpha=0.12, linewidth=0)
        ax.set(xscale="log", yscale="log", title=title, xlabel=r"Width $n$", ylabel=ylabel)
        ax.xaxis.set_major_locator(FixedLocator(args.widths))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.grid(axis="y", color="#e9e6df", linewidth=0.7)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=12, loc="best", handlelength=2.5)
    title = "One SGD step:\ndirect coordinate updates" if mobile else "One SGD step: direct coordinate updates"
    fig.suptitle(title, fontsize=17, x=0.5, y=0.99)
    subtitle = (
        f"Fixed batches · {args.seeds} initializations"
        + ("\nMedian and interquartile range · " if mobile else " · median and interquartile range · ")
        + rf"$\eta={args.eta:g}$"
    )
    fig.text(
        0.5, 0.90, subtitle,
        ha="center", fontsize=11.5, color="#69655e",
    )
    if mobile:
        fig.subplots_adjust(left=0.20, right=0.97, bottom=0.055, top=0.83, hspace=0.62)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.76, wspace=0.58)
    output = args.mobile_figure if mobile else args.figure
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def report(summary, args):
    print("layer,param,median_at_min_width,median_at_max_width,ratio_max/min,loglog_slope")
    slopes = {}
    for layer in LAYERS:
        for parameterization in PARAMETERIZATIONS:
            median = summary[layer, parameterization][1]
            slope = float(np.polyfit(np.log(args.widths), np.log(median), 1)[0])
            slopes[layer, parameterization] = slope
            print(f"{layer},{parameterization},{median[0]:.8g},{median[-1]:.8g},"
                  f"{median[-1]/median[0]:.5g},{slope:+.4f}")
    # A bounded coordinate check should be approximately flat. This is a loose
    # numerical diagnostic for the default width range, not a theoretical proof.
    if max(abs(slopes[layer, "muP"]) for layer in LAYERS) > 0.35:
        print("NOTE: at least one muP slope exceeds 0.35 on this finite width range.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument("--data-seed", type=int, default=314159)
    parser.add_argument("--weight-seed", type=int, default=271828)
    parser.add_argument("--csv", type=Path, default=Path("coordinate-check.csv"))
    parser.add_argument("--figure", type=Path, default=Path("coordinate-check.svg"))
    parser.add_argument("--mobile-figure", type=Path, help="Optional three-row figure, using the same measurements.")
    parser.add_argument("--font", type=Path, help="Optional local TTF font for the SVG labels.")
    args = parser.parse_args()
    args.widths = sorted(set(args.widths))
    if len(args.widths) < 2 or min(args.widths) < 1:
        parser.error("--widths needs at least two distinct positive widths")
    if min(args.seeds, args.dimension, args.batch_size, args.probe_size) < 1 or args.eta <= 0:
        parser.error("seeds, dimensions, batch sizes, and eta must be positive")
    return args


def main():
    args = parse_args()
    relative_error, max_error, margin = finite_difference_check()
    print(f"Finite differences: relative error={relative_error:.3e}; "
          f"max absolute error={max_error:.3e}; min |preactivation|={margin:.3e}")
    rows = coordinate_check(args)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows, args.widths)
    report(summary, args)
    draw_figure(summary, args)
    if args.mobile_figure:
        draw_figure(summary, args, mobile=True)
    print(f"Wrote {len(rows)} per-seed measurements to {args.csv}")
    print(f"Wrote figure to {args.figure}")
    if args.mobile_figure:
        print(f"Wrote mobile figure to {args.mobile_figure}")


if __name__ == "__main__":
    main()
