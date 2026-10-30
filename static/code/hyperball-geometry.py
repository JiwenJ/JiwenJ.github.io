#!/usr/bin/env python3
"""Verify Hyperball's fixed-radius wrapper and reproduce its geometry figures.

Dependencies: NumPy and Matplotlib only (tested with 2.0.2 and 3.9.4).
From the repository root:

    MPLCONFIGDIR=/private/tmp/hyperball-mpl \
        python3 static/code/hyperball-geometry.py

Optional explicit outputs / PNG previews for visual review:

    MPLCONFIGDIR=/private/tmp/hyperball-mpl \
        python3 static/code/hyperball-geometry.py \
        --csv static/data/hyperball/geometry.csv \
        --report static/data/hyperball/verification.json \
        --figure static/images/hyperball/geometry.svg \
        --mobile-figure static/images/hyperball/geometry-mobile.svg \
        --preview-dir /private/tmp/hyperball-preview

For nonzero U, this FP64 reference implements exactly
    W_tilde = W - eta * R * U / ||U||_F
    W_next = R * W_tilde / ||W_tilde||_F.
The intended input satisfies ||W||_F = R. A zero U is a no-op; nonpositive
R, negative eta, nonfinite inputs, and a zero trial W_tilde are rejected.

U is a supplied update DIRECTION, with descent sign handled by subtraction.
There is no base optimizer, optimizer state, weight decay, training loss,
mixed precision, or epsilon approximation in this numerical reference.
All plotted values are geometric calculations, not training measurements.

If c = <W/R, U/||U||_F> and ||W||_F = R, the exact nonnegative angle is
    atan2(eta * sqrt(1-c**2), 1-eta*c).
The singular case eta = c = 1 has no projected point. Plots use eta < 1.

For comparison, the tangent variant removes the radial component from the
UNIT update before taking the step, without renormalizing its tangent part:
    T = U/||U||_F - <W/R, U/||U||_F> * W/R
    W_tangent = R * (W/R - eta*T) / ||W/R - eta*T||_F.
That is a different operation; its angle is atan(eta*sqrt(1-c**2)).
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
SEED = 20261030
INK = "#383832"
SECONDARY = "#74746c"
GREEN = "#5c7964"
GRID = "#e7e8e2"


def frobenius(x: np.ndarray) -> float:
    """Compute a Frobenius norm in FP64, avoiding overflow when squaring."""
    scale = float(np.max(np.abs(x), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.sqrt(np.sum((x / scale) ** 2, dtype=np.float64)))


def hyperball_step(w: np.ndarray, u: np.ndarray, eta: float, radius: float) -> np.ndarray:
    """Apply only the normalized-update / radial-projection wrapper.

    Returns a new FP64 matrix, leaving both inputs untouched. With zero U,
    returns W unchanged; callers should initialize W on the radius-R sphere.
    """
    w = np.asarray(w, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if w.ndim != 2 or w.size == 0 or w.shape != u.shape:
        raise ValueError("W and U must be nonempty matrices of the same shape.")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(u)):
        raise ValueError("W and U must contain only finite values.")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("R must be finite and strictly positive.")
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and nonnegative.")
    u_norm = frobenius(u)
    if u_norm == 0.0:
        return w.copy()
    if not np.isfinite(u_norm):
        raise ValueError("The update norm must be representable in FP64.")
    trial = w - (eta * radius) * (u / u_norm)
    trial_norm = frobenius(trial)
    if trial_norm == 0.0 or not np.isfinite(trial_norm):
        raise ValueError("The trial must have a finite, nonzero norm.")
    return radius * (trial / trial_norm)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Measure the angle with atan2, including angles near zero and pi."""
    a = a / frobenius(a)
    b = b / frobenius(b)
    cosine = float(np.sum(a * b, dtype=np.float64))
    perpendicular = b - cosine * a
    return float(np.arctan2(frobenius(perpendicular), cosine))


def analytic_angle(eta: float, c: float | np.ndarray) -> float | np.ndarray:
    """Exact angle on the radius-R sphere, away from a zero trial point."""
    c = np.asarray(c, dtype=np.float64)
    if (eta < 0.0 or not np.isfinite(eta) or not np.all(np.isfinite(c))
            or np.any(np.abs(c) > 1.0)):
        raise ValueError("Require eta >= 0 and -1 <= c <= 1.")
    if eta == 1.0 and np.any(c == 1.0):
        raise ValueError("eta = c = 1 gives a zero trial point.")
    return np.arctan2(eta * np.sqrt(np.maximum(0.0, 1.0 - c * c)), 1.0 - eta * c)


def tangent_step(w: np.ndarray, u: np.ndarray, eta: float, radius: float) -> np.ndarray:
    """Comparison only: project unit U onto the tangent space before stepping."""
    w_hat = w / radius
    u_hat = u / frobenius(u)
    c = float(np.sum(w_hat * u_hat))
    tangent = u_hat - c * w_hat
    trial = w_hat - eta * tangent
    return radius * trial / frobenius(trial)


def verify() -> dict:
    """Run deterministic algebraic checks, not optimizer/training benchmarks."""
    rng = np.random.default_rng(SEED)
    maxima = {"norm_relative_error": 0.0, "positive_scale_relative_error": 0.0,
              "angle_absolute_error_rad": 0.0, "repeated_norm_relative_error": 0.0}
    count = 0
    for shape in ((3, 5), (16, 7), (9, 9)):
        for radius in (0.3, 1.0, 9.0):
            for eta in (0.0, 0.1, 0.5, 0.95, 1.5):
                for _ in range(10):
                    w = rng.normal(size=shape)
                    w = radius * w / frobenius(w)
                    u = rng.normal(size=shape)
                    next_w = hyperball_step(w, u, eta, radius)
                    c = float(np.clip(np.sum((w / radius) * (u / frobenius(u))), -1.0, 1.0))
                    maxima["norm_relative_error"] = max(maxima["norm_relative_error"],
                        abs(frobenius(next_w) / radius - 1.0))
                    maxima["angle_absolute_error_rad"] = max(maxima["angle_absolute_error_rad"],
                        abs(angle_between(w, next_w) - float(analytic_angle(eta, c))))
                    for scale in (1e-6, 0.1, 17.0, 1e6):
                        scaled = hyperball_step(w, scale * u, eta, radius)
                        maxima["positive_scale_relative_error"] = max(
                            maxima["positive_scale_relative_error"], frobenius(next_w - scaled) / radius)
                    count += 1

    radius = 3.7
    w = rng.normal(size=(13, 8))
    w = radius * w / frobenius(w)
    for _ in range(500):
        w = hyperball_step(w, rng.normal(size=w.shape), 0.5, radius)
        maxima["repeated_norm_relative_error"] = max(maxima["repeated_norm_relative_error"],
            abs(frobenius(w) / radius - 1.0))

    # Boundary cases are tested separately because random matrices rarely align.
    w = np.array([[1.0, 0.0]], dtype=np.float64)
    for eta in (0.0, 0.1, 0.5, 0.95, 1.5):
        for c in (-1.0, 0.0, 1.0):
            u = np.array([[c, -np.sqrt(1.0 - c * c)]])
            actual = angle_between(w, hyperball_step(w, u, eta, 1.0))
            assert abs(actual - float(analytic_angle(eta, c))) < 2e-14
    zero_result = hyperball_step(w, np.zeros_like(w), 0.5, 1.0)
    assert np.array_equal(zero_result, w) and not np.shares_memory(zero_result, w)
    for invalid_radius in (0.0, -1.0):
        try:
            hyperball_step(w, w, 0.5, invalid_radius)
        except ValueError:
            pass
        else:
            raise AssertionError("A nonpositive radius was accepted.")
    for eta, u in ((1.0, w), (-0.1, w)):
        try:
            hyperball_step(w, u, eta, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("A zero trial or negative step was accepted.")

    eta, c = 0.5, 0.6
    u = np.array([[c, -np.sqrt(1.0 - c * c)]])
    radial = hyperball_step(w, u, eta, 1.0)
    tangent = tangent_step(w, u, eta, 1.0)
    tangent_angle = angle_between(w, tangent)
    assert abs(tangent_angle - np.arctan(eta * np.sqrt(1.0 - c * c))) < 2e-14
    assert frobenius(radial - tangent) > 0.1
    for value in maxima.values():
        assert value < 2e-14, maxima
    return {
        "status": "PASS", "scope": "FP64 geometric identities; no training data",
        "seed": SEED, "random_matrix_cases": count, "repeated_steps": 500,
        "positive_update_scales": [1e-6, 0.1, 17.0, 1e6],
        "max_errors": maxima,
        "boundary_checks": ["zero update is a no-op", "nonpositive R rejected",
                            "zero trial rejected", "negative eta rejected",
                            "aligned and orthogonal directions including eta > 1"],
        "tangent_comparison": {
            "eta": eta, "c": c, "tangent_renormalized_before_step": False,
            "hyperball_angle_rad": angle_between(w, radial),
            "tangent_angle_rad": tangent_angle,
            "endpoint_distance_at_R1": frobenius(radial - tangent),
        },
    }


def write_csv(path: Path) -> tuple[list[dict], float]:
    """Save analytic / independently measured values for both plotted curves."""
    rows = []
    w = np.array([[1.0, 0.0]])
    for eta in (0.1, 0.5):
        for c in np.linspace(-1.0, 1.0, 201):
            u = np.array([[c, -np.sqrt(max(0.0, 1.0 - c * c))]])
            next_w = hyperball_step(w, u, eta, 1.0)
            analytic = float(analytic_angle(eta, c))
            numeric = angle_between(w, next_w)
            rows.append({"eta": eta, "c": float(c), "analytic_angle_rad": analytic,
                         "numeric_angle_rad": numeric, "angle_error_rad": abs(analytic - numeric),
                         "next_norm": frobenius(next_w),
                         "tangent_variant_angle_rad": float(np.arctan(eta * np.sqrt(max(0.0, 1.0-c*c))))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    max_error = max(row["angle_error_rad"] for row in rows)
    assert max_error < 2e-14
    return rows, max_error


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
        "ytick.color": SECONDARY, "axes.edgecolor": "#c1c4bb",
        "axes.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "path", "svg.hashsalt": "hyperball-geometry-20261030",
        "savefig.facecolor": "white", "savefig.dpi": 150,
    })


def draw_geometry(ax):
    from matplotlib.patches import Arc, Circle, FancyArrowPatch
    c, eta = 0.6, 0.5
    w = np.array([1.0, 0.0])
    trial = w - eta * np.array([c, -np.sqrt(1.0 - c * c)])
    next_w = trial / np.linalg.norm(trial)
    theta = float(analytic_angle(eta, c))
    ax.set_aspect("equal")
    ax.set_xlim(-1.13, 1.56)
    ax.set_ylim(-1.15, 1.19)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="#afb8ac", lw=1.2))
    ax.plot([-1.06, 1.1], [0, 0], color=GRID, lw=0.8, zorder=0)
    ax.plot([0, 0], [-1.06, 1.06], color=GRID, lw=0.8, zorder=0)
    ax.plot([0, w[0]], [0, w[1]], color=SECONDARY, lw=1.3)
    ax.plot([0, next_w[0]], [0, next_w[1]], color=GREEN, lw=1.4)
    ax.add_patch(FancyArrowPatch(w, trial, arrowstyle="-|>", mutation_scale=12,
                                color=INK, lw=1.7, shrinkA=3, shrinkB=3))
    ax.add_patch(FancyArrowPatch(trial, next_w, arrowstyle="-|>", mutation_scale=12,
                                color=GREEN, lw=2, shrinkA=3, shrinkB=3))
    ax.add_patch(Arc((0, 0), 0.73, 0.73, theta1=0,
                     theta2=np.degrees(theta), color=GREEN, lw=1.4))
    ax.scatter([0, w[0]], [0, w[1]], s=[15, 32], color=INK, zorder=5)
    ax.scatter(*trial, s=32, color="white", edgecolor=INK, lw=1.4, zorder=6)
    ax.scatter(*next_w, s=38, color=GREEN, zorder=6)
    ax.text(-0.06, -0.11, "$0$", ha="right", va="top")
    ax.text(1.07, -0.04, "$W=(1,0)$", ha="left", va="top", fontsize=11)
    ax.text(0.4, 0.07, r"$\theta$", color=GREEN, fontsize=14)
    ax.annotate(r"$-\eta\,\widehat U$", xy=(0.88, 0.17), xytext=(1.1, 0.2),
                color=INK, fontsize=13, va="center",
                arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.7})
    ax.annotate(r"Trial $\widetilde{W}$", xy=trial, xytext=(0.12, 0.65), ha="center", fontsize=11,
                arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.7,
                            "shrinkB": 5})
    ax.annotate("Radial projection", xy=(trial + next_w)/2, xytext=(0.45, 1.08),
                ha="center", va="center", color=GREEN, fontsize=11,
                arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 0.8,
                            "connectionstyle": "angle3,angleA=-90,angleB=0", "shrinkB": 7})
    ax.text(next_w[0]+0.08, next_w[1]+0.07, "$W_{+}$", color=GREEN, fontsize=14)
    ax.text(0, -0.84, r"$\|W\|_F=R=1$", ha="center", color=SECONDARY, fontsize=11)
    ax.text(0.06, -0.35, r"$c=0.6,\quad\eta=0.5$", fontsize=12)
    ax.text(0.06, -0.52, r"$\widehat U=(0.6,-0.8)$", color=SECONDARY, fontsize=11)


def draw_angles(ax):
    c = np.linspace(-1.0, 1.0, 801)
    for eta, color in ((0.1, SECONDARY), (0.5, GREEN)):
        ax.plot(c, analytic_angle(eta, c), color=color, lw=2.1,
                label=rf"$\eta={eta}$ (exact)")
        marker_c = np.linspace(-1.0, 1.0, 17)
        numeric = []
        w = np.array([[1.0, 0.0]])
        for value in marker_c:
            u = np.array([[value, -np.sqrt(max(0, 1-value*value))]])
            numeric.append(angle_between(w, hyperball_step(w, u, eta, 1.0)))
        ax.scatter(marker_c, numeric, s=21, facecolor="white", edgecolor=color,
                   linewidth=1.1, zorder=3, label="FP64 check" if eta == 0.5 else None)
    ax.axvline(0, color="#bfc5ba", ls=(0, (3, 4)), lw=0.9, zorder=0)
    ax.annotate(r"$c=0:\ \theta=\arctan\eta$", xy=(0, np.arctan(0.5)),
                xytext=(-0.86, 0.59), color=GREEN, fontsize=12,
                arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 0.8})
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-0.012, 0.66)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xlabel(r"Alignment $c=\langle\widehat W,\widehat U\rangle$", labelpad=7)
    ax.set_ylabel(r"Angle $\theta$ (radians)", labelpad=7)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, labelsize=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.48, 0.57),
              fontsize=10, handlelength=1.6, labelspacing=0.6)


def make_figure(path: Path, mobile: bool, preview_dir: Path | None):
    import matplotlib.pyplot as plt
    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(5.15, 9.6),
                                 gridspec_kw={"height_ratios": [1.08, 1]})
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.145, top=0.835, hspace=0.30)
        title_y, subtitle_y = 0.975, 0.915
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.95),
                                 gridspec_kw={"width_ratios": [1.08, 1]})
        fig.subplots_adjust(left=0.035, right=0.975, bottom=0.21, top=0.74, wspace=0.27)
        title_y, subtitle_y = 0.975, 0.865
    title = "Hyperball: a fixed-radius step" if mobile else "Hyperball: one step on a fixed-radius sphere"
    subtitle = ("Analytic geometry\nNormalized update + radial projection" if mobile
                else "Analytic geometry · normalized update + radial projection")
    fig.suptitle(title, x=0.5, y=title_y, fontsize=16 if mobile else 17)
    fig.text(0.5, subtitle_y, subtitle, linespacing=1.5,
             ha="center", color=SECONDARY, fontsize=11)
    draw_geometry(axes[0])
    draw_angles(axes[1])
    for ax, title in zip(axes, ("(a) Trial point and projection", "(b) Exact angular displacement")):
        ax.set_title(title, loc="left", fontsize=12, pad=17)
    fig.text(0.5, 0.035,
             r"$\theta=\operatorname{atan2}\!\left(\eta\sqrt{1-c^2},\ 1-\eta c\right)$",
             ha="center", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None,
        "Title": "Hyperball fixed-radius update: analytic geometry",
        "Description": "Circle geometry and exact angular displacement. Lines are analytic; markers are FP64 checks. No training data."})
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{path.stem}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=STATIC / "data/hyperball/geometry.csv")
    parser.add_argument("--report", type=Path, default=STATIC / "data/hyperball/verification.json")
    parser.add_argument("--figure", type=Path, default=STATIC / "images/hyperball/geometry.svg")
    parser.add_argument("--mobile-figure", type=Path, default=STATIC / "images/hyperball/geometry-mobile.svg")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    report = verify()
    rows, max_csv_error = write_csv(args.csv)
    report["csv_rows"] = len(rows)
    report["csv_max_angle_error_rad"] = max_csv_error
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    configure_plotting()
    make_figure(args.figure, mobile=False, preview_dir=args.preview_dir)
    make_figure(args.mobile_figure, mobile=True, preview_dir=args.preview_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"CSV: {args.csv}\nReport: {args.report}\nDesktop: {args.figure}\nMobile: {args.mobile_figure}")


if __name__ == "__main__":
    main()
