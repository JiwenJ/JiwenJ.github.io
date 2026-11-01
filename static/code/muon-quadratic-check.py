#!/usr/bin/env python3
"""Reproduce a solvable matrix model of a momentum-free exact polar update.

Dependencies: NumPy and Matplotlib only (tested with 2.0.2 and 3.9.4).
Run from the repository root:

    MPLCONFIGDIR=/private/tmp/muon-quadratic-mpl \
        python3 static/code/muon-quadratic-check.py \
        --preview-dir /private/tmp/muon-quadratic-preview

Model, with m >= 2k, h > 0 and noise energy sigma2 > 0:
    W^T W = (r/k) I, f(W) = h ||W||_F^2 / 2,
    G = h W + E, E[E | W] = 0,
    W^T E = 0, E^T E = (sigma2/k) I,
    U = polar(G), W_next = W - eta U.
An equiprobable pair +/- E in an orthogonal complement of W satisfies the
conditional mean assumption exactly. The Gram condition is preserved for
either sign. Consequently the second-moment/loss recursion is pathwise,
not a Monte Carlo estimate:
    r_next = r - 2 eta h sqrt(k) r / sqrt(h^2 r + sigma2) + k eta^2.

Its minimizer and minimum are
    eta_star(r) = h r / (sqrt(k) sqrt(h^2 r + sigma2)),
    g(r) = sigma2 r / (h^2 r + sigma2).
The exact decomposition r_next = g(r) + k (eta-eta_star(r))^2 and the
monotonicity of g make the greedy schedule globally optimal for this model.
For r0 > 0, c = sigma2/(h^2 r0), the closed forms are
    r_t = sigma2 r0 / (sigma2 + t h^2 r0),
    eta_t = sqrt(sigma2)/(h sqrt(k) sqrt((t+c)(t+c+1))).

The figure uses h=sigma2=r0=1, k=4 and N=100 actual updates, t=0,...,99.
All four curves share five prescribed warmup updates, eta_t=peak*t/5.
The common peak solves peak=eta_star(r_5(peak)), so the exact optimal tail
joins the warmup without a jump. The other tails are linear, cosine, and
constant at the same peak, over the remaining 95 updates. The optimum is
conditional on this fixed prefix; the prefix and peak are not optimized.
Linear and cosine reach zero at the unused t=N budget boundary. Matrix QA
uses fixed-seed orthogonal bases and paired signs, not sampled loss estimates.

This optimality statement is ONLY for this isotropic tangent-noise matrix
model, with no momentum and an exact polar factor. It does not establish
global optimality for neural-network Muon, approximate Newton-Schulz updates,
shape-scaled variants, weight decay, or other gradient/noise models.
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
SEED = 20261101
N, K, WARMUP = 100, 4, 5
ATOL, RTOL = 3e-12, 3e-11
INK, SECONDARY, GRID = "#383832", "#74746c", "#e7e8e2"
STYLES = {
    "exact": {"label": "Optimal tail", "color": INK, "ls": "-", "lw": 2.25},
    "linear": {"label": "Linear", "color": "#b56b30", "ls": (0, (5, 2)), "lw": 2.0},
    "cosine": {"label": "Cosine", "color": "#517b8b", "ls": (0, (5, 2, 1, 2)), "lw": 2.0},
    "constant": {"label": "Constant", "color": "#94826d", "ls": (0, (1, 2)), "lw": 2.0},
}


def check(actual, expected, name: str, errors: dict):
    actual, expected = np.asarray(actual), np.asarray(expected)
    delta = float(np.max(np.abs(actual - expected), initial=0.0))
    scale = float(np.max(np.abs(expected), initial=0.0))
    entry = errors.setdefault(name, {"max_absolute_error": 0.0, "max_scaled_error": 0.0})
    entry["max_absolute_error"] = max(entry["max_absolute_error"], delta)
    entry["max_scaled_error"] = max(entry["max_scaled_error"], delta / max(1.0, scale))
    if delta > ATOL + RTOL * scale:
        raise AssertionError(f"{name}: error={delta:g}, reference scale={scale:g}")


def optimal_rate(r, h=1.0, sigma2=1.0, k=K):
    return h * r / (np.sqrt(k) * np.sqrt(h * h * r + sigma2))


def minimum_map(r, h=1.0, sigma2=1.0):
    return sigma2 * r / (h * h * r + sigma2)


def step(r, eta, h=1.0, sigma2=1.0, k=K):
    """Stable completed-square form of the exact pathwise scalar recurrence."""
    return minimum_map(r, h, sigma2) + k * (eta - optimal_rate(r, h, sigma2, k))**2


def trajectory(eta, h=1.0, sigma2=1.0, r0=1.0, k=K):
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 1 or np.any(eta < 0.0) or not np.all(np.isfinite(eta)):
        raise ValueError("Learning rates must be a finite nonnegative vector.")
    r = np.empty(len(eta) + 1)
    r[0] = r0
    for t, rate in enumerate(eta):
        r[t + 1] = step(r[t], rate, h, sigma2, k)
    return r


def verify_matrices(rng):
    shapes = ((2, 1), (4, 2), (7, 3), (8, 4), (11, 4), (9, 2))
    errors, pairs, updates, sequences = {}, 0, 0, 0
    for m, k in shapes:
        for h in (0.3, 1.0, 2.5):
            for sigma2 in (0.2, 1.0, 3.0):
                for r in (0.0, 0.05, 1.0, 4.0):
                    q, _ = np.linalg.qr(rng.normal(size=(m, 2 * k)), mode="reduced")
                    w = np.sqrt(r / k) * q[:, :k]
                    noise = np.sqrt(sigma2 / k) * q[:, k:2*k]
                    check((noise + (-noise)) / 2.0, np.zeros_like(noise), "paired_conditional_mean", errors)
                    check(w.T @ w, (r / k) * np.eye(k), "initial_isotropic_gram", errors)
                    rates = (0.0, optimal_rate(r, h, sigma2, k),
                             0.5 * optimal_rate(r, h, sigma2, k),
                             2.0 * optimal_rate(r, h, sigma2, k), 0.7)
                    sign_norms = []
                    for sign in (-1.0, 1.0):
                        e, g = sign * noise, h * w + sign * noise
                        check(w.T @ e, np.zeros((k, k)), "tangent_noise", errors)
                        check(e.T @ e, (sigma2 / k) * np.eye(k), "isotropic_noise_gram", errors)
                        check(g.T @ g, ((h*h*r + sigma2) / k) * np.eye(k), "gradient_gram", errors)
                        left, _, right_t = np.linalg.svd(g, full_matrices=False)
                        u = left @ right_t
                        check(u, np.sqrt(k) * g / np.sqrt(h*h*r + sigma2), "svd_vs_closed_polar", errors)
                        check(u.T @ u, np.eye(k), "polar_column_orthogonality", errors)
                        values = []
                        for eta in rates:
                            next_w = w - eta * u
                            next_r = step(r, eta, h, sigma2, k)
                            check(next_w.T @ next_w, (next_r / k) * np.eye(k), "gram_invariance", errors)
                            actual_r = float(np.sum(next_w**2))
                            check(actual_r, next_r, "matrix_vs_scalar_recurrence", errors)
                            direct_r = r - 2.0 * eta * h * np.sqrt(k) * r / np.sqrt(h*h*r + sigma2) + k * eta**2
                            check(next_r, direct_r, "completed_square_identity", errors)
                            values.append(actual_r)
                            updates += 1
                        sign_norms.append(values)
                    check(sign_norms[0], sign_norms[1], "both_noise_signs_have_same_loss", errors)
                    pairs += 1
        # Track real matrix/SVD updates, rebuilding an exact tangent complement
        # at each iterate. Each sign pattern is admissible; no averages are used.
        for h, sigma2, r0 in ((1.0, 1.0, 1.0), (0.5, 0.2, 3.0), (2.0, 3.0, 0.4)):
            for alternating in (False, True):
                q, _ = np.linalg.qr(rng.normal(size=(m, k)), mode="reduced")
                w = np.sqrt(r0 / k) * q
                c = sigma2 / (h*h*r0)
                for t in range(N):
                    basis, _ = np.linalg.qr(w, mode="complete")
                    sign = -1.0 if alternating and t % 2 else 1.0
                    e = sign * np.sqrt(sigma2 / k) * basis[:, k:2*k]
                    check(w.T @ e, np.zeros((k, k)), "trajectory_tangent_noise", errors)
                    eta = np.sqrt(sigma2) / (h * np.sqrt(k) * np.sqrt((t+c)*(t+c+1.0)))
                    g = h * w + e
                    left, _, right_t = np.linalg.svd(g, full_matrices=False)
                    w = w - eta * (left @ right_t)
                    exact_r = sigma2 * r0 / (sigma2 + (t+1) * h*h*r0)
                    check(w.T @ w, (exact_r / k) * np.eye(k), "trajectory_gram_vs_closed_form", errors)
                    check(float(np.sum(w*w)), exact_r, "matrix_trajectory_vs_closed_form", errors)
                sequences += 1
    return {"shapes": [list(s) for s in shapes], "paired_noise_cases": pairs,
            "matrix_updates_checked": updates, "full_svd_trajectories": sequences,
            "updates_per_trajectory": N, "errors": errors}


def verify_closed_forms():
    errors, cases = {}, 0
    grid = np.concatenate(([0.0], np.logspace(-6, 3, 501)))
    for h in (0.3, 1.0, 2.5):
        for sigma2 in (0.2, 1.0, 3.0):
            for r0 in (0.05, 1.0, 4.0):
                for k in (1, 3, 4):
                    c = sigma2 / (h*h*r0)
                    t = np.arange(N, dtype=np.float64)
                    eta = np.sqrt(sigma2) / (h * np.sqrt(k) * np.sqrt((t+c)*(t+c+1.0)))
                    r = trajectory(eta, h, sigma2, r0, k)
                    exact_r = sigma2 * r0 / (sigma2 + np.arange(N+1) * h*h*r0)
                    check(r, exact_r, "recurrence_vs_closed_second_moment", errors)
                    check(eta, optimal_rate(exact_r[:-1], h, sigma2, k), "closed_vs_greedy_rate", errors)
                    values = minimum_map(grid, h, sigma2)
                    derivative = sigma2**2 / (h*h*grid + sigma2)**2
                    assert np.all(np.diff(values) > 0.0) and np.all(derivative > 0.0)
                    cases += 1
    return {"parameter_cases": cases, "monotonicity_grid_points": len(grid),
            "minimum_map": "g(r)=sigma2*r/(h^2*r+sigma2)",
            "positive_derivative": "g'(r)=sigma2^2/(h^2*r+sigma2)^2 > 0",
            "global_optimality_reason": "q(r,eta)>=g(r), g is increasing, and eta_star attains equality",
            "errors": errors}


def warmup_peak():
    """Prescribe a prefix whose endpoint joins the conditional optimum."""
    def gap(peak):
        prefix = peak * np.arange(WARMUP, dtype=float) / WARMUP
        return peak - optimal_rate(trajectory(prefix)[-1])

    left, right = 0.0, 1.0 / np.sqrt(K)
    assert gap(left) < 0.0 < gap(right)
    for _ in range(80):
        middle = (left + right) / 2.0
        if gap(middle) > 0.0:
            right = middle
        else:
            left = middle
    return (left + right) / 2.0


def compare_schedules():
    peak = warmup_peak()
    prefix = peak * np.arange(WARMUP, dtype=float) / WARMUP
    warm_squares = trajectory(prefix)
    r_warm = warm_squares[-1]
    tail_steps = N - WARMUP
    j = np.arange(tail_steps, dtype=float)
    c_warm = 1.0 / r_warm
    tail_exact = 1.0 / (np.sqrt(K) * np.sqrt((j+c_warm)*(j+c_warm+1.0)))
    shapes = {"exact": tail_exact / peak,
              "linear": 1.0 - j/tail_steps,
              "cosine": .5 * (1.0 + np.cos(np.pi*j/tail_steps)),
              "constant": np.ones(tail_steps)}
    records, schedules, errors = {}, {}, {}
    closed_tail = 1.0 / (c_warm + np.arange(tail_steps+1))
    for name, shape in shapes.items():
        eta = np.concatenate((prefix, peak * shape))
        r = trajectory(eta)
        check(eta[:WARMUP], prefix, "identical_warmup_rates", errors)
        check(r[:WARMUP+1], warm_squares, "identical_warmup_states", errors)
        check(eta[WARMUP], peak, "continuous_warmup_join", errors)
        assert len(eta) == N and np.all(np.isfinite(eta))
        assert np.all((eta >= 0.0) & (eta <= peak + 1e-14))
        records[name] = {"alpha": peak, "eta": eta, "r": r}
        schedules[name] = {"common_peak": peak,
                           "final_loss": float(r[-1]/2.0)}
    check(records["exact"]["r"][WARMUP:], closed_tail,
          "conditional_optimum_closed_tail", errors)
    check(records["exact"]["eta"][WARMUP:],
          optimal_rate(records["exact"]["r"][WARMUP:-1]),
          "conditional_optimum_greedy_rates", errors)
    for name, data in records.items():
        assert np.all(data["r"] >= records["exact"]["r"] - 1e-14)
        schedules[name]["ratio_to_exact"] = float(data["r"][-1] / records["exact"]["r"][-1])
    return records, {"parameters": {"h": 1.0, "sigma2": 1.0, "r0": 1.0, "k": K,
                                     "updates": N, "update_indices": [0, N-1]},
                      "warmup": {"updates": WARMUP, "formula": "eta_t=peak*t/5, t=0,...,4",
                                 "common_peak": peak, "state_after_warmup": float(r_warm),
                                 "selection": "peak=optimal_rate(r_5(peak)) ensures continuous join",
                                 "fixed_prefix_optimized": False},
                      "remaining_updates": tail_steps,
                      "conditional_exact_formula": "eta_(5+j)=1/(sqrt(k)*sqrt((j+c_w)*(j+c_w+1))), c_w=1/r_5",
                      "conditional_closed_final_loss": float(closed_tail[-1]/2.0),
                      "baseline_formulas": {"linear": "peak*(1-j/95)",
                                            "cosine": "peak*(1+cos(pi*j/95))/2", "constant": "peak"},
                      "baseline_amplitudes_tuned": False,
                      "all_baselines_above_exact_at_every_iterate": True,
                      "schedules": schedules, "errors": errors}


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["schedule", "alpha", "step", "learning_rate", "squared_frobenius_norm", "expected_loss"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, data in records.items():
            for t, r in enumerate(data["r"]):
                writer.writerow({"schedule": name, "alpha": data["alpha"], "step": t,
                                 "learning_rate": data["eta"][t] if t < N else "",
                                 "squared_frobenius_norm": r, "expected_loss": r/2.0})
    return len(records)*(N+1)


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    font = STATIC / "fonts/source-serif-4/SourceSerif4-Regular.ttf"
    family = "DejaVu Serif"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    matplotlib.rcParams.update({"font.family": family, "font.size": 11, "mathtext.fontset": "stix",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": SECONDARY, "ytick.color": SECONDARY,
        "axes.edgecolor": "#c1c4bb", "axes.linewidth": 0.8, "figure.facecolor": "white",
        "axes.facecolor": "white", "svg.fonttype": "path", "svg.hashsalt": "muon-quadratic-20261101",
        "savefig.facecolor": "white", "savefig.dpi": 160})


def make_figure(path, records, mobile, preview_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(5.4, 9.5))
        fig.subplots_adjust(left=0.19, right=0.965, top=0.775, bottom=0.15, hspace=0.43)
        title_y, subtitle_y, legend_y = 0.977, 0.925, 0.85
        subtitle = r"$f(W)=\frac{1}{2}\|W\|_F^2$ · $k=4$" + "\n" + r"$\sigma^2=r_0=1$ · $100$ updates"
        footer = "Shared 5-step warmup and peak; 95 tail updates.\nOptimal tail conditional on the fixed prefix.\nExact polar update · no momentum · this model only."
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
        fig.subplots_adjust(left=0.09, right=0.975, top=0.69, bottom=0.22, wspace=0.28)
        title_y, subtitle_y, legend_y = 0.975, 0.875, 0.80
        subtitle = r"$f(W)=\frac{1}{2}\|W\|_F^2$ · $k=4$ · $\sigma^2=r_0=1$ · $100$ updates"
        footer = "Shared 5-step warmup and peak; optimal tail conditional on the fixed prefix.\nExact polar update · no momentum · isotropic tangent noise · this matrix model only."
    fig.suptitle("A solvable Muon model with warmup", y=title_y, fontsize=16 if mobile else 17)
    fig.text(0.5, subtitle_y, subtitle, ha="center", va="top", fontsize=11, color=SECONDARY, linespacing=1.6)
    peak = records["exact"]["alpha"]
    for loss, ax in enumerate(axes):
        curves = {}
        for name, data in records.items():
            values = data["r"]/2.0 if loss else data["eta"]
            if not loss:
                boundary = (optimal_rate(data["r"][-1]) if name == "exact" else
                            data["eta"][-1] if name == "constant" else 0.0)
                values = np.append(values, boundary)
            curves[name] = values
            ax.plot(np.arange(len(values)), values, **STYLES[name])
        ax.axvspan(0, WARMUP, color="#f0f0ec", zorder=0)
        ax.axvline(WARMUP, color="#b7bbb1", ls=(0, (3, 3)), lw=.8, zorder=1)
        ax.set_yscale("linear")
        ax.set_xlim(0, N)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Updates completed, $t$" if loss else "Update index, $t$", labelpad=7)
        if loss:
            ax.set_ylim(0, .525)
            ax.set_yticks(np.arange(0, .51, .1))
            ax.set_ylabel(r"Expected loss $\mathbb{E}[f(W_t)]$", labelpad=8)
        else:
            ax.set_ylim(0, peak * 1.18)
            ax.set_yticks([0, .05, .1, .15])
            ax.set_ylabel(r"Learning rate $\eta_t$", labelpad=8)
            ax.annotate("Warmup ends", xy=(WARMUP, peak), xytext=(15, peak*1.10),
                        fontsize=9, color=SECONDARY, va="center",
                        arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": .7})
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=3, labelsize=10)
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("(b) Expected loss" if loss else "(a) Learning-rate schedules", loc="left", fontsize=12, pad=13)
        # The full LR range now resolves every curve; zoom only the late loss.
        if loss:
            zoom = ax.inset_axes([.39, .44, .58, .47])
            for name, values in curves.items():
                zoom.plot(np.arange(len(values)), values, **{**STYLES[name], "lw": 1.5})
            zoom.set_xlim(50, N)
            zoom.set_ylim(0, .1)
            zoom.set_xticks([50, 75, 100])
            zoom.set_yticks([0, .05, .1])
            zoom.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            zoom.tick_params(labelsize=8, length=2)
            zoom.set_title("Late loss (linear zoom)", fontsize=9, pad=5)
            zoom.grid(axis="y", color=GRID, linewidth=.5)
            zoom.set_axisbelow(True)
            zoom.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.52, legend_y),
               ncol=2 if mobile else 4, frameon=False, fontsize=10.5, handlelength=2.6,
               columnspacing=1.7, labelspacing=0.9)
    fig.text(0.5, 0.035, footer, ha="center", fontsize=10.5, color=SECONDARY, linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None, "Title": "Solvable Muon matrix model with a common warmup",
        "Description": "Exact polar update with no momentum and isotropic tangent noise. Shared prescribed five-step linear warmup and common peak, followed by 95 updates. The exact tail is optimal conditional on this prefix; other tail shapes share the same peak and are not amplitude tuned. All axes are linear. Learning rates extend to the unused t=100 budget boundary. Optimality applies only to this matrix model, not neural-network training."})
    path.write_text("".join(line.rstrip() + "\n" for line in path.read_text(encoding="utf-8").splitlines()),
                    encoding="utf-8")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{path.stem}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=OUTPUT / "muon-quadratic-verification.json")
    parser.add_argument("--csv", type=Path, default=OUTPUT / "muon-quadratic-comparison.csv")
    parser.add_argument("--desktop-figure", type=Path, default=OUTPUT / "muon-quadratic-desktop.svg")
    parser.add_argument("--mobile-figure", type=Path, default=OUTPUT / "muon-quadratic-mobile.svg")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    records, comparison = compare_schedules()
    report = {"status": "PASS", "seed": SEED, "tolerances": {"absolute": ATOL, "relative": RTOL},
              "scope": "Exact polar update, no momentum, isotropic tangent-noise matrix quadratic; no neural-network guarantee",
              "assumptions": ["m>=2k", "h>0", "sigma2>0", "W^T W=(r/k)I", "E[E|W]=0", "W^T E=0", "E^T E=(sigma2/k)I"],
              "noise_construction": "Equiprobable +/- orthogonal-complement matrix; both signs explicitly checked",
              "loss_evaluation": "Exact pathwise scalar recurrence, not Monte Carlo",
              "matrix_checks": verify_matrices(np.random.default_rng(SEED)),
              "closed_form_checks": verify_closed_forms(), "comparison": comparison,
              "csv_rows": write_csv(args.csv, records)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    configure_plotting()
    make_figure(args.desktop_figure, records, False, args.preview_dir)
    make_figure(args.mobile_figure, records, True, args.preview_dir)
    print(json.dumps({"status": report["status"], "csv_rows": report["csv_rows"],
                      "matrix_checks": {key: value for key, value in report["matrix_checks"].items() if key != "errors"},
                      "schedules": comparison["schedules"], "report": str(args.report)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
