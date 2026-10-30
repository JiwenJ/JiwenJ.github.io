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
Each linear, cosine, and constant baseline independently tunes its amplitude
alpha in [0,1/sqrt(k)] for the exact loss AFTER all N updates. Linear and
cosine have a formal zero at t=N, which is not an applied update. Tuning
scans two dense derivative grids, bisects all bracketed stationary points,
and compares with both endpoints. This is a numerical tuning check, not a
formal root isolation certificate. Matrix QA uses fixed-seed orthogonal
bases and paired signs, not stochastic loss estimates.

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
N, K = 100, 4
ATOL, RTOL = 3e-12, 3e-11
INK, SECONDARY, GRID = "#383832", "#74746c", "#e7e8e2"
STYLES = {
    "exact": {"label": "Exact optimum", "color": INK, "ls": "-", "lw": 2.25},
    "linear": {"label": "Linear (tuned)", "color": "#b56b30", "ls": (0, (5, 2)), "lw": 2.0},
    "cosine": {"label": "Cosine (tuned)", "color": "#517b8b", "ls": (0, (5, 2, 1, 2)), "lw": 2.0},
    "constant": {"label": "Constant (tuned)", "color": "#94826d", "ls": (0, (1, 2)), "lw": 2.0},
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


def shapes():
    t = np.arange(N, dtype=np.float64)
    return {"linear": 1.0-t/N, "cosine": 0.5*(1.0+np.cos(np.pi*t/N)), "constant": np.ones(N)}


def final_loss_and_derivative(alpha, shape):
    """Exact forward sensitivity includes the evolving r at every update."""
    alpha = np.asarray(alpha, dtype=np.float64)
    r, derivative = np.ones_like(alpha), np.zeros_like(alpha)
    for coefficient in shape:
        eta = alpha * coefficient
        denominator = r + 1.0
        partial_r = 1.0 - eta * np.sqrt(K) * (r + 2.0) / denominator**1.5
        partial_eta = -2.0 * np.sqrt(K) * r / np.sqrt(denominator) + 2.0 * K * eta
        derivative = partial_r * derivative + partial_eta * coefficient
        r = step(r, eta)
    return r / 2.0, derivative / 2.0


def tune(shape, grid_points):
    grid = np.linspace(0.0, 1.0/np.sqrt(K), grid_points)
    losses, derivatives = final_loss_and_derivative(grid, shape)
    roots = []
    for index in np.flatnonzero(derivatives[:-1] * derivatives[1:] <= 0.0):
        left, right = float(grid[index]), float(grid[index+1])
        f_left = float(derivatives[index])
        if f_left == 0.0:
            roots.append(left)
            continue
        if float(derivatives[index+1]) == 0.0:
            roots.append(right)
            continue
        for _ in range(80):
            middle = (left + right) / 2.0
            f_middle = float(final_loss_and_derivative(middle, shape)[1])
            if f_middle == 0.0 or right-left <= 2e-15:
                break
            if (f_middle < 0.0) == (f_left < 0.0):
                left, f_left = middle, f_middle
            else:
                right = middle
        roots.append(middle)
    candidates = [0.0, 1.0/np.sqrt(K), *roots]
    values = [float(final_loss_and_derivative(alpha, shape)[0]) for alpha in candidates]
    best = int(np.argmin(values))
    return {"alpha": candidates[best], "final_loss": values[best], "roots": roots,
            "best_grid_loss": float(np.min(losses)), "endpoint_losses": values[:2]}


def compare_schedules():
    t = np.arange(N, dtype=np.float64)
    eta = 1.0 / (np.sqrt(K) * np.sqrt((t+1.0)*(t+2.0)))
    records = {"exact": {"alpha": None, "eta": eta, "r": trajectory(eta)}}
    report = {"exact": {"formula": "1/(sqrt(k)*sqrt((t+1)*(t+2)))",
                         "final_loss": float(records["exact"]["r"][-1]/2),
                         "closed_form_final_loss": 1.0/(2*(N+1)), "ratio_to_exact": 1.0}}
    errors = {}
    check(records["exact"]["r"], 1.0/(np.arange(N+1)+1), "plotted_exact_closed_form", errors)
    for name, shape in shapes().items():
        coarse, fine = tune(shape, 32769), tune(shape, 65537)
        check(coarse["alpha"], fine["alpha"], "coarse_vs_fine_amplitude", errors)
        alpha = fine["alpha"]
        eta = alpha * shape
        r = trajectory(eta)
        loss, derivative = final_loss_and_derivative(alpha, shape)
        check(loss, r[-1]/2.0, "sensitivity_vs_trajectory_loss", errors)
        check(derivative, 0.0, "stationary_amplitude_derivative", errors)
        assert float(loss) <= fine["best_grid_loss"] + 1e-15
        assert np.all(r >= records["exact"]["r"] - 1e-14)
        assert np.all((eta >= 0.0) & (eta <= 1.0/np.sqrt(K)))
        # A central finite difference independently checks the sensitivity code.
        delta = 1e-6
        finite_difference = (trajectory((alpha+delta)*shape)[-1] - trajectory((alpha-delta)*shape)[-1]) / (4.0*delta)
        assert abs(finite_difference) < 2e-8
        records[name] = {"alpha": alpha, "eta": eta, "r": r}
        report[name] = {"optimal_alpha": alpha, "final_loss": float(r[-1]/2.0),
                        "ratio_to_exact": float(r[-1]/records["exact"]["r"][-1]),
                        "stationary_points_found": fine["roots"],
                        "derivative_at_alpha": float(derivative),
                        "central_difference_at_alpha": float(finite_difference),
                        "best_dense_grid_loss": fine["best_grid_loss"],
                        "endpoint_losses": fine["endpoint_losses"]}
    return records, {"parameters": {"h": 1.0, "sigma2": 1.0, "r0": 1.0, "k": K,
                                     "updates": N, "update_indices": [0, N-1]},
                      "amplitude_interval": [0.0, 1.0/np.sqrt(K)], "derivative_grids": [32769, 65537],
                      "central_difference_check": {"step": 1e-6, "absolute_tolerance": 2e-8},
                      "tuning_method": "dense derivative scan; bisect all sign changes; compare roots and endpoints",
                      "formal_global_root_certificate": False,
                      "baseline_formulas": {"linear": "alpha*(1-t/N)", "cosine": "alpha*(1+cos(pi*t/N))/2", "constant": "alpha"},
                      "all_baselines_above_exact_at_every_iterate": True, "schedules": report, "errors": errors}


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
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 9.7) if mobile else (9.0, 8.5))
    fig.subplots_adjust(left=0.19 if mobile else 0.12, right=0.965,
                        top=0.755 if mobile else 0.765, bottom=0.16, hspace=0.48)
    fig.suptitle("An exactly solvable Muon model", y=0.977, fontsize=16 if mobile else 17)
    subtitle = (r"$f(W)=\frac{1}{2}\|W\|_F^2$ · $k=4$" + "\n" + r"$\sigma^2=r_0=1$ · $100$ updates" if mobile
                else r"$f(W)=\frac{1}{2}\|W\|_F^2$ · $k=4$ · $\sigma^2=r_0=1$ · $100$ updates")
    fig.text(0.5, 0.922, subtitle, ha="center", va="top", fontsize=11, color=SECONDARY, linespacing=1.6)
    for loss, ax in enumerate(axes):
        for name, data in records.items():
            values = data["r"]/2.0 if loss else data["eta"]
            ax.plot(np.arange(len(values)), values, **STYLES[name])
        ax.set_yscale("log")
        ax.set_xlim(0, N)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Updates completed, $t$" if loss else "Update index, $t$", labelpad=7)
        if loss:
            ax.set_ylim(0.0038, 0.7)
            ax.set_yticks([0.005, 0.01, 0.05, 0.1, 0.5])
            ax.set_ylabel(r"Expected loss $\mathbb{E}[f(W_t)]$ (log scale)", labelpad=8)
        else:
            ax.set_ylim(4e-6, 0.55)
            ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1])
            ax.set_ylabel(r"Learning rate $\eta_t$ (log scale)", labelpad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=3, labelsize=10)
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("(b) Expected loss" if loss else "(a) Learning-rate schedules", loc="left", fontsize=12, pad=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.52, 0.835 if mobile else 0.847),
               ncol=2 if mobile else 4, frameon=False, fontsize=10.5, handlelength=2.6,
               columnspacing=1.7, labelspacing=0.9)
    footer = ("Exact polar update · no momentum\nIsotropic tangent noise · this matrix model only\nEach baseline is tuned for the final loss." if mobile
              else "Exact polar update · no momentum · isotropic tangent noise\nEach baseline is tuned for the final loss on this matrix model only.")
    fig.text(0.5, 0.035, footer, ha="center", fontsize=10.5, color=SECONDARY, linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None, "Title": "Solvable isotropic matrix model of a Muon polar update",
        "Description": "Exact polar update with no momentum and isotropic tangent noise. Independently tuned baseline schedules; exact pathwise loss recurrence, no Monte Carlo. Both vertical axes logarithmic. Optimality applies only to this matrix model, not neural-network training."})
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
