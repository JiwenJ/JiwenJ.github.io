#!/usr/bin/env python3
"""Illustrate an optimal schedule in a conditional tangent-alignment model.

This is not a MuonH training experiment. The model assumes unit tangent
updates with constant target-direction drift a, so its exact scalar closure
is m_next=(m+a*epsilon)/sqrt(1+epsilon**2). A prescribed warmup is followed
by a monotone tail with a common peak, zero final applied update, and an
artificial maximum decrement per step. None of those constraints is inferred
from the optimizer. All comparisons have the same admissible tail constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from schedule_plot_style import (
    INK, MUTED, GRID, LINEAR, COSINE, CONSTANT, WARMUP_FILL, WARMUP_EDGE,
    configure,
)

STATIC = Path(__file__).resolve().parents[1]
PREVIEWS = Path(__file__).resolve().parents[3] / "artifacts"
TOTAL, WARMUP, TAIL = 3200, 160, 3040
PEAK, DRIFT, INITIAL_TAIL_ALIGNMENT, MAX_DROP = .03, .0075, .25, 6e-5
STYLES = {
    "optimum": {"label": "Model optimum", "color": INK, "ls": "-", "lw": 2.25},
    "linear": {"label": "Linear", "color": LINEAR, "ls": (0, (5, 2)), "lw": 2.0},
    "cosine": {"label": "Cosine", "color": COSINE, "ls": (0, (5, 2, 1, 2)), "lw": 2.0},
    "air": {"label": "AIR", "color": CONSTANT, "ls": (0, (1, 2)), "lw": 2.0},
}


def tail_schedules(n, peak, delta):
    j = np.arange(n, dtype=float)
    progress = j/(n-1)
    unconstrained = peak/np.sqrt(1+peak*peak*j)
    terminal_cap = delta*(n-1-j)
    alpha = (n-1)/8
    q = 1/np.sqrt(1+j/alpha)
    curves = {
        "optimum": np.minimum(unconstrained, terminal_cap),
        "linear": peak*(1-progress),
        "cosine": .5*peak*(1+np.cos(np.pi*progress)),
        "air": peak*(q-q[-1])/(1-q[-1]),
    }
    for values in curves.values():
        values[0], values[-1] = peak, 0.0
    return curves, unconstrained, terminal_cap


def trajectory(initial, rates, drift):
    states = np.empty(len(rates)+1)
    states[0] = initial
    for j, rate in enumerate(rates):
        states[j+1] = (states[j]+drift*rate)/np.sqrt(1+rate*rate)
    return states


def terminal_loss_gradient(initial, rates, drift):
    states = trajectory(initial, rates, drift)
    gradient = np.empty_like(rates)
    adjoint = -1.0  # derivative of the terminal model loss 1-m
    for j in range(len(rates)-1, -1, -1):
        rate = rates[j]
        gradient[j] = adjoint*(drift-states[j]*rate)/(1+rate*rate)**1.5
        adjoint /= np.sqrt(1+rate*rate)
    return 1-states[-1], gradient


def check_feasible(values, peak, delta):
    drops = values[:-1]-values[1:]
    np.testing.assert_allclose(values[[0, -1]], [peak, 0.0], rtol=0, atol=1e-14)
    assert np.min(values) >= -1e-14 and np.max(values) <= peak+1e-14
    assert np.min(drops) >= -1e-14, "Tail must be monotone."
    assert np.max(drops) <= delta+1e-14, "Maximum-decrement constraint violated."
    return float(np.max(drops))


def small_slsqp_check():
    """Independent finite-dimensional check; the algebra, not SLSQP, proves optimality."""
    n, initial, peak, drift, delta = 32, .25, .3, .075, .06
    curves, _, _ = tail_schedules(n, peak, delta)
    exact = curves["optimum"]
    exact_loss = 1-trajectory(initial, exact, drift)[-1]
    difference = np.eye(n-1, n)-np.eye(n-1, n, k=1)
    offset = difference[:, 0]*peak
    constraint = LinearConstraint(difference[:, 1:-1], -offset, delta-offset)

    def objective(interior):
        rates = np.concatenate(([peak], interior, [0.0]))
        value, gradient = terminal_loss_gradient(initial, rates, drift)
        return value, gradient[1:-1]

    checks = []
    for name in ("linear", "cosine", "air"):
        check_feasible(curves[name], peak, delta)
        result = minimize(objective, curves[name][1:-1], jac=True, method="SLSQP",
                          bounds=Bounds(0, peak), constraints=[constraint],
                          options={"maxiter": 1500, "ftol": 1e-13})
        candidate = np.concatenate(([peak], result.x, [0.0]))
        drops = candidate[:-1]-candidate[1:]
        feasibility_error = max(0.0, float(-drops.min()), float(drops.max()-delta))
        states = trajectory(initial, candidate, drift)
        assert np.max(states*states+drift*drift) <= 1+1e-12
        checks.append({"start": name, "success": bool(result.success),
                       "status": int(result.status), "iterations": int(result.nit),
                       "terminal_loss": float(result.fun),
                       "loss_gap_to_formula": float(result.fun-exact_loss),
                       "schedule_max_error": float(np.max(np.abs(candidate-exact))),
                       "constraint_error": feasibility_error})
    assert all(x["success"] and abs(x["loss_gap_to_formula"]) < 1e-9
               and x["constraint_error"] < 1e-10 for x in checks)
    return {"n": n, "m0": initial, "a": drift, "peak": peak, "delta": delta,
            "formula_terminal_loss": float(exact_loss), "solver_checks": checks}


def build_and_verify():
    rng = np.random.default_rng(20261102)
    m = rng.uniform(.01, .9, 1000)
    a = rng.uniform(.001, .1, 1000)
    e = rng.uniform(0, .6, 1000)
    lhs = ((m+a*e)/np.sqrt(1+e*e))**2
    rhs = m*m+a*a-(m*e-a)**2/(1+e*e)
    identity_error = float(np.max(np.abs(lhs-rhs)))
    assert identity_error < 1e-14

    tails, free, cap = tail_schedules(TAIL, PEAK, MAX_DROP)
    warmup = np.linspace(0, PEAK, WARMUP)
    initial = INITIAL_TAIL_ALIGNMENT
    for rate in reversed(warmup):
        initial = initial*np.sqrt(1+rate*rate)-DRIFT*rate
    records, terminal_losses, drops = {}, {}, {}
    for name, tail in tails.items():
        drops[name] = check_feasible(tail, PEAK, MAX_DROP)
        rates = np.concatenate((warmup, tail))
        states = trajectory(initial, rates, DRIFT)
        np.testing.assert_allclose(states[WARMUP], INITIAL_TAIL_ALIGNMENT, rtol=0, atol=1e-14)
        assert np.min(states) > 0 and np.max(states*states+DRIFT*DRIFT) <= 1+1e-14
        assert rates[-1] == 0 and len(rates) == TOTAL
        records[name] = {"rates": rates, "alignment": states, "loss": 1-states}
        terminal_losses[name] = float(1-states[-1])
    assert DRIFT == PEAK*INITIAL_TAIL_ALIGNMENT
    assert INITIAL_TAIL_ALIGNMENT**2+TAIL*DRIFT**2 <= 1
    assert free[0]-free[1] <= MAX_DROP and PEAK <= MAX_DROP*(TAIL-1)
    switch = int(np.flatnonzero(cap <= free)[0])
    np.testing.assert_allclose(tails["optimum"][:switch], free[:switch], rtol=0, atol=1e-14)
    np.testing.assert_allclose(tails["optimum"][switch:], cap[switch:], rtol=0, atol=1e-14)
    assert all(terminal_losses["optimum"] <= value+1e-13 for value in terminal_losses.values())
    summary = {"identity_max_error": identity_error, "total_updates": TOTAL,
               "fixed_warmup_updates": WARMUP, "initial_alignment_before_warmup": float(initial),
               "tail_initial_alignment": INITIAL_TAIL_ALIGNMENT, "a": DRIFT,
               "peak": PEAK, "artificial_decrement_cap": MAX_DROP,
               "switch_tail_index": switch, "switch_global_update": WARMUP+switch,
               "maximum_tail_decrements": drops, "terminal_model_losses": terminal_losses,
               "small_slsqp": small_slsqp_check()}
    return records, summary


def figure(records, summary, output, mobile, preview_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(5.4, 9.5))
        fig.subplots_adjust(left=.19, right=.965, top=.775, bottom=.13, hspace=.43)
        title_y, subtitle_y, legend_y = .977, .925, .85
        subtitle = "Constant drift · tangent unit updates\n" + r"$a=0.0075$ · $m_{\mathrm{tail}}=0.25$"
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
        fig.subplots_adjust(left=.09, right=.975, top=.69, bottom=.19, wspace=.28)
        title_y, subtitle_y, legend_y = .975, .875, .80
        subtitle = r"Constant drift · tangent unit updates · $a=0.0075$ · $m_{\mathrm{tail}}=0.25$"
    fig.suptitle("Tangent alignment model", y=title_y, fontsize=16 if mobile else 17)
    fig.text(.5, subtitle_y, subtitle, ha="center", va="top", color=MUTED, fontsize=11, linespacing=1.6)
    ax_rate, ax_loss = axes
    x = np.arange(TOTAL+1)
    for name, record in records.items():
        ax_rate.plot(x, np.append(record["rates"], 0)/PEAK, **STYLES[name])
        ax_loss.plot(x, record["loss"], **STYLES[name])
    for ax in axes:
        ax.axvspan(0, WARMUP, color=WARMUP_FILL, zorder=0)
        ax.axvline(WARMUP, color=WARMUP_EDGE, lw=.85, ls=(0, (3, 3)), zorder=1)
        ax.set_xlim(0, TOTAL)
        ax.set_xticks([0, 800, 1600, 2400, 3200])
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3, labelsize=10)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax_rate.set_xlabel("Update index, $t$", labelpad=7)
    ax_loss.set_xlabel("Updates completed, $t$", labelpad=7)
    ax_rate.set_title("(a) Normalized effective learning rate", loc="left", fontsize=12, pad=13)
    ax_loss.set_title("(b) Model loss", loc="left", fontsize=12, pad=13)
    ax_rate.set_ylabel(r"Normalized ELR $\varepsilon_t/E$", labelpad=8)
    ax_loss.set_ylabel(r"Model loss $1-m_t$", labelpad=8)
    ax_rate.set_ylim(0, 1.22)
    ax_rate.set_yticks([0, .25, .5, .75, 1])
    lo = min(record["loss"].min() for record in records.values())
    hi = max(record["loss"].max() for record in records.values())
    ax_loss.set_ylim(lo-.06*(hi-lo), hi+.28*(hi-lo))
    ax_loss.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_rate.annotate("Warmup", xy=(WARMUP, 1), xytext=(.18*TOTAL, 1.09), fontsize=9,
                     color=MUTED, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .7})
    switch = summary["switch_global_update"]
    switch_height = records["optimum"]["rates"][switch]/PEAK
    ax_rate.annotate("Terminal cap", xy=(switch, switch_height), xytext=(.61*TOTAL, .89),
                     fontsize=9, color=MUTED,
                     arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .7})
    # The full curves occupy the lower-right area, so keep the zoom above them.
    inset = ax_loss.inset_axes([.45, .58, .52, .34])
    for name, record in records.items():
        inset.plot(x, record["loss"], **{**STYLES[name], "lw": 1.5})
    late = x >= TOTAL//2
    late_lo = min(record["loss"][late].min() for record in records.values())
    late_hi = max(record["loss"][late].max() for record in records.values())
    inset.set_xlim(TOTAL//2, TOTAL)
    inset.set_ylim(late_lo-.06*(late_hi-late_lo), late_hi+.06*(late_hi-late_lo))
    inset.set_xticks([1600, 2400, 3200])
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
    inset.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    inset.tick_params(labelsize=8, length=2)
    inset.set_title("Late loss (linear zoom)", fontsize=9, pad=5)
    inset.grid(axis="y", color=GRID, linewidth=.5)
    inset.set_axisbelow(True)
    inset.spines[["top", "right"]].set_visible(False)
    handles, labels = ax_rate.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.52, legend_y),
               ncol=2 if mobile else 4, frameon=False, fontsize=10.5,
               handlelength=2.6, columnspacing=1.7, labelspacing=.9)
    footer = ("Prescribed 5% warmup; same peak and zero final update.\n"
              + r"Artificial tail constraint: $0\leq\varepsilon_j-\varepsilon_{j+1}\leq6\times10^{-5}$.") if mobile else (
              "Prescribed 5% warmup; same peak and zero final update; "
              + r"artificial maximum decrement $\delta=6\times10^{-5}$.")
    fig.text(.5, .04, footer, ha="center", color=MUTED, fontsize=10.5, linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None, "Title": "Tangent alignment model: constrained ELR schedules",
        "Description": "Conditional scalar alignment model, not actual MuonH training. Constant target drift, tangent unit updates, a prescribed 160-update warmup, and an artificial tail maximum decrement of 0.00006. All curves share the same peak and zero final applied rate. The inverse-square-root plus terminal-cap curve is the model optimum under those assumptions; comparisons use linear, cosine, and affine inverse-square-root (AIR) tails. All axes are linear."})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines())+"\n")
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_dir/f"{output.stem}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEWS)
    args = parser.parse_args()
    records, summary = build_and_verify()
    configure("muonh-alignment-model-20261102")
    for mobile in (False, True):
        variant = "mobile" if mobile else "desktop"
        output = STATIC/"images/lr-schedules"/f"muonh-alignment-{variant}.svg"
        figure(records, summary, output, mobile, args.preview_dir)
    print(json.dumps({"status": "PASS", **summary}, indent=2))


if __name__ == "__main__":
    main()
