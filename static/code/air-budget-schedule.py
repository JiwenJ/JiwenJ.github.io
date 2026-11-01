#!/usr/bin/env python3
"""Compare AIR with a finite-grid global optimum at the same ELR budget.

This is a deterministic tangent-alignment model, not a MuonH training run:
    m_next = (m + a * epsilon) / sqrt(1 + epsilon**2).
Warmup is prescribed. Every tail has a fixed first peak, a zero final applied
rate, a common cap, and an exactly equal sum of rates. AIR is rounded once to
the 64-level grid to choose the common physical budget. Refinements retain
that budget and original quantized baseline. For the displayed comparison,
continuous AIR's alpha is calibrated to this budget, so quantization cannot
explain the improvement over AIR. The finest (256-level) DP is displayed.

The dynamic program stores the greatest alignment for each spent integer
budget. Keeping only that scalar is exact because the next-state function is
strictly increasing in alignment. It optimizes all orders on the grid. An
exchange argument permits sorting the recovered interior rates in descending
order without decreasing terminal alignment; this is checked independently.
An analytical O(h**2) gap bound also encloses the continuous model optimum.
For the finest grid, rational coefficient checks and outward-rounded
float64 dynamic programming enclose its value; the analytic gap constant is
rounded upward as well. No actual-optimizer global optimality is claimed.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from schedule_plot_style import (
    CONSTANT, GRID, INK, MUTED, WARMUP_EDGE, WARMUP_FILL, configure,
)

STATIC = Path(__file__).resolve().parents[1]
PREVIEWS = Path(__file__).resolve().parents[3] / "artifacts"
TOTAL, WARMUP, TAIL = 160, 8, 152
PEAK, DRIFT, TAIL_ALIGNMENT = .2, .05, .25
BASE_LEVELS = 64
STYLES = {
    "optimum": {"label": "DP schedule (K = 256)", "color": INK,
                "ls": "-", "lw": 2.25},
    "air": {"label": "AIR (budget matched)", "color": CONSTANT,
            "ls": (0, (5, 2)), "lw": 2.0},
}


def trajectory(initial, rates, drift):
    """Direct recurrence, independent of the DP's precomputed affine form."""
    states = np.empty(len(rates)+1)
    states[0] = initial
    for index, rate in enumerate(rates):
        states[index+1] = (states[index]+drift*rate)/np.sqrt(1+rate*rate)
    return states


def air_schedule(n, peak, alpha):
    inverse_root = 1/np.sqrt(1+np.arange(n)/alpha)
    rates = peak*(inverse_root-inverse_root[-1])/(1-inverse_root[-1])
    rates[0], rates[-1] = peak, 0.0
    return rates


def grid_optimum(n, levels, budget, initial, peak, drift):
    """Exact finite-action Bellman recurrence, evaluated in float64.

    P_i[b] = max_{0 <= k <= L} A_k P_{i-1}[b-k] + B_k,
    where A_k=(1+epsilon_k**2)**(-1/2), B_k=a*epsilon_k*A_k.
    The first peak and final zero are fixed, leaving n-2 free updates.
    Complexity is O(n * budget * levels); backpointers need O(n * budget).
    """
    assert n >= 2 and 0 < levels < 65535
    assert levels <= budget <= (n-1)*levels
    free_steps, free_budget = n-2, budget-levels
    rates = peak*np.arange(levels+1)/levels
    scale = 1/np.sqrt(1+rates*rates)
    offset = drift*rates*scale
    previous = np.full(free_budget+1, -np.inf)
    previous[0] = (initial+drift*peak)/np.sqrt(1+peak*peak)
    choices = np.full((free_steps, free_budget+1), 65535, dtype=np.uint16)
    for step in range(free_steps):
        current = np.full_like(previous, -np.inf)
        # Future rates can spend at most levels per remaining free update.
        lower = max(0, free_budget-(free_steps-step-1)*levels)
        upper = min(free_budget, (step+1)*levels)
        for level in range(levels+1):
            start, stop = max(lower, level), upper+1
            if start >= stop:
                continue
            candidate = previous[start-level:stop-level]*scale[level]+offset[level]
            view = current[start:stop]
            improve = candidate > view
            view[improve] = candidate[improve]
            choices[step, start:stop][improve] = level
        previous = current
    optimum = float(previous[free_budget])
    assert np.isfinite(optimum)
    path = np.empty(n, dtype=int)
    path[0], path[-1] = levels, 0
    spent = free_budget
    for step in range(free_steps-1, -1, -1):
        level = int(choices[step, spent])
        assert 0 <= level <= levels
        path[step+1] = level
        spent -= level
    assert spent == 0 and path.sum() == budget
    unsorted_alignment = trajectory(initial, peak*path/levels, drift)[-1]
    path[1:-1] = np.sort(path[1:-1])[::-1]
    sorted_alignment = trajectory(initial, peak*path/levels, drift)[-1]
    np.testing.assert_allclose([unsorted_alignment, sorted_alignment], optimum,
                               rtol=0, atol=2e-13)
    assert np.all(np.diff(path) <= 0)
    return path, optimum


def exhaustive_check():
    """Enumerate every interior sequence in a separate small problem."""
    n, levels, budget = 7, 4, 12
    best, feasible = -np.inf, 0
    for middle in product(range(levels+1), repeat=n-2):
        if sum(middle)+levels != budget:
            continue
        feasible += 1
        rates = PEAK*np.array((levels, *middle, 0))/levels
        best = max(best, trajectory(TAIL_ALIGNMENT, rates, DRIFT)[-1])
    path, value = grid_optimum(n, levels, budget, TAIL_ALIGNMENT, PEAK, DRIFT)
    error = float(abs(best-value))
    assert error < 2e-14
    return {"tail_updates": n, "levels": levels, "budget_units": budget,
            "enumerated_sequences": (levels+1)**(n-2),
            "budget_feasible_sequences": feasible, "max_error": error,
            "optimal_grid_rates": path.tolist()}


def continuous_gap_constants():
    """Conservative Hessian bounds for budget-preserving balanced rounding.

    At a continuous optimum, the linear Taylor term vanishes: free-control
    gradients share one budget multiplier, and controls at a box boundary
    are already on the grid. A balanced rounding has sum(delta)=0,
    ||delta||_1 <= n*h/2, and ||delta||_2**2 <= n*h**2/4.
    The remaining Taylor term is bounded by
        C_h = h**2 / 8 * (n * H_diag + n**2 * H_offdiag).
    This is an analytical model bound; float64 evaluation is not an
    interval-arithmetic certificate. The reported bracket adds 1e-12.
    """
    assert PEAK <= 1/np.sqrt(2)
    free = TAIL-2
    start = (TAIL_ALIGNMENT+DRIFT*PEAK)/np.sqrt(1+PEAK*PEAK)
    upper_alignment = np.sqrt(start*start+free*DRIFT*DRIFT)
    rate_derivative = max(DRIFT, upper_alignment*PEAK-DRIFT)
    return {"free_updates": free, "alignment_after_fixed_peak": float(start),
            "alignment_bound_M": float(upper_alignment),
            "first_derivative_bound": float(rate_derivative),
            "diagonal_hessian_bound": float(upper_alignment+3*DRIFT*PEAK),
            "offdiagonal_hessian_bound": float(PEAK*rate_derivative),
            "floating_point_safety_margin": 1e-12,
            "arithmetic_scope": "analytical bound evaluated in float64; not interval-certified"}


def rational_float_bound(value, upper):
    """Convert an exact rational to a float with a checked directed bound."""
    result = float(value)
    direction = np.inf if upper else -np.inf
    while (Fraction(result) < value if upper else Fraction(result) > value):
        result = float(np.nextafter(result, direction))
    return result


def interval_coefficients(levels):
    """Exact rational checks certify bounds on r=(1+eta²)^(-1/2), b=a*eta*r."""
    drift, peak = Fraction(1, 20), Fraction(1, 5)
    r_lo, r_hi, b_lo, b_hi = (np.empty(levels+1) for _ in range(4))
    for level in range(levels+1):
        rate = peak*level/levels
        square_denominator = 1+rate*rate
        lower = upper = float(1/np.sqrt(float(square_denominator)))
        while Fraction(lower)**2*square_denominator > 1:
            lower = float(np.nextafter(lower, -np.inf))
        while Fraction(upper)**2*square_denominator < 1:
            upper = float(np.nextafter(upper, np.inf))
        assert Fraction(lower)**2*square_denominator <= 1
        assert Fraction(upper)**2*square_denominator >= 1
        r_lo[level], r_hi[level] = lower, upper
        b_lo[level] = rational_float_bound(drift*rate*Fraction(lower), upper=False)
        b_hi[level] = rational_float_bound(drift*rate*Fraction(upper), upper=True)
    return r_lo, r_hi, b_lo, b_hi


def interval_affine(values, scale, offset, upper):
    """Round multiplication and addition separately, preserving unreachable states."""
    direction = np.inf if upper else -np.inf
    finite = np.isfinite(values)
    product = np.nextafter(values*scale, direction)
    result = np.nextafter(product+offset, direction)
    return np.where(finite, result, -np.inf)


def interval_certificate(levels, budget, chosen_path):
    """Enclose the finite-grid optimum and a continuous-model optimality gap.

    Exact tail parameters are m0=1/4, E=1/5, a=1/20. Coefficient inequalities
    are verified with Fraction. Subsequent arithmetic uses float64 plus one
    outward nextafter after each operation, under IEEE-754 round-to-nearest.
    Maxima preserve interval bounds. Unreachable states remain -infinity.
    """
    started = perf_counter()
    free, free_budget = TAIL-2, budget-levels
    r_lo, r_hi, b_lo, b_hi = interval_coefficients(levels)
    previous_lo = np.full(free_budget+1, -np.inf)
    previous_hi = np.full(free_budget+1, -np.inf)
    start_lo = interval_affine(np.array([.25]), r_lo[levels], b_lo[levels], False)[0]
    start_hi = interval_affine(np.array([.25]), r_hi[levels], b_hi[levels], True)[0]
    previous_lo[0], previous_hi[0] = start_lo, start_hi
    for step in range(free):
        current_lo = np.full_like(previous_lo, -np.inf)
        current_hi = np.full_like(previous_hi, -np.inf)
        lower = max(0, free_budget-(free-step-1)*levels)
        upper = min(free_budget, (step+1)*levels)
        for level in range(levels+1):
            start, stop = max(lower, level), upper+1
            if start >= stop:
                continue
            source = slice(start-level, stop-level)
            low_candidate = interval_affine(previous_lo[source], r_lo[level], b_lo[level], False)
            high_candidate = interval_affine(previous_hi[source], r_hi[level], b_hi[level], True)
            np.maximum(current_lo[start:stop], low_candidate, out=current_lo[start:stop])
            np.maximum(current_hi[start:stop], high_candidate, out=current_hi[start:stop])
        previous_lo, previous_hi = current_lo, current_hi
    grid_lo, grid_hi = float(previous_lo[free_budget]), float(previous_hi[free_budget])
    policy_lo, policy_hi = np.array([.25]), np.array([.25])
    for level in chosen_path:
        policy_lo = interval_affine(policy_lo, r_lo[level], b_lo[level], False)
        policy_hi = interval_affine(policy_hi, r_hi[level], b_hi[level], True)
    policy_lo, policy_hi = float(policy_lo[0]), float(policy_hi[0])
    assert policy_lo <= policy_hi and grid_lo <= grid_hi
    assert policy_lo <= grid_hi and grid_lo <= policy_hi+1e-12

    # M² is rational because the fixed first peak's squared alignment is rational.
    initial, peak, drift = Fraction(1, 4), Fraction(1, 5), Fraction(1, 20)
    squared_start = (initial+drift*peak)**2/(1+peak*peak)
    squared_M = squared_start+free*drift*drift
    upper_M = float(np.sqrt(float(squared_M)))
    while Fraction(upper_M)**2 < squared_M:
        upper_M = float(np.nextafter(upper_M, np.inf))
    assert Fraction(upper_M)**2 >= squared_M
    M = Fraction(upper_M)
    first_derivative_bound = max(drift, M*peak-drift)
    diagonal_bound = M+3*drift*peak
    offdiagonal_bound = peak*first_derivative_bound
    spacing = peak/levels
    exact_conservative_gap = spacing*spacing/8*(free*diagonal_bound+free*free*offdiagonal_bound)
    gap_upper = rational_float_bound(exact_conservative_gap, upper=True)
    continuous_upper = float(np.nextafter(grid_hi+gap_upper, np.inf))
    policy_gap = float(np.nextafter(continuous_upper-policy_lo, np.inf))
    loss_lower = float(np.nextafter(1-continuous_upper, -np.inf))
    loss_upper = float(np.nextafter(1-policy_lo, np.inf))
    return {"levels": levels, "exact_parameters": {"m0": "1/4", "E": "1/5", "a": "1/20"},
            "arithmetic": "rational coefficient checks; separately outward-rounded float64 operations",
            "grid_optimum_alignment_interval": [grid_lo, grid_hi],
            "chosen_policy_alignment_interval": [policy_lo, policy_hi],
            "analytical_gap_upper_bound": gap_upper,
            "continuous_optimum_alignment_upper_bound": continuous_upper,
            "chosen_policy_continuous_optimality_gap_upper_bound": policy_gap,
            "continuous_optimum_loss_interval": [loss_lower, loss_upper],
            "runtime_seconds": perf_counter()-started}


def build_and_verify():
    alpha = (TAIL-1)/8
    original_air = air_schedule(TAIL, PEAK, alpha)
    air_levels = np.rint(BASE_LEVELS*original_air/PEAK).astype(int)
    air_levels[0], air_levels[-1] = BASE_LEVELS, 0
    budget = int(air_levels.sum())
    physical_budget = PEAK*budget/BASE_LEVELS
    air_rates = PEAK*air_levels/BASE_LEVELS
    air_alignment = float(trajectory(TAIL_ALIGNMENT, air_rates, DRIFT)[-1])
    matched_alpha = brentq(lambda value: air_schedule(TAIL, PEAK, value).sum()-physical_budget,
                           1e-6, 1e5, xtol=1e-12)
    matched_air_rates = air_schedule(TAIL, PEAK, matched_alpha)
    matched_air_alignment = float(trajectory(TAIL_ALIGNMENT, matched_air_rates, DRIFT)[-1])
    np.testing.assert_allclose(matched_air_rates.sum(), physical_budget, rtol=0, atol=2e-13)
    warmup = np.linspace(0, PEAK, WARMUP)
    initial = TAIL_ALIGNMENT
    for rate in reversed(warmup):
        initial = initial*np.sqrt(1+rate*rate)-DRIFT*rate
    assert initial > 0
    assert TAIL_ALIGNMENT**2+TAIL*DRIFT**2 <= 1

    refinements, best_rates = [], None
    certificate = continuous_gap_constants()
    for levels in (64, 128, 256):
        started = perf_counter()
        scaled_budget = budget*(levels//BASE_LEVELS)
        path, alignment = grid_optimum(TAIL, levels, scaled_budget,
                                       TAIL_ALIGNMENT, PEAK, DRIFT)
        elapsed = perf_counter()-started
        rates = PEAK*path/levels
        best_rates = rates
        np.testing.assert_allclose(rates.sum(), physical_budget, rtol=0, atol=2e-14)
        assert alignment >= air_alignment-1e-13
        assert alignment > matched_air_alignment
        if refinements:
            assert alignment >= refinements[-1]["terminal_alignment"]-1e-13
        spacing = PEAK/levels
        free = certificate["free_updates"]
        gap = spacing*spacing/8*(free*certificate["diagonal_hessian_bound"]
                                +free*free*certificate["offdiagonal_hessian_bound"])
        safety = certificate["floating_point_safety_margin"]
        refinements.append({"levels": levels, "budget_units": scaled_budget,
                            "terminal_alignment": alignment, "terminal_loss": 1-alignment,
                            "alignment_gain_over_grid_air": alignment-air_alignment,
                            "alignment_gain_over_budget_matched_air": alignment-matched_air_alignment,
                            "grid_spacing": spacing, "continuous_alignment_gap_bound": gap,
                            "continuous_alignment_bracket": [alignment-safety, alignment+gap+safety],
                            "continuous_loss_bracket": [1-alignment-gap-safety, 1-alignment+safety],
                            "runtime_seconds": elapsed,
                            "tail_level_sequence": path.tolist()})

    rigorous_certificate = interval_certificate(256, budget*4,
                                                 np.array(refinements[-1]["tail_level_sequence"]))
    records = {}
    for name, tail in (("optimum", best_rates), ("air", matched_air_rates)):
        assert tail[0] == PEAK and tail[-1] == 0
        assert np.min(tail) >= 0 and np.max(tail) <= PEAK
        assert np.all(np.diff(tail) <= 1e-15)
        rates = np.concatenate((warmup, tail))
        states = trajectory(initial, rates, DRIFT)
        np.testing.assert_allclose(states[WARMUP], TAIL_ALIGNMENT, rtol=0, atol=1e-14)
        assert len(rates) == TOTAL and rates[-1] == 0
        assert np.max(states*states+DRIFT*DRIFT) < 1
        records[name] = {"rates": rates, "alignment": states, "loss": 1-states}
    np.testing.assert_allclose(records["optimum"]["rates"].sum(),
                               records["air"]["rates"].sum(), rtol=0, atol=2e-14)

    summary = {"status": "PASS", "model": "deterministic tangent alignment",
               "optimality_scope": "global on each finite rate grid; not a MuonH training optimum",
               "total_updates": TOTAL, "warmup_updates": WARMUP, "tail_updates": TAIL,
               "peak": PEAK, "constant_drift": DRIFT,
               "initial_alignment": float(initial), "post_warmup_alignment": TAIL_ALIGNMENT,
               "air_original_alpha": alpha, "air_budget_matched_alpha": float(matched_alpha),
               "base_grid_levels": BASE_LEVELS, "plotted_optimum_levels": 256,
               "tail_budget_units_on_base_grid": budget, "tail_elr_sum": physical_budget,
               "warmup_elr_sum": float(warmup.sum()),
               "full_elr_sum": float(physical_budget+warmup.sum()),
               "smooth_air_tail_elr_sum_before_rounding": float(original_air.sum()),
               "original_smooth_air_terminal_loss": float(1-trajectory(TAIL_ALIGNMENT, original_air, DRIFT)[-1]),
               "grid_air_terminal_alignment": air_alignment,
               "grid_air_terminal_loss": 1-air_alignment,
               "budget_matched_air_terminal_alignment": matched_air_alignment,
               "budget_matched_air_terminal_loss": 1-matched_air_alignment,
               "grid_air_tail_level_sequence": air_levels.tolist(),
               "continuous_gap_bound_constants": certificate,
               "outward_rounded_certificate": rigorous_certificate,
               "refinements": refinements, "exhaustive_check": exhaustive_check()}
    return records, summary


def figure(records, summary, output, mobile, preview_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(5.4, 9.5))
        fig.subplots_adjust(left=.19, right=.965, top=.775, bottom=.13, hspace=.43)
        title_y, subtitle_y, legend_y = .977, .925, .85
        subtitle = "Tangent alignment model · constant drift\n"+r"$a=0.05$ · $m_{\mathrm{tail}}=0.25$"
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
        fig.subplots_adjust(left=.09, right=.975, top=.69, bottom=.19, wspace=.28)
        title_y, subtitle_y, legend_y = .975, .875, .80
        subtitle = r"Tangent alignment model · constant drift $a=0.05$ · $m_{\mathrm{tail}}=0.25$"
    fig.suptitle("AIR under a fixed ELR budget", y=title_y, fontsize=16 if mobile else 17)
    fig.text(.5, subtitle_y, subtitle, ha="center", va="top", color=MUTED,
             fontsize=11, linespacing=1.6)
    ax_rate, ax_loss = axes
    x = np.arange(TOTAL+1)
    for name, record in records.items():
        ax_rate.plot(x, np.append(record["rates"], 0)/PEAK, **STYLES[name])
        ax_loss.plot(x, record["loss"], **STYLES[name])
    for ax in axes:
        ax.axvspan(0, WARMUP, color=WARMUP_FILL, zorder=0)
        ax.axvline(WARMUP, color=WARMUP_EDGE, lw=.85, ls=(0, (3, 3)), zorder=1)
        ax.set_xlim(0, TOTAL)
        ax.set_xticks([0, 40, 80, 120, 160])
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
    low = min(record["loss"].min() for record in records.values())
    high = max(record["loss"].max() for record in records.values())
    ax_loss.set_ylim(low-.06*(high-low), high+.28*(high-low))
    ax_loss.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_rate.annotate("Warmup", xy=(WARMUP, 1), xytext=(.18*TOTAL, 1.09), fontsize=9,
                     color=MUTED, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .7})
    inset = ax_loss.inset_axes([.45, .58, .52, .34])
    for name, record in records.items():
        inset.plot(x, record["loss"], **{**STYLES[name], "lw": 1.5})
    late = x >= TOTAL//2
    late_low = min(record["loss"][late].min() for record in records.values())
    late_high = max(record["loss"][late].max() for record in records.values())
    inset.set_xlim(TOTAL//2, TOTAL)
    inset.set_ylim(late_low-.06*(late_high-late_low), late_high+.06*(late_high-late_low))
    inset.set_xticks([80, 120, 160])
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
    inset.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    inset.tick_params(labelsize=8, length=2)
    inset.set_title("Late loss (linear zoom)", fontsize=9, pad=5)
    inset.grid(axis="y", color=GRID, linewidth=.5)
    inset.set_axisbelow(True)
    inset.spines[["top", "right"]].set_visible(False)
    handles, labels = ax_rate.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.52, legend_y),
               ncol=1 if mobile else 2, frameon=False, fontsize=10.5,
               handlelength=2.6, columnspacing=1.7, labelspacing=.9)
    budget = summary["full_elr_sum"]
    footer = (f"Same 5% warmup, peak, final zero and total ELR ({budget:g}).\n"
              "Budget-matched AIR; grid DP for the alignment model.")
    fig.text(.5, .04, footer, ha="center", color=MUTED, fontsize=10.5, linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None,
        "Title": "AIR and dynamic programming under a fixed ELR budget",
        "Description": "Conditional deterministic tangent-alignment model, not actual MuonH training. "
        "Both schedules have the same prescribed 8-update warmup, peak 0.2, zero final applied rate, "
        "and cumulative ELR. The common budget is chosen from AIR rounded once to a 64-level grid. "
        "The displayed continuous AIR calibrates alpha to match that exact budget; dynamic programming "
        "computes a schedule on the nested grid with 256 intervals (257 allowed rates). All axes are linear."})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines())+"\n")
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_dir/f"{output.stem}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEWS)
    parser.add_argument("--report", type=Path, default=Path("/private/tmp/air-budget-check.json"))
    args = parser.parse_args()
    records, summary = build_and_verify()
    configure("air-fixed-budget-grid-model-20261102")
    for mobile in (False, True):
        variant = "mobile" if mobile else "desktop"
        figure(records, summary, STATIC/"images/lr-schedules"/f"air-budget-{variant}.svg",
               mobile, args.preview_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2)+"\n")
    compact = {k: v for k, v in summary.items()
               if k not in ("refinements", "grid_air_tail_level_sequence")}
    compact["refinements"] = [{k: v for k, v in row.items() if k != "tail_level_sequence"}
                              for row in summary["refinements"]]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
