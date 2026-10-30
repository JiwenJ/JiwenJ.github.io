#!/usr/bin/env python3
"""Verify matrix identities and a frozen-profile LR bound used for Muon.

Dependency: NumPy only (tested with 2.0.2). From the repository root:

    python3 static/code/muon-schedule-check.py

Optional report location:

    python3 static/code/muon-schedule-check.py --report /private/tmp/muon-verification.json

For each supplied matrix G this reference uses a reduced SVD, with singular
values above 1e-12 * sigma_max retained consistently. It forms

    H = (G G^T)^(1/2), H_pinv = H^dagger, U = H_pinv G,

and verifies the nuclear energy, partial-isometry norm, range reconstruction,
and the variable-metric identity for Z_next = Z - w U. Tests include tall,
wide, square, rank-deficient, and zero matrices. The H constructed here uses
the retained singular values; the difference from the untruncated square
root is also checked at the stated numerical tolerance.

The temporal test verifies a telescoping upper bound with
    R^2 = max_{t=1,...,T} ||E_t||_F^2,
    Lambda = lambda_max(H_1)
             + sum_{t=2,...,T} max(lambda_max(H_t-H_{t-1}), 0).
The terminal E_{T+1} is excluded from the definition of R.

Separately, for fixed A > 0 and a_t > 0 it verifies the exact minimizer of
    B(w) = (A + sum_t a_t w_t^2) / (2 sum_t w_t),
namely w_t = sqrt(A)/(a_t sqrt(Q)), Q = sum_t 1/a_t. Equal a_t gives
eta_t = w_t * sum_{s=t+1}^T w_s / sum_{s=1}^T w_s = w_1*(1-t/T).
These schedule indices are t=1,...,T; eta_T=0 is the formal endpoint.

An additional fixed-coefficient surrogate includes a nonnegative linear term:
    B_b(w) = (A + sum_t a_t w_t^2 + 2 sum_t b_t w_t) / (2 sum_t w_t).
On the positive branch lambda > max_t b_t, its candidate is
    w_t = (lambda-b_t)/a_t,
    sum_t (lambda-b_t)^2/a_t = A.
The checks construct lambda=max(b)+1, choose A from this equation, and verify
the optimum and completed-square gap. The coefficients b are supplied fixed
numbers, not momentum measured from an optimizer or a training experiment.

This is deterministic FP64 algebraic QA. It does not implement momentum,
Newton-Schulz approximations, shape scaling, weight decay, or neural-network
training. It does not prove that actual Muon gradient statistics stay fixed
when a schedule is changed, or that the bound is globally optimal for any
particular training problem. No existing figures are generated or modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


STATIC = Path(__file__).resolve().parents[1]
SEED = 20261031
RANK_RTOL = 1e-12
CHECK_ATOL = 2e-12
CHECK_RTOL = 2e-11
SHAPES = ((3, 5), (5, 3), (4, 4), (1, 7), (7, 1))


def require_close(actual, expected, label: str, errors: dict) -> None:
    """Record elementwise absolute and scaled errors, then enforce tolerance."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    delta = float(np.max(np.abs(actual - expected), initial=0.0))
    scale = float(np.max(np.abs(expected), initial=0.0))
    item = errors.setdefault(label, {"max_absolute_error": 0.0, "max_scaled_error": 0.0})
    item["max_absolute_error"] = max(item["max_absolute_error"], delta)
    item["max_scaled_error"] = max(item["max_scaled_error"], delta / max(1.0, scale))
    if delta > CHECK_ATOL + CHECK_RTOL * scale:
        raise AssertionError(f"{label}: absolute error {delta:g}, reference scale {scale:g}")


def make_matrix(rng: np.random.Generator, shape: tuple[int, int], rank: int, scale: float = 1.0):
    m, n = shape
    if rank == 0:
        return np.zeros(shape, dtype=np.float64)
    left, _ = np.linalg.qr(rng.normal(size=(m, rank)), mode="reduced")
    right, _ = np.linalg.qr(rng.normal(size=(n, rank)), mode="reduced")
    singular = scale * np.exp(rng.uniform(-1.0, 1.0, size=rank))
    return (left * singular) @ right.T


def polar_metric(g: np.ndarray) -> dict:
    """Use one SVD and one rank mask for H, H^dagger, and the polar factor."""
    g = np.asarray(g, dtype=np.float64)
    if g.ndim != 2 or not g.size or not np.all(np.isfinite(g)):
        raise ValueError("G must be a nonempty finite matrix.")
    left, singular, right_t = np.linalg.svd(g, full_matrices=False)
    threshold = RANK_RTOL * float(singular[0])
    keep = singular > threshold
    active_left, active_right = left[:, keep], right_t[keep, :]
    active_singular = singular[keep]
    h = (active_left * active_singular) @ active_left.T
    h_pinv = (active_left / active_singular) @ active_left.T
    u = active_left @ active_right
    return {"h": h, "h_pinv": h_pinv, "u": u, "rank": int(np.count_nonzero(keep)),
            "nuclear": float(np.sum(active_singular)),
            "all_singular": singular, "left": left,
            "threshold": threshold}


def inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))


def metric_square(e: np.ndarray, h: np.ndarray) -> float:
    return inner(e, h @ e)


def verify_matrix_identities(rng: np.random.Generator) -> dict:
    errors, matrices, identities = {}, 0, 0
    ranks_seen = {"zero": 0, "rank_deficient_nonzero": 0, "full_rank": 0}
    for shape in SHAPES:
        k = min(shape)
        for rank in sorted({0, 1, max(0, k - 1), k}):
            for scale in (0.2, 1.0, 3.0):
                for _ in range(5):
                    g = make_matrix(rng, shape, rank, scale)
                    result = polar_metric(g)
                    h, h_pinv, u = result["h"], result["h_pinv"], result["u"]
                    assert result["rank"] == rank
                    category = "zero" if rank == 0 else ("full_rank" if rank == k else "rank_deficient_nonzero")
                    ranks_seen[category] += 1
                    full_root = (result["left"] * result["all_singular"]) @ result["left"].T
                    require_close(h, full_root, "retained_vs_untruncated_root", errors)
                    require_close(h @ h, g @ g.T, "square_root_identity", errors)
                    require_close(h_pinv @ g, u, "pseudoinverse_vs_polar_update", errors)
                    require_close(h @ h_pinv @ g, g, "range_reconstruction", errors)
                    require_close(h @ u, g, "metric_times_update", errors)
                    require_close(inner(g, h_pinv @ g), result["nuclear"], "nuclear_energy", errors)
                    require_close(result["nuclear"], np.sum(result["all_singular"]),
                                  "retained_vs_full_nuclear_norm", errors)
                    require_close(inner(u, u), rank, "polar_frobenius_square_equals_rank", errors)
                    require_close(metric_square(u, h), result["nuclear"], "update_metric_energy", errors)
                    for w in (0.0, 0.15, 1.3):
                        e = rng.normal(size=shape)
                        e_next = e - w * u
                        lhs = 2.0 * w * inner(g, e)
                        rhs = (metric_square(e, h) - metric_square(e_next, h)
                               + w**2 * result["nuclear"])
                        require_close(lhs, rhs, "variable_metric_one_step_identity", errors)
                        identities += 1
                    matrices += 1
    return {"matrices": matrices, "one_step_identities": identities,
            "shapes": [list(s) for s in SHAPES], "ranks_seen": ranks_seen, "errors": errors}


def verify_temporal_bound(rng: np.random.Generator) -> dict:
    errors, sequences, indefinite_changes, decreasing_changes = {}, 0, 0, 0
    minimum_slack, maximum_violation = float("inf"), 0.0
    for shape in SHAPES:
        k = min(shape)
        for horizon in (5, 12, 29):
            for trial in range(6):
                # Repeated zero/full/partial ranks force changes of support;
                # each temporal sequence nevertheless keeps one matrix shape.
                pattern = (0, k, 1, k, 0, max(1, k - 1))
                updates = [polar_metric(make_matrix(rng, shape, pattern[(t + trial) % len(pattern)],
                                                   float(rng.uniform(0.25, 2.0))))
                           for t in range(horizon)]
                weights = rng.uniform(0.0, 1.0, size=horizon) / np.sqrt(horizon)
                es = [rng.normal(size=shape)]
                potential_sum = 0.0
                identity_lhs, identity_energy = 0.0, 0.0
                for result, weight in zip(updates, weights):
                    e_next = es[-1] - weight * result["u"]
                    potential_sum += metric_square(es[-1], result["h"]) - metric_square(e_next, result["h"])
                    # G = H U, reconstructed using the exact same rank policy.
                    identity_lhs += 2.0 * weight * inner(result["h"] @ result["u"], es[-1])
                    identity_energy += weight**2 * result["nuclear"]
                    es.append(e_next)
                first_h, last_h = updates[0]["h"], updates[-1]["h"]
                telescoped = metric_square(es[0], first_h) - metric_square(es[-1], last_h)
                variation = max(0.0, float(np.linalg.eigvalsh(first_h)[-1]))
                for t in range(1, horizon):
                    difference = updates[t]["h"] - updates[t - 1]["h"]
                    eigenvalues = np.linalg.eigvalsh(difference)
                    largest, smallest = float(eigenvalues[-1]), float(eigenvalues[0])
                    variation += max(0.0, largest)
                    telescoped += metric_square(es[t], difference)
                    indefinite_changes += int(smallest < -1e-10 and largest > 1e-10)
                    decreasing_changes += int(smallest < -1e-10 and largest <= 1e-10)
                r_squared = max(inner(e, e) for e in es[:-1])
                upper = r_squared * variation
                slack = upper - potential_sum
                minimum_slack = min(minimum_slack, slack)
                maximum_violation = max(maximum_violation, -slack)
                assert slack >= -CHECK_ATOL - CHECK_RTOL * abs(upper)
                require_close(potential_sum, telescoped, "telescoping_identity", errors)
                require_close(identity_lhs, potential_sum + identity_energy,
                              "summed_variable_metric_identity", errors)
                sequences += 1
    assert indefinite_changes > 0 and decreasing_changes > 0
    return {"sequences": sequences, "horizons": [5, 12, 29],
            "bound": "sum_t(||E_t||^2_Ht-||E_(t+1)||^2_Ht) <= R^2 * Lambda",
            "R_definition": "max pre-update Frobenius norm; terminal E_(T+1) excluded",
            "Lambda_definition": "lambda_max(H_1) + sum_(t=2)^T max(lambda_max(H_t-H_(t-1)),0)",
            "indefinite_metric_changes": indefinite_changes,
            "decreasing_metric_changes": decreasing_changes,
            "minimum_bound_slack": minimum_slack, "maximum_bound_violation": maximum_violation,
            "errors": errors}


def objective(weights: np.ndarray, energy: np.ndarray, a_constant: float):
    weights = np.asarray(weights, dtype=np.float64)
    total = np.sum(weights, axis=-1)
    return (a_constant + np.sum(energy * weights**2, axis=-1)) / (2.0 * total)


def schedule_from_weights(weights: np.ndarray) -> np.ndarray:
    future = np.zeros_like(weights)
    future[:-1] = np.cumsum(weights[:0:-1])[::-1]
    return weights * future / np.sum(weights)


def verify_frozen_profiles(rng: np.random.Generator) -> dict:
    errors, cases, perturbations = {}, [], 0
    smallest_gap = float("inf")
    for horizon in (3, 17, 100):
        coordinate = np.linspace(0.0, 1.0, horizon)
        profiles = {"constant": np.full(horizon, 2.0),
                    "decreasing": 0.3 + 2.2 * (1.0 - coordinate),
                    "middle_peak": 0.5 + 1.8 * np.exp(-((coordinate - 0.5) / 0.16)**2),
                    "alternating": 0.3 + 2.0 * (np.arange(horizon) % 2)}
        for name, energy in profiles.items():
            for a_constant in (0.05, 1.0, 9.0):
                q = float(np.sum(1.0 / energy))
                weights = np.sqrt(a_constant) / (energy * np.sqrt(q))
                best_value = np.sqrt(a_constant / q)
                total = float(np.sum(weights))
                numerator = a_constant + float(np.sum(energy * weights**2))
                gradient = (2.0 * energy * weights * total - numerator) / (2.0 * total**2)
                require_close(objective(weights, energy, a_constant), best_value,
                              "candidate_value_equals_closed_form", errors)
                require_close(gradient, np.zeros_like(gradient), "candidate_gradient_is_zero", errors)
                # Positive multiplicative perturbations preserve the domain.
                sampled_weights = weights[None, :] * np.exp(rng.normal(0.0, 0.7, size=(512, horizon)))
                values = objective(sampled_weights, energy, a_constant)
                gaps = values - best_value
                square_gaps = (np.sum(energy * (sampled_weights - weights)**2, axis=-1)
                               / (2.0 * np.sum(sampled_weights, axis=-1)))
                require_close(gaps, square_gaps, "completed_square_gap_identity", errors)
                assert float(np.min(gaps)) >= -CHECK_ATOL
                smallest_gap = min(smallest_gap, float(np.min(gaps)))
                perturbations += len(sampled_weights)
                schedule = schedule_from_weights(weights)
                assert schedule[-1] == 0.0 and np.all(schedule >= 0.0)
                if name == "constant":
                    linear = weights[0] * (1.0 - np.arange(1, horizon + 1) / horizon)
                    require_close(schedule, linear, "constant_profile_gives_linear_schedule", errors)
                cases.append({"profile": name, "horizon": horizon, "A": a_constant,
                              "Q": q, "minimum_bound": best_value,
                              "max_abs_gradient": float(np.max(np.abs(gradient))),
                              "smallest_sampled_perturbation_gap": float(np.min(gaps))})
    return {"cases": len(cases), "positive_perturbations": perturbations,
            "minimum_perturbation_gap": smallest_gap,
            "assumptions": "A > 0 and all a_t > 0 fixed independently of the candidate weights",
            "candidate": "w_t=sqrt(A)/(a_t*sqrt(Q)), Q=sum_t(1/a_t)",
            "optimal_value": "sqrt(A/Q)",
            "global_gap_identity": "B(w)-sqrt(A/Q)=sum_t a_t*(w_t-w_t_star)^2/(2*sum_t w_t)",
            "schedule_indices": "t=1,...,T; eta_T=0 is the formal endpoint",
            "errors": errors, "case_results": cases}


def verify_fixed_momentum_surrogate(rng: np.random.Generator) -> dict:
    """Check only the all-positive branch with a fixed nonnegative b profile."""
    errors, cases, perturbations = {}, [], 0
    smallest_gap = float("inf")
    for horizon in (3, 17, 100):
        for trial in range(6):
            energy = np.exp(rng.normal(0.0, 0.6, size=horizon))
            b = rng.uniform(0.0, 2.0, size=horizon)
            if trial == 0:
                b = np.zeros(horizon)
            elif trial == 1:
                b = np.full(horizon, 0.7)
            optimum = float(np.max(b)) + 1.0
            a_constant = float(np.sum((optimum - b)**2 / energy))
            weights = (optimum - b) / energy
            assert np.all(energy > 0.0) and np.all(b >= 0.0)
            assert np.all(weights > 0.0) and optimum > float(np.max(b))
            total = float(np.sum(weights))
            numerator = a_constant + float(np.sum(energy * weights**2 + 2.0 * b * weights))
            gradient = ((2.0 * energy * weights + 2.0 * b) * total - numerator) / (2.0 * total**2)
            require_close(numerator / (2.0 * total), optimum,
                          "candidate_value_equals_lambda", errors)
            require_close(gradient, np.zeros_like(gradient), "candidate_gradient_is_zero", errors)
            # Independent scalar quadratic solution for the positive branch.
            q = float(np.sum(1.0 / energy))
            p = float(np.sum(b / energy))
            c = float(np.sum(b**2 / energy))
            root = (p + np.sqrt(p**2 + q * (a_constant - c))) / q
            require_close(root, optimum, "scalar_constraint_positive_root", errors)
            sampled_weights = weights[None, :] * np.exp(rng.normal(0.0, 0.7, size=(512, horizon)))
            sums = np.sum(sampled_weights, axis=-1)
            values = ((a_constant + np.sum(energy * sampled_weights**2 + 2.0 * b * sampled_weights, axis=-1))
                      / (2.0 * sums))
            gaps = values - optimum
            squared_gaps = np.sum(energy * (sampled_weights - weights)**2, axis=-1)
            require_close(2.0 * sums * gaps, squared_gaps,
                          "weighted_completed_square_gap_identity", errors)
            assert float(np.min(gaps)) >= -CHECK_ATOL
            smallest_gap = min(smallest_gap, float(np.min(gaps)))
            perturbations += len(sampled_weights)
            cases.append({"horizon": horizon, "b_profile": "zero" if trial == 0 else ("constant" if trial == 1 else "random_nonnegative"),
                          "A": a_constant, "lambda": optimum, "maximum_b": float(np.max(b)),
                          "positive_branch_margin": optimum - float(np.max(b)),
                          "max_abs_gradient": float(np.max(np.abs(gradient))),
                          "smallest_sampled_perturbation_gap": float(np.min(gaps))})
    return {"cases": len(cases), "positive_perturbations": perturbations,
            "minimum_perturbation_gap": smallest_gap,
            "assumptions": "A > 0, a_t > 0, b_t >= 0 all fixed; lambda > max_t b_t",
            "objective": "(A+sum_t a_t*w_t^2+2*sum_t b_t*w_t)/(2*sum_t w_t)",
            "scalar_constraint": "sum_t (lambda-b_t)^2/a_t=A",
            "candidate": "w_t_star=(lambda-b_t)/a_t",
            "optimal_value": "lambda",
            "global_gap_identity": "2*sum_t w_t*(B_b(w)-lambda)=sum_t a_t*(w_t-w_t_star)^2",
            "scope": "Fixed-coefficient surrogate; b is not a measured optimizer momentum trajectory",
            "errors": errors, "case_results": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=STATIC / "images/lr-schedules/muon-verification.json")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    report = {"status": "PASS", "seed": SEED,
              "scope": "FP64 matrix identities and exact optimization of a frozen-profile upper bound",
              "excluded_claim": "No neural-network experiment or global optimality claim for actual Muon training",
              "numerical_rank_policy": {"relative_cutoff": RANK_RTOL,
                                        "criterion": "retain sigma > relative_cutoff * sigma_max; rank(0)=0",
                                        "shared_by": ["H", "H_pseudoinverse", "polar_update"]},
              "tolerances": {"absolute": CHECK_ATOL, "relative": CHECK_RTOL},
              "matrix_identities": verify_matrix_identities(rng),
              "temporal_bound": verify_temporal_bound(rng),
              "frozen_profile_bound": verify_frozen_profiles(rng),
              "fixed_momentum_surrogate": verify_fixed_momentum_surrogate(rng)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report),
                      "matrices": report["matrix_identities"]["matrices"],
                      "one_step_identities": report["matrix_identities"]["one_step_identities"],
                      "temporal_sequences": report["temporal_bound"]["sequences"],
                      "frozen_profile_cases": report["frozen_profile_bound"]["cases"],
                      "positive_perturbations": report["frozen_profile_bound"]["positive_perturbations"],
                      "fixed_momentum_cases": report["fixed_momentum_surrogate"]["cases"],
                      "fixed_momentum_positive_perturbations": report["fixed_momentum_surrogate"]["positive_perturbations"]},
                     indent=2))


if __name__ == "__main__":
    main()
