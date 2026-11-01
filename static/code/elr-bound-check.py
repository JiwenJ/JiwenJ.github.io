#!/usr/bin/env python3
"""Check the algebra of directly controlled relative-update ELR.

Dependency: NumPy. This checks identities and optimization of a supplied
deterministic bound; it does not establish uniform bounds for neural training
or certify a globally optimal training schedule.

For nonzero W and U, write r=||W||_F, q=||U||_F, D=(r/q)U,
K=(q/r)H and Delta=G-KD. Thus a requested relative step e uses the external
learning rate eta=e*r/q, and ||eta*U||_F/||W||_F=e. This is norm feedback,
not an assertion that an open-loop external LR has deterministic ELR.

Muon checks use ordinary/Nesterov momentum, a retained-rank partial polar
factor, and the five-step polynomial Newton--Schulz map in FP64. The latter
is an algebraic reference, not a bitwise reproduction of a BF16 GPU kernel.
No weight decay is included. Zero parameter/update norms are excluded from
the normalization and explicitly checked as error cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEED = 20261101
ATOL, RTOL = 2e-10, 2e-10


def inner(a, b):
    return float(np.sum(a * b))


def check(actual, expected, name, errors):
    actual, expected = np.asarray(actual), np.asarray(expected)
    absolute = float(np.max(np.abs(actual - expected), initial=0.0))
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    result = errors.setdefault(name, {"max_absolute_error": 0.0,
                                      "max_scaled_error": 0.0})
    result["max_absolute_error"] = max(result["max_absolute_error"], absolute)
    result["max_scaled_error"] = max(result["max_scaled_error"], absolute / scale)
    if absolute > ATOL + RTOL * scale:
        raise AssertionError(f"{name}: error={absolute:g}, scale={scale:g}")


def normalized_update(w, u, apply_h):
    r, q = float(np.linalg.norm(w)), float(np.linalg.norm(u))
    if r <= 0.0 or q <= 0.0:
        raise ValueError("Positive parameter and update norms are required.")
    direction = (r / q) * u

    def apply_k(x):
        return (q / r) * apply_h(x)

    return direction, apply_k


def verify_identity(w, g, u, apply_h, rng, errors, label):
    direction, apply_k = normalized_update(w, u, apply_h)
    residual = g - apply_k(direction)
    energy = inner(direction, apply_k(direction))
    check(np.linalg.norm(direction), np.linalg.norm(w), label + "/relative_norm", errors)
    check(energy, np.linalg.norm(w) / np.linalg.norm(u) * inner(u, apply_h(u)),
          label + "/energy_rescaling", errors)
    e = rng.normal(size=w.shape)
    weight = float(rng.uniform(0.01, 1.5))
    next_e = e - weight * direction
    lhs = 2 * weight * inner(g, e)
    rhs = (inner(e, apply_k(e)) - inner(next_e, apply_k(next_e))
           + weight**2 * energy + 2 * weight * inner(residual, e))
    check(lhs, rhs, label + "/distance_identity", errors)
    elr = float(rng.uniform(0.001, 0.5))
    eta = elr * np.linalg.norm(w) / np.linalg.norm(u)
    check(eta * u, elr * direction, label + "/controlled_update", errors)
    check(np.linalg.norm(eta * u) / np.linalg.norm(w), elr,
          label + "/realized_elr", errors)
    return direction, apply_k, energy, residual


def partial_polar_and_root(m):
    left, singular, right = np.linalg.svd(m, full_matrices=False)
    keep = singular > singular[0] * 1e-12
    polar = left[:, keep] @ right[keep]
    root = (left * singular) @ left.T
    return polar, root, int(np.count_nonzero(keep)), float(np.sum(singular))


def ns5(m):
    x = m / (np.linalg.norm(m) + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(5):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    return x.T if transposed else x


def optimizer_checks(rng):
    errors, counts = {}, {"sgd": 0, "adafactor": 0, "adam": 0, "muon": 0}
    for shape in ((3, 7), (7, 3), (4, 4)):
        for _ in range(12):
            w = rng.normal(size=shape)
            g = rng.normal(size=shape)
            _, _, energy, residual = verify_identity(
                w, g, g, lambda x: x, rng, errors, "sgd")
            check(energy, np.linalg.norm(w) * np.linalg.norm(g), "sgd/energy", errors)
            check(residual, np.zeros_like(g), "sgd/no_residual", errors)
            counts["sgd"] += 1

            # Actual row/column second-moment accumulation and reconstruction.
            rows, columns = np.zeros((shape[0], 1)), np.zeros((1, shape[1]))
            for t in range(1, 8):
                g = rng.normal(size=shape)
                squared = g*g + 1e-8
                beta = 1 - t**(-0.8)
                rows = beta * rows + (1-beta) * squared.sum(axis=1, keepdims=True)
                columns = beta * columns + (1-beta) * squared.sum(axis=0, keepdims=True)
            v = rows * columns / rows.sum()
            h = np.sqrt(v)
            u = g / h
            direction, apply_k, energy, residual = verify_identity(
                w, g, u, lambda x: h*x, rng, errors, "adafactor")
            check(energy, np.linalg.norm(w) / np.linalg.norm(u) * np.sum(g*g/h),
                  "adafactor/energy", errors)
            check(residual, np.zeros_like(g), "adafactor/no_residual", errors)
            parameter_scale = max(1e-3, float(np.sqrt(np.mean(w*w))))
            for clip_threshold in (0.05, 1.0, 5.0):
                clip = max(1.0, float(np.sqrt(np.mean(u*u))) / clip_threshold)
                scaled_u = parameter_scale * u / clip
                scaled_h = h * clip / parameter_scale
                actual_d, actual_k = normalized_update(w, scaled_u, lambda x: scaled_h*x)
                check(actual_d, direction, "adafactor/scalar_cancellation_direction", errors)
                check(actual_k(g), apply_k(g), "adafactor/scalar_cancellation_metric", errors)
                rho = 0.02
                realized = np.linalg.norm(rho*scaled_u) / np.linalg.norm(w)
                expected = rho*parameter_scale*np.sqrt(np.mean((u/clip)**2))/np.sqrt(np.mean(w*w))
                check(realized, expected, "adafactor/relative_step_to_elr", errors)
            counts["adafactor"] += 1

            m, v = np.zeros(shape), np.zeros(shape)
            for t in range(1, 12):
                g = rng.normal(size=shape)
                m, v = 0.9*m+0.1*g, 0.99*v+0.01*g*g
            m_hat, v_hat = m/(1-0.9**t), v/(1-0.99**t)
            h = np.sqrt(v_hat) + 1e-8
            u = m_hat / h
            _, _, energy, residual = verify_identity(w, g, u, lambda x: h*x, rng, errors, "adam")
            check(energy, np.linalg.norm(w) / np.linalg.norm(u) * np.sum(m_hat*m_hat/h),
                  "adam/energy", errors)
            check(residual, g-m_hat, "adam/momentum_residual", errors)
            counts["adam"] += 1

        # Retain momentum state across several gradients, including rank-deficient histories.
        for rank in (1, min(shape)):
            left = rng.normal(size=(shape[0], rank))
            right = rng.normal(size=(rank, shape[1]))
            for beta in (0.0, 0.9, 0.95):
                buffer = np.zeros(shape)
                for _ in range(9):
                    g = left @ (rng.normal(size=(rank, rank)) @ right)
                    buffer = beta*buffer + (1-beta)*g
                for nesterov in (False, True):
                    m = (1-beta)*g + beta*buffer if nesterov else buffer
                    polar, root, numerical_rank, nuclear = partial_polar_and_root(m)
                    w = rng.normal(size=shape)
                    for name, phi in (("polar", polar), ("ns5", ns5(m))):
                        previous_d, previous_kg = None, None
                        for scale in (0.1, 1.0, 3.0):
                            u = scale*phi
                            apply_h = lambda x: root @ x / scale
                            direction, apply_k, energy, residual = verify_identity(
                                w, g, u, apply_h, rng, errors, "muon/"+name)
                            check(residual, g-root@phi, "muon/"+name+"/residual", errors)
                            if name == "polar":
                                check(np.linalg.norm(polar)**2, numerical_rank, "muon/polar/rank_norm", errors)
                                check(energy, np.linalg.norm(w)*nuclear/np.sqrt(numerical_rank),
                                      "muon/polar/energy", errors)
                                check(residual, g-m, "muon/polar/momentum_residual", errors)
                            if previous_d is not None:
                                check(direction, previous_d, "muon/"+name+"/scale_cancellation_direction", errors)
                                check(apply_k(g), previous_kg, "muon/"+name+"/scale_cancellation_metric", errors)
                            previous_d, previous_kg = direction, apply_k(g)
                            counts["muon"] += 1

    excluded = 0
    for w, u in ((np.zeros((2, 2)), np.ones((2, 2))),
                 (np.ones((2, 2)), np.zeros((2, 2)))):
        try:
            normalized_update(w, u, lambda x: x)
        except ValueError:
            excluded += 1
    assert excluded == 2
    return {"cases": counts, "zero_norm_cases_rejected": excluded, "errors": errors}


def solve_bound(a, b, initial_bound):
    left, right = float(np.min(b)), float(np.max(b)+np.sqrt(initial_bound*np.max(a)))
    for _ in range(100):
        lam = (left+right)/2
        if np.sum(np.maximum(lam-b, 0)**2/a) < initial_bound:
            left = lam
        else:
            right = lam
    lam = (left+right)/2
    return lam, np.maximum(lam-b, 0)/a


def objective(w, a, b, initial_bound):
    return (initial_bound + np.sum(a*w*w) + 2*np.sum(b*w))/(2*np.sum(w))


def bound_checks(rng):
    errors, cases, comparisons, inactive_cases = {}, 0, 0, 0
    for n in (3, 10, 95):
        for _ in range(20):
            a = np.exp(rng.uniform(-2, 2, size=n))
            b = rng.uniform(0, 2, size=n)
            for initial_bound in (0.001, 0.2, 10.0, 100.0):
                lam, w_star = solve_bound(a, b, initial_bound)
                active = w_star > 0
                inactive_cases += int(not np.all(active))
                value = objective(w_star, a, b, initial_bound)
                check(np.sum(np.maximum(lam-b, 0)**2/a), initial_bound, "kkt/root", errors)
                check(value, lam, "kkt/objective_value", errors)
                derivative_numerator = a*w_star+b-value
                check(derivative_numerator[active], np.zeros(active.sum()), "kkt/active_gradient", errors)
                assert np.all(derivative_numerator[~active] >= -1e-12)
                for _ in range(12):
                    candidate = np.exp(rng.uniform(-4, 2, size=n))
                    if rng.random() < .5:
                        candidate[rng.random(n) < .3] = 0
                    if candidate.sum() == 0:
                        candidate[0] = 1
                    actual = 2*candidate.sum()*(objective(candidate, a, b, initial_bound)-lam)
                    expected = np.sum(a*(candidate-w_star)**2) + 2*np.sum((b[~active]-lam)*candidate[~active])
                    check(actual, expected, "kkt/completed_square_with_inactive_term", errors)
                    assert actual >= -1e-10
                    comparisons += 1
                cases += 1
    for n in (5, 95, 100):
        for constant_a, constant_b, initial_bound in ((1.0, 0.0, 1.0), (2.5, .4, 3.0)):
            a, b = np.full(n, constant_a), np.full(n, constant_b)
            lam, w = solve_bound(a, b, initial_bound)
            closed_w = np.sqrt(initial_bound/(constant_a*n))
            check(w, np.full(n, closed_w), "constant/weights", errors)
            remaining = np.cumsum(w[::-1])[::-1]-w
            elr = w*remaining/w.sum()
            check(elr, closed_w*(1-np.arange(1, n+1)/n), "constant/linear_elr", errors)
            check(lam, constant_b+np.sqrt(initial_bound*constant_a/n), "constant/bound_value", errors)
    return {"profile_cases": cases, "cases_with_inactive_weights": inactive_cases,
            "nonnegative_candidate_comparisons": comparisons, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("/private/tmp/elr-bound-verification.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    report = {"status": "PASS", "seed": SEED,
              "scope": "Normalized-update identities and deterministic bound minimization only; no neural-training optimality claim",
              "optimizers": optimizer_checks(rng), "bound": bound_checks(rng)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    all_errors = list(report["optimizers"]["errors"].values()) + list(report["bound"]["errors"].values())
    print(json.dumps({"status": report["status"], "optimizer_cases": report["optimizers"]["cases"],
                      "bound_profiles": report["bound"]["profile_cases"],
                      "inactive_weight_profiles": report["bound"]["cases_with_inactive_weights"],
                      "max_scaled_error": max(e["max_scaled_error"] for e in all_errors),
                      "report": str(args.report)}))


if __name__ == "__main__":
    main()
