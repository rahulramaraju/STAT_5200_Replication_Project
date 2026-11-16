#!/usr/bin/env python3
"""Solve the nonlinear debt decision rule on a six-dimensional state grid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from posterior_reader import read_parameters


@dataclass(frozen=True)
class ModelParameters:
    h: float
    s_tilde: float
    gamma_tau_L: float
    gamma_g_L: float
    rho_A: float
    rho_g: float
    rho_z: float
    rho_tau: float
    sigma_A_p: float
    sigma_g_p: float
    sigma_z_p: float
    sigma_tau_p: float
    delta: float


@dataclass(frozen=True)
class Calibration:
    beta: float
    A_bar: float
    n_bar: float
    y_bar_quarterly: float
    g_bar: float
    b_bar: float
    tau_bar: float
    z_bar: float
    c_bar: float
    phi: float
    gamma_tau: float
    gamma_g: float
    sigma_A: float
    sigma_g: float
    sigma_z: float
    sigma_tau: float
    eta_1: float
    eta_2: float


STATE_NAMES = ("b_d", "c_lag", "A", "u_g", "z", "u_tau")


def load_model_parameters(case: str) -> ModelParameters:
    """Load the selected posterior medians and the paper's default calibration."""
    values = read_parameters(case)

    h, s_tilde, gamma_tau_L, gamma_g_L, rho_A, rho_g, rho_z, rho_tau, \
        sigma_A_p, sigma_g_p, sigma_z_p, sigma_tau_p = map(float, values)

    # The table's annualized default-rate cases correspond to quarterly partial
    # default shares delta=0.075 (30%) and delta=0.05 (20%) in the paper.
    delta = 0.075 if case == "nonlinear" else 0.05
    
    return ModelParameters(
        h, s_tilde, gamma_tau_L, gamma_g_L, rho_A, rho_g, rho_z, rho_tau,
        sigma_A_p, sigma_g_p, sigma_z_p, sigma_tau_p, delta,
    )


def make_calibration(p: ModelParameters) -> Calibration:
    """Build level steady states from the paper's calibration table.

    Normalize steady-state productivity to one. The paper reports a 25% work
    share, g/y=0.181, debt/y=1.095*4 (quarterly GDP units), tau=0.333, and
    beta=0.99. Transfers are the steady-state residual in the government
    budget. The paper's text says 25% of time is spent working
    """
    beta = 0.99
    A_bar = 1.0
    n_bar = 0.25
    y_bar = A_bar * n_bar
    g_bar = 0.181 * y_bar
    b_bar = 1.095 * 4.0 * y_bar
    tau_bar = 0.333
    c_bar = y_bar - g_bar

    # Steady government budget: tau*A*n + beta*b = b + g + z.
    z_bar = tau_bar * A_bar * n_bar + beta * b_bar - b_bar - g_bar

    # Calibrate leisure weight phi from the household intratemporal FOC at
    # the steady state: phi*(c-h*c)/(1-n) = A*(1-tau).
    phi = A_bar * (1.0 - tau_bar) * (1.0 - n_bar) / (c_bar * (1.0 - p.h))

    # Convert estimated long-run fiscal responses to the level coefficients.
    gamma_tau = p.gamma_tau_L * tau_bar * (1.0 - p.rho_tau) / b_bar
    gamma_g = p.gamma_g_L * g_bar * (1.0 - p.rho_g) / b_bar

    # Table 2 reports shock standard deviations relative to their steady states.
    sigma_A = p.sigma_A_p * A_bar
    sigma_g = p.sigma_g_p * g_bar
    sigma_z = p.sigma_z_p * z_bar
    sigma_tau = p.sigma_tau_p * tau_bar

    # The logistic fiscal-limit CDF passes through (s_tilde, .30) and
    # (s_tilde + .60, .999), as specified in Section 4.2.1.
    s_hat = p.s_tilde + 0.60
    p_tilde, p_hat = 0.30, 0.999
    eta_2 = (
        np.log(p_tilde * (1.0 - p_hat) / (p_hat * (1.0 - p_tilde)))
        / (p.s_tilde - s_hat)
    )
    eta_1 = np.log(p_tilde / (1.0 - p_tilde)) - eta_2 * p.s_tilde

    return Calibration(
        beta, A_bar, n_bar, y_bar, g_bar, b_bar, tau_bar, z_bar, c_bar, phi,
        gamma_tau, gamma_g, sigma_A, sigma_g, sigma_z, sigma_tau, eta_1, eta_2,
    )


def make_state_grid(p: ModelParameters, c: Calibration, counts: tuple[int, ...]):
    """Create the six-dimensional grid and return axes plus one state per row. Bounds are choosen based of me knowing the end results didnt see any info on that in the paper"""
    bounds = (
    (0.899, 1.387),    # b_d
    (0.178, 0.222),    # c_lag
    (0.829, 1.095),    # A
    (0.0357, 0.0583),  # u_g
    (-0.0365, 0.0906), # z
    (0.2866, 0.3708),  # u_tau
    )
    axes = tuple(np.linspace(lo, hi, int(n)) for (lo, hi), n in zip(bounds, counts))
    mesh = np.meshgrid(*axes, indexing="ij")
    states = np.stack(mesh, axis=-1).reshape(-1, 6)
    return axes, states


def consumption_and_output(
    b_d: float, c_lag: float, A: float, u_g: float, u_tau: float,
    p: ModelParameters, c: Calibration,
) -> tuple[float, float, float, float, float]:
    """Evaluate fiscal rules and equations"""
    tau = u_tau + c.gamma_tau * (b_d - c.b_bar)
    g = u_g - c.gamma_g * (b_d - c.b_bar)
    denominator = 1.0 + c.phi - tau
    if denominator <= 0.0 or A <= 0.0:
        raise ValueError("Invalid state: consumption equation has a nonpositive denominator.")

    c_t = (c.phi * p.h * c_lag + (A - g) * (1.0 - tau)) / denominator
    n_t = (c_t + g) / A  # Goods-market clearing: c_t + g_t = A_t*n_t.
    y_t = A * n_t
    if c_t <= p.h * c_lag or not (0.0 < n_t < 1.0) or y_t <= 0.0:
        raise ValueError(
            "Invalid state: household consumption, labor, or output is outside "
            "the model's feasible region. Adjust the starter grid bounds."
        )
    return tau, g, c_t, n_t, y_t


def normal_quadrature(order: int, sigma: float):
    """Shock generation for an N(0, sigma^2)."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return np.sqrt(2.0) * sigma * nodes, weights / np.sqrt(np.pi)


def expected_bond_price(
    state: np.ndarray,
    b_old: float,
    p: ModelParameters,
    c: Calibration,
    shock_order: int,
) -> tuple[float, float, float, float, float]:
    """Evaluate the household Euler price using old debt rule and shock quadrature.

    Integrates over productivity, spending, and tax shocks as in Appendix A.
    The stochastic fiscal-limit draw is integrated exactly as its two outcomes:
    no default and partial default, weighted by the logistic default probability.
    """
    b_d, c_lag, A_t, u_g, _z_t, u_tau = map(float, state)
    tau_t, g_t, c_t, n_t, y_t = consumption_and_output(
        b_d, c_lag, A_t, u_g, u_tau, p, c
    )

    debt_gdp = b_old / (4.0 * y_t)
    linear_index = np.clip(c.eta_1 + c.eta_2 * debt_gdp, -700.0, 700.0)
    p_default = 1.0 / (1.0 + np.exp(-linear_index))

    a_nodes, a_weights = normal_quadrature(shock_order, c.sigma_A)
    g_nodes, g_weights = normal_quadrature(shock_order, c.sigma_g)
    tau_nodes, tau_weights = normal_quadrature(shock_order, c.sigma_tau)

    expected_repayment_ratio = 0.0
    for ia, ig, itau in product(range(shock_order), repeat=3):
        shock_weight = a_weights[ia] * g_weights[ig] * tau_weights[itau]
        A_next = (1.0 - p.rho_A) * c.A_bar + p.rho_A * A_t + a_nodes[ia]
        u_g_next = (1.0 - p.rho_g) * c.g_bar + p.rho_g * u_g + g_nodes[ig]
        u_tau_next = (1.0 - p.rho_tau) * c.tau_bar + p.rho_tau * u_tau + tau_nodes[itau]

        # No-default branch: Delta=0, so all current debt remains outstanding.
        b_d_no_default = b_old
        tau_no_default, g_no_default, c_next_no_default, _, _ = consumption_and_output(
            b_d_no_default, c_t, A_next, u_g_next, u_tau_next, p, c
        )
        denom_no_default = c_next_no_default - p.h * c_t
        repayment_no_default = 1.0 / denom_no_default

        # Default branch: Delta=delta; the remaining debt is (1-delta)*b_t.
        b_d_default = (1.0 - p.delta) * b_old
        _, _, c_next_default, _, _ = consumption_and_output(
            b_d_default, c_t, A_next, u_g_next, u_tau_next, p, c
        )
        denom_default = c_next_default - p.h * c_t
        repayment_default = (1.0 - p.delta) / denom_default

        expected_repayment_ratio += shock_weight * (
            (1.0 - p_default) * repayment_no_default
            + p_default * repayment_default
        )

    q_t = c.beta * (c_t - p.h * c_lag) * expected_repayment_ratio
    return q_t, tau_t, g_t, c_t, n_t


def solve_policy(
    p: ModelParameters,
    c: Calibration,
    axes: tuple[np.ndarray, ...],
    states: np.ndarray,
    shock_order: int = 3,
    tolerance: float = 1e-5,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Iterate equation (A.9) at every grid point until the debt rule converges."""
    counts = tuple(len(axis) for axis in axes)
    old_debt_rule = np.full(int(np.prod(counts)), c.b_bar, dtype=float)
    new_debt_rule = np.empty_like(old_debt_rule)
    last_prices = np.empty_like(old_debt_rule)

    for iteration in range(1, max_iterations + 1):
        for i, state in enumerate(states):
            b_d, _c_lag, A_t, u_g, z_t, u_tau = map(float, state)
            b_old = old_debt_rule[i]
            q_demand, tau_t, g_t, c_t, n_t = expected_bond_price(
                state, b_old, p, c, shock_order
            )
            funds = b_d + z_t + g_t - tau_t * A_t * n_t
            if not np.isfinite(q_demand) or q_demand <= 0.0:
                raise ValueError(f"Nonpositive/nonfinite household bond price at grid row {i}.")
            b_new = funds / q_demand
            if not np.isfinite(b_new) or b_new <= 0.0:
                raise ValueError(
                    f"Nonpositive/nonfinite debt at grid row {i}; inspect grid bounds."
                )
            new_debt_rule[i] = b_new
            last_prices[i] = funds / b_new

        error = float(np.max(np.abs(new_debt_rule - old_debt_rule)))
        if error < tolerance:
            return new_debt_rule.copy(), last_prices.copy(), iteration, error
        old_debt_rule, new_debt_rule = new_debt_rule, old_debt_rule

    raise RuntimeError(
        f"Debt rule did not converge in {max_iterations} iterations; "
        f"last max change was {error:.3g}. Try wider/different grid bounds or inspect parameters."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("nonlinear", "nonlinear_0.2"), default="nonlinear"
    )
    parser.add_argument("--n-b", type=int, default=3)
    parser.add_argument("--n-c", type=int, default=3)
    parser.add_argument("--n-A", type=int, default=3)
    parser.add_argument("--n-g", type=int, default=3)
    parser.add_argument("--n-z", type=int, default=3)
    parser.add_argument("--n-tau", type=int, default=3)
    args = parser.parse_args()

    model = load_model_parameters(args.case)
    calibration = make_calibration(model)
    counts = (args.n_b, args.n_c, args.n_A, args.n_g, args.n_z, args.n_tau)
    axes, states = make_state_grid(model, calibration, counts)
    debt_rule, bond_prices, iterations, error = solve_policy(
        model, calibration, axes, states
    )

    output = Path("model_solution.npz")
    np.savez_compressed(
        output,
        states=states,
        debt_rule=debt_rule,
        bond_prices=bond_prices,
        grid_shape=np.asarray(counts),
        state_names=np.asarray(STATE_NAMES),
        iterations=iterations,
        max_rule_change=error,
        posterior_case=args.case,
    )
    print(f"Solved {len(states):,} grid states in {iterations} iterations.")
    print(f"Maximum debt-rule change: {error:.3e}")
    print(f"Saved grid solution to {output.resolve()}")
    s = np.load("model_solution.npz")
    print("Debt range:", s["debt_rule"].min(), s["debt_rule"].max())
    print("Bond price range:", s["bond_prices"].min(), s["bond_prices"].max())
    print("All finite:", np.isfinite(s["debt_rule"]).all(),
          np.isfinite(s["bond_prices"]).all())

if __name__ == "__main__":
    main()
