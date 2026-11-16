#!/usr/bin/env python3
"""Plot the model's four-quarter 2011 rate and default-risk forecasts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from model_solver import (
    STATE_NAMES,
    consumption_and_output,
    load_model_parameters,
    make_calibration,
)
from state_filter import (
    interpolate_grid,
    load_solution,
)


def load_2011_debt_gdp(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read the observed 2011 quarterly debt/GDP ratios (percent units)."""
    quarters: list[str] = []
    ratios: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        for row in rows:
            quarters.append(row["quarter"].strip())
            ratios.append(float(row["debt_gdp_percent"]))
    expected = ("2011Q1", "2011Q2", "2011Q3", "2011Q4")
    if tuple(quarters) != expected or not np.all(np.isfinite(ratios)):
        raise ValueError(f"{path} must contain finite 2011Q1-2011Q4 ratios in order.")
    return np.asarray(ratios, dtype=float) / 100.0, expected


def fiscal_limit_probability(debt_gdp: float, s_tilde: float) -> float:
    """Evaluate the paper's logistic fiscal-limit CDF at a debt/GDP ratio."""
    p_tilde, p_hat = 0.30, 0.999
    s_hat = s_tilde + 0.60
    eta_2 = np.log(p_tilde * (1.0 - p_hat) / (p_hat * (1.0 - p_tilde))) / (
        s_tilde - s_hat
    )
    eta_1 = np.log(p_tilde / (1.0 - p_tilde)) - eta_2 * s_tilde
    index = np.clip(eta_1 + eta_2 * debt_gdp, -700.0, 700.0)
    return float(1.0 / (1.0 + np.exp(-index)))


def table2_stilde(case: str) -> tuple[float, float, float]:
    """Return the Table 2 posterior median and 5th/95th percentiles of s_tilde."""
    if case == "nonlinear":
        median_column = "Nonlinear delta_A=0.3 median"
        interval_column = "Nonlinear delta_A=0.3 [5th, 95th]"
    elif case == "nonlinear_0.2":
        median_column = "Nonlinear delta_A=0.2 median"
        interval_column = "Nonlinear delta_A=0.2 [5th, 95th]"
    else:
        raise ValueError("Observed fiscal-limit risk requires a nonlinear case.")

    table_path = Path(__file__).with_name("posterior_data.txt")
    if not table_path.exists():
        table_path = Path.cwd() / "greece_estimates_table2.txt"
    with table_path.open("r", encoding="utf-8", newline="") as file:
        next(file)
        rows = csv.DictReader(file, delimiter="\t")
        for row in rows:
            if row["Parameter"].strip() == "s_tilde":
                median = float(row[median_column])
                low, high = (float(value.strip()) for value in row[interval_column].strip("[]").split(","))
                return median, low, high
    raise ValueError("Could not find s_tilde in greece_estimates_table2.txt.")


def observed_debt_default_risk(
    debt_gdp_ratios: np.ndarray, case: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map observed debt ratios into median and approximate 90% risk intervals.

    The interval uses the rounded marginal s_tilde 5th/95th percentiles in
    Table 2.  Since default risk is monotone in s_tilde, its interval reverses
    those threshold endpoints.
    """
    median, stilde_5, stilde_95 = table2_stilde(case)
    median_risk = np.asarray(
        [fiscal_limit_probability(s, median) for s in debt_gdp_ratios]
    )
    lower_risk = np.asarray(
        [fiscal_limit_probability(s, stilde_95) for s in debt_gdp_ratios]
    )
    upper_risk = np.asarray(
        [fiscal_limit_probability(s, stilde_5) for s in debt_gdp_ratios]
    )
    return median_risk, lower_risk, upper_risk


def bounded_interpolate(axes, values: np.ndarray, state: np.ndarray) -> float:
    """Interpolate at the nearest in-grid state instead of extrapolating."""
    bounded_state = np.asarray(
        [np.clip(value, axis[0], axis[-1]) for axis, value in zip(axes, state)],
        dtype=float,
    )
    return interpolate_grid(axes, values, bounded_state)


def forecast_transition(
    states: np.ndarray,
    axes,
    debt_grid: np.ndarray,
    p,
    c,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance paths, using boundary policy values for out-of-grid states."""
    n_paths = len(states)
    next_states = np.empty_like(states)
    eps_A = rng.normal(0.0, c.sigma_A, n_paths)
    eps_g = rng.normal(0.0, c.sigma_g, n_paths)
    eps_z = rng.normal(0.0, c.sigma_z, n_paths)
    eps_tau = rng.normal(0.0, c.sigma_tau, n_paths)

    for i, state in enumerate(states):
        b_d, c_lag, A_t, u_g, z_t, u_tau = map(float, state)
        tau_t, g_t, c_t, n_t, y_t = consumption_and_output(
            b_d, c_lag, A_t, u_g, u_tau, p, c
        )
        b_t = bounded_interpolate(axes, debt_grid, state)
        debt_gdp = b_t / (4.0 * y_t)
        logit = np.clip(c.eta_1 + c.eta_2 * debt_gdp, -700.0, 700.0)
        probability_default = 1.0 / (1.0 + np.exp(-logit))
        delta_t1 = p.delta if rng.random() < probability_default else 0.0

        next_states[i] = (
            (1.0 - delta_t1) * b_t,
            c_t,
            (1.0 - p.rho_A) * c.A_bar + p.rho_A * A_t + eps_A[i],
            (1.0 - p.rho_g) * c.g_bar + p.rho_g * u_g + eps_g[i],
            (1.0 - p.rho_z) * c.z_bar + p.rho_z * z_t + eps_z[i],
            (1.0 - p.rho_tau) * c.tau_bar + p.rho_tau * u_tau + eps_tau[i],
        )
    return next_states


def load_fitted_2010_state(
    path: Path, case: str
) -> tuple[np.ndarray, float, int, np.ndarray | None]:
    """Load the fitted 2010Q4 state and, if present, the smoothed history."""
    with np.load(path, allow_pickle=False) as saved:
        saved_case = str(saved["posterior_case"].item())
        if saved_case != case:
            raise ValueError(
                f"State file uses case {saved_case!r}, but --case is {case!r}."
            )
        particles = np.asarray(saved["terminal_particles"], dtype=float)
        weights = np.asarray(saved["terminal_weights"], dtype=float)
        if "smoothed_states" in saved.files:
            smoothed_states = np.asarray(saved["smoothed_states"], dtype=float)
            fitted_state = smoothed_states[-1]
        elif "filtered_means" in saved.files:
            smoothed_states = None
            fitted_state = np.asarray(saved["filtered_means"][-1], dtype=float)
        else:
            smoothed_states = None
            fitted_state = np.asarray(weights @ particles, dtype=float)


    weights = weights / weights.sum()
    start_ess = float(1.0 / np.sum(weights**2))
    if smoothed_states is not None and (
        smoothed_states.ndim != 2
        or smoothed_states.shape[1] != len(STATE_NAMES)
        or smoothed_states.shape[0] != 40
        or not np.all(np.isfinite(smoothed_states))
    ):
    return fitted_state, start_ess, len(weights), smoothed_states


def evaluate_forecast_states(
    states: np.ndarray,
    axes,
    debt_grid: np.ndarray,
    price_grid: np.ndarray,
    p,
    c,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return default risk, rate deviation, annual rate, debt ratio, and flags."""
    n_paths = len(states)
    default_probability = np.empty(n_paths, dtype=float)
    rate_deviation = np.empty(n_paths, dtype=float)
    annual_rate_percent = np.empty(n_paths, dtype=float)
    debt_gdp_ratio = np.empty(n_paths, dtype=float)

    outside_grid = np.zeros(n_paths, dtype=bool)
    for dimension, axis in enumerate(axes):
        outside_grid |= (states[:, dimension] < axis[0]) | (
            states[:, dimension] > axis[-1]
        )

    for i, state in enumerate(states):
        b_d, c_lag, A_t, u_g, z_t, u_tau = map(float, state)
        _, _, _, _, y_t = consumption_and_output(
            b_d, c_lag, A_t, u_g, u_tau, p, c
        )
        b_t = bounded_interpolate(axes, debt_grid, state)
        q_t = bounded_interpolate(axes, price_grid, state)
        if not np.isfinite(q_t) or q_t <= 0.0:
            raise ValueError("Forecast produced a nonpositive or nonfinite bond price.")

        # As in the transition rule, fiscal-limit risk is based on newly issued
        # debt relative to annualized output.
        ratio = b_t / (4.0 * y_t)
        logit = np.clip(c.eta_1 + c.eta_2 * ratio, -700.0, 700.0)
        default_probability[i] = 1.0 / (1.0 + np.exp(-logit))
        debt_gdp_ratio[i] = ratio

        # Match the transformed R observable: deviation in the gross real
        # return from its deterministic model reference, (1 / beta).
        rate_deviation[i] = c.beta / q_t - 1.0

        # q_t prices a one-quarter-ahead unit payoff. Annualize the quarterly
        # net rate by multiplying by four, and report percentage points.
        annual_rate_percent[i] = 400.0 * (1.0 / q_t - 1.0)

    return (
        default_probability,
        rate_deviation,
        annual_rate_percent,
        debt_gdp_ratio,
        outside_grid,
    )


def run_forecast(
    fitted_2010_state: np.ndarray,
    axes,
    debt_grid: np.ndarray,
    price_grid: np.ndarray,
    p,
    c,
    horizon_quarters: int,
    n_paths: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate forecast paths beginning in 2010Q4 and entering 2011Q1."""
    rng = np.random.default_rng(seed)
    # The paper forecasts from the fitted 2010Q4 state, then generates
    # uncertainty from future shocks and default events.
    states = np.repeat(fitted_2010_state[None, :], n_paths, axis=0)

    default_paths = np.empty((horizon_quarters, n_paths), dtype=float)
    rate_deviation_paths = np.empty((horizon_quarters, n_paths), dtype=float)
    rate_paths = np.empty((horizon_quarters, n_paths), dtype=float)
    debt_paths = np.empty((horizon_quarters, n_paths), dtype=float)
    outside_grid_share = np.empty(horizon_quarters, dtype=float)

    for h in range(horizon_quarters):
        # Move from the fitted state at 2010Q4 into 2011Q1, then forward one
        # quarter at a time. This draws the model shocks and default event.
        states = forecast_transition(states, axes, debt_grid, p, c, rng)
        (
            default_paths[h],
            rate_deviation_paths[h],
            rate_paths[h],
            debt_paths[h],
            out_of_grid,
        ) = evaluate_forecast_states(states, axes, debt_grid, price_grid, p, c)
        outside_grid_share[h] = np.mean(out_of_grid)

    def intervals(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.quantile(paths, 0.50, axis=1),
            np.quantile(paths, 0.05, axis=1),
            np.quantile(paths, 0.95, axis=1),
        )

    p_med, p_low, p_high = intervals(default_paths)
    rd_med, rd_low, rd_high = intervals(rate_deviation_paths)
    r_med, r_low, r_high = intervals(rate_paths)
    b_med, b_low, b_high = intervals(debt_paths)

    return {
        "default_paths": default_paths,
        "rate_deviation_paths": rate_deviation_paths,
        "rate_paths": rate_paths,
        "debt_paths": debt_paths,
        "default_median": p_med,
        "default_5": p_low,
        "default_95": p_high,
        "rate_deviation_median": rd_med,
        "rate_deviation_5": rd_low,
        "rate_deviation_95": rd_high,
        "rate_median": r_med,
        "rate_5": r_low,
        "rate_95": r_high,
        "debt_gdp_median": b_med,
        "debt_gdp_5": b_low,
        "debt_gdp_95": b_high,
        "outside_grid_share": outside_grid_share,
    }


def plot_forecast(result: dict[str, np.ndarray], path: Path) -> None:
    """Plot the rate forecast and observed-debt-based default-risk estimates."""
    import matplotlib.pyplot as plt

    x = np.arange(4)
    labels = ("2011 Q1", "2011 Q2", "2011 Q3", "2011 Q4")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    axes[0].fill_between(x, result["rate_5"], result["rate_95"], color="tab:blue", alpha=0.2,
                         label="90% interval")
    axes[0].plot(x, result["rate_median"], color="tab:blue", marker="o", label="Median")
    axes[0].set_title("Real interest rate")
    axes[0].set_ylabel("Percent")

    axes[1].fill_between(
        x,
        100.0 * result["default_5"],
        100.0 * result["default_95"],
        color="tab:orange",
        alpha=0.2,
        label="90% posterior interval",
    )
    axes[1].plot(
        x,
        100.0 * result["default_median"],
        color="tab:orange",
        marker="o",
        label="Median, observed debt/GDP",
    )
    axes[1].set_title("Model-implied sovereign default probability")
    axes[1].set_ylabel("Percent")
    axes[1].set_ylim(0, 100)

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    figure.suptitle("Greece: 2011 forecast")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("nonlinear", "nonlinear_0.2"), default="nonlinear"
    )
    parser.add_argument("--paths", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--state-file", type=Path, default=Path("state_smoother_output.npz")
    )
    parser.add_argument("--solution", type=Path, default=Path("model_solution.npz"))
    parser.add_argument(
        "--debt-gdp-data",
        type=Path,
        default=Path(__file__).with_name("greece_2011_debt_gdp.csv"),
    )
    parser.add_argument("--plot", type=Path, default=Path("forecast_2011.png"))
    args = parser.parse_args()

    p = load_model_parameters(args.case)
    c = make_calibration(p)
    axes, debt_grid, price_grid = load_solution(args.solution, args.case)
    fitted_2010_state, starting_ess, n_terminal, _smoothed_states = load_fitted_2010_state(
        args.state_file, args.case
    )
    result = run_forecast(
        fitted_2010_state,
        axes,
        debt_grid,
        price_grid,
        p,
        c,
        horizon_quarters=4,
        n_paths=args.paths,
        seed=args.seed,
    )
    observed_debt_gdp, quarters = load_2011_debt_gdp(args.debt_gdp_data)
    (
        result["default_median"],
        result["default_5"],
        result["default_95"],
    ) = observed_debt_default_risk(observed_debt_gdp, args.case)
    plot_forecast(result, args.plot)

    print("2011: rate forecast and observed-debt default risk")
    print("Rate intervals are simulated; risk intervals use the Table 2 s_tilde range.")
    for i, quarter in enumerate(quarters):
        print(
            f"{quarter}: observed debt/GDP {100 * observed_debt_gdp[i]:.1f}% | "
            f"rate {result['rate_median'][i]:.2f}% "
            f"[{result['rate_5'][i]:.2f}, {result['rate_95'][i]:.2f}] | "
            f"default {100 * result['default_median'][i]:.2f}% "
            f"[{100 * result['default_5'][i]:.2f}, "
            f"{100 * result['default_95'][i]:.2f}]"
        )
    print(f"Saved plot to {args.plot.resolve()}")


if __name__ == "__main__":
    main()
