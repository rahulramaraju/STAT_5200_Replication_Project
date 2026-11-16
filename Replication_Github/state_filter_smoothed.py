#!/usr/bin/env python3
"""Particle filter plus a backward smoother just as the paper does
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from observed_data_reader import read_data
from model_solver import (
    STATE_NAMES,
    ModelParameters,
    Calibration,
    load_model_parameters,
    make_calibration,
)
from state_filter import (
    initial_state_from_first_observation,
    load_solution,
    log_observation_weight,
    normalized_weights,
    predict_observables,
    transition_particles,
)


def systematic_resample_indices(
    weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return parent indices selected by systematic resampling."""
    n_particles = len(weights)
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def run_filter_and_smoother(
    data: np.ndarray,
    p: ModelParameters,
    c: Calibration,
    axes,
    debt_grid: np.ndarray,
    price_grid: np.ndarray,
    n_particles: int = 5000,
    seed: int = 12345,
):
    rng = np.random.default_rng(seed)
    measurement_sd = np.maximum(0.20 * np.std(data, axis=0, ddof=1), 1e-8)
    n_periods = len(data)
    particles = initial_state_from_first_observation(
        data[0], p, c, n_particles, rng
    )

    # history[t] holds the particles at quarter t before weighting on y_t.
    history = np.empty((n_periods, n_particles, 6), dtype=float)
    # parents[t, j] is the particle index at t-1 that generated particle j at t.
    parents = np.full((n_periods, n_particles), -1, dtype=np.int64)
    filtered_observables = np.empty((n_periods, 5), dtype=float)
    effective_sample_sizes = np.empty(n_periods, dtype=float)
    terminal_weights = np.full(n_particles, 1.0 / n_particles)
    log_likelihood = 0.0

    for t, observed in enumerate(data):
        history[t] = particles
        log_weights = np.full(n_particles, -np.inf, dtype=float)
        predictions = np.zeros((n_particles, 5), dtype=float)
        for j, state in enumerate(particles):
            try:
                prediction = predict_observables(
                    state, axes, debt_grid, price_grid, p, c
                )
                predictions[j] = prediction
                log_weights[j] = log_observation_weight(
                    observed, prediction, measurement_sd
                )
            except (ValueError, FloatingPointError, ZeroDivisionError):
                pass

        terminal_weights, log_increment = normalized_weights(log_weights)
        log_likelihood += log_increment
        filtered_observables[t] = terminal_weights @ predictions
        effective_sample_sizes[t] = 1.0 / np.sum(terminal_weights**2)

        if t < n_periods - 1:
            parent_indices = systematic_resample_indices(terminal_weights, rng)
            parents[t + 1] = parent_indices
            particles = transition_particles(
                particles[parent_indices], axes, debt_grid, p, c, rng
            )
            terminal_weights = np.full(n_particles, 1.0 / n_particles)

    # smoothing part introduced in paper
    smoothed_observables = np.zeros((n_periods, 5), dtype=float)
    smoothed_states = np.zeros((n_periods, 6), dtype=float)
    for terminal_index, terminal_weight in enumerate(terminal_weights):
        if terminal_weight == 0.0:
            continue
        particle_index = terminal_index
        for t in range(n_periods - 1, -1, -1):
            state = history[t, particle_index]
            smoothed_states[t] += terminal_weight * state
            smoothed_observables[t] += terminal_weight * predict_observables(
                state, axes, debt_grid, price_grid, p, c
            )
            if t > 0:
                particle_index = int(parents[t, particle_index])

    return {
        "log_likelihood": log_likelihood,
        "terminal_particles": history[-1],
        "terminal_weights": terminal_weights,
        "filtered_observables": filtered_observables,
        "smoothed_observables": smoothed_observables,
        "smoothed_states": smoothed_states,
        "effective_sample_sizes": effective_sample_sizes,
        "measurement_sd": measurement_sd,
    }


def plot_smoothed_fit(
    data: np.ndarray,
    smoothed_observables: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save five panels comparing observations with smoothed model estimates."""
    import matplotlib.pyplot as plt

    labels = (
        "Government Debt",
        "Government Spending",
        "Tax Revenue",
        "Output",
        "Real Interest Rate",
    )
    # Data columns are Y, G, TR, B, R; display in B, G, TR, Y, R order.
    order = (3, 1, 2, 0, 4)
    quarters = np.arange(len(data))
    tick_positions = np.arange(0, len(data), 8)
    tick_labels = [f"Q1'{2001 + int(t // 4)}" for t in tick_positions]

    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    flat_axes = axes.ravel()
    for panel, column in enumerate(order):
        ax = flat_axes[panel]
        ax.plot(quarters, data[:, column], color="black", linewidth=1.2, label="Data")
        ax.plot(
            quarters,
            smoothed_observables[:, column],
            color="red",
            linestyle="--",
            linewidth=1.4,
            label="Smoothed fit",
        )
        ax.set_title(labels[panel])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, alpha=0.25)
        ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
        if panel == 0:
            ax.legend(frameon=False, fontsize=8)
    flat_axes[-1].axis("off")
    figure.suptitle("Observed Data and Smoothed Model Fit", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("nonlinear", "nonlinear_0.2"), default="nonlinear"
    )
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--solution", type=Path, default=Path("model_solution.npz"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("state_smoother_output.npz")
    )
    parser.add_argument(
        "--plot-output", type=Path, default=Path("smoothed_fit_vs_data.png")
    )
    args = parser.parse_args()

    p = load_model_parameters(args.case)
    c = make_calibration(p)
    axes, debt_grid, price_grid = load_solution(args.solution, args.case)
    if args.data is None:
        observations = np.column_stack(read_data())
    else:
        observations = np.loadtxt(args.data, dtype=float)

    result = run_filter_and_smoother(
        observations,
        p,
        c,
        axes,
        debt_grid,
        price_grid,
        n_particles=args.particles,
        seed=args.seed,
    )
    np.savez_compressed(
        args.output,
        terminal_particles=result["terminal_particles"],
        terminal_weights=result["terminal_weights"],
        filtered_observables=result["filtered_observables"],
        smoothed_observables=result["smoothed_observables"],
        smoothed_states=result["smoothed_states"],
        effective_sample_sizes=result["effective_sample_sizes"],
        measurement_sd=result["measurement_sd"],
        log_likelihood=result["log_likelihood"],
        state_names=np.asarray(STATE_NAMES),
        posterior_case=args.case,
        n_particles=args.particles,
    )
    plot_smoothed_fit(
        observations, result["smoothed_observables"], args.plot_output
    )
    print(f"Filtered and smoothed {len(observations)} quarters.")
    print(f"Approximate log likelihood: {result['log_likelihood']:.6f}")
    print(
        "Terminal effective sample size: "
        f"{result['effective_sample_sizes'][-1]:.1f} / {args.particles}"
    )
    print(f"Saved smoothed states to {args.output.resolve()}")
    print(f"Saved smoothed fit plot to {args.plot_output.resolve()}")
    s = np.load("state_smoother_output.npz")
    print(s["effective_sample_sizes"])
    print("Smoothed productivity:", s["smoothed_states"][:, 2])

if __name__ == "__main__":
    main()
