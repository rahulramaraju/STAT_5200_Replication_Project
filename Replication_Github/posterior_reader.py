#!/usr/bin/env python3
"""Read posterior medians from greece_estimates_table2.txt.

Returned tuple order:
h, s_tilde, gamma_tau_L, gamma_g_L, rho_A, rho_g, rho_z, rho_tau,
sigma_A_p, sigma_g_p, sigma_z_p, sigma_tau_p
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TABLE_PATH = Path(__file__).with_name("posterior_data.txt")


def read_parameters(case: str = "nonlinear") -> tuple[float | None, ...]:
    if case == "nonlinear":
        median_column = "Nonlinear delta_A=0.3 median"
    elif case == "nonlinear_0.2":
        median_column = "Nonlinear delta_A=0.2 median"
    elif case == "linear":
        median_column = "Linear median"

    values: list[float | None] = []
    with TABLE_PATH.open("r", encoding="utf-8", newline="") as file:
        next(file)  # Skip the title line.
        rows = csv.DictReader(file, delimiter="\t")
        for row in rows:
            value = (row[median_column] or "").strip()
            values.append(None if value in {"", "-"} else float(value))

    return tuple(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("nonlinear", "nonlinear_0.2", "linear"),
        default="nonlinear",
    )
    args = parser.parse_args()
    print(read_parameters(args.case))

