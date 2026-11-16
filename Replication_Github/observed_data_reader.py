#!/usr/bin/env python3
""" Read the 5-column files since the data is already transformed"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATA_PATH = "paper_data.txt"


def read_data(path: str | Path = DEFAULT_DATA_PATH):
    """Return five NumPy arrays in the README's column order: Y, G, TR, B, R."""
    data = np.loadtxt(path, dtype=float)
    if data.shape != (40, 5):
        raise ValueError(f"Expected 40 rows and 5 columns; got {data.shape}")

    Y, G, TR, B, R = data.T
    return Y, G, TR, B, R


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    Y, G, TR, B, R = read_data(args.data)
    print(f"Loaded {len(Y)} quarters (2001Q1-2010Q4).")
    print("Column order: Y, G, TR, B, R")
    print("First row:", Y[0], G[0], TR[0], B[0], R[0])

