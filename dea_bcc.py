"""Input-oriented BCC (variable returns to scale) DEA for La Liga clubs.

Efficiency is computed separately for every season, so each club is only
compared with the other clubs of the same season.

Model (envelopment form), solved once per club k:

    min  theta
    s.t. sum_j lambda_j * x_ij <= theta * x_ik      for every input i
         sum_j lambda_j * y_rj >= y_rk              for every output r
         sum_j lambda_j = 1                         (VRS / BCC)
         lambda_j >= 0

Inputs:  LCPD (spending limit), Wartosc_kadry (squad market value)
Output:  Punkty (league points)

Running this file recomputes the scores from the raw inputs and compares them
with the ``Efektywnosc`` column of ``wyniki_DEA.csv``. The process exits with
a non-zero status if the largest absolute difference exceeds ``--tol``.

Usage:
    python dea_bcc.py [--csv PATH] [--tol 1e-9]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

DEFAULT_CSV = Path(__file__).resolve().parent / "wyniki_DEA.csv"

INPUT_COLUMNS = ["LCPD", "Wartosc_kadry"]
OUTPUT_COLUMNS = ["Punkty"]
REQUIRED_COLUMNS = ["Sezon", "Klub", *INPUT_COLUMNS, *OUTPUT_COLUMNS, "Efektywnosc"]


class DEAError(Exception):
    """Raised when the data are invalid or a linear program cannot be solved."""


def load_data(csv_path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Read the results CSV and validate it.

    Raises:
        DEAError: if the file is missing, columns are absent, values are
            missing or non-positive, or a club appears twice in one season.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DEAError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DEAError(f"Missing required columns: {missing}")

    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Check the data are suitable for DEA.

    Inputs must be strictly positive (a zero input makes the ratio form
    degenerate), outputs non-negative, there must be no missing values and
    each (season, club) pair must be unique.

    Raises:
        DEAError: on the first violated condition.
    """
    numeric = [*INPUT_COLUMNS, *OUTPUT_COLUMNS]
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DEAError(f"Column {col} must be numeric")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise DEAError("Missing values in required columns")
    if (df[INPUT_COLUMNS] <= 0).any().any():
        raise DEAError("Inputs must be strictly positive")
    if (df[OUTPUT_COLUMNS] < 0).any().any():
        raise DEAError("Outputs must be non-negative")
    if df.duplicated(["Sezon", "Klub"]).any():
        raise DEAError("Duplicate (Sezon, Klub) rows found")


def dea_bcc_input(inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Input-oriented BCC efficiency scores for one set of comparable units.

    Args:
        inputs: array of shape (n_units, n_inputs), strictly positive.
        outputs: array of shape (n_units, n_outputs).

    Returns:
        Array of shape (n_units,) with theta in (0, 1]; 1 means the unit lies
        on the efficient frontier.

    Raises:
        DEAError: on shape mismatch or if any linear program fails.
    """
    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    if inputs.ndim != 2 or outputs.ndim != 2 or len(inputs) != len(outputs):
        raise DEAError("inputs and outputs must be 2-D arrays with equal row counts")

    n, n_in = inputs.shape
    n_out = outputs.shape[1]

    # Decision vector: [lambda_1 .. lambda_n, theta]; objective: minimise theta.
    c = np.zeros(n + 1)
    c[-1] = 1.0
    a_eq = np.append(np.ones(n), 0.0).reshape(1, -1)
    b_eq = [1.0]
    bounds = [(0, None)] * (n + 1)

    thetas = np.empty(n)
    for k in range(n):
        a_ub = np.zeros((n_in + n_out, n + 1))
        b_ub = np.zeros(n_in + n_out)
        a_ub[:n_in, :n] = inputs.T            # X lambda - theta x_k <= 0
        a_ub[:n_in, -1] = -inputs[k]
        a_ub[n_in:, :n] = -outputs.T          # -Y lambda <= -y_k
        b_ub[n_in:] = -outputs[k]

        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            raise DEAError(f"LP failed for unit {k}: {res.message}")
        thetas[k] = res.fun
    return thetas


def compute_efficiency(df: pd.DataFrame) -> pd.Series:
    """Compute BCC input-oriented efficiency separately for each season.

    Returns:
        Series aligned with ``df.index``.
    """
    scores = pd.Series(np.nan, index=df.index, name="Efektywnosc_recomputed")
    for _, group in df.groupby("Sezon"):
        scores.loc[group.index] = dea_bcc_input(
            group[INPUT_COLUMNS].to_numpy(float),
            group[OUTPUT_COLUMNS].to_numpy(float),
        )
    if scores.isna().any():
        raise DEAError("Some observations received no efficiency score")
    return scores


def verify(df: pd.DataFrame, tol: float = 1e-9) -> float:
    """Recompute scores and compare with the ``Efektywnosc`` column.

    Prints a per-season summary and returns the maximum absolute difference.
    """
    recomputed = compute_efficiency(df)
    diff = (recomputed - df["Efektywnosc"]).abs()

    print("=== DEA VERIFICATION (input-oriented BCC, per season) ===")
    print(f"Inputs: {', '.join(INPUT_COLUMNS)} | Output: {', '.join(OUTPUT_COLUMNS)}")
    for season, idx in df.groupby("Sezon").groups.items():
        print(f"{season}: n = {len(idx):2d}, max |diff| = {diff.loc[idx].max():.3e}")
    print(f"Observations compared: {len(df)}")
    print(f"Maximum absolute difference: {diff.max():.3e}")
    print(f"Observations within tolerance {tol:g}: {int((diff <= tol).sum())} of {len(df)}")
    return float(diff.max())


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=DEFAULT_CSV, type=Path,
                        help="path to wyniki_DEA.csv (default: next to this script)")
    parser.add_argument("--tol", default=1e-9, type=float,
                        help="maximum accepted absolute difference")
    args = parser.parse_args(argv)

    try:
        df = load_data(args.csv)
        max_diff = verify(df, args.tol)
    except DEAError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if max_diff > args.tol:
        print("FAIL: recomputed scores differ from the CSV.", file=sys.stderr)
        return 2
    print("OK: the Efektywnosc column is reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
