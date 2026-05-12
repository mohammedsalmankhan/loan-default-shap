"""Generate a synthetic loan-default dataset for the explainability demo.

The dataset mimics realistic distributions of features that drive default risk
in published lending datasets (e.g. LendingClub, German Credit). It is
deliberately small and synthetic to keep the repository self-contained and
free of any privacy or licensing concerns.

Run:
    python src/generate_data.py --rows 2000 --out sample_data/loans.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_loan_data(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic loan dataset with realistic feature relationships.

    Features:
        Age           — applicant age, 21–70
        Income        — annual income in pounds, log-normal distribution
        LoanAmount    — requested loan amount, correlated with income
        CreditScore   — 300–850, weakly positive with income
        MonthsEmployed — months at current job, 0–360
        NumCreditLines — number of active credit lines, 1–12
        Default       — target: 1 if defaulted, 0 otherwise

    Default risk is a logistic function of:
        - high loan-to-income ratio (positive)
        - low credit score (positive)
        - short employment history (positive)
        - many open credit lines (positive)
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, size=n_rows)
    income = np.clip(rng.lognormal(mean=10.6, sigma=0.55, size=n_rows), 12_000, 250_000).round(0)
    loan_amount = np.clip(income * rng.uniform(0.3, 4.0, size=n_rows), 1_000, 500_000).round(0)
    credit_score = np.clip(
        rng.normal(loc=650 + 0.0003 * income, scale=80, size=n_rows),
        300, 850,
    ).round(0)
    months_employed = rng.integers(0, 361, size=n_rows)
    num_credit_lines = rng.integers(1, 13, size=n_rows)

    # Default probability driven by realistic risk factors
    loan_to_income = loan_amount / income
    risk = (
        2.0 * loan_to_income
        - 0.012 * (credit_score - 600)
        - 0.008 * months_employed / 12
        + 0.15 * num_credit_lines
        - 5.0
    )
    default_prob = 1.0 / (1.0 + np.exp(-risk))
    default = (rng.random(n_rows) < default_prob).astype(int)

    df = pd.DataFrame(
        {
            "Age": age,
            "Income": income.astype(int),
            "LoanAmount": loan_amount.astype(int),
            "CreditScore": credit_score.astype(int),
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "Default": default,
        }
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic loan default dataset.")
    parser.add_argument("--rows", type=int, default=2000, help="Number of rows (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--out",
        type=str,
        default="sample_data/loans.csv",
        help="Output CSV path (default: sample_data/loans.csv)",
    )
    args = parser.parse_args()

    df = generate_loan_data(n_rows=args.rows, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    default_rate = df["Default"].mean()
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Default rate: {default_rate:.1%}")
    print(f"Feature ranges:")
    for col in df.columns:
        if col == "Default":
            continue
        print(f"  {col:<16} min={df[col].min():>8}  max={df[col].max():>8}  mean={df[col].mean():>10.1f}")


if __name__ == "__main__":
    main()
