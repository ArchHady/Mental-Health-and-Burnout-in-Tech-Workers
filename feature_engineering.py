"""
Shared feature engineering transformations.

Imported by both train_model.py and app.py so the same transforms
are applied identically during training and at prediction time.

All operations are row-wise — no statistics are learned from data —
so there is zero train/test leakage when calling engineer_features
before or after a train/test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Ordinal mappings ────────────────────────────────────────────────────────
# seniority_level has a natural career-progression order.
SENIORITY_ORDER: dict[str, int] = {
    "Junior": 1,
    "Mid": 2,
    "Senior": 3,
    "Lead": 4,
    "Manager": 5,
    "Principal": 6,
}

# company_size has a natural headcount order.
COMPANY_SIZE_ORDER: dict[str, int] = {
    "Startup (1-50)": 1,
    "Small (51-200)": 2,
    "Mid (201-1000)": 3,
    "Large (1001-5000)": 4,
    "Enterprise (5000+)": 5,
}

# Categorical columns that stay as strings after engineering (used by OHE).
MODEL_CATEGORICAL_COLS: list[str] = [
    "gender",
    "country",
    "job_role",
    "industry",
    "work_mode",
]

# Original categorical columns shown in the prediction form (includes the two
# that get ordinal-encoded so the user can still pick them from a dropdown).
RAW_CATEGORICAL_COLS: list[str] = [
    "gender",
    "country",
    "job_role",
    "seniority_level",
    "company_size",
    "industry",
    "work_mode",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to *df* and return the result.

    Input must contain the original raw columns.  Both the original and
    new columns are kept; callers select the subset they need via
    feature_order from feature_meta.json.
    """
    df = df.copy()

    # ── 1. Ordinal encoding ──────────────────────────────────────────────────
    # Replace seniority_level / company_size with integer ranks.
    # Unseen values default to the middle of the scale (3).
    df["seniority_ord"] = df["seniority_level"].map(SENIORITY_ORDER).fillna(3.0)
    df["company_size_ord"] = df["company_size"].map(COMPANY_SIZE_ORDER).fillna(3.0)

    # ── 2. Composite interaction features ───────────────────────────────────
    # Work-to-recovery ratio: captures the overwork × sleep-deprivation axis.
    sleep_safe = df["sleep_hours_per_night"].clip(lower=1.0)
    df["work_recovery_ratio"] = df["work_hours_per_week"] / sleep_safe

    # Net pressure after social + managerial support is subtracted.
    df["pressure_vs_support"] = (
        df["deadline_pressure_score"]
        - df["manager_support_score"]
        - df["social_support_score"]
    )

    # Composite protective buffer (sleep, exercise, vacation, balance).
    df["resilience_index"] = (
        df["sleep_hours_per_night"] / 10.0
        + df["exercise_days_per_week"] / 7.0
        + df["vacation_days_taken"] / 30.0
        + df["work_life_balance_score"] / 10.0
    ) / 4.0

    # Tenure relative to total experience — low ratio flags job-hopping.
    df["tenure_experience_ratio"] = df["years_at_company"] / (df["years_experience"] + 1.0)

    # Salary adjusted for experience — low value flags potential under-payment.
    df["salary_per_experience"] = df["salary_usd"] / (df["years_experience"] + 1.0)

    # Raw meeting load (density × hours).
    df["meetings_burden"] = df["meetings_per_day"] * df["work_hours_per_week"]

    # ── 3. Log transforms (reduce right-skew for linear models) ─────────────
    df["log_salary"] = np.log1p(df["salary_usd"])
    df["log_team_size"] = np.log1p(df["team_size"])

    # ── 4. Binary threshold flags ────────────────────────────────────────────
    # Burnout research identifies specific thresholds, not linear gradients.
    df["overworked"] = (df["work_hours_per_week"] > 50).astype(int)
    df["sleep_deprived"] = (df["sleep_hours_per_night"] < 6).astype(int)
    df["no_vacation"] = (df["vacation_days_taken"] < 5).astype(int)
    df["high_meetings"] = (df["meetings_per_day"] > 7).astype(int)
    df["no_exercise"] = (df["exercise_days_per_week"] == 0).astype(int)

    # ── 5. Polynomial interaction terms ─────────────────────────────────────
    # Hand-picked cross-products of the top predictors. Multiplying two
    # features creates a nonlinear term that helps logistic regression and
    # gives tree models a pre-computed axis to split on.
    df["stress_x_hours"] = df["stress_score"] * df["work_hours_per_week"]
    df["stress_x_sleep_inv"] = df["stress_score"] / sleep_safe
    df["pressure_x_imbalance"] = (
        df["deadline_pressure_score"] * (11.0 - df["work_life_balance_score"])
    )
    df["stress_x_no_support"] = (
        df["stress_score"] * (11.0 - df["manager_support_score"])
    )

    return df
