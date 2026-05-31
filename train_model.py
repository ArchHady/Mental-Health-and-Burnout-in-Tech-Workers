"""
Mental Health Burnout — training pipeline.

Reads mental_health_burnout_tech_2026.csv, runs the full data-science workflow
(missing values -> duplicates -> univariate analysis -> outlier handling ->
feature engineering -> preprocessing -> model selection -> evaluation), and
saves all artifacts the Streamlit dashboard needs.

Run:
    python train_model.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from feature_engineering import (
    MODEL_CATEGORICAL_COLS,
    RAW_CATEGORICAL_COLS,
    engineer_features,
)
from models import SoftVotingEnsemble

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "mental_health_burnout_tech_2026.csv"
ART_DIR = HERE / "artifacts"
ART_DIR.mkdir(exist_ok=True)

TARGET = "burnout_level"
# Columns that leak the target or are not legitimate inputs.
LEAKAGE_COLS = [
    "burnout_score",
    "phq9_score",
    "phq9_category",
    "gad7_score",
    "gad7_category",
    "seeks_mental_health_support",
    "job_change_intention",
]
ID_COLS = ["employee_id"]

# Categorical columns split by cardinality for the preprocessor.
# Low-cardinality → OneHotEncoder; high-cardinality → TargetEncoder.
LOW_CARD_CATS: list[str] = ["gender", "work_mode"]
HIGH_CARD_CATS: list[str] = ["country", "job_role", "industry"]

# Columns where IQR-based outlier detection is meaningful.
IQR_COLS = [
    "salary_usd",
    "work_hours_per_week",
    "meetings_per_day",
    "team_size",
    "sleep_hours_per_night",
    "vacation_days_taken",
    "years_experience",
    "years_at_company",
]

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    print(f"[1/9] Loading {CSV_PATH.name} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"      Loaded shape: {df.shape}")
    return df


def report_missing(df: pd.DataFrame) -> None:
    print("[2/9] Missing value check ...")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("      No missing values found (imputers still kept in pipeline as safety net).")
    else:
        print(miss.to_string())


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/9] Dropping duplicates ...")
    before = len(df)
    # Ignore employee_id when checking content-level duplicates.
    deduped = df.drop_duplicates(subset=[c for c in df.columns if c not in ID_COLS])
    after = len(deduped)
    print(f"      {before} -> {after} rows ({before - after} duplicates removed)")
    return deduped.reset_index(drop=True)


def univariate_analysis(df: pd.DataFrame) -> dict:
    print("[4/9] Univariate analysis ...")
    report: dict = {"numeric": {}, "categorical": {}, "target": {}}

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        s = df[col]
        report["numeric"][col] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "skew": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
        }

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        vc = df[col].value_counts().head(10)
        report["categorical"][col] = {
            "cardinality": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.items()},
        }

    tgt = df[TARGET].value_counts()
    report["target"] = {str(k): int(v) for k, v in tgt.items()}

    (ART_DIR / "univariate.json").write_text(json.dumps(report, indent=2))
    print(f"      Saved {ART_DIR / 'univariate.json'}")
    return report


def detect_outliers_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag rows that are IQR outliers in >= 2 of the IQR_COLS and drop them.

    Returns the cleaned DataFrame plus the per-column bounds (saved for the
    dashboard so it can warn on out-of-range inputs).
    """
    print("[5/9] Outlier detection (IQR, multi-column rule) ...")
    bounds: dict[str, dict] = {}
    flag_matrix = np.zeros((len(df), len(IQR_COLS)), dtype=bool)
    for j, col in enumerate(IQR_COLS):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bounds[col] = {"low": float(low), "high": float(high), "q1": float(q1), "q3": float(q3)}
        flag_matrix[:, j] = (df[col] < low) | (df[col] > high)

    flagged_count = flag_matrix.sum(axis=1)
    drop_mask = flagged_count >= 2
    cleaned = df.loc[~drop_mask].reset_index(drop=True)
    print(
        f"      Dropped {int(drop_mask.sum())} rows "
        f"({drop_mask.mean():.2%}) flagged as outliers in >=2 columns. "
        f"Remaining: {len(cleaned)}"
    )
    return cleaned, bounds


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build a ColumnTransformer with three branches:

    - Numeric  → median impute → StandardScaler
    - Low-cardinality categorical (gender, work_mode) → mode impute → OneHotEncoder
    - High-cardinality categorical (country, job_role, industry) → mode impute →
      TargetEncoder(multiclass), which encodes each category as a 4-dim probability
      vector (one per burnout class) — far denser signal than 30+ OHE dummy columns.
    """
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    ohe_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    target_enc_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "target_enc",
                TargetEncoder(
                    target_type="multiclass",
                    smooth="auto",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    low_card  = [c for c in categorical_cols if c in LOW_CARD_CATS]
    high_card = [c for c in categorical_cols if c in HIGH_CARD_CATS]

    transformers = [("num", numeric_pipe, numeric_cols)]
    if low_card:
        transformers.append(("ohe", ohe_pipe, low_card))
    if high_card:
        transformers.append(("target", target_enc_pipe, high_card))

    return ColumnTransformer(transformers)



def train_and_select(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[str, object, dict]:
    """Fit LR / HGB / XGBoost plus a soft-voting ensemble; return the best."""
    print("[7/9] Training & comparing models ...")

    # Pre-compute balanced sample weights for XGBoost.
    sample_weights = compute_sample_weight("balanced", y_train)

    candidates = {
        # C=3.0: less L2 regularisation now that we have 39 engineered features.
        "LogisticRegression": LogisticRegression(
            max_iter=2000, n_jobs=-1, C=3.0, class_weight="balanced"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        # XGBoost has no class_weight param — pass sample_weight at fit time.
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    leaderboard: dict[str, dict] = {}
    fitted_pipes: dict[str, Pipeline] = {}
    best_name, best_obj, best_f1 = None, None, -1.0

    # ── Train three individual models ────────────────────────────────────────
    for name, clf in candidates.items():
        pre = build_preprocessor(numeric_cols, categorical_cols)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        t0 = time.time()
        if name == "XGBoost":
            pipe.fit(X_train, y_train, clf__sample_weight=sample_weights)
        else:
            pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1  = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        elapsed = time.time() - t0
        leaderboard[name] = {"macro_f1": float(f1), "accuracy": float(acc), "fit_seconds": elapsed}
        fitted_pipes[name] = pipe
        print(f"      {name:22s}  acc={acc:.3f}  macro-F1={f1:.3f}  ({elapsed:.1f}s)")
        if f1 > best_f1:
            best_f1, best_name, best_obj = f1, name, pipe

    # ── Soft-voting ensemble (averages probabilities, no re-training) ────────
    print("      Building soft-voting ensemble ...")
    t0 = time.time()
    ensemble = SoftVotingEnsemble(list(fitted_pipes.values()))
    ens_probas = ensemble.predict_proba(X_test)
    ens_preds  = np.argmax(ens_probas, axis=1)
    ens_f1  = f1_score(y_test, ens_preds, average="macro")
    ens_acc = accuracy_score(y_test, ens_preds)
    elapsed = time.time() - t0
    leaderboard["Ensemble(LR+HGB+XGB)"] = {
        "macro_f1": float(ens_f1),
        "accuracy": float(ens_acc),
        "fit_seconds": elapsed,
    }
    print(f"      {'Ensemble(LR+HGB+XGB)':22s}  acc={ens_acc:.3f}  macro-F1={ens_f1:.3f}  ({elapsed:.1f}s)")
    if ens_f1 > best_f1:
        best_f1, best_name, best_obj = ens_f1, "Ensemble(LR+HGB+XGB)", ensemble

    print(f"      Winner: {best_name} (macro-F1={best_f1:.3f})")
    return best_name, best_obj, leaderboard


def evaluate(
    model: object,          # Pipeline OR SoftVotingEnsemble
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> dict:
    print("[8/9] Evaluating winner & computing permutation importance ...")
    preds = model.predict(X_test)
    acc         = accuracy_score(y_test, preds)
    macro_f1    = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")
    report = classification_report(
        y_test, preds, target_names=le.classes_.tolist(), output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, preds).tolist()

    # Permutation importance — uses sklearn's utility for Pipelines; for the
    # ensemble we use the first sub-pipeline as a representative proxy.
    sample_size = min(3000, len(X_test))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=sample_size, replace=False)
    perm_model = model.pipelines[0] if isinstance(model, SoftVotingEnsemble) else model
    perm = permutation_importance(
        perm_model,
        X_test.iloc[idx],
        y_test[idx],
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="f1_macro",
    )
    importance = sorted(
        [
            {"feature": col, "importance": float(m), "std": float(s)}
            for col, m, s in zip(X_test.columns, perm.importances_mean, perm.importances_std)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm,
        "class_labels": le.classes_.tolist(),
        "permutation_importance": importance,
    }


def build_feature_meta(
    raw_df: pd.DataFrame,
    eng_df: pd.DataFrame,
    model_numeric_cols: list[str],
    model_categorical_cols: list[str],
    raw_numeric_cols: list[str],
    iqr_bounds: dict,
) -> dict:
    """Build metadata the dashboard uses for both the prediction form and EDA.

    ``raw_df``  — original dataset (before engineering) — used for form ranges/options.
    ``eng_df``  — engineered dataset — used to compute ranges for derived features.
    ``feature_order`` is the exact column sequence the saved model expects.
    ``numeric`` and ``categorical`` describe the *raw* inputs the prediction form shows.
    """
    meta: dict = {
        "numeric": {},
        "categorical": {},
        "derived_numeric": {},
        "iqr_bounds": iqr_bounds,
        "feature_order": model_numeric_cols + model_categorical_cols,
    }

    # Raw numeric — shown as sliders in the prediction form.
    for col in raw_numeric_cols:
        s = raw_df[col]
        meta["numeric"][col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "is_integer": bool(pd.api.types.is_integer_dtype(s)),
        }

    # Raw categorical — shown as dropdowns in the prediction form.
    for col in RAW_CATEGORICAL_COLS:
        meta["categorical"][col] = sorted(raw_df[col].dropna().unique().tolist())

    # Derived numeric — informational only (not shown in form).
    derived_cols = [c for c in model_numeric_cols if c not in raw_numeric_cols]
    for col in derived_cols:
        s = eng_df[col]
        meta["derived_numeric"][col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
        }

    return meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_data()
    report_missing(df)
    df = drop_duplicates(df)
    univariate_analysis(df)

    df, iqr_bounds = detect_outliers_iqr(df)

    # Step 6: feature engineering (row-wise, no leakage risk).
    print("[6/9] Feature engineering ...")
    raw_df = df.copy()  # keep original for feature_meta ranges / form options
    eng_df = engineer_features(df)
    derived_names = [c for c in eng_df.columns if c not in df.columns]
    print(f"      Added {len(derived_names)} engineered features: {derived_names}")

    # Build X / y from engineered data.
    drop_all = LEAKAGE_COLS + ID_COLS + [TARGET] + RAW_CATEGORICAL_COLS
    # RAW_CATEGORICAL_COLS includes seniority_level & company_size, which are now
    # represented as seniority_ord / company_size_ord in the numeric block.
    feature_df = eng_df.drop(columns=drop_all)
    # Re-add the 5 categorical columns that stay as strings.
    feature_df = pd.concat(
        [feature_df, eng_df[MODEL_CATEGORICAL_COLS]], axis=1
    )

    raw_numeric_cols = [
        c for c in raw_df.drop(columns=LEAKAGE_COLS + ID_COLS + [TARGET]).columns
        if c not in RAW_CATEGORICAL_COLS
    ]
    model_numeric_cols = [c for c in feature_df.columns if c not in MODEL_CATEGORICAL_COLS]

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    best_name, best_pipe, leaderboard = train_and_select(
        X_train, y_train, X_test, y_test, model_numeric_cols, MODEL_CATEGORICAL_COLS
    )

    metrics = evaluate(best_pipe, X_test, y_test, le)
    metrics["winning_model"] = best_name
    metrics["leaderboard"] = leaderboard

    feature_meta = build_feature_meta(
        raw_df, eng_df, model_numeric_cols, MODEL_CATEGORICAL_COLS,
        raw_numeric_cols, iqr_bounds,
    )

    print("[9/9] Saving artifacts ...")
    joblib.dump(best_pipe, ART_DIR / "model.joblib")
    joblib.dump(le, ART_DIR / "label_encoder.joblib")
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ART_DIR / "feature_meta.json").write_text(json.dumps(feature_meta, indent=2))
    print(f"      Saved to {ART_DIR}/")

    print("\n=== Summary ===")
    print(f"Winning model : {best_name}")
    print(f"Accuracy      : {metrics['accuracy']:.3f}")
    print(f"Macro F1      : {metrics['macro_f1']:.3f}")
    print(f"Weighted F1   : {metrics['weighted_f1']:.3f}")
    print("Top 5 features by permutation importance:")
    for row in metrics["permutation_importance"][:5]:
        print(f"  {row['feature']:32s}  {row['importance']:+.4f}")


if __name__ == "__main__":
    main()
