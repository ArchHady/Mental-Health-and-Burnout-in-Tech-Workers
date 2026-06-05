"""
Mental Health Burnout — training pipeline.

────────────────────────────────────────────────────────────────────
USER CONFIGURATION  (edit the block below, then run: python train_model.py)
────────────────────────────────────────────────────────────────────
  FEATURE_COLS  – list the column names you want as model inputs, or
                  leave as None to auto-select everything that is not
                  the target or a known leakage column.

  TARGET        – the column the model should predict.

  N_ITER_SEARCH – number of random hyperparameter combinations tried
                  for each model (more = better params, longer runtime;
                  10 iterations ≈ 5-8 min on a modern laptop).

  CV_FOLDS      – number of cross-validation folds (5 is standard).
────────────────────────────────────────────────────────────────────
The pipeline handles everything internally:
  data cleaning -> feature engineering -> train/test split ->
  RandomizedSearchCV with StratifiedKFold CV ->
  soft-voting ensemble -> evaluation -> save artefacts
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
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

# ══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION — change these, leave everything else as-is
# ══════════════════════════════════════════════════════════════════════════════

# Set to a list of column names, e.g. ['stress_score', 'work_hours_per_week'],
# or None to automatically use all non-leakage, non-target columns.
FEATURE_COLS: list[str] | None = None

TARGET = "burnout_level"

# Tuning budget (more iterations = better model, longer runtime).
N_ITER_SEARCH = 10   # random hyperparameter combinations per model
CV_FOLDS      = 5    # stratified k-fold splits

# ══════════════════════════════════════════════════════════════════════════════
# Internal config (no need to change)
# ══════════════════════════════════════════════════════════════════════════════
HERE      = Path(__file__).resolve().parent
CSV_PATH  = HERE / "mental_health_burnout_tech_2026.csv"
ART_DIR   = HERE / "artifacts"
ART_DIR.mkdir(exist_ok=True)

LEAKAGE_COLS = [
    "burnout_score", "phq9_score", "phq9_category",
    "gad7_score", "gad7_category",
    "seeks_mental_health_support", "job_change_intention",
]
ID_COLS = ["employee_id"]

LOW_CARD_CATS  = ["gender", "work_mode"]
HIGH_CARD_CATS = ["country", "job_role", "industry"]

IQR_COLS = [
    "salary_usd", "work_hours_per_week", "meetings_per_day", "team_size",
    "sleep_hours_per_night", "vacation_days_taken", "years_experience", "years_at_company",
]

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Step helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"[1/9] Loading {CSV_PATH.name} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"      Shape: {df.shape}")
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
    deduped = df.drop_duplicates(subset=[c for c in df.columns if c not in ID_COLS])
    after = len(deduped)
    print(f"      {before:,} -> {after:,} rows  ({before - after} duplicates removed)")
    return deduped.reset_index(drop=True)


def univariate_analysis(df: pd.DataFrame) -> dict:
    print("[4/9] Univariate analysis ...")
    report: dict = {"numeric": {}, "categorical": {}, "target": {}}
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col]
        report["numeric"][col] = {
            "count": int(s.count()), "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "q25": float(s.quantile(0.25)),
            "median": float(s.median()), "q75": float(s.quantile(0.75)),
            "max": float(s.max()), "skew": float(s.skew()), "kurtosis": float(s.kurtosis()),
        }
    for col in df.select_dtypes(include="object").columns:
        vc = df[col].value_counts().head(10)
        report["categorical"][col] = {
            "cardinality": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.items()},
        }
    report["target"] = {str(k): int(v) for k, v in df[TARGET].value_counts().items()}
    (ART_DIR / "univariate.json").write_text(json.dumps(report, indent=2))
    print(f"      Saved univariate.json")
    return report


def detect_outliers_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("[5/9] Outlier detection (IQR, >=2-column rule) ...")
    bounds: dict = {}
    flag_mat = np.zeros((len(df), len(IQR_COLS)), dtype=bool)
    for j, col in enumerate(IQR_COLS):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bounds[col] = {"low": float(lo), "high": float(hi), "q1": float(q1), "q3": float(q3)}
        flag_mat[:, j] = (df[col] < lo) | (df[col] > hi)
    drop_mask = flag_mat.sum(axis=1) >= 2
    cleaned = df.loc[~drop_mask].reset_index(drop=True)
    print(f"      Dropped {int(drop_mask.sum()):,} rows ({drop_mask.mean():.2%}) -> {len(cleaned):,} remaining")
    return cleaned, bounds


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Three-branch transformer: numeric / OHE (low-card) / TargetEncoder (high-card)."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    ohe_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    target_enc_pipe = Pipeline([
        ("impute",     SimpleImputer(strategy="most_frequent")),
        ("target_enc", TargetEncoder(target_type="multiclass", smooth="auto",
                                      random_state=RANDOM_STATE)),
    ])
    low_card  = [c for c in categorical_cols if c in LOW_CARD_CATS]
    high_card = [c for c in categorical_cols if c in HIGH_CARD_CATS]
    transformers = [("num", numeric_pipe, numeric_cols)]
    if low_card:  transformers.append(("ohe",    ohe_pipe,        low_card))
    if high_card: transformers.append(("target", target_enc_pipe, high_card))
    return ColumnTransformer(transformers)


def tune_and_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test:  pd.DataFrame,
    y_test:  np.ndarray,
    model_numeric_cols: list[str],
    model_categorical_cols: list[str],
) -> tuple[str, object, dict, dict]:
    """
    Run RandomizedSearchCV on HGB and XGBoost, then build a soft-voting ensemble.

    Returns
    -------
    best_name  : name of the winning model/ensemble
    best_obj   : fitted model / SoftVotingEnsemble
    leaderboard: holdout metrics for each candidate
    tuning_info: CV scores, best params, search results for the dashboard
    """
    print(f"[7/9] Hyperparameter tuning  "
          f"({N_ITER_SEARCH} iterations × {CV_FOLDS}-fold StratifiedKFold) ...")

    cv             = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sample_weights = compute_sample_weight("balanced", y_train)

    # ── HistGradientBoosting ──────────────────────────────────────────────────
    print("      [1/3] Tuning HistGradientBoosting ...")
    t0 = time.time()
    hgb_search = RandomizedSearchCV(
        Pipeline([
            ("pre", build_preprocessor(model_numeric_cols, model_categorical_cols)),
            ("clf", HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        param_distributions={
            "clf__max_iter":          randint(200, 600),
            "clf__learning_rate":     loguniform(0.02, 0.20),
            "clf__max_depth":         [4, 5, 6, 7, 8],
            "clf__min_samples_leaf":  randint(10, 60),
            "clf__l2_regularization": loguniform(0.001, 2.0),
        },
        n_iter=N_ITER_SEARCH, cv=cv, scoring="f1_macro",
        n_jobs=-1, random_state=RANDOM_STATE, refit=True, verbose=0,
    )
    hgb_search.fit(X_train, y_train)
    print(f"      HGB best CV macro-F1: {hgb_search.best_score_:.3f}  ({time.time()-t0:.0f}s)")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("      [2/3] Tuning XGBoost ...")
    t0 = time.time()
    xgb_search = RandomizedSearchCV(
        Pipeline([
            ("pre", build_preprocessor(model_numeric_cols, model_categorical_cols)),
            ("clf", XGBClassifier(
                eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        param_distributions={
            "clf__n_estimators":      randint(200, 500),
            "clf__learning_rate":     loguniform(0.02, 0.15),
            "clf__max_depth":         randint(4, 9),
            "clf__subsample":         [0.6, 0.7, 0.8, 0.9],
            "clf__colsample_bytree":  [0.6, 0.7, 0.8, 0.9],
            "clf__reg_alpha":         loguniform(0.001, 1.0),
        },
        n_iter=N_ITER_SEARCH, cv=cv, scoring="f1_macro",
        n_jobs=-1, random_state=RANDOM_STATE, refit=True, verbose=0,
    )
    xgb_search.fit(X_train, y_train, clf__sample_weight=sample_weights)
    print(f"      XGB best CV macro-F1: {xgb_search.best_score_:.3f}  ({time.time()-t0:.0f}s)")

    # ── Baseline LogisticRegression (fast, no tuning needed) ──────────────────
    print("      [3/3] Fitting baseline LogisticRegression ...")
    t0 = time.time()
    lr_pipe = Pipeline([
        ("pre", build_preprocessor(model_numeric_cols, model_categorical_cols)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, C=3.0, class_weight="balanced")),
    ])
    lr_pipe.fit(X_train, y_train)
    print(f"      LR done  ({time.time()-t0:.1f}s)")

    # ── Holdout evaluation for each candidate ─────────────────────────────────
    fitted_pipes = {
        "LogisticRegression":  lr_pipe,
        "HistGradientBoosting": hgb_search.best_estimator_,
        "XGBoost":             xgb_search.best_estimator_,
    }
    leaderboard: dict = {}
    best_name, best_obj, best_f1 = None, None, -1.0

    for name, pipe in fitted_pipes.items():
        preds = pipe.predict(X_test)
        f1  = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        leaderboard[name] = {"macro_f1": float(f1), "accuracy": float(acc)}
        print(f"      {name:<24s}  acc={acc:.3f}  macro-F1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_name, best_obj = f1, name, pipe

    # ── Soft-voting ensemble ──────────────────────────────────────────────────
    ensemble  = SoftVotingEnsemble(list(fitted_pipes.values()))
    ens_preds = ensemble.predict(X_test)
    ens_f1    = f1_score(y_test, ens_preds, average="macro")
    ens_acc   = accuracy_score(y_test, ens_preds)
    leaderboard["Ensemble(LR+HGB+XGB)"] = {
        "macro_f1": float(ens_f1), "accuracy": float(ens_acc)}
    print(f"      {'Ensemble(LR+HGB+XGB)':<24s}  acc={ens_acc:.3f}  macro-F1={ens_f1:.3f}")
    if ens_f1 > best_f1:
        best_f1, best_name, best_obj = ens_f1, "Ensemble(LR+HGB+XGB)", ensemble
    print(f"      Winner: {best_name}  (macro-F1={best_f1:.3f})")

    # ── CV fold scores (stability check on the best individual model) ─────────
    best_ind_name = max(
        [(n, p) for n, p in fitted_pipes.items()],
        key=lambda x: leaderboard[x[0]]["macro_f1"],
    )[0]
    best_ind_pipe = fitted_pipes[best_ind_name]
    cv_fold_scores = cross_val_score(
        best_ind_pipe, X_train, y_train,
        cv=cv, scoring="f1_macro", n_jobs=-1,
    )
    print(f"      CV fold scores ({best_ind_name}): "
          f"{[f'{s:.3f}' for s in cv_fold_scores]}  "
          f"mean={cv_fold_scores.mean():.3f} +/- {cv_fold_scores.std():.3f}")

    # ── Package tuning info for metrics.json ──────────────────────────────────
    def _clean_params(params: dict) -> dict:
        """Convert numpy types to plain Python so json.dumps works."""
        out = {}
        for k, v in params.items():
            key = k.replace("clf__", "")
            out[key] = float(v) if hasattr(v, "item") else v
        return out

    tuning_info = {
        "cv_fold_scores":       [float(s) for s in cv_fold_scores],
        "cv_mean":              float(cv_fold_scores.mean()),
        "cv_std":               float(cv_fold_scores.std()),
        "cv_model":             best_ind_name,
        "best_params": {
            "HistGradientBoosting": _clean_params(hgb_search.best_params_),
            "XGBoost":             _clean_params(xgb_search.best_params_),
        },
        "search_scores": {
            "HistGradientBoosting": {
                "mean_cv_scores": [float(v) for v in hgb_search.cv_results_["mean_test_score"]],
                "std_cv_scores":  [float(v) for v in hgb_search.cv_results_["std_test_score"]],
            },
            "XGBoost": {
                "mean_cv_scores": [float(v) for v in xgb_search.cv_results_["mean_test_score"]],
                "std_cv_scores":  [float(v) for v in xgb_search.cv_results_["std_test_score"]],
            },
        },
    }
    return best_name, best_obj, leaderboard, tuning_info


def evaluate(
    model: object,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> dict:
    print("[8/9] Evaluating winner & computing permutation importance ...")
    preds       = model.predict(X_test)
    acc         = accuracy_score(y_test, preds)
    macro_f1    = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")
    report      = classification_report(
        y_test, preds, target_names=le.classes_.tolist(),
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, preds).tolist()

    # Permutation importance (use first sub-pipeline for ensemble).
    sample_size = min(3000, len(X_test))
    idx         = np.random.RandomState(RANDOM_STATE).choice(
        len(X_test), size=sample_size, replace=False)
    perm_model  = model.pipelines[0] if isinstance(model, SoftVotingEnsemble) else model
    perm = permutation_importance(
        perm_model, X_test.iloc[idx], y_test[idx],
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=1, scoring="f1_macro",
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
        "accuracy":               float(acc),
        "macro_f1":               float(macro_f1),
        "weighted_f1":            float(weighted_f1),
        "classification_report":  report,
        "confusion_matrix":       cm,
        "class_labels":           le.classes_.tolist(),
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
    meta: dict = {
        "numeric":        {},
        "categorical":    {},
        "derived_numeric":{},
        "iqr_bounds":     iqr_bounds,
        "feature_order":  model_numeric_cols + model_categorical_cols,
    }
    for col in raw_numeric_cols:
        s = raw_df[col]
        meta["numeric"][col] = {
            "min":        float(s.min()),
            "max":        float(s.max()),
            "median":     float(s.median()),
            "is_integer": bool(pd.api.types.is_integer_dtype(s)),
        }
    for col in RAW_CATEGORICAL_COLS:
        meta["categorical"][col] = sorted(raw_df[col].dropna().unique().tolist())
    for col in [c for c in model_numeric_cols if c not in raw_numeric_cols]:
        s = eng_df[col]
        meta["derived_numeric"][col] = {
            "min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
        }
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_data()
    report_missing(df)
    df = drop_duplicates(df)
    univariate_analysis(df)
    df, iqr_bounds = detect_outliers_iqr(df)

    # ── Step 6: feature engineering ───────────────────────────────────────────
    print("[6/9] Feature engineering ...")
    raw_df = df.copy()
    eng_df = engineer_features(df)
    derived = [c for c in eng_df.columns if c not in df.columns]
    print(f"      Added {len(derived)} engineered features")

    # ── Step 7: build feature matrix from user-specified columns ──────────────
    # If FEATURE_COLS is None, auto-select all non-leakage, non-target columns.
    if FEATURE_COLS is None:
        auto_raw = [
            c for c in raw_df.columns
            if c not in LEAKAGE_COLS + ID_COLS + [TARGET]
        ]
        print(f"      FEATURE_COLS=None -> auto-selected {len(auto_raw)} raw columns")
    else:
        missing = [c for c in FEATURE_COLS if c not in raw_df.columns]
        if missing:
            raise ValueError(f"FEATURE_COLS contains columns not in dataset: {missing}")
        auto_raw = FEATURE_COLS
        print(f"      Using {len(auto_raw)} user-specified columns")

    # Determine which raw columns are categorical (will stay as strings)
    raw_cats_in_use = [c for c in auto_raw if c in RAW_CATEGORICAL_COLS]
    raw_nums_in_use = [c for c in auto_raw if c not in RAW_CATEGORICAL_COLS]

    # Decide which engineered features to include:
    # always add derived columns (ordinal / composites / flags based on the
    # raw columns that ARE in use).
    always_derived = [c for c in derived]  # all 19 are row-wise & safe

    # Build feature_df: all eng_df columns EXCEPT leakage / ID / target /
    # raw categoricals (represented via ordinal or retained as MODEL_CATEGORICAL_COLS)
    drop_all   = LEAKAGE_COLS + ID_COLS + [TARGET] + RAW_CATEGORICAL_COLS
    feature_df = eng_df.drop(columns=drop_all)

    # Restrict to user-selected raw numeric cols + always keep engineered cols
    eng_numeric = [c for c in always_derived if c in feature_df.columns]
    raw_numeric_present = [c for c in raw_nums_in_use if c in feature_df.columns]
    feature_df = feature_df[raw_numeric_present + eng_numeric]

    # Re-attach the 5 model categoricals that stay as strings
    model_cats_in_use = [c for c in MODEL_CATEGORICAL_COLS if c in raw_cats_in_use]
    if not model_cats_in_use:
        # If user didn't include any categoricals, add them back anyway — they carry signal
        model_cats_in_use = MODEL_CATEGORICAL_COLS
    feature_df = pd.concat([feature_df, eng_df[model_cats_in_use]], axis=1)

    model_numeric_cols   = [c for c in feature_df.columns if c not in model_cats_in_use]
    raw_numeric_cols_out = [c for c in model_numeric_cols if c not in always_derived]

    print(f"      Feature matrix: {feature_df.shape[1]} columns "
          f"({len(model_numeric_cols)} numeric + {len(model_cats_in_use)} categorical)")

    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}  "
          f"(stratified 80/20, seed={RANDOM_STATE})")

    # ── Step 7 continued: tune + train ────────────────────────────────────────
    best_name, best_obj, leaderboard, tuning_info = tune_and_train(
        X_train, y_train, X_test, y_test,
        model_numeric_cols, model_cats_in_use,
    )

    # ── Step 8: evaluate ──────────────────────────────────────────────────────
    metrics = evaluate(best_obj, X_test, y_test, le)
    metrics.update({
        "winning_model": best_name,
        "leaderboard":   leaderboard,
        **tuning_info,       # cv_fold_scores, cv_mean, cv_std, best_params, search_scores
    })

    # ── Step 9: save artefacts ─────────────────────────────────────────────────
    print("[9/9] Saving artefacts ...")
    feature_meta = build_feature_meta(
        raw_df, eng_df, model_numeric_cols, model_cats_in_use,
        raw_numeric_cols_out, iqr_bounds,
    )
    joblib.dump(best_obj, ART_DIR / "model.joblib")
    joblib.dump(le,       ART_DIR / "label_encoder.joblib")
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ART_DIR / "feature_meta.json").write_text(json.dumps(feature_meta, indent=2))
    print(f"      Saved to {ART_DIR}/")

    print("\n=== Summary ===")
    print(f"Winning model  : {best_name}")
    print(f"Accuracy       : {metrics['accuracy']:.3f}")
    print(f"Macro F1       : {metrics['macro_f1']:.3f}")
    print(f"Weighted F1    : {metrics['weighted_f1']:.3f}")
    print(f"CV mean F1     : {metrics['cv_mean']:.3f} +/- {metrics['cv_std']:.3f}  "
          f"({CV_FOLDS} folds, {metrics['cv_model']})")
    print("Top 5 features by permutation importance:")
    for row in metrics["permutation_importance"][:5]:
        print(f"  {row['feature']:32s}  {row['importance']:+.4f}")


if __name__ == "__main__":
    main()
