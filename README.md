# Tech Burnout — ML Pipeline + Streamlit Dashboard

End-to-end machine learning project on the `mental_health_burnout_tech_2026.csv`
dataset (100k tech-worker records). Predicts **burnout level**
(Low / Moderate / High / Severe) and serves the model through an interactive
Streamlit dashboard.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (writes artifacts/)
python train_model.py

# 3. Launch the dashboard
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>.

## Files

| File | Purpose |
| --- | --- |
| `mental_health_burnout_tech_2026.csv` | Source dataset (100k rows). |
| `train_model.py` | Full pipeline: load → missing-value handling → drop duplicates → univariate analysis → IQR outlier detection → preprocessing → train+compare three models → evaluate winner → save artifacts. |
| `app.py` | 4-page Streamlit dashboard (Overview, EDA, Model Performance, Predict). |
| `requirements.txt` | Pinned dependencies. |
| `artifacts/` | Produced by `train_model.py`. |

## Pipeline steps (in `train_model.py`)

1. **Load** the CSV.
2. **Missing value handling** — printed report; `SimpleImputer` (median / most-frequent) baked into the saved pipeline.
3. **Drop duplicates** ignoring `employee_id`.
4. **Univariate analysis** — numeric stats (mean, std, skew, kurtosis), categorical value counts, target balance → `artifacts/univariate.json`.
5. **Outlier detection** — IQR rule on 8 unbounded numeric columns; rows flagged in **≥2** columns are dropped (\~0.4% of data). Bounds saved for the dashboard.
6. **Preprocessing** — `ColumnTransformer` with `SimpleImputer → StandardScaler` for numeric and `SimpleImputer → OneHotEncoder` for categorical features.
7. **Model selection** — train and compare `LogisticRegression`, `RandomForestClassifier`, and `HistGradientBoostingClassifier` on an 80/20 stratified split. Winner is chosen by macro-F1.
8. **Evaluation** — accuracy, macro-F1, weighted-F1, per-class report, confusion matrix, and permutation importance.

Features dropped before training (to prevent target leakage): `burnout_score`,
`phq9_score`, `phq9_category`, `gad7_score`, `gad7_category`,
`seeks_mental_health_support`, `job_change_intention`, `employee_id`.

## Dashboard pages

- **Overview** — KPIs, sidebar filters, filtered table, CSV download.
- **EDA** — burnout distribution, group breakdowns, histograms, correlation heatmap, box plots.
- **Model Performance** — leaderboard, confusion matrix, per-class metrics, top-15 permutation importance.
- **Predict** — dynamic form (sliders + dropdowns) built from `feature_meta.json`; outputs predicted class with probability bar chart and out-of-range warnings.
