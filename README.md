# CreditGuard

End-to-end Credit Risk & Loan Delinquency Prediction system for banking.

## Problem

Predict whether a loan applicant will default (delinquent) using structured banking data. The system compares three classifiers, selects an operating threshold from the precision-recall curve for delinquent-borrower detection, and serves predictions via a REST API.

## Dataset

**Home Credit Default Risk** — Kaggle multi-table dataset:
- `application_train` — main application (target = TARGET)
- `bureau` — previous credits at other agencies
- `bureau_balance` — monthly bureau snapshots
- `previous_application` — previous Home Credit applications
- `installments_payment` — payment history

Default rate ≈ 8%. Not included in the repo — download with:

```bash
python -m src.data.download   # uses kagglehub, no API key needed
```

Or place the 7 CSVs manually in `data/raw/`.

## Architecture

```
PostgreSQL (raw tables)
    │
    ▼  SQL feature-engineering views (sql/feature_engineering.sql)
v_feature_matrix (bureau + prev_app + installments aggregates)
    │
    ▼  src/features/engineer.py (encoding + imputation)
Numeric + encoded categorical matrix
    │
    ▼  src/modeling/train.py (SMOTE + 3 models)
saved sklearn Pipeline → models/creditguard_model.pkl
    │
    ▼  src/api/app.py (FastAPI)
POST /predict → {default_probability, risk_score, risk_band}
```

## Setup

```bash
# 1. PostgreSQL running locally
# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python -m src.data.download

# 4. Ingest into DB
python -m src.data.ingest

# 5. Run SQL feature engineering
psql $DATABASE_URL -f sql/feature_engineering.sql

# 6. Train models
python -m src.modeling.train

# 7. Run API
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

## API

`POST /predict` — JSON body with borrower features (see `src/api/app.py` for full schema).

Response:
```json
{
  "default_probability": 0.12,
  "risk_score": 34,
  "risk_band": "Low",
  "threshold_used": 0.3125
}
```

`GET /health` — liveness check.

## Models compared

1. **Logistic Regression** — baseline, interpretable
2. **Random Forest** — non-linear, robust to outliers
3. **XGBoost** — gradient boosting, best for tabular data

Class imbalance handled via SMOTE (training) + `class_weight` (all models). Primary metric: **PR-AUC**.

## Risk score & bands

| Score | Band   |
|-------|--------|
| 0–40  | Low    |
| 41–70 | Medium |
| 71–100| High   |

## Testing

```bash
pytest -q
```

Tests use synthetic data — no DB or dataset download needed.

## Docker

```bash
docker compose up --build
# API at http://localhost:8000
# DB at localhost:5432
```