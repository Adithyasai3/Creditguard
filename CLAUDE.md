# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in the CreditGuard repository.

CreditGuard is an end-to-end Credit Risk & Loan Delinquency Prediction system. It ingests customer/loan data from PostgreSQL, engineers features via SQL, trains and compares three classifiers (Logistic Regression, Random Forest, XGBoost), selects an operating threshold from the precision-recall curve for delinquent-borrower detection, and exposes predictions through a FastAPI REST API.

## Project layout

```
creditguard/
├── sql/                      # PostgreSQL schema + SQL feature-engineering views
├── data/raw/                 # Downloaded source datasets (not committed)
├── data/processed/           # Engineered feature matrices exported from the DB
├── models/                   # Saved sklearn Pipeline artifact (.pkl)
├── src/
│   ├── data/                 # Dataset download, DB ingest, feature loading
│   ├── features/             # SQL feature engineering + Pandas preprocessing
│   ├── modeling/             # Training, evaluation, threshold selection
│   ├── api/                  # FastAPI app + Pydantic schemas
│   └── utils/                # Config, DB connection helpers
└── tests/                    # pytest suite
```

## Commands

Install (after `pip install -r requirements.txt`):

```bash
# Download the dataset (see Data note below)
python -m src.data.download

# Load raw tables into PostgreSQL
python -m src.data.ingest

# Run SQL feature-engineering views
psql $DATABASE_URL -f sql/feature_engineering.sql

# Export the engineered feature matrix
python -m src.data.load_features

# Train + evaluate all three models
python -m src.modeling.train

# Run the test suite
pytest -q
```

Run the API:

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
# or via Docker: docker compose up
```

API docs are at `http://localhost:8000/docs` (OpenAPI/Swagger).

## Data note

The project uses the **Home Credit Default Risk** dataset (Kaggle), a multi-table banking dataset ideal for SQL-based feature engineering across applications, bureau, previous loans, and installment payments. It is NOT committed to the repo. Provide it by running `python -m src.data.download` (uses `kagglehub`, no API key needed) or by placing the 7 CSVs in `data/raw/`. The script tells you exactly what it needs.

## Key conventions

- **No leakage**: splits are time-based on `DAYS_BIRTH`/application date, not random. Preprocessing is fit on train only and applied to val/test via a serialized sklearn Pipeline.
- **Class imbalance**: handled with SMOTE + classifier `class_weight`; PR-AUC is the primary metric because default rates are low (~8% in this dataset).
- **Operating threshold**: chosen from the precision-recall curve to maximize recall on delinquent borrowers (business objective), then mapped to a 0-100 risk score and Low/Medium/High band.
- **One artifact**: `models/creditguard_model.pkl` holds the full preprocessing + model pipeline so the API loads a single object.
- Tests use synthetic data fixtures, not the real dataset, so they run without downloads.

## Environment

- Python 3.13, FastAPI, pandas, scikit-learn, xgboost, imbalanced-learn, SQLAlchemy, psycopg2, pytest.
- PostgreSQL must be running and reachable via `DATABASE_URL` (see `.env.example`).
- Docker Compose file spins up PostgreSQL + the API in one command.