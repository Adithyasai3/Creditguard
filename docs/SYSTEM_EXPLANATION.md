# CreditGuard: Credit Risk & Loan Delinquency Prediction System

## Executive Summary

CreditGuard is an end-to-end machine learning system designed to predict loan applicant default risk using structured banking data. The system implements a complete pipeline from raw data ingestion through model training, evaluation, and API deployment. It addresses key banking requirements including class imbalance handling, proper train/validation/test separation, and production-ready model serving.

## System Architecture Overview

```
┌─────────────────┐    ┌─────────────────────────────┐    ┌─────────────────────┐
│   Raw Data      │───▶│   PostgreSQL Database       │───▶│    Feature          │
│   (CSV Files)   │    │   (creditguard schema)      │    │   Engineering SQL   │
└─────────────────┘    └─────────────────────────────┘    └─────────────────────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────────┐
                                                              │  Feature Matrix │
                                                              │ (v_feature_matrix│
                                                              │      View)      │
                                                              └─────────────────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────────┐
                                                              │  Feature        │
                                                              │  Engineering    │
                                                              │ (Python/Pandas) │
                                                              └─────────────────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────────┐
                                                              │   Model Training│
                                                              │   & Evaluation  │
                                                              └─────────────────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────────┐
                                                              │   Model Artifact│
                                                              │ (creditguard_   │
                                                              │     model.pkl)  │
                                                              └─────────────────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────────┐
                                                              │   FastAPI       │
                                                              │   Prediction    │
                                                              │   Service       │
                                                              └─────────────────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────────┐
                                                              │   REST API      │
                                                              │  /predict       │
                                                              └─────────────────┘
```

## Detailed Component Breakdown

### 1. Data Layer & SQL Feature Engineering

**Purpose**: Transform raw banking data into predictive features using SQL for efficient set-based operations.

**Source Data** (Home Credit Default Risk Dataset):
- `application_train.csv` - Main application records with TARGET (1=default, 0=paid)
- `bureau.csv` - Previous credit bureau records
- `bureau_balance.csv` - Monthly bureau payment snapshots
- `previous_application.csv` - Previous Home Credit applications
- `installments_payment.csv` - Payment history records

**Key SQL Views** (`sql/feature_engineering.sql`):
1. `v_bureau` - Aggregates bureau data per applicant (counts, active loans, overdue amounts)
2. `v_bureau_balance` - Monthly bureau status aggregation (paid, delays)
3. `v_prev_app` - Previous application statistics (approved/refused counts, amounts)
4. `v_installments` - Payment behavior aggregations (late payments, amounts)
5. `v_feature_matrix` - Master view joining all features to application base

**Sample Features Generated**:
- Bureau: `bureau_count`, `bureau_overdue_cnt`, `bureau_total_debt`
- Previous apps: `prev_app_cnt`, `prev_approved_cnt`, `prev_avg_credit`
- Installments: `inst_late_cnt`, `inst_avg_late_days`, `inst_avg_amount`
- Derived: `credit_to_income_ratio`, `applicant_age_years`, `employment_years`

### 2. Feature Engineering Pipeline

**Preprocessing Steps** (`src/features/engineer.py`):
1. **Data Loading**: Pull `v_feature_matrix` view into Pandas DataFrame
2. **Time-Based Splitting**: Sort by `DAYS_BIRTH` (proxy for application date) to prevent temporal leakage:
   - Train: Oldest 60% of applications
   - Validation: Middle 20%
   - Test: Newest 20%
3. **Categorical Encoding**: Ordinal encoding using target mean statistics (fit on training only)
4. **Numerical Imputation**: Median imputation (fit on training only)
5. **Feature Assembly**: Combine encoded categoricals + imputed numerics

**Key Design**: All preprocessing steps are encapsulated in a scikit-learn Pipeline to prevent data leakage and ensure identical transformation at inference time.

### 3. Model Training & Evaluation

**Training Process** (`src/modeling/train.py`):
1. **Class Imbalance Handling**: 
   - SMOTE oversampling on training data only
   - Class weighting in all models (`class_weight='balanced'`)
2. **Model Comparison**:
   - Logistic Regression (baseline, interpretable)
   - Random Forest (non-linear, robust)
   - XGBoost (gradient boosting, typically best for tabular)
3. **Evaluation Metrics** (with focus on imbalanced class):
   - Precision: Of predicted defaults, % actually default
   - Recall: Of actual defaults, % correctly identified
   - F1-Score: Harmonic mean of precision & recall
   - ROC-AUC: Overall ranking ability
   - PR-AUC: **Primary metric** for imbalanced datasets (focuses on positive class)
4. **Threshold Selection**:
   - Precision-Recall curve analysis
   - Select threshold maximizing recall while maintaining minimum precision (≥0.25)
   - Business objective: Maximize detection of delinquent borrowers
5. **Risk Score Mapping**:
   - Probability → 0-100 score: `score = min(100, max(0, round(prob/threshold * 50)))`
   - Risk Bands: Low (0-40), Medium (41-70), High (71-100)

### 4. Model Persistence & API

**Model Artifact**: 
- Single file: `models/creditguard_model.pkl`
- Contains entire sklearn Pipeline (preprocessing + model)
- Enables one-line loading in API: `joblib.load(MODEL_PATH)`

**FastAPI Service** (`src/api/app.py`):
- **Endpoint**: `POST /predict`
- **Input**: Borrower feature JSON (matches training schema)
- **Processing**:
  1. Feature derivation (ratios, ages from raw fields)
  2. Categorical encoding (using precomputed target means)
  3. Feature vector assembly in training order
  4. Model inference → probability
  5. Risk score & band calculation
- **Output**: 
  ```json
  {
    "default_probability": 0.1234,
    "risk_score": 34,
    "risk_band": "Low",
    "threshold_used": 0.3125
  }
  ```

## Key Technical Innovations

### 1. Leakage Prevention
- **Time-based splitting** instead of random split
- **Pipeline encapsulation** ensures training statistics don't leak to validation/test
- **SQL views** computed independently of ML process

### 2. Production-Ready Design
- **Single model artifact** simplifies deployment
- **Lazy model loading** in API for fast startup
- **Comprehensive input validation** via Pydantic models
- **Health check endpoint** for monitoring
- **Dockerized deployment** for environment consistency

### 3. Banking-Specific Considerations
- **PR-AUC optimization** appropriate for low default rate (~8%)
- **Threshold selection** aligned with business goal (catching defaulters)
- **Interpretable features** from domain-knowledge engineering
- **Audit trail** via SQL-based feature generation

## Deployment & Usage

### Local Development
```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download data (uses kagglehub, no API key)
python -m src.data.download

# 3. Load into PostgreSQL
python -m src.data.ingest

# 4. Generate features
psql $DATABASE_URL -f sql/feature_engineering.sql

# 5. Train model
python -m src.modeling.train

# 6. Start API
uvicorn src.api.app:app --reload
```

### Docker Deployment
```bash
docker compose up --build
# API available at http://localhost:8000
```

### API Example
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 30000,
    "DAYS_BIRTH": -7300,
    "CNT_CHILDREN": 1,
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "Y",
    "NAME_INCOME_TYPE": "Working"
  }'
```

## Validation & Testing

### Test Suite (`tests/`)
- `test_engineer.py`: Feature engineering correctness, no leakage verification
- `test_api.py`: API endpoint validation, error handling, response format

### Synthetic Data Validation
All tests use synthetically generated data that mirrors the schema and distributions of the real Home Credit dataset, ensuring:
- No external dependencies for CI/CD
- Deterministic test outcomes
- Validation of preprocessing logic
- Verification of API contract compliance

## Files Summary

```
creditguard/
├── CLAUDE.md                  # This file - guidance for Claude Code
├── README.md                  # Project overview & setup instructions
├── requirements.txt           # Python dependencies
├── Dockerfile                 # API container definition
├── docker-compose.yml         # Full stack orchestration
├── .env.example               # Environment variables template
│
├── sql/
│   └── feature_engineering.sql # SQL views for feature generation
│
├── data/
│   ├── raw/                   # Downloaded CSV datasets (gitignored)
│   └── processed/             # Exported feature matrices (gitignored)
│
├── models/                    # Generated model artifacts (gitignored)
│   └── creditguard_model.pkl  # Serialized sklearn Pipeline
│
├── src/
│   ├── data/                  # Download & ingest scripts
│   ├── features/              # Feature engineering logic
│   ├── modeling/              # Training, evaluation, threshold selection
│   └── api/                   # FastAPI application
│
└── tests/
    ├── test_engineer.py       # Feature engineering tests
    └── test_api.py            # API endpoint tests
```

## Notes on Reproducibility

1. **Dataset**: Home Credit Default Risk via kagglehub (no API key needed)
2. **Random Seeds**: Fixed seeds throughout for reproducible splits and model initialization
3. **Environment**: Requirements pinned; Docker ensures dependency consistency
4. **Version Control**: All source code, SQL, and configuration committed; data and models excluded
5. **Documentation**: Complete setup instructions in README.md

This implementation satisfies all core requirements while maintaining production-grade code quality, proper ML practices, and deployable architecture.