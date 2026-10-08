"""Central configuration for CreditGuard.

All settings (paths, DB URL, model hyperparameters) live here so the
pipeline and API share one source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "creditguard_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://creditguard:creditguard@localhost:5432/creditguard",
)

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
# Home Credit default rate is ~8%; SMOTE balances the training set.
SMOTE_RANDOM_STATE = 42
# Time-based split: train on applications older than this cutoff.
TRAIN_CUTOFF_DAYS = -1500  # ~4 years before most recent application
VAL_CUTOFF_DAYS = -750

# Risk score: 0-100, higher = riskier.
RISK_BANDS = {
    "low": (0, 40),
    "medium": (41, 70),
    "high": (71, 100),
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Kaggle dataset slug for kagglehub.download_dataset
DATASET_SLUG = "bchi522/home-credit-default-risk"
REQUIRED_TABLES = (
    "application_train",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payment",
)