"""Feature engineering utilities for CreditGuard.

- Pull the SQL view v_feature_matrix from PostgreSQL into a pandas DataFrame.
- Encode categoricals, impute numerics, and split time-based train/val/test.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.utils.config import (
    DATABASE_URL,
    RANDOM_STATE,
    TRAIN_CUTOFF_DAYS,
    VAL_CUTOFF_DAYS,
)

CAT_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "EMERGENCYSTATE_MODE",
]
NUM_COLS = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY", "REGION_POPULATION_RELATIVE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",
    "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAREA_AVG",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "credit_to_income_ratio", "annuity_to_credit_ratio",
    "goods_to_credit_ratio", "applicant_age_years", "employment_years",
    "bureau_count", "bureau_active_cnt", "bureau_overdue_cnt",
    "bureau_avg_debt", "bureau_max_debt", "bureau_total_debt",
    "bureau_avg_age_days", "bureau_avg_limit", "bureau_total_overdue",
    "prev_app_cnt", "prev_avg_credit", "prev_avg_annuity",
    "prev_avg_down_payment", "prev_avg_decision_age",
    "prev_approved_cnt", "prev_refused_cnt",
    "prev_avg_installment_ratio",
    "inst_count", "inst_avg_deferred", "inst_avg_late_days",
    "inst_late_cnt", "inst_avg_amount", "inst_avg_payment",
]

TARGET = "TARGET"
SK_ID = "SK_ID_CURR"


def load_feature_matrix(engine) -> pd.DataFrame:
    """Pull v_feature_matrix from PostgreSQL and cast columns."""
    df = pd.read_sql(text("SELECT * FROM creditguard.v_feature_matrix"), engine)
    for col in NUM_COLS + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preprocess(
    df: pd.DataFrame,
    *,
    fit: bool = True,
    encoders: dict | None = None,
    imputers: dict | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Return a numeric DataFrame ready for modeling.

    Fills categoricals with 'MISSING', ordinal-encode with mean target
    (fit-only), and imputes numerics with median (fit-only).
    """
    work = df.copy()

    # Categoricals
    cat_encoders = encoders if encoders is not None else {}
    for col in CAT_COLS:
        if col not in work.columns:
            continue
        work[col] = work[col].fillna("MISSING").astype(str)
        if fit:
            mean_target = (
                work.groupby(col)[TARGET].mean().sort_values().to_dict()
            )
            cat_encoders[col] = mean_target
        mapping = cat_encoders[col]
        work[col + "_enc"] = work[col].map(mapping)
        if fit:
            # reserve new categories
            work[col + "_enc"] = work[col + "_enc"].fillna(
                max(mapping.values()) + 1 if mapping else 0
            )
        else:
            work[col + "_enc"] = work[col + "_enc"].fillna(
                max(mapping.values()) if mapping else 0
            )

    # Numerics
    num_imputers = imputers if imputers is not None else {}
    for col in NUM_COLS:
        if col not in work.columns:
            continue
        if fit:
            median = work[col].median()
            num_imputers[col] = median
        work[col] = work[col].fillna(num_imputers[col])

    feature_cols = [c + "_enc" for c in CAT_COLS if c in work.columns] + NUM_COLS
    X = work[feature_cols].copy()
    return X, cat_encoders, num_imputers


def time_based_split(
    df: pd.DataFrame,
    *,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by DAYS_BIRTH cutoff — no leakage across time.

    Older DAYS_BIRTH => train, middle => val, newer => test.
    """
    df = df.sort_values("DAYS_BIRTH", ascending=True).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]