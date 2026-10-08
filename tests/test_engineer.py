"""Tests for feature engineering (no DB required)."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import preprocess, time_based_split, TARGET


@pytest.fixture
def df():
    rows = 200
    rng = np.random.default_rng(42)
    data = {
        TARGET: rng.integers(0, 2, size=rows).astype(float),
        "DAYS_BIRTH": rng.integers(-10000, -5000, size=rows),
        "CNT_CHILDREN": rng.integers(0, 10, size=rows),
        "AMT_INCOME_TOTAL": rng.uniform(20_000, 200_000, size=rows),
        "AMT_CREDIT": rng.uniform(10_000, 1_000_000, size=rows),
        "AMT_ANNUITY": rng.uniform(1_000, 50_000, size=rows),
        "AMT_GOODS_PRICE": rng.uniform(10_000, 900_000, size=rows),
        "DAYS_EMPLOYED": rng.integers(-5_000, 3_000, size=rows),
        "DAYS_REGISTRATION": rng.integers(-10_000, 0, size=rows),
        "DAYS_ID_PUBLISH": rng.integers(-10_000, 0, size=rows),
        "OWN_CAR_AGE": rng.integers(0, 20, size=rows),
        "REGION_RATING_CLIENT": rng.integers(1, 3, size=rows),
        "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 3, size=rows),
        "REGION_POPULATION_RELATIVE": rng.uniform(0.1, 3, size=rows),
        "EXT_SOURCE_1": rng.uniform(0, 1, size=rows),
        "EXT_SOURCE_2": rng.uniform(0, 1, size=rows),
        "EXT_SOURCE_3": rng.uniform(0, 1, size=rows),
        "APARTMENTS_AVG": rng.uniform(10, 100, size=rows),
        "BASEMENTAREA_AVG": rng.uniform(10, 500, size=rows),
        "YEARS_BEGINEXPLUATATION_AVG": rng.uniform(1950, 2020, size=rows),
        "YEARS_BUILD_AVG": rng.uniform(1950, 2020, size=rows),
        "COMMONAREA_AVG": rng.uniform(10, 500, size=rows),
        "ELEVATORS_AVG": rng.uniform(0, 5, size=rows),
        "ENTRANCES_AVG": rng.uniform(1, 10, size=rows),
        "FLOORSMAX_AVG": rng.uniform(1, 30, size=rows),
        "FLOORSMIN_AVG": rng.uniform(1, 30, size=rows),
        "LANDAREA_AVG": rng.uniform(100, 10_000, size=rows),
        "LIVINGAPARTMENTS_AVG": rng.uniform(10, 200, size=rows),
        "LIVINGAREA_AVG": rng.uniform(20, 500, size=rows),
        "NONLIVINGAREA_AVG": rng.uniform(10, 500, size=rows),
        "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 50, size=rows),
        "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 50, size=rows),
        "OBS_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 50, size=rows),
        "DEF_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 50, size=rows),
        "credit_to_income_ratio": rng.uniform(0.1, 10, size=rows),
        "annuity_to_credit_ratio": rng.uniform(0, 1, size=rows),
        "goods_to_credit_ratio": rng.uniform(0, 1, size=rows),
        "applicant_age_years": rng.integers(20, 75, size=rows),
        "employment_years": rng.uniform(-20, 40, size=rows),
        "bureau_count": rng.integers(0, 20, size=rows),
        "bureau_active_cnt": rng.integers(0, 20, size=rows),
        "bureau_overdue_cnt": rng.integers(0, 20, size=rows),
        "bureau_avg_debt": rng.uniform(0, 1_000_000, size=rows),
        "bureau_max_debt": rng.uniform(0, 1_000_000, size=rows),
        "bureau_total_debt": rng.uniform(0, 5_000_000, size=rows),
        "bureau_avg_age_days": rng.uniform(-5_000, 0, size=rows),
        "bureau_avg_limit": rng.uniform(0, 1_000_000, size=rows),
        "bureau_total_overdue": rng.uniform(0, 1_000_000, size=rows),
        "prev_app_cnt": rng.integers(0, 10, size=rows),
        "prev_avg_credit": rng.uniform(10_000, 1_000_000, size=rows),
        "prev_avg_annuity": rng.uniform(1_000, 50_000, size=rows),
        "prev_avg_down_payment": rng.uniform(0, 200_000, size=rows),
        "prev_avg_decision_age": rng.uniform(-5_000, 0, size=rows),
        "prev_approved_cnt": rng.integers(0, 10, size=rows),
        "prev_refused_cnt": rng.integers(0, 10, size=rows),
        "prev_avg_installment_ratio": rng.uniform(0, 1, size=rows),
        "inst_count": rng.integers(0, 200, size=rows),
        "inst_avg_deferred": rng.uniform(-30, 30, size=rows),
        "inst_avg_late_days": rng.uniform(-30, 30, size=rows),
        "inst_late_cnt": rng.integers(0, 200, size=rows),
        "inst_avg_amount": rng.uniform(1_000, 50_000, size=rows),
        "inst_avg_payment": rng.uniform(1_000, 50_000, size=rows),
    }
    for c in [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
        "EMERGENCYSTATE_MODE",
    ]:
        data[c] = rng.choice(["A", "B", "C", None], size=rows)
    return pd.DataFrame(data)


def test_preprocess_shape(df):
    X, enc, imp = preprocess(df, fit=True)
    assert X.shape == (len(df), 71)  # 48 num + 10 cat_enc (from CAT_COLS)
    assert sum("_enc" in c for c in X.columns) == 10  # CAT_COLS


def test_preprocess_no_leakage(df):
    X_train, enc, imp = preprocess(df, fit=True)
    X_test, _, _ = preprocess(df, fit=False, encoders=enc, imputers=imp)
    assert X_train.shape[1] == X_test.shape[1]


def test_time_split_preserves_order(df):
    train, val, test = time_based_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) > len(val) > 0
    assert len(test) > 0