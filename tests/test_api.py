"""Tests for the FastAPI prediction endpoint (no DB required)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.api.app import app
from src.utils.config import MODEL_DIR, THRESHOLD_PATH, MODEL_PATH, METRICS_PATH
from src.features.engineer import NUM_COLS, CAT_COLS


@pytest.fixture(autouse=True)
def ensure_model():
    """Create a minimal saved model if none exists, so tests
    can import the API without a full training run."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from src.modeling.train import _build_preprocessor

        feature_names = NUM_COLS + CAT_COLS
        rng = np.random.default_rng(0)
        n = 200
        data = {c: rng.uniform(-3, 3, n) for c in NUM_COLS}
        for c in CAT_COLS:
            data[c] = rng.choice(["A", "B", "C"], size=n)
        X = pd.DataFrame(data, columns=feature_names)
        y = rng.integers(0, 2, n).astype(float)
        preprocessor = _build_preprocessor()
        pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=500))])
        pipe.fit(X, y)
        joblib.dump(pipe, MODEL_PATH)
        THRESHOLD_PATH.write_text(json.dumps({"threshold": 0.5}))
        METRICS_PATH.write_text(json.dumps({"model": "test"}))
        # Force lazy load to pick up the artifact we just wrote.
        import src.api.app as _app_module

        _app_module._ensure_loaded()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_expected_fields(client):
    payload = {
        "AMT_INCOME_TOTAL": 150_000.0,
        "AMT_CREDIT": 500_000.0,
        "AMT_ANNUITY": 30_000.0,
        "AMT_GOODS_PRICE": 450_000.0,
        "DAYS_BIRTH": -7_300,
        "DAYS_EMPLOYED": -1_800,
        "DAYS_REGISTRATION": -5_000,
        "DAYS_ID_PUBLISH": -3_000,
        "CNT_CHILDREN": 1,
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "REGION_POPULATION_RELATIVE": 1.0,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.4,
        "EXT_SOURCE_3": 0.3,
        "APARTMENTS_AVG": 50.0,
        "BASEMENTAREA_AVG": 100.0,
        "YEARS_BEGINEXPLUATATION_AVG": 1990.0,
        "YEARS_BUILD_AVG": 2000.0,
        "COMMONAREA_AVG": 200.0,
        "ELEVATORS_AVG": 2.0,
        "ENTRANCES_AVG": 3.0,
        "FLOORSMAX_AVG": 5.0,
        "FLOORSMIN_AVG": 1.0,
        "LANDAREA_AVG": 5_000.0,
        "LIVINGAPARTMENTS_AVG": 80.0,
        "LIVINGAREA_AVG": 120.0,
        "NONLIVINGAREA_AVG": 40.0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 5.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": 1.0,
        "OBS_60_CNT_SOCIAL_CIRCLE": 3.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "credit_to_income_ratio": 3.33,
        "annuity_to_credit_ratio": 0.06,
        "goods_to_credit_ratio": 0.9,
        "applicant_age_years": 20,
        "employment_years": 5,
        "bureau_count": 2,
        "bureau_active_cnt": 1,
        "bureau_overdue_cnt": 0,
        "bureau_avg_debt": 50_000.0,
        "bureau_max_debt": 100_000.0,
        "bureau_total_debt": 150_000.0,
        "bureau_avg_age_days": -365,
        "bureau_avg_limit": 200_000.0,
        "bureau_total_overdue": 0.0,
        "prev_app_cnt": 1,
        "prev_avg_credit": 300_000.0,
        "prev_avg_annuity": 18_000.0,
        "prev_avg_down_payment": 50_000.0,
        "prev_avg_decision_age": -730,
        "prev_approved_cnt": 1,
        "prev_refused_cnt": 0,
        "prev_avg_installment_ratio": 16.67,
        "inst_count": 12,
        "inst_avg_deferred": 0.0,
        "inst_avg_late_days": -5,
        "inst_late_cnt": 0,
        "inst_avg_amount": 30_000.0,
        "inst_avg_payment": 29_000.0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "default_probability" in data
    assert "risk_score" in data
    assert "risk_band" in data
    assert data["risk_band"] in ("Low", "Medium", "High")
    assert 0 <= data["risk_score"] <= 100
    assert 0 <= data["default_probability"] <= 1


def test_predict_invalid_income(client):
    payload = {
        "AMT_INCOME_TOTAL": -1.0,
        "AMT_CREDIT": 500_000.0,
        "AMT_ANNUITY": 30_000.0,
        "AMT_GOODS_PRICE": 450_000.0,
        "DAYS_BIRTH": -7_300,
        "DAYS_EMPLOYED": -1_800,
        "DAYS_REGISTRATION": -5_000,
        "DAYS_ID_PUBLISH": -3_000,
        "CNT_CHILDREN": 0,
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "N",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_RATING_CLIENT": 1,
        "REGION_RATING_CLIENT_W_CITY": 1,
        "REGION_POPULATION_RELATIVE": 0.5,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.4,
        "EXT_SOURCE_3": 0.3,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422  # Pydantic validation error