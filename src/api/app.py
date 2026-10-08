"""CreditGuard FastAPI — serve default probability, risk score, risk band."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.config import MODEL_PATH, THRESHOLD_PATH
from src.features.engineer import NUM_COLS, CAT_COLS

app = FastAPI(title="CreditGuard API", version="0.1.0")

# Lazy-loaded; tests create the artifact before importing the app fixture.
_model = None
_threshold: float | None = None
_feature_names: list[str] | None = None


def _ensure_loaded():
    global _model, _threshold, _feature_names
    global _model, _threshold, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _threshold = json.loads(THRESHOLD_PATH.read_text())["threshold"]
        _feature_names = NUM_COLS + [c + "_enc" for c in CAT_COLS]


class BorrowerFeatures(BaseModel):
    # Application-level
    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str = "M"
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "N"
    CNT_CHILDREN: int = 0
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: float = Field(..., ge=0)
    AMT_GOODS_PRICE: float = Field(..., ge=0)
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    OCCUPATION_TYPE: str | None = None
    DAYS_BIRTH: int = Field(..., le=0)
    DAYS_EMPLOYED: int = Field(..., le=0)
    DAYS_REGISTRATION: int = Field(..., le=0)
    DAYS_ID_PUBLISH: int = Field(..., le=0)
    OWN_CAR_AGE: int | None = None
    REGION_RATING_CLIENT: int = 1
    REGION_RATING_CLIENT_W_CITY: int = 1
    REGION_POPULATION_RELATIVE: float = 0.5
    EXT_SOURCE_1: float | None = None
    EXT_SOURCE_2: float | None = None
    EXT_SOURCE_3: float | None = None
    APARTMENTS_AVG: float | None = None
    BASEMENTAREA_AVG: float | None = None
    YEARS_BEGINEXPLUATATION_AVG: float | None = None
    YEARS_BUILD_AVG: float | None = None
    COMMONAREA_AVG: float | None = None
    ELEVATORS_AVG: float | None = None
    ENTRANCES_AVG: float | None = None
    FLOORSMAX_AVG: float | None = None
    FLOORSMIN_AVG: float | None = None
    LANDAREA_AVG: float | None = None
    LIVINGAPARTMENTS_AVG: float | None = None
    LIVINGAREA_AVG: float | None = None
    NONLIVINGAREA_AVG: float | None = None
    EMERGENCYSTATE_MODE: str | None = None
    OBS_30_CNT_SOCIAL_CIRCLE: float | None = None
    DEF_30_CNT_SOCIAL_CIRCLE: float | None = None
    OBS_60_CNT_SOCIAL_CIRCLE: float | None = None
    DEF_60_CNT_SOCIAL_CIRCLE: float | None = None

    # SQL-engineered features (optional)
    credit_to_income_ratio: float | None = None
    annuity_to_credit_ratio: float | None = None
    goods_to_credit_ratio: float | None = None
    applicant_age_years: int | None = None
    employment_years: float | None = None
    bureau_count: int | None = 0
    bureau_active_cnt: int | None = 0
    bureau_overdue_cnt: int | None = 0
    bureau_avg_debt: float | None = 0
    bureau_max_debt: float | None = 0
    bureau_total_debt: float | None = 0
    bureau_avg_age_days: int | None = 0
    bureau_avg_limit: float | None = 0
    bureau_total_overdue: float | None = 0
    prev_app_cnt: int | None = 0
    prev_avg_credit: float | None = 0
    prev_avg_annuity: float | None = 0
    prev_avg_down_payment: float | None = 0
    prev_avg_decision_age: int | None = 0
    prev_approved_cnt: int | None = 0
    prev_refused_cnt: int | None = 0
    prev_avg_installment_ratio: float | None = 0
    inst_count: int | None = 0
    inst_avg_deferred: float | None = 0
    inst_avg_late_days: int | None = 0
    inst_late_cnt: int | None = 0
    inst_avg_amount: float | None = 0
    inst_avg_payment: float | None = 0


def _risk_band(score: int) -> str:
    if score <= 40:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(b: BorrowerFeatures):
    _ensure_loaded()
    row = b.model_dump()

    if row.get("credit_to_income_ratio") is None:
        row["credit_to_income_ratio"] = round(
            row["AMT_CREDIT"] / row["AMT_INCOME_TOTAL"], 2
        )
    if row.get("annuity_to_credit_ratio") is None:
        row["annuity_to_credit_ratio"] = round(
            row["AMT_ANNUITY"] / max(row["AMT_CREDIT"], 1), 4
        )
    if row.get("goods_to_credit_ratio") is None:
        row["goods_to_credit_ratio"] = round(
            row["AMT_GOODS_PRICE"] / max(row["AMT_CREDIT"], 1), 4
        )
    if row.get("applicant_age_years") is None:
        row["applicant_age_years"] = int(-row["DAYS_BIRTH"] / 365)
    if row.get("employment_years") is None:
        row["employment_years"] = None if row["DAYS_EMPLOYED"] >= -365 else int(-row["DAYS_EMPLOYED"] / 365)

    raw_row = {c: row.get(c, 0) for c in NUM_COLS}
    for c in CAT_COLS:
        raw_row[c] = str(row.get(c, "MISSING"))
    X = pd.DataFrame([raw_row])
    prob = float(_model.predict_proba(X)[0, 1])
    score = int(min(max(round(prob / max(_threshold, 1e-9) * 50), 0), 100))
    band = _risk_band(score)

    return {
        "default_probability": round(prob, 4),
        "risk_score": score,
        "risk_band": band,
        "threshold_used": _threshold,
    }