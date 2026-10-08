"""Train, evaluate, and save a CreditGuard model.

Compares Logistic Regression, Random Forest, and XGBoost,
handles class imbalance with SMOTE + class weights, selects
an operating threshold from the PR curve, and persists the
full sklearn Pipeline so the API can load one artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from xgboost import XGBClassifier

from src.features.engineer import (
    CAT_COLS,
    NUM_COLS,
    TARGET,
    load_feature_matrix,
    preprocess,
    time_based_split,
)
from src.utils.config import (
    DATABASE_URL,
    MODEL_DIR,
    MODEL_PATH,
    METRICS_PATH,
    THRESHOLD_PATH,
    RANDOM_STATE,
    SMOTE_RANDOM_STATE,
)


class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    """Ordinal encode categoricals by mean target, fit-only."""

    def __init__(self):
        self.mapping_: dict[str, dict[str, float]] = {}

    def fit(self, X, y=None):
        if y is None:
            return self
        df = pd.DataFrame(X, columns=CAT_COLS)
        df["_target"] = np.asarray(y)
        for col in CAT_COLS:
            if col in df.columns:
                self.mapping_[col] = (
                    df.groupby(col)["_target"].mean().to_dict()
                )
        return self

    def transform(self, X, y=None):
        out = np.zeros((len(X), len(CAT_COLS)))
        df = pd.DataFrame(X, columns=CAT_COLS)
        for i, col in enumerate(CAT_COLS):
            if col in df.columns and col in self.mapping_:
                out[:, i] = df[col].map(self.mapping_[col]).fillna(0).values
        return out


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SkPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUM_COLS,
            ),
            ("cat", MeanTargetEncoder(), CAT_COLS),
        ],
        remainder="drop",
    )


def _evaluate(model, X_val, y_val, label: str) -> dict:
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "model": label,
        "precision": round(float(precision_score(y_val, y_pred)), 4),
        "recall": round(float(recall_score(y_val, y_pred)), 4),
        "f1": round(float(f1_score(y_val, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_val, y_prob)), 4),
    }
    print(f"\n--- {label} ---")
    print(classification_report(y_val, y_pred, digits=4))
    return metrics


def _select_threshold(model, X_val, y_val) -> dict:
    """Pick threshold maximizing recall while precision >= 0.25."""
    y_prob = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    candidates = [
        (t, p, r)
        for t, p, r in zip(thresholds, precisions, recalls)
        if p >= 0.25
    ]
    if not candidates:
        candidates = list(zip(thresholds, precisions, recalls))
    best = max(candidates, key=lambda x: x[2])
    return {
        "threshold": round(float(best[0]), 4),
        "precision_at_threshold": round(float(best[1]), 4),
        "recall_at_threshold": round(float(best[2]), 4),
    }


def _score_to_band(score: float) -> str:
    if score <= 40:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


def _prob_to_score(prob: float, threshold: float) -> int:
    """Map probability to 0-100 risk score."""
    raw = prob / max(threshold, 1e-9) * 50 + 50 * (prob / max(threshold, 1e-9))
    return int(min(max(round(raw), 0), 100))


def train() -> dict:
    engine = create_engine(DATABASE_URL)
    df = load_feature_matrix(engine)
    X, encoders, imputers = preprocess(df, fit=True)
    y = df[TARGET].values

    X_train_full, X_val, X_test = time_based_split(X)
    y_train_full = y[X_train_full.index]
    y_val = y[X_val.index]
    y_test = y[X_test.index]

    # Balance train only
    X_res, y_res = SMOTE(random_state=SMOTE_RANDOM_STATE).fit_resample(
        X_train_full, y_train_full
    )

    preprocessor = _build_preprocessor()
    results = []
    best_model, best_f1, best_label = None, -1, ""

    configs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=float(sum(y_res == 0) / max(sum(y_res == 1), 1)),
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }

    for label, estimator in configs.items():
        pipe = SkPipeline([("prep", preprocessor), ("clf", estimator)])
        pipe.fit(X_res, y_res)
        metrics = _evaluate(pipe, X_val, y_val, label)
        results.append(metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = pipe
            best_label = label

    print(f"\nBest model: {best_label} (F1={best_f1:.4f})")

    threshold_info = _select_threshold(best_model, X_val, y_val)
    print("Threshold selected:", threshold_info)

    # Final eval on test
    test_metrics = _evaluate(best_model, X_test, y_test, f"{best_label} (test)")

    # Persist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    artifact = {
        "model": best_label,
        "threshold": threshold_info["threshold"],
        "val_metrics": results,
        "test_metrics": test_metrics,
        "preprocessor": "StandardScaler + MeanTargetEncoder",
        "feature_columns": NUM_COLS,
        "categorical_columns": CAT_COLS,
    }
    METRICS_PATH.write_text(json.dumps(artifact, indent=2, default=str))
    THRESHOLD_PATH.write_text(json.dumps(threshold_info, indent=2))
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved threshold -> {THRESHOLD_PATH}")
    return artifact


if __name__ == "__main__":
    train()