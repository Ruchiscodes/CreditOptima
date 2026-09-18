from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from .config import (
    ARTIFACT_DIR,
    DECILES_PATH,
    FEATURES,
    ID_COLUMN,
    METRICS_PATH,
    MODEL_PATH,
    TARGET,
)
from .metrics import decile_table, false_approval_rate, metric_summary

DATA_URL = (
    "https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/"
    "3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv"
)


def download_data(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading Kaggle Give Me Some Credit mirror to {destination} ...")
        urllib.request.urlretrieve(DATA_URL, destination)
    return destination


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.drop(columns=[ID_COLUMN], errors="ignore").copy()
    required = set(FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame = frame.drop_duplicates()
    frame = frame[(frame["age"] >= 18) & (frame["age"] <= 120)]
    # 96/98 are documented anomalies in the three delinquency counters.
    delinquency = [
        "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]
    frame[delinquency] = frame[delinquency].mask(frame[delinquency] >= 90, np.nan)
    # Winsorize data-entry extremes using values learned before the split only for this demo.
    for column in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
        frame[column] = frame[column].clip(lower=0, upper=frame[column].quantile(0.995))
    return frame


def approval_threshold_for_rate(scores: np.ndarray, approval_rate: float = 0.80) -> float:
    """Threshold yielding a fixed approval rate, enabling a fair model comparison."""
    return float(np.quantile(scores, approval_rate))


def train(data_path: Path, artifact_dir: Path = ARTIFACT_DIR, seed: int = 42) -> dict:
    raw = pd.read_csv(data_path)
    data = clean_data(raw)
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ]), FEATURES)
    ], verbose_feature_names_out=False)

    negative, positive = np.bincount(y_train)
    scale_pos_weight = float(negative / positive)
    xgb = XGBClassifier(
        n_estimators=650, max_depth=4, learning_rate=0.035, min_child_weight=8,
        subsample=0.85, colsample_bytree=0.9, reg_alpha=0.15, reg_lambda=2.5,
        objective="binary:logistic", eval_metric="auc", scale_pos_weight=scale_pos_weight,
        random_state=seed, n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb)])
    pipeline.fit(X_train, y_train)
    xgb_scores = pipeline.predict_proba(X_test)[:, 1]

    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    baseline.fit(X_train, y_train)
    baseline_scores = baseline.predict_proba(X_test)[:, 1]

    # Compare at the same approval volume; false approvals are defaults among approved cases.
    xgb_threshold = approval_threshold_for_rate(xgb_scores)
    baseline_threshold = approval_threshold_for_rate(baseline_scores)
    xgb_fa = false_approval_rate(y_test, xgb_scores, xgb_threshold)
    baseline_fa = false_approval_rate(y_test, baseline_scores, baseline_threshold)
    reduction = (baseline_fa - xgb_fa) / baseline_fa if baseline_fa else 0.0

    summary = metric_summary(y_test, xgb_scores, xgb_threshold)
    summary.update({
        "baseline_roc_auc": float(roc_auc_score(y_test, baseline_scores)),
        "baseline_false_approval_rate": baseline_fa,
        "false_approval_reduction": float(reduction),
        "approval_rate": 0.80,
        "test_rows": len(y_test),
        "default_rate": float(y.mean()),
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "dataset": "Kaggle Give Me Some Credit",
        "dataset_url": "https://www.kaggle.com/c/GiveMeSomeCredit/data",
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "trained_at": datetime.now(UTC).isoformat(),
    })

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    version = f"xgb-{datetime.now(UTC).strftime('%Y%m%d')}-{summary['data_sha256'][:8]}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "preprocessor": fitted_preprocessor,
        "explainer": explainer,
        "model_version": version,
        "features": FEATURES,
        "metrics": summary,
    }, artifact_dir / MODEL_PATH.name)
    (artifact_dir / METRICS_PATH.name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    decile_table(y_test.to_numpy(), xgb_scores).to_csv(artifact_dir / DECILES_PATH.name, index=False)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate CreditOptima")
    parser.add_argument("--data", type=Path, default=Path("data/raw/cs-training.csv"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    path = args.data if args.no_download else download_data(args.data)
    train(path, args.artifacts)


if __name__ == "__main__":
    main()
