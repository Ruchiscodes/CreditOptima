from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    index = int(np.argmax(tpr - fpr))
    return float(tpr[index] - fpr[index]), float(thresholds[index])


def decile_table(y_true, y_score) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": y_true, "probability": y_score})
    frame["decile"] = pd.qcut(
        frame["probability"].rank(method="first", ascending=False), 10, labels=range(1, 11)
    )
    grouped = frame.groupby("decile", observed=True)
    table = grouped.agg(
        accounts=("actual", "size"), defaults=("actual", "sum"),
        avg_probability=("probability", "mean"), min_probability=("probability", "min"),
        max_probability=("probability", "max"),
    ).reset_index()
    table["default_rate"] = table["defaults"] / table["accounts"]
    table["cumulative_defaults_pct"] = table["defaults"].cumsum() / table["defaults"].sum()
    return table


def false_approval_rate(y_true, y_score, threshold: float) -> float:
    predicted_default = np.asarray(y_score) >= threshold
    tn, _fp, fn, _tp = confusion_matrix(y_true, predicted_default).ravel()
    # Defaults incorrectly approved / all approvals.
    return float(fn / (tn + fn)) if (tn + fn) else 0.0


def metric_summary(y_true, y_score, threshold: float) -> dict[str, float]:
    ks, ks_threshold = ks_statistic(y_true, y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ks_statistic": ks,
        "ks_threshold": ks_threshold,
        "decision_threshold": threshold,
        "false_approval_rate": false_approval_rate(y_true, y_score, threshold),
    }
