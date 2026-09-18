from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, MODEL_PATH, REASON_CODES


def probability_to_score(pd: float) -> int:
    """Map probability of default monotonically onto a conventional 300–850 scale."""
    return int(np.clip(round(850 - 550 * pd), 300, 850))


def risk_policy(pd: float) -> tuple[str, str]:
    if pd < 0.05:
        return "A", "APPROVE"
    if pd < 0.10:
        return "B", "APPROVE"
    if pd < 0.20:
        return "C", "REFER"
    if pd < 0.35:
        return "D", "REFER"
    return "E", "DECLINE"


@dataclass
class RiskEngine:
    bundle: dict[str, Any]

    @classmethod
    def load(cls, path=MODEL_PATH) -> RiskEngine:
        return cls(joblib.load(path))

    @property
    def version(self) -> str:
        return self.bundle["model_version"]

    def predict(self, values: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
        frame = pd.DataFrame([values], columns=FEATURES)
        pipeline = self.bundle["pipeline"]
        pd_value = float(pipeline.predict_proba(frame)[0, 1])
        reasons = self._reasons(frame)
        return pd_value, reasons

    def _reasons(self, frame: pd.DataFrame, limit: int = 4) -> list[dict[str, Any]]:
        """Return features that push default risk upward, ordered by local SHAP impact."""
        transformed = self.bundle["preprocessor"].transform(frame)
        explainer = self.bundle.get("explainer")
        if explainer is None:
            return []
        values = np.asarray(explainer.shap_values(transformed))
        if values.ndim == 3:
            values = values[:, :, 1]
        row = values[0]
        ranked = [i for i in np.argsort(row)[::-1] if row[i] > 0][:limit]
        output = []
        for idx in ranked:
            feature = FEATURES[int(idx)]
            code, description = REASON_CODES[feature]
            output.append({
                "code": code,
                "feature": feature,
                "description": description,
                "contribution": round(float(row[idx]), 6),
            })
        return output
