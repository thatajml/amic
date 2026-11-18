"""Tier 1: calibrated gate model over the FeatureVector.

Head 1: p = P(memory required)  (LogReg or LightGBM, then isotonic/Platt calibration on a
        held-out split, so p is an honest probability that cost-sensitive tau can act on).
Head 2: depth_p = P(DEEP | required), trained on required examples only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

SCHEMA_VERSION = 1


def _base(kind: Literal["logreg", "lgbm"], seed: int = 0) -> Any:
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        )
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )


@dataclass
class Tier1Model:
    features: list[str]
    gate: Any  # calibrated classifier
    depth: Any | None  # P(DEEP | required), may be None if too few DEEP labels
    kind: str
    calibration: str
    schema_version: int = SCHEMA_VERSION
    meta: dict[str, Any] = field(default_factory=dict)

    def _x(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([[feats[f] for f in self.features]], dtype=np.float32)

    def predict(self, feats: dict[str, float]) -> tuple[float, float]:
        x = self._x(feats)
        p = float(self.gate.predict_proba(x)[0, 1])
        d = float(self.depth.predict_proba(x)[0, 1]) if self.depth is not None else 0.5
        return p, d

    def predict_many(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.gate.predict_proba(X)[:, 1]
        d = self.depth.predict_proba(X)[:, 1] if self.depth is not None else np.full(len(X), 0.5)
        return p, d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> Tier1Model:
        m = joblib.load(path)
        if m.schema_version != SCHEMA_VERSION:
            raise ValueError(f"tier1 schema {m.schema_version} != {SCHEMA_VERSION}; retrain")
        return m


def train_tier1(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    features: list[str],
    *,
    kind: Literal["logreg", "lgbm"] = "logreg",
    calibration: Literal["isotonic", "platt"] = "isotonic",
    depth_tr: np.ndarray | None = None,
    seed: int = 0,
) -> Tier1Model:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    base = _base(kind, seed).fit(X_tr, y_tr)
    method = "isotonic" if calibration == "isotonic" else "sigmoid"
    gate = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X_cal, y_cal)
    depth = None
    if depth_tr is not None:
        mask = y_tr.astype(bool)
        yd = depth_tr[mask]
        if mask.sum() >= 20 and 0 < yd.sum() < len(yd):
            depth = _base("logreg", seed).fit(X_tr[mask], yd)
    return Tier1Model(features, gate, depth, kind, calibration)
