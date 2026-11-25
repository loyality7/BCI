"""
Hivemind — Stage 1 Production Inference & Calibration Module
============================================================
Provides a clean, reusable interface for training, saving, loading, and
running single-window inference with the winning Stage 1 PyRiemann model.
"""

import time
from pathlib import Path
import joblib
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

CLASS_NAMES = {0: "Rest", 1: "Left", 2: "Right"}


def build_pipeline():
    """Create the winning Stage 1 PyRiemann (OAS -> TangentSpace -> LDA) pipeline."""
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[1 / 3, 1 / 3, 1 / 3]),
    )


def save_model(model, path: str | Path, metadata: dict | None = None) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, out_path)
    return out_path


def load_model(path: str | Path):
    payload = joblib.load(path)
    return payload["model"], payload.get("metadata", {})


def predict_window(model, window_64ch: np.ndarray, active_threshold: float = 0.50) -> dict:
    """
    Predict a single EEG window of shape (64, n_times) or (1, 64, n_times).
    Applies a confidence gate: if max(P(Left), P(Right)) < active_threshold, returns Rest.
    """
    x = window_64ch[np.newaxis, :, :] if window_64ch.ndim == 2 else window_64ch
    t0 = time.perf_counter()
    probs = model.predict_proba(x.astype(np.float64))[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Gate active commands to protect against false activations during Rest
    if max(probs[1], probs[2]) < active_threshold:
        pred_cls = 0
    else:
        pred_cls = int(np.argmax(probs))

    return {
        "class_id": pred_cls,
        "label": CLASS_NAMES[pred_cls],
        "probabilities": {"Rest": float(probs[0]), "Left": float(probs[1]), "Right": float(probs[2])},
        "latency_ms": latency_ms,
    }

