"""Create a tiny synthetic-data ML classifier for website testing.
This model is NOT a clinical ECG model. It is only a local/deployment fallback until
real MIT-BIH CNN weights are trained.
"""
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(123)
fs = 360
L = 360
classes = ["Normal", "Supraventricular", "PVC", "Fusion"]


def features(x):
    from scipy.signal import find_peaks
    from scipy.stats import skew, kurtosis
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x)
    peaks, _ = find_peaks(x, prominence=max(0.10 * float(np.std(x)), 1e-6), distance=20)
    return np.array([
        np.mean(x), np.std(x), np.min(x), np.max(x), np.sqrt(np.mean(x*x)), np.ptp(x),
        np.mean(np.abs(dx)), np.std(dx), len(peaks),
        skew(x, bias=False) if np.std(x) > 0 else 0,
        kurtosis(x, bias=False) if np.std(x) > 0 else 0,
    ], dtype=np.float32)


def make_window(label):
    t = np.arange(L) / fs
    x = 0.03 * np.sin(2 * np.pi * 1.2 * t) + 0.025 * rng.normal(size=L)
    center = 180 + rng.integers(-6, 7)
    width = {"Normal": 8, "Supraventricular": 7, "PVC": 15, "Fusion": 12}[label]
    amp = {"Normal": 1.0, "Supraventricular": 0.9, "PVC": 1.35, "Fusion": 1.05}[label]
    idx = np.arange(L)
    x += amp * np.exp(-0.5 * ((idx - center) / width) ** 2)
    if label == "Supraventricular":
        x += 0.3 * np.exp(-0.5 * ((idx - (center - 24)) / 9) ** 2)
    elif label == "PVC":
        x += 0.20 * np.exp(-0.5 * ((idx - (center + 35)) / 18) ** 2)
    elif label == "Fusion":
        x += 0.40 * np.exp(-0.5 * ((idx - (center + 10)) / 10) ** 2)
    return x

X, y = [], []
for label in classes:
    for _ in range(600):
        w = make_window(label)
        X.append(features(w))
        y.append(label)

model = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced")
model.fit(np.asarray(X), np.asarray(y))
joblib.dump(model, MODEL_DIR / "demo_classifier.joblib")
(MODEL_DIR / "demo_meta.json").write_text(
    json.dumps({"window_size": L, "sampling_rate": fs, "classes": classes, "model": "RandomForest synthetic demo"}, indent=2),
    encoding="utf-8",
)

print("Demo ML model created in models/")
