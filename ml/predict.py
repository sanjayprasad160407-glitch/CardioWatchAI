import json
from pathlib import Path

import joblib
import numpy as np

from .preprocess import extract_center_window, preprocess_ecg, preprocess_multilead

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
BINARY_MODEL = MODEL_DIR / "binary_abnormal_model.keras"
BINARY_ENCODER = MODEL_DIR / "binary_abnormal_model_encoder.joblib"
BINARY_META = MODEL_DIR / "binary_abnormal_model_meta.json"
SUBTYPE_MODEL = MODEL_DIR / "rhythm3_model.keras"
SUBTYPE_ENCODER = MODEL_DIR / "rhythm3_model_encoder.joblib"
SUBTYPE_META = MODEL_DIR / "rhythm3_model_meta.json"
LEGACY_MODEL = MODEL_DIR / "arrhythmia_model.keras"
LEGACY_ENCODER = MODEL_DIR / "label_encoder.joblib"
LEGACY_META = MODEL_DIR / "model_meta.json"
DEMO_MODEL = MODEL_DIR / "demo_classifier.joblib"
DEMO_META = MODEL_DIR / "demo_meta.json"


def assets_status():
    if BINARY_MODEL.exists() and BINARY_ENCODER.exists() and BINARY_META.exists():
        return {"ready": True, "type": "Real 2-stage CNN", "message": "Real ECG CNN screening model is ready."}
    if LEGACY_MODEL.exists() and LEGACY_ENCODER.exists() and LEGACY_META.exists():
        return {"ready": True, "type": "Legacy Real CNN", "message": "Legacy real ECG CNN model is ready."}
    if DEMO_MODEL.exists() and DEMO_META.exists():
        return {"ready": True, "type": "Demo ML", "message": "Demo ML model is ready. Train the real CNN to switch modes."}
    return {"ready": False, "type": "None", "message": "Run setup_demo.py or train the real model."}


def read_csv_ecg(path):
    import pandas as pd
    try:
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            values = df[numeric].to_numpy(dtype=np.float32)
        else:
            df = pd.read_csv(path, header=None)
            values = pd.DataFrame({"ecg": pd.to_numeric(df.iloc[:, 0], errors="coerce")}).to_numpy(dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Could not read ECG CSV: {exc}") from exc
    finite_mask = np.all(np.isfinite(values), axis=1)
    values = values[finite_mask]
    if len(values) < 720:
        raise ValueError("ECG must contain at least 720 numeric samples for the real 2-second model.")
    return values


def signal_quality_score(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    scores = []
    for ch in range(x.shape[1]):
        s = x[:, ch]
        std = float(np.std(s))
        if std < 1e-7:
            scores.append(0.0)
            continue
        z = np.abs((s - np.mean(s)) / (std + 1e-8))
        clipping = float(np.mean(z > 5.0))
        diff_ratio = float(np.std(np.diff(s)) / (std + 1e-8))
        noise_penalty = min(0.55, max(0.0, (diff_ratio - 0.25) * 0.25))
        clipping_penalty = min(0.35, clipping * 3.0)
        scores.append(float(np.clip(1.0 - noise_penalty - clipping_penalty, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0


def _top_regions(values, fs, n=4):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return []
    idx = np.argsort(values)[-max(1, n):]
    return sorted({round(float(i) / float(fs), 2) for i in idx})


def _saliency(window, model, class_index):
    import tensorflow as tf
    x = tf.convert_to_tensor(window[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        output = model(x, training=False)
        score = output[:, int(class_index)]
    grad = tape.gradient(score, x)
    if grad is None:
        return np.zeros(window.shape[0], dtype=np.float32)
    values = np.max(np.abs(grad.numpy()[0]), axis=-1)
    maximum = float(np.max(values)) if len(values) else 0.0
    return (values / maximum if maximum > 0 else values).astype(np.float32)


def _predict_two_stage(window):
    from tensorflow.keras.models import load_model

    primary = load_model(BINARY_MODEL, compile=False)
    encoder = joblib.load(BINARY_ENCODER)
    with BINARY_META.open(encoding="utf-8") as f:
        meta = json.load(f)

    probs = primary.predict(window[None, ...], verbose=0)[0]
    threshold = float(meta.get("threshold", 0.5))
    abnormal_index = int(np.where(np.asarray(encoder.classes_) == "Abnormal")[0][0])
    normal_index = int(np.where(np.asarray(encoder.classes_) == "Normal")[0][0])
    abnormal_probability = float(probs[abnormal_index])
    class_index = abnormal_index if abnormal_probability >= threshold else normal_index
    primary_label = str(encoder.inverse_transform([class_index])[0])
    primary_confidence = abnormal_probability if primary_label == "Abnormal" else float(probs[normal_index])

    subtype = None
    subtype_confidence = None
    subtype_model = None
    if primary_label == "Abnormal" and SUBTYPE_MODEL.exists() and SUBTYPE_ENCODER.exists() and SUBTYPE_META.exists():
        subtype_model = load_model(SUBTYPE_MODEL, compile=False)
        subtype_encoder = joblib.load(SUBTYPE_ENCODER)
        sub_probs = subtype_model.predict(window[None, ...], verbose=0)[0]
        sub_idx = int(np.argmax(sub_probs))
        subtype = str(subtype_encoder.inverse_transform([sub_idx])[0])
        subtype_confidence = float(sub_probs[sub_idx])

    label = "Normal" if primary_label == "Normal" else "Abnormal"
    if subtype == "Ventricular":
        label = "Abnormal · Ventricular"
    elif subtype == "OtherAbnormal":
        label = "Abnormal · Other"

    saliency = _saliency(window, primary, class_index)
    return {
        "prediction": label,
        "primary_prediction": primary_label,
        "confidence": float(primary_confidence),
        "subtype": subtype,
        "subtype_confidence": subtype_confidence,
        "saliency": saliency,
        "model_type": "Real 2-stage CNN",
        "class_index": class_index,
    }


def _predict_legacy(window):
    from tensorflow.keras.models import load_model
    model = load_model(LEGACY_MODEL, compile=False)
    encoder = joblib.load(LEGACY_ENCODER)
    with LEGACY_META.open(encoding="utf-8") as f:
        meta = json.load(f)
    expected = int(meta.get("window_size", len(window)))
    expected_channels = int(meta.get("channels", 1))
    if len(window) != expected:
        raise ValueError(f"Model expects {expected} samples but received {len(window)}.")
    if window.ndim != 2 or window.shape[1] != expected_channels:
        raise ValueError(f"Legacy model expects {expected_channels} ECG leads.")
    probs = model.predict(window[None, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    label = str(encoder.inverse_transform([idx])[0])
    return {
        "prediction": label,
        "primary_prediction": label,
        "confidence": float(probs[idx]),
        "subtype": None,
        "subtype_confidence": None,
        "saliency": _saliency(window, model, idx),
        "model_type": "Legacy Real CNN",
        "class_index": idx,
    }


def _predict_demo(window):
    model = joblib.load(DEMO_MODEL)
    with DEMO_META.open(encoding="utf-8") as f:
        meta = json.load(f)
    one = window[:, 0] if window.ndim > 1 else window
    from scipy.signal import find_peaks
    from scipy.stats import skew, kurtosis
    peaks, _ = find_peaks(one, prominence=max(0.15 * float(np.std(one)), 1e-6), distance=20)
    diffs = np.diff(one)
    features = np.array([
        np.mean(one), np.std(one), np.min(one), np.max(one),
        np.sqrt(np.mean(one * one)), np.ptp(one),
        np.mean(np.abs(diffs)) if len(diffs) else 0.0,
        np.std(diffs) if len(diffs) else 0.0,
        float(len(peaks)),
        float(skew(one, bias=False)) if np.std(one) > 0 else 0.0,
        float(kurtosis(one, bias=False)) if np.std(one) > 0 else 0.0,
    ], dtype=np.float32)
    probs = model.predict_proba(features[None, :])[0]
    classes = list(meta["classes"])
    idx = int(np.argmax(probs))
    sal = np.abs(np.gradient(one)).astype(np.float32)
    if np.max(sal) > 0:
        sal /= np.max(sal)
    return {
        "prediction": classes[idx],
        "primary_prediction": classes[idx],
        "confidence": float(probs[idx]),
        "subtype": None,
        "subtype_confidence": None,
        "saliency": sal,
        "model_type": "Demo ML",
        "class_index": idx,
    }


def analyze_ecg(path, fs):
    status = assets_status()
    if not status["ready"]:
        raise FileNotFoundError("No ML model is available. Run setup_demo.py or train the real model.")

    raw = read_csv_ecg(path)
    real_two_stage = BINARY_MODEL.exists() and BINARY_ENCODER.exists() and BINARY_META.exists()
    legacy_real = LEGACY_MODEL.exists() and LEGACY_ENCODER.exists() and LEGACY_META.exists()

    if real_two_stage:
        with BINARY_META.open(encoding="utf-8") as f:
            meta = json.load(f)
        expected_channels = int(meta.get("channels", 2))
        window_size = int(meta.get("window_size", 720))
        if raw.shape[1] != expected_channels:
            raise ValueError(
                f"The real model expects {expected_channels} ECG leads, but this CSV has {raw.shape[1]}. "
                "Use make_real_csv.py to create a two-lead test file."
            )
        processed = preprocess_multilead(raw, fs)
        window, start = extract_center_window(processed, window_size)
        result = _predict_two_stage(window)
    elif legacy_real:
        with LEGACY_META.open(encoding="utf-8") as f:
            meta = json.load(f)
        window_size = int(meta.get("window_size", 360))
        expected_channels = int(meta.get("channels", 1))
        if raw.shape[1] != expected_channels:
            raise ValueError(f"Legacy model expects {expected_channels} ECG leads.")
        processed = preprocess_multilead(raw, fs)
        window, start = extract_center_window(processed, window_size)
        result = _predict_legacy(window)
    else:
        processed = preprocess_ecg(raw[:, 0], fs)
        window, start = extract_center_window(processed, 360)
        window = window[:, None]
        result = _predict_demo(window)

    sqi = signal_quality_score(window)
    sal = result["saliency"]
    stride = max(1, int(np.ceil(len(sal) / 1200)))
    return {
        "prediction": result["prediction"],
        "primary_prediction": result["primary_prediction"],
        "confidence": result["confidence"],
        "subtype": result["subtype"],
        "subtype_confidence": result["subtype_confidence"],
        "signal_quality": sqi,
        "signal": raw,
        "display_signal": raw[:, 0].astype(float).tolist(),
        "saliency": sal[::stride].astype(float).tolist(),
        "salient_seconds": _top_regions(sal, fs, n=4),
        "window_start": start,
        "window_size": len(window),
        "model_type": result["model_type"],
        "class_index": result["class_index"],
        "channels": int(raw.shape[1]),
    }
