"""Train robust, patient-disjoint ECG screening models on MIT-BIH.

Primary task:
    Normal vs Abnormal

Secondary task:
    Normal / Ventricular ectopic / Other abnormal

Why two stages?
    MIT-BIH has strong class imbalance. A single 4-class model can achieve high
    headline accuracy by mostly predicting Normal while doing poorly on rare
    classes. The primary binary task is a more stable first-stage screening
    target, while the secondary 3-class model provides a useful subtype for
    abnormal windows.

This remains an educational/research prototype and is not clinically validated.

Run:
    python -m ml.train_model --epochs 15 --max-per-class 5000
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import wfdb
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks, layers, models

from .preprocess import extract_window, preprocess_multilead

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "mitdb"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = MODEL_DIR / "real_model_report"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
    "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
    "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]

RAW_NORMAL = {"N", "L", "R", "e", "j"}
RAW_VENTRICULAR = {"V", "E"}
RAW_OTHER = {"A", "a", "J", "S", "F"}
THREE_CLASSES = ["Normal", "Ventricular", "OtherAbnormal"]
BINARY_CLASSES = ["Normal", "Abnormal"]


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def raw_to_three(symbol: str) -> str | None:
    if symbol in RAW_NORMAL:
        return "Normal"
    if symbol in RAW_VENTRICULAR:
        return "Ventricular"
    if symbol in RAW_OTHER:
        return "OtherAbnormal"
    return None


def load_record_samples(record_name: str, window_size: int):
    path = DATA_DIR / record_name
    header = wfdb.rdheader(str(path))
    channels = [0, 1] if header.n_sig >= 2 else [0]
    record = wfdb.rdrecord(str(path), channels=channels)
    signals = record.p_signal.astype(np.float32)
    signals = preprocess_multilead(signals, float(record.fs))
    if signals.ndim == 1:
        signals = signals[:, None]

    ann = wfdb.rdann(str(path), "atr")
    xs, ys, groups = [], [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        label = raw_to_three(symbol)
        if label is None:
            continue
        window = extract_window(signals, int(sample), window_size)
        if window is None:
            continue
        # Always use two channels for consistent model input. Duplicate the
        # first channel only for an unusual one-lead record.
        if window.shape[1] == 1:
            window = np.repeat(window, 2, axis=1)
        elif window.shape[1] > 2:
            window = window[:, :2]
        xs.append(window)
        ys.append(label)
        groups.append(record_name)
    if not xs:
        return np.empty((0, window_size, 2), dtype=np.float32), np.empty((0,), dtype=str), np.empty((0,), dtype=str)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys), np.asarray(groups)


def collect_records(records, window_size):
    xs, ys, gs = [], [], []
    for record in records:
        print(f"Loading record {record}...")
        x, y, g = load_record_samples(record, window_size)
        if len(x) == 0:
            continue
        xs.append(x)
        ys.append(y)
        gs.append(g)
        print("  samples:", len(x), "classes:", {c: int(np.sum(y == c)) for c in THREE_CLASSES})
    if not xs:
        raise RuntimeError("No usable ECG samples were loaded.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(gs, axis=0)


def choose_validation_records(ds1_records, window_size, n_val=5):
    counts = {}
    for record in ds1_records:
        x, y, _ = load_record_samples(record, window_size)
        counts[record] = {c: int(np.sum(y == c)) for c in THREE_CLASSES}

    total = {c: sum(counts[r][c] for r in ds1_records) for c in THREE_CLASSES}
    target_ratio = n_val / len(ds1_records)
    best = None
    for combo in itertools.combinations(ds1_records, n_val):
        val_counts = {c: sum(counts[r][c] for r in combo) for c in THREE_CLASSES}
        missing = sum(val_counts[c] == 0 for c in THREE_CLASSES)
        ratio_error = sum(abs((val_counts[c] / max(total[c], 1)) - target_ratio) for c in THREE_CLASSES)
        size_error = abs((sum(val_counts.values()) / max(sum(total.values()), 1)) - target_ratio)
        score = missing * 10.0 + ratio_error + size_error
        key = (score, tuple(combo))
        if best is None or key < best[0]:
            best = (key, combo)
    assert best is not None
    val_records = list(best[1])
    train_records = [r for r in ds1_records if r not in val_records]
    return train_records, val_records, counts


def balanced_sample(X, y_text, classes, target_per_class, seed=42):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cls in classes:
        idx = np.flatnonzero(y_text == cls)
        if len(idx) == 0:
            raise RuntimeError(f"No training samples for class {cls}.")
        chosen = rng.choice(idx, size=target_per_class, replace=len(idx) < target_per_class)
        xs.append(X[chosen])
        ys.append(y_text[chosen])
    Xb = np.concatenate(xs, axis=0)
    yb = np.concatenate(ys, axis=0)
    order = rng.permutation(len(yb))
    return Xb[order].astype(np.float32), yb[order]


def augment(x, y):
    x = tf.cast(x, tf.float32)
    scale = tf.random.uniform([], 0.96, 1.04)
    noise = tf.random.normal(tf.shape(x), stddev=0.008)
    shift = tf.random.uniform([], -8, 9, dtype=tf.int32)
    n = tf.shape(x)[0]
    t = tf.cast(tf.range(n), tf.float32) / tf.cast(n, tf.float32)
    drift_amp = tf.random.uniform([], 0.0, 0.015)
    drift = drift_amp * tf.sin(2.0 * np.pi * t)
    x = x * scale + noise + drift[:, None]
    x = tf.roll(x, shift=shift, axis=0)
    return x, y


def make_ds(X, y, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_cnn(window_size, channels, n_classes, name):
    inp = layers.Input(shape=(window_size, channels))
    x = layers.Conv1D(32, 11, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    for filters in (64, 96, 128):
        residual = x
        x = layers.Conv1D(filters, 7, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.12)(x)
        x = layers.Conv1D(filters, 7, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        if residual.shape[-1] != filters:
            residual = layers.Conv1D(filters, 1, padding="same", use_bias=False)(residual)
            residual = layers.BatchNormalization()(residual)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)

    avg = layers.GlobalAveragePooling1D()(x)
    mx = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg, mx])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def train_one(X_train_text, X_val_text, X_test_text, train_labels, val_labels, test_labels,
              classes, model_path, report_dir, epochs, target_per_class, seed, binary=False):
    X_train, y_train_text = balanced_sample(X_train_text, train_labels, classes, target_per_class, seed)

    encoder = LabelEncoder()
    encoder.fit(classes)
    y_train = encoder.transform(y_train_text)
    y_val = encoder.transform(val_labels)
    y_test = encoder.transform(test_labels)

    model = make_cnn(X_train.shape[1], X_train.shape[2], len(classes), "cardiowatch_stage")
    ckpt = model_path
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    print(f"\nTraining {'binary' if binary else '3-class'} model")
    print("Balanced train:", X_train.shape)
    history = model.fit(
        make_ds(X_train, y_train),
        validation_data=(X_val_text, y_val),
        epochs=epochs,
        callbacks=cb,
        verbose=1,
    )

    model = tf.keras.models.load_model(ckpt, compile=False)
    probs = model.predict(X_test_text, verbose=0)

    threshold = 0.5
    if binary:
        # LabelEncoder sorts classes alphabetically, so for BINARY_CLASSES
        # the class order is ['Abnormal', 'Normal']. Always threshold the
        # actual Abnormal probability rather than assuming it is column 1.
        abnormal_index = int(np.where(np.asarray(encoder.classes_) == "Abnormal")[0][0])
        normal_index = int(np.where(np.asarray(encoder.classes_) == "Normal")[0][0])
        val_probs = model.predict(X_val_text, verbose=0)[:, abnormal_index]
        best_bal = -1.0
        for t in np.arange(0.25, 0.76, 0.01):
            preds = np.where(val_probs >= t, abnormal_index, normal_index)
            score = balanced_accuracy_score(y_val, preds)
            if score > best_bal:
                best_bal, threshold = float(score), float(t)
        predictions = np.where(probs[:, abnormal_index] >= threshold, abnormal_index, normal_index)
    else:
        predictions = np.argmax(probs, axis=1)

    bal_acc = float(balanced_accuracy_score(y_test, predictions))
    report = classification_report(
        y_test, predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_test, predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(len(encoder.classes_))).tolist()

    encoder_path = model_path.with_name(model_path.stem + "_encoder.joblib")
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    joblib.dump(encoder, encoder_path)
    save_json(meta_path, {
        "model": "2-lead 1D CNN",
        "window_size": int(X_train.shape[1]),
        "channels": 2,
        "sampling_rate": 360,
        "classes": list(encoder.classes_),
        "task": "Normal vs Abnormal" if binary else "Normal vs Ventricular vs OtherAbnormal",
        "threshold": threshold,
        "dataset": "MIT-BIH Arrhythmia Database",
        "test_protocol": "patient-disjoint DS2",
    })
    save_json(report_dir / ("binary_metrics.json" if binary else "three_class_metrics.json"), {
        "balanced_accuracy": bal_acc,
        "threshold": threshold,
    })
    save_json(report_dir / ("binary_report.json" if binary else "three_class_report.json"), report)
    save_json(report_dir / ("binary_confusion_matrix.json" if binary else "three_class_confusion_matrix.json"), matrix)
    save_json(report_dir / ("binary_history.json" if binary else "three_class_history.json"), {k: [float(v) for v in vals] for k, vals in history.history.items()})
    (report_dir / ("binary_report.txt" if binary else "three_class_report.txt")).write_text(report_text, encoding="utf-8")

    print("Test balanced accuracy:", round(bal_acc, 5))
    print(report_text)
    print("Confusion matrix:")
    print(matrix)
    print("Saved:", model_path)
    return bal_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=720)
    parser.add_argument("--max-per-class", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.hea")):
        raise RuntimeError("MIT-BIH data is missing. Run: python download_mitdb.py")

    missing = [r for r in DS1_RECORDS + DS2_RECORDS if not (DATA_DIR / f"{r}.hea").exists() or not (DATA_DIR / f"{r}.atr").exists()]
    if missing:
        raise RuntimeError(f"Missing MIT-BIH records/annotations: {missing}. Re-run python download_mitdb.py")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print("\n=== CARDIOWATCH AI: TWO-STAGE REAL ECG TRAINING ===")
    print("Window:", args.window_size, "samples (2 seconds at 360 Hz)")

    train_records, val_records, record_counts = choose_validation_records(DS1_RECORDS, args.window_size)
    print("Train records:", train_records)
    print("Validation records:", val_records)
    print("Test records:", DS2_RECORDS)

    X_train, y_train, _ = collect_records(train_records, args.window_size)
    X_val, y_val, _ = collect_records(val_records, args.window_size)
    X_test, y_test, _ = collect_records(DS2_RECORDS, args.window_size)

    y_train_bin = np.where(y_train == "Normal", "Normal", "Abnormal")
    y_val_bin = np.where(y_val == "Normal", "Normal", "Abnormal")
    y_test_bin = np.where(y_test == "Normal", "Normal", "Abnormal")

    print("\nNatural train counts:", {c: int(np.sum(y_train == c)) for c in THREE_CLASSES})
    print("Validation counts:", {c: int(np.sum(y_val == c)) for c in THREE_CLASSES})
    print("Test counts:", {c: int(np.sum(y_test == c)) for c in THREE_CLASSES})

    primary_target = min(args.max_per_class, 6000)
    subtype_target = min(args.max_per_class, 4000)

    binary_acc = train_one(
        X_train, X_val, X_test,
        y_train_bin, y_val_bin, y_test_bin,
        BINARY_CLASSES,
        MODEL_DIR / "binary_abnormal_model.keras",
        REPORT_DIR, args.epochs, primary_target, args.seed, binary=True,
    )
    three_acc = train_one(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        THREE_CLASSES,
        MODEL_DIR / "rhythm3_model.keras",
        REPORT_DIR, args.epochs, subtype_target, args.seed, binary=False,
    )

    save_json(REPORT_DIR / "training_summary.json", {
        "window_size": args.window_size,
        "sampling_rate": 360,
        "channels": 2,
        "binary_balanced_accuracy": binary_acc,
        "three_class_balanced_accuracy": three_acc,
        "train_records": train_records,
        "validation_records": val_records,
        "test_records": DS2_RECORDS,
        "record_class_counts": record_counts,
        "note": "Primary endpoint is balanced accuracy on patient-disjoint DS2. Results are research/educational only.",
    })
    print("\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
