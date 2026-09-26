"""Train a patient-aware two-lead ECG beat classifier on MIT-BIH.

Improvements over the first prototype:
- uses both ECG leads when available
- keeps the test set patient-disjoint (DS2)
- chooses validation records from DS1 before sampling beats
- balances ONLY the training set, leaving validation/test representative
- uses focal loss + label smoothing to emphasize hard examples
- uses ECG augmentation during training
- saves accuracy, balanced accuracy, precision/recall/F1 and confusion matrix

Run:
    python -m ml.train_model --epochs 20 --max-per-class 2500
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
from tensorflow.keras import callbacks, layers, models, regularizers

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

CLASS_MAP = {
    "N": "Normal", "L": "Normal", "R": "Normal", "e": "Normal", "j": "Normal",
    "A": "Supraventricular", "a": "Supraventricular", "J": "Supraventricular", "S": "Supraventricular",
    "V": "PVC", "E": "PVC",
    "F": "Fusion",
}
CLASS_ORDER = ["Normal", "Supraventricular", "PVC", "Fusion"]


def load_record_samples(record_name: str, window_size: int):
    path = DATA_DIR / record_name
    record = wfdb.rdrecord(str(path), channels=[0, 1] if wfdb.rdheader(str(path)).n_sig >= 2 else [0])
    signals = record.p_signal.astype(np.float32)
    signals = preprocess_multilead(signals, float(record.fs))
    ann = wfdb.rdann(str(path), "atr")

    samples, labels, groups = [], [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        label = CLASS_MAP.get(symbol)
        if label is None:
            continue
        window = extract_window(signals, int(sample), window_size)
        if window is None:
            continue
        samples.append(window)
        labels.append(label)
        groups.append(record_name)

    return np.asarray(samples, dtype=np.float32), np.asarray(labels), np.asarray(groups)


def record_class_counts(records, window_size):
    result = {}
    for record in records:
        x, y, _ = load_record_samples(record, window_size)
        result[record] = {c: int(np.sum(y == c)) for c in CLASS_ORDER}
    return result


def choose_validation_records(ds1_records, window_size, n_val=5, seed=42):
    """Pick whole records for validation, targeting ~20% of DS1 class counts."""
    counts = record_class_counts(ds1_records, window_size)
    total = {c: sum(counts[r][c] for r in ds1_records) for c in CLASS_ORDER}
    target_ratio = n_val / len(ds1_records)
    best = None

    for combo in itertools.combinations(ds1_records, n_val):
        val_counts = {c: sum(counts[r][c] for r in combo) for c in CLASS_ORDER}
        # Require at least two validation classes and preferably all classes.
        missing = sum(val_counts[c] == 0 for c in CLASS_ORDER)
        ratio_error = sum(abs((val_counts[c] / max(total[c], 1)) - target_ratio) for c in CLASS_ORDER)
        size_error = abs((sum(val_counts.values()) / max(sum(total.values()), 1)) - target_ratio)
        score = missing * 5.0 + ratio_error + size_error
        key = (score, tuple(combo))
        if best is None or key < best[0]:
            best = (key, combo, counts)

    assert best is not None
    val_records = list(best[1])
    train_records = [r for r in ds1_records if r not in val_records]
    return train_records, val_records, counts


def collect_records(records, window_size):
    xs, ys, gs = [], [], []
    for record in records:
        print(f"Loading record {record}...")
        x, y, g = load_record_samples(record, window_size)
        xs.append(x)
        ys.append(y)
        gs.append(g)
        print("  samples:", len(x), "classes:", {c: int(np.sum(y == c)) for c in CLASS_ORDER})
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(gs, axis=0)


def class_counts(y):
    return {c: int(np.sum(y == c)) for c in CLASS_ORDER}


def balance_training(X, y, target_per_class=2500, seed=42):
    rng = np.random.default_rng(seed)
    parts_x, parts_y = [], []
    for label in CLASS_ORDER:
        idx = np.flatnonzero(y == label)
        if len(idx) == 0:
            raise RuntimeError(f"Training data has no samples for class {label}.")
        replace = len(idx) < target_per_class
        chosen = rng.choice(idx, size=target_per_class, replace=replace)
        parts_x.append(X[chosen])
        parts_y.append(y[chosen])
    order = rng.permutation(target_per_class * len(CLASS_ORDER))
    Xb = np.concatenate(parts_x, axis=0)[order]
    yb = np.concatenate(parts_y, axis=0)[order]
    return Xb.astype(np.float32), yb


def focal_loss(gamma=1.5, label_smoothing=0.04):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        if label_smoothing:
            depth = tf.cast(tf.shape(y_pred)[-1], tf.float32)
            y = y * (1.0 - label_smoothing) + label_smoothing / depth
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return loss_fn


def build_model(window_size, channels, n_classes):
    inp = layers.Input(shape=(window_size, channels))
    x = layers.Conv1D(64, 9, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    def block(t, filters, stride=1, kernel=5):
        shortcut = t
        y = layers.Conv1D(filters, kernel, strides=stride, padding="same", use_bias=False)(t)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.SpatialDropout1D(0.08)(y)
        y = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        if int(shortcut.shape[-1]) != filters or stride != 1:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        y = layers.Add()([shortcut, y])
        return layers.Activation("relu")(y)

    x = block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = block(x, 128)
    x = layers.MaxPooling1D(2)(x)
    x = block(x, 256)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="cardiowatch_resnet1d_two_lead")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=1.0),
        loss=focal_loss(),
        metrics=["accuracy"],
    )
    return model


def augment(x, y):
    x = tf.cast(x, tf.float32)
    scale = tf.random.uniform([], 0.94, 1.06)
    noise = tf.random.normal(tf.shape(x), stddev=0.012)
    shift = tf.random.uniform([], -6, 7, dtype=tf.int32)
    baseline_amp = tf.random.uniform([], 0.0, 0.025)
    n = tf.shape(x)[0]
    t = tf.cast(tf.range(n), tf.float32) / tf.cast(n, tf.float32)
    drift = baseline_amp * tf.sin(2.0 * np.pi * t)
    x = x * scale + noise + drift[:, None]
    x = tf.roll(x, shift=shift, axis=0)
    return x, y


def make_train_ds(X, y, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=360)
    parser.add_argument("--max-per-class", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.hea")):
        raise RuntimeError("MIT-BIH data is missing. Run: python download_mitdb.py")
    missing = [r for r in DS1_RECORDS + DS2_RECORDS if not (DATA_DIR / f"{r}.hea").exists() or not (DATA_DIR / f"{r}.atr").exists()]
    if missing:
        raise RuntimeError(f"Missing MIT-BIH records/annotations: {missing}. Re-run python download_mitdb.py")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print("\n=== CARDIOWATCH AI: IMPROVED REAL ECG TRAINING ===")

    train_records, val_records, record_counts = choose_validation_records(DS1_RECORDS, args.window_size, n_val=5, seed=args.seed)
    print("Train records:", train_records)
    print("Validation records:", val_records)
    print("Test records:", DS2_RECORDS)

    X_train_raw, y_train_raw, _ = collect_records(train_records, args.window_size)
    X_val, y_val_text, _ = collect_records(val_records, args.window_size)
    X_test, y_test_text, _ = collect_records(DS2_RECORDS, args.window_size)

    X_train, y_train_text = balance_training(X_train_raw, y_train_raw, args.max_per_class, args.seed)

    encoder = LabelEncoder()
    encoder.fit(CLASS_ORDER)
    y_train = encoder.transform(y_train_text)
    y_val = encoder.transform(y_val_text)
    y_test = encoder.transform(y_test_text)

    print("\nClasses:", list(encoder.classes_))
    print("Balanced train:", X_train.shape, class_counts(y_train_text))
    print("Validation:", X_val.shape, class_counts(y_val_text))
    print("Test:", X_test.shape, class_counts(y_test_text))

    model = build_model(args.window_size, X_train.shape[-1], len(encoder.classes_))
    model.summary()

    ckpt = MODEL_DIR / "arrhythmia_model.keras"
    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", mode="max", save_best_only=True),
    ]

    history = model.fit(
        make_train_ds(X_train, y_train),
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        callbacks=cb,
        verbose=1,
    )

    model = tf.keras.models.load_model(ckpt, custom_objects={"loss_fn": focal_loss()})
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report_dict = classification_report(
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
    bal_acc = float(balanced_accuracy_score(y_test, predictions))

    print("\nTest loss:", round(float(loss), 5))
    print("Test accuracy:", round(float(accuracy), 5))
    print("Balanced accuracy:", round(bal_acc, 5))
    print("\nClassification report\n", report_text)
    print("Confusion matrix\n", matrix)

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    save_json(MODEL_DIR / "model_meta.json", {
        "window_size": args.window_size,
        "sampling_rate": 360,
        "channels": 2,
        "classes": list(encoder.classes_),
        "model": "Two-lead 1D ResNet-style CNN with focal loss",
        "dataset": "MIT-BIH Arrhythmia Database",
        "evaluation": "patient-disjoint DS2 test; whole-record validation from DS1",
        "train_records": train_records,
        "validation_records": val_records,
        "test_records": DS2_RECORDS,
    })
    (REPORT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_json(REPORT_DIR / "classification_report.json", report_dict)
    save_json(REPORT_DIR / "confusion_matrix.json", matrix)
    save_json(REPORT_DIR / "metrics.json", {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "balanced_accuracy": bal_acc,
    })
    save_json(REPORT_DIR / "record_split.json", {
        "train_records": train_records,
        "validation_records": val_records,
        "test_records": DS2_RECORDS,
        "record_class_counts": record_counts,
    })
    save_json(REPORT_DIR / "training_history.json", {k: [float(v) for v in vals] for k, vals in history.history.items()})

    print("\nSaved model:", ckpt)
    print("Saved reports:", REPORT_DIR)


if __name__ == "__main__":
    main()
