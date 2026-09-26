import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def _safe_filter(signal, fs, low=0.5, high=40.0, order=4):
    x = np.asarray(signal, dtype=np.float32)
    if len(x) < max(30, int(fs * 2)):
        return x
    nyquist = 0.5 * float(fs)
    high = min(float(high), nyquist * 0.90)
    if high <= low:
        return x
    try:
        b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
        return filtfilt(b, a, x).astype(np.float32)
    except Exception:
        return x


def _safe_notch(signal, fs, frequency=50.0, q=30.0):
    x = np.asarray(signal, dtype=np.float32)
    if len(x) < max(30, int(fs * 2)) or frequency >= fs / 2:
        return x
    try:
        b, a = iirnotch(frequency / (fs / 2), q)
        return filtfilt(b, a, x).astype(np.float32)
    except Exception:
        return x


def normalize(signal):
    x = np.asarray(signal, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(np.mean(x))
    std = float(np.std(x))
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def preprocess_ecg(signal, fs):
    x = _safe_filter(signal, fs)
    x = _safe_notch(x, fs)
    return normalize(x)


def preprocess_multilead(signals, fs):
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim == 1:
        return preprocess_ecg(x, fs)[:, None]
    out = np.zeros_like(x, dtype=np.float32)
    for ch in range(x.shape[1]):
        out[:, ch] = preprocess_ecg(x[:, ch], fs)
    return out


def extract_center_window(signal, window_size):
    x = np.asarray(signal, dtype=np.float32)
    if len(x) < window_size:
        raise ValueError(f"ECG must contain at least {window_size} samples.")
    start = (len(x) - window_size) // 2
    return x[start:start + window_size], start


def extract_window(signal, center, window_size):
    x = np.asarray(signal, dtype=np.float32)
    half = window_size // 2
    start = int(center) - half
    end = start + window_size
    if start < 0 or end > len(x):
        return None
    return x[start:end]
