from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
fs = 360
duration = 20
rng = np.random.default_rng(42)
t = np.arange(0, duration, 1 / fs)

# Synthetic ECG-like signal for software demonstration only.
period = 0.82
phase = np.mod(t, period)
p = 0.12 * np.exp(-0.5 * ((phase - 0.20) / 0.035) ** 2)
q = -0.10 * np.exp(-0.5 * ((phase - 0.30) / 0.012) ** 2)
r = 1.10 * np.exp(-0.5 * ((phase - 0.325) / 0.012) ** 2)
s = -0.24 * np.exp(-0.5 * ((phase - 0.355) / 0.015) ** 2)
tw = 0.28 * np.exp(-0.5 * ((phase - 0.50) / 0.065) ** 2)
baseline = 0.03 * np.sin(2 * np.pi * 0.25 * t)
noise = 0.025 * rng.normal(size=len(t))

# Add a few wider complexes to create a visible non-normal demo pattern.
signal = baseline + p + q + r + s + tw + noise
for center in [5.1, 11.7, 16.6]:
    mask = np.abs(t - center) < 0.08
    signal[mask] += 0.8 * np.exp(-0.5 * ((t[mask] - center) / 0.035) ** 2)

out = BASE_DIR / "test_ecg.csv"
pd.DataFrame({"ecg": signal}).to_csv(out, index=False)
print(f"Created {out} with {len(signal)} samples at {fs} Hz")
