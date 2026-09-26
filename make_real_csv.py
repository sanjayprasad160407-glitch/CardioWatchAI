import argparse
from pathlib import Path

import pandas as pd
import wfdb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "mitdb"

parser = argparse.ArgumentParser(description="Create a two-lead CSV from a MIT-BIH record.")
parser.add_argument("--record", default="100")
parser.add_argument("--seconds", type=float, default=20)
args = parser.parse_args()

record_base = DATA_DIR / args.record
header = wfdb.rdheader(str(record_base))
if header.n_sig < 2:
    raise RuntimeError(f"Record {args.record} does not contain two ECG leads.")

record = wfdb.rdrecord(str(record_base), channels=[0, 1])
fs = float(record.fs)
count = max(360, int(args.seconds * fs))
values = record.p_signal[:count, :2]

out = BASE_DIR / f"real_ecg_{args.record}_two_lead.csv"
pd.DataFrame({"lead1": values[:, 0], "lead2": values[:, 1]}).to_csv(out, index=False)
print(f"Created {out}")
print(f"Record={args.record}, samples={len(values)}, leads=2, sampling_rate={fs}")
