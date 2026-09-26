from pathlib import Path
import wfdb

BASE_DIR = Path(__file__).resolve().parent
TARGET = BASE_DIR / "data" / "mitdb"
TARGET.mkdir(parents=True, exist_ok=True)

print(f"Downloading MIT-BIH Arrhythmia Database to: {TARGET}")
wfdb.dl_database("mitdb", dl_dir=str(TARGET))
print("Download complete.")
