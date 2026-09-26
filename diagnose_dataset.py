"""Print MIT-BIH class counts for the CardioWatch AI training/test records."""
from collections import Counter
from pathlib import Path
import wfdb

DATA_DIR = Path(__file__).resolve().parent / "data" / "mitdb"
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
MAP = {"N":"Normal", "L":"Normal", "R":"Normal", "e":"Normal", "j":"Normal", "A":"Supraventricular", "a":"Supraventricular", "J":"Supraventricular", "S":"Supraventricular", "V":"PVC", "E":"PVC", "F":"Fusion"}

for name, records in (("DS1", DS1), ("DS2", DS2)):
    total = Counter()
    print(f"\n{name}")
    for r in records:
        ann = wfdb.rdann(str(DATA_DIR / r), "atr")
        c = Counter(MAP[s] for s in ann.symbol if s in MAP)
        total.update(c)
        print(r, dict(c))
    print("TOTAL", dict(total))
