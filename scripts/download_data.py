"""Download the three Metrica Sports open sample games into data/raw/ (about 180 MB).

Source: https://github.com/metrica-sports/sample-data. The data is not committed to this
repository; run this once after cloning.

    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = {
    1: ["RawEventsData.csv", "RawTrackingData_Away_Team.csv", "RawTrackingData_Home_Team.csv"],
    2: ["RawEventsData.csv", "RawTrackingData_Away_Team.csv", "RawTrackingData_Home_Team.csv"],
    3: ["events.json", "metadata.xml", "tracking.txt"],
}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    for game, suffixes in FILES.items():
        folder = RAW / f"g{game}"
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in suffixes:
            name = f"Sample_Game_{game}_{suffix}"
            target = folder / name
            if target.exists() and target.stat().st_size > 0:
                print(f"  have {target.relative_to(RAW.parent)}")
                continue
            print(f"  get  {name} ...", flush=True)
            urllib.request.urlretrieve(f"{BASE}/Sample_Game_{game}/{name}", target)
    print("done:", RAW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
