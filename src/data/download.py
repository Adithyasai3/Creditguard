"""Download the Home Credit Default Risk dataset into data/raw/.

Run:  python -m src.data.download
Uses kagglehub (no API key required). Idempotent — skips if files exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.utils.config import DATA_RAW, DATASET_SLUG, REQUIRED_TABLES


def download() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # kagglehub is a lazy import so the module can be imported for tests
    # without the dependency installed.
    import kagglehub

    print(f"Downloading {DATASET_SLUG} via kagglehub ...")
    path = kagglehub.dataset_download(DATASET_SLUG)
    downloaded = sorted(Path(path).glob("*.csv"))

    for csv in downloaded:
        dest = DATA_RAW / csv.name
        if not dest.exists():
            dest.write_bytes(csv.read_bytes())
            print(f"  saved {dest.name}")
        else:
            print(f"  already present: {dest.name}")

    present = {p.stem for p in DATA_RAW.glob("*.csv")}
    missing = [t for t in REQUIRED_TABLES if t not in present]
    if missing:
        print("\nWARNING: expected tables missing:", ", ".join(missing))
        sys.exit(1)
    print("\nAll required tables present.")


if __name__ == "__main__":
    download()