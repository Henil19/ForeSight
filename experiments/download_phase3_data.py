"""Download one AAPL snapshot for the isolated Phase 3 experiments.

This script refuses to overwrite an existing snapshot. Experiment code reads
the saved cleaned CSV and never contacts Yahoo Finance.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module


DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "aapl_5y_yfinance_raw.csv"
CLEAN_PATH = DATA_DIR / "aapl_5y_clean.csv"
METADATA_PATH = DATA_DIR / "aapl_5y_metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if any(path.exists() for path in (RAW_PATH, CLEAN_PATH, METADATA_PATH)):
        raise FileExistsError(
            "A Phase 3 data snapshot already exists; refusing to download or overwrite it."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc)
    raw = yf.download("AAPL", period="5y", progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no AAPL observations.")
    raw.to_csv(RAW_PATH, index_label="Date")

    original_download = data_module.yf.download
    data_module.yf.download = lambda *args, **kwargs: raw.copy()
    try:
        cleaned = data_module.download_stock_data("AAPL", period="5y")
    finally:
        data_module.yf.download = original_download
    cleaned.to_csv(CLEAN_PATH, index_label="Date", float_format="%.12g")

    metadata = {
        "ticker": "AAPL",
        "requested_period": "5y",
        "downloaded_at_utc": downloaded_at.isoformat(),
        "raw_first_date": str(raw.index.min()),
        "raw_last_date": str(raw.index.max()),
        "clean_first_date": str(cleaned.index.min()),
        "clean_last_date": str(cleaned.index.max()),
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(cleaned)),
        "raw_file": RAW_PATH.name,
        "raw_sha256": _sha256(RAW_PATH),
        "clean_file": CLEAN_PATH.name,
        "clean_sha256": _sha256(CLEAN_PATH),
        "source": "Yahoo Finance via yfinance; auto_adjust=False",
        "cleaning_steps": [
            "Flatten yfinance OHLCV MultiIndex columns when present.",
            "Normalize required fields to Open, High, Low, Close, and Volume.",
            "Convert index values to dates and discard invalid dates.",
            "Sort chronologically and keep the last occurrence of duplicate dates.",
            "Convert required OHLCV fields to numeric values.",
            "Drop rows with missing or non-numeric required OHLCV values.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
