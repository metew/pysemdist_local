from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import pandas as pd

REQUIRED_FIELDS = {"id", "text"}


def load_petitions(path: str | Path) -> pd.DataFrame:
    """
    Load petitions from a JSON / JSONL / CSV file.

    Expected columns:
      - id
      - text
    Optionally:
      - category
      - locale
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        df = pd.read_json(path)
    elif suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use .json, .jsonl, or .csv")

    # Basic sanity check
    required = {"id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")

    return df


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def to_parquet_append(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs([path])
    if path.exists():
        existing = pd.read_parquet(path)
        pd.concat([existing, df], ignore_index=True).to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)
