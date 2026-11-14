"""Small file helpers shared by the stages."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a dataframe, used to tie a model run to its exact data."""
    hashed = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.md5(hashed.tobytes()).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Pipe table without needing the optional `tabulate` dependency."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
