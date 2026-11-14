"""A small feature store, modelled on the offline/online split used by Feast.

offline  data/features/<snapshot>/<table>.parquet
         Full point-in-time snapshots, one per reference time. Training and
         tuning read from here.
online   data/features/online_store.db (SQLite)
         The latest snapshot keyed by entity id, for single-user lookups at
         prediction time.

Both sides are written from the same frames, so a model sees identical
feature values whether it is being trained or asked for a recommendation.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config, load_feature_registry
from src.common.io import read_json, utc_now, write_json
from src.validation.schemas import CANONICAL_GENRES

KEY_COLUMNS = {"user_id", "item_id"}
CONTEXT_COLUMNS = {"interaction_features": {"rating", "event_time"}}
DTYPE_FAMILY = {"int": "i", "float": "f", "string": "O"}


class FeatureContractError(ValueError):
    pass


class FeatureStore:
    def __init__(self, cfg: Config):
        self.root: Path = cfg.path("features")
        self.registry = load_feature_registry()
        self.online_path = self.root / "online_store.db"

    # ---------------------------------------------------------- registry
    def registered_columns(self) -> dict[str, dict[str, str]]:
        """storage table -> {column: dtype} with the <genre> wildcards expanded."""
        out: dict[str, dict[str, str]] = {}
        for view in self.registry["feature_views"].values():
            table = view["storage"]
            if table == "computed at request time":
                continue
            cols = out.setdefault(table, {})
            for f in view["features"]:
                if "<genre>" in f["name"]:
                    for g in CANONICAL_GENRES:
                        cols[f["name"].replace("<genre>", g)] = f["dtype"]
                else:
                    cols[f["name"]] = f["dtype"]
        return out

    def check_contract(self, frames: dict[str, pd.DataFrame]) -> None:
        """Fail loudly if a computed table drifts from the registry."""
        problems = []
        for table, expected in self.registered_columns().items():
            df = frames.get(table)
            if df is None:
                problems.append(f"{table}: missing")
                continue
            allowed = set(expected) | KEY_COLUMNS | CONTEXT_COLUMNS.get(table, set())
            extra = set(df.columns) - allowed
            missing = set(expected) - set(df.columns)
            if extra:
                problems.append(f"{table}: unregistered columns {sorted(extra)}")
            if missing:
                problems.append(f"{table}: registered but not produced {sorted(missing)}")
            for col, dtype in expected.items():
                if col in df.columns:
                    kind = df[col].dtype.kind
                    if DTYPE_FAMILY[dtype] == "O":
                        ok = kind in "OUT" or pd.api.types.is_string_dtype(df[col])
                    else:
                        ok = kind == DTYPE_FAMILY[dtype]
                    if not ok:
                        problems.append(f"{table}.{col}: expected {dtype}, got {df[col].dtype}")
        if problems:
            raise FeatureContractError("; ".join(problems))

    # ----------------------------------------------------------- offline
    def materialize(self, snapshot: str, frames: dict[str, pd.DataFrame], reference_time: pd.Timestamp,
                    source: dict) -> dict:
        self.check_contract(frames)
        snap_dir = self.root / snapshot
        snap_dir.mkdir(parents=True, exist_ok=True)
        for name, df in frames.items():
            df.to_parquet(snap_dir / f"{name}.parquet", index=False)
        meta = {"snapshot": snapshot, "reference_time": reference_time.isoformat(), "created_at": utc_now(),
                "registry_version": self.registry["version"], "source": source,
                "tables": {k: {"rows": len(v), "columns": len(v.columns)} for k, v in frames.items()}}
        write_json(meta, snap_dir / "_metadata.json")
        return meta

    def load(self, snapshot: str) -> dict[str, pd.DataFrame]:
        snap_dir = self.root / snapshot
        return {p.stem: pd.read_parquet(p) for p in sorted(snap_dir.glob("*.parquet"))}

    def metadata(self, snapshot: str) -> dict:
        return read_json(self.root / snapshot / "_metadata.json")

    def training_frame(self, snapshot: str) -> pd.DataFrame:
        """Observed ratings joined with the user and item features known at that time."""
        f = self.load(snapshot)
        return (f["interaction_features"]
                .merge(f["user_features"], on="user_id", how="left")
                .merge(f["item_features"], on="item_id", how="left"))

    # ------------------------------------------------------------ online
    def publish_online(self, snapshot: str) -> None:
        frames = self.load(snapshot)
        meta = self.metadata(snapshot)
        with sqlite3.connect(self.online_path) as con:
            for name in ("user_features", "user_genre_affinity", "item_features", "item_genres"):
                frames[name].to_sql(name, con, if_exists="replace", index=False)
                key = "user_id" if name.startswith("user") else "item_id"
                con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ix_{name} ON {name}({key})")
            pd.DataFrame([{"snapshot": snapshot, "reference_time": meta["reference_time"],
                           "published_at": utc_now()}]).to_sql("_store_info", con, if_exists="replace", index=False)

    def _online(self, table: str, key: str, ids) -> pd.DataFrame:
        ids = [int(i) for i in np.atleast_1d(ids)]
        marks = ",".join("?" * len(ids))
        with sqlite3.connect(self.online_path) as con:
            return pd.read_sql_query(f"SELECT * FROM {table} WHERE {key} IN ({marks})", con, params=ids)

    def online_user_features(self, user_ids) -> pd.DataFrame:
        return self._online("user_features", "user_id", user_ids)

    def online_genre_affinity(self, user_ids) -> pd.DataFrame:
        return self._online("user_genre_affinity", "user_id", user_ids)

    def online_item_features(self, item_ids) -> pd.DataFrame:
        return self._online("item_features", "item_id", item_ids)

    def online_info(self) -> dict:
        with sqlite3.connect(self.online_path) as con:
            return pd.read_sql_query("SELECT * FROM _store_info", con).iloc[0].to_dict()
