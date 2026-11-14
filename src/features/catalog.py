"""Generates reports/feature_catalog.{md,csv} from the registry plus real statistics."""
from __future__ import annotations

import pandas as pd

from src.common.io import df_to_markdown
from src.validation.schemas import CANONICAL_GENRES


def _stats(series: pd.Series) -> dict:
    s = series.dropna()
    if pd.api.types.is_numeric_dtype(series):
        return {"null_pct": series.isna().mean() * 100, "min": s.min(), "mean": s.mean(), "max": s.max(), "top_value": ""}
    top = s.value_counts()
    return {"null_pct": series.isna().mean() * 100, "min": None, "mean": None, "max": None,
            "top_value": f"{top.index[0]} ({top.iloc[0] / len(s):.0%})" if len(top) else ""}


def build_catalog(registry: dict, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for view_name, view in registry["feature_views"].items():
        table = view["storage"]
        df = frames.get(table)
        for f in view["features"]:
            row = {"feature": f["name"], "feature_view": view_name, "entity": view["entity"], "dtype": f["dtype"],
                   "description": f["description"], "logic": f["logic"], "source": f["source"], "storage": table}
            if df is not None and "<genre>" in f["name"]:
                cols = [f["name"].replace("<genre>", g) for g in CANONICAL_GENRES]
                vals = df[cols].stack()
                row.update(_stats(vals))
                row["feature"] = f["name"] + f" ({len(cols)} columns)"
            elif df is not None and f["name"] in df.columns:
                row.update(_stats(df[f["name"]]))
            else:
                row.update({"null_pct": None, "min": None, "mean": None, "max": None, "top_value": "computed per request"})
            rows.append(row)
    return pd.DataFrame(rows)


def write_catalog(registry: dict, frames: dict[str, pd.DataFrame], snapshot_meta: dict, out_dir) -> pd.DataFrame:
    cat = build_catalog(registry, frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    cat.to_csv(out_dir / "feature_catalog.csv", index=False)

    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float):
            return f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.0f}"
        return str(v)

    view = cat.copy()
    view["stats (min / mean / max or top)"] = [
        t if t else f"{fmt(a)} / {fmt(b)} / {fmt(c)}" for a, b, c, t in
        zip(view["min"], view["mean"], view["max"], view["top_value"])]
    view["null %"] = view["null_pct"].map(lambda x: "" if x is None or pd.isna(x) else f"{x:.1f}")
    cols = ["feature", "entity", "dtype", "description", "logic", "source", "null %", "stats (min / mean / max or top)"]

    lines = ["# Feature Catalog", "",
             f"Generated from `configs/feature_registry.yaml` (version {registry['version']}). Statistics are from "
             f"the `{snapshot_meta['snapshot']}` snapshot, computed as of {snapshot_meta['reference_time']} "
             "using only interactions on or before that time.", "",
             "Offline store: `data/features/<snapshot>/*.parquet`. Online store: `data/features/online_store.db` (SQLite).", ""]
    for view_name in cat["feature_view"].unique():
        part = view[view["feature_view"] == view_name]
        storage = part["storage"].iloc[0]
        lines += [f"## {view_name}", "", f"Stored in: `{storage}`", "", df_to_markdown(part[cols]), ""]
    (out_dir / "feature_catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cat
