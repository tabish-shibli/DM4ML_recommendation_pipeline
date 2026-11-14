"""Stage 1 - Ingestion.

Downloads MovieLens 1M, converts the three `::`-delimited files into landing
CSVs with consistent column names, optionally simulates a dirty feed, and
writes a manifest describing exactly what landed.

Output: data/raw/landing/{users,items,interactions}.csv + _manifest.json
"""
from __future__ import annotations

import pandas as pd

from src.common.config import PROJECT_ROOT, Config, load_config
from src.common.io import file_md5, utc_now, write_json
from src.common.logging_setup import get_logger
from src.ingestion.dirty_feed import DirtyFeedSimulator
from src.ingestion.download import download_archive, extract_archive

log = get_logger(__name__)

# MovieLens 1M file layouts (see the dataset README)
LAYOUTS = {
    "users": ("users.dat", ["user_id", "gender", "age", "occupation", "zip_code"]),
    "items": ("movies.dat", ["item_id", "title", "genres"]),
    "interactions": ("ratings.dat", ["user_id", "item_id", "rating", "timestamp"]),
}


def _read_dat(path, columns) -> pd.DataFrame:
    # movies.dat contains Latin-1 characters (e.g. accented French titles);
    # everything is kept as text here, typing is the validation stage's job
    rows = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            rows.append(line.rstrip("\n").split("::"))
    return pd.DataFrame(rows, columns=columns)


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    icfg = cfg["ingestion"]
    raw_dir = cfg.path("raw")
    landing = raw_dir / "landing"
    landing.mkdir(parents=True, exist_ok=True)

    archive = download_archive(icfg["source_url"], raw_dir / "ml-1m.zip", icfg.get("md5"))
    extract_archive(archive, raw_dir)
    src_dir = raw_dir / icfg["archive_dir"]

    frames = {}
    for name, (fname, cols) in LAYOUTS.items():
        frames[name] = _read_dat(src_dir / fname, cols)
        log.info("Read %-12s %9d rows from %s", name, len(frames[name]), fname)
    frames["interactions"]["event_type"] = "rating"

    injected = None
    if icfg.get("simulate_dirty_feed"):
        sim = DirtyFeedSimulator(icfg["dirty_feed"], seed=icfg["dirty_feed"]["seed"])
        frames["users"] = sim.users(frames["users"])
        frames["items"] = sim.items(frames["items"])
        frames["interactions"] = sim.interactions(frames["interactions"])
        injected = sim.ground_truth()
        injected.to_csv(landing / "_injected_anomalies.csv", index=False)
        log.info("Dirty-feed simulation added %d logged defects", len(injected))

    manifest = {
        "stage": "ingestion",
        "source": icfg["source_name"],
        "source_url": icfg["source_url"],
        "archive_md5": file_md5(archive),
        "ingested_at": utc_now(),
        "dirty_feed_simulated": bool(icfg.get("simulate_dirty_feed")),
        "files": {},
    }
    for name, df in frames.items():
        out = landing / f"{name}.csv"
        df.to_csv(out, index=False)
        manifest["files"][name] = {"path": out.relative_to(PROJECT_ROOT).as_posix(), "rows": len(df),
                                   "columns": list(df.columns), "md5": file_md5(out)}
        log.info("Landed %-12s %9d rows -> %s", name, len(df), out.name)
    if injected is not None:
        manifest["injected_defects"] = injected.groupby(["table", "anomaly"]).size().reset_index(name="count").to_dict("records")

    write_json(manifest, landing / "_manifest.json")
    return manifest


if __name__ == "__main__":
    run()
