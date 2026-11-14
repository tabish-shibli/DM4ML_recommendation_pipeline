"""Stage 5 - Feature management.

Builds two point-in-time snapshots with the same code:
  tuning    as of the tuning cutoff, from the inner training window
  training  as of the test cutoff, from the full training window

Both are checked against configs/feature_registry.yaml before they are
written. The training snapshot is then published to the online store and
documented in reports/feature_catalog.md.
"""
from __future__ import annotations

import pandas as pd

from src.common.config import Config, load_config
from src.common.io import frame_fingerprint, read_json
from src.common.logging_setup import get_logger
from src.features.build import build_all
from src.features.catalog import write_catalog
from src.features.store import FeatureStore

log = get_logger(__name__)

SNAPSHOTS = {"tuning": ("train_inner", "tuning_split"), "training": ("train", "test_split")}


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    pdir = cfg.path("processed")
    users = pd.read_parquet(pdir / "users_clean.parquet")
    items = pd.read_parquet(pdir / "items_clean.parquet")
    split_info = read_json(pdir / "splits" / "split_info.json")
    store = FeatureStore(cfg)

    metas = {}
    for snapshot, (split_name, info_key) in SNAPSHOTS.items():
        window = pd.read_parquet(pdir / "splits" / f"{split_name}.parquet")
        ref = pd.Timestamp(split_info[info_key]["cutoff"])
        frames = build_all(window, users, items, ref, cfg["features"])
        source = {"split": split_name, "rows": len(window),
                  "fingerprint": frame_fingerprint(window[["user_id", "item_id", "rating", "timestamp"]])}
        metas[snapshot] = store.materialize(snapshot, frames, ref, source)
        log.info("Snapshot %-8s as of %s: %s", snapshot, ref.date(),
                 ", ".join(f"{k}={v['rows']}" for k, v in metas[snapshot]["tables"].items()))

    store.publish_online("training")
    log.info("Published 'training' snapshot to %s", store.online_path.name)

    write_catalog(store.registry, store.load("training"), metas["training"], cfg.path("reports"))
    log.info("Feature catalog written")
    return {"stage": "features", "snapshots": metas}


if __name__ == "__main__":
    run()
