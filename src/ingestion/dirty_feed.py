"""Adds a seeded, logged set of realistic defects to the landing data.

MovieLens 1M is a curated research dataset and has almost nothing for a
validation stage to catch. A production feed is never that tidy, so this
module simulates the usual suspects: double-sent events, empty keys, ratings
typed as text, timestamps from a broken clock, events for users or movies the
catalogue has never heard of, and inconsistent formatting in the dimension
tables.

Every defect is written to a ground-truth log so the validation stage can be
checked against it. The downloaded archive itself is never changed.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _epoch(year: int, rng: np.random.Generator) -> str:
    start = datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()
    return str(int(start + rng.integers(0, 365 * 86400)))


class DirtyFeedSimulator:
    def __init__(self, spec: dict, seed: int):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.log: list[dict] = []

    def _record(self, table: str, anomaly: str, handling: str, **keys) -> None:
        self.log.append({"table": table, "anomaly": anomaly, "expected_handling": handling, **keys})

    def _take(self, pool: list[int], n: int) -> list[int]:
        """Pop n distinct row positions so two defects never hit the same row."""
        picked = pool[:n]
        del pool[:n]
        return picked

    # ------------------------------------------------------------------ users
    def users(self, users: pd.DataFrame) -> pd.DataFrame:
        s = self.spec["users"]
        df = users.copy()
        pool = list(self.rng.permutation(len(df)))

        for pos in self._take(pool, s["missing_gender"]):
            df.iat[pos, df.columns.get_loc("gender")] = ""
            self._record("users", "missing_gender", "impute", user_id=df.iat[pos, 0])
        messy = ["male", "female", "m", " F", "Female ", "MALE"]
        for pos in self._take(pool, s["messy_gender"]):
            df.iat[pos, df.columns.get_loc("gender")] = str(self.rng.choice(messy))
            self._record("users", "messy_gender", "standardise", user_id=df.iat[pos, 0])
        for pos in self._take(pool, s["invalid_age"]):
            df.iat[pos, df.columns.get_loc("age")] = str(self.rng.choice(["99", "0", "-1", "200"]))
            self._record("users", "invalid_age", "impute", user_id=df.iat[pos, 0])
        for pos in self._take(pool, s["missing_zip"]):
            df.iat[pos, df.columns.get_loc("zip_code")] = ""
            self._record("users", "missing_zip", "impute", user_id=df.iat[pos, 0])

        extra = []
        dup = df.iloc[self.rng.choice(len(df), s["exact_duplicates"], replace=False)].copy()
        for uid in dup["user_id"]:
            self._record("users", "exact_duplicate", "quarantine", user_id=uid)
        extra.append(dup)
        broken = df.iloc[self.rng.choice(len(df), s["missing_user_id"], replace=False)].copy()
        broken["user_id"] = ""
        for _ in range(len(broken)):
            self._record("users", "missing_user_id", "quarantine", user_id="")
        extra.append(broken)
        return pd.concat([df, *extra], ignore_index=True)

    # ------------------------------------------------------------------ items
    def items(self, items: pd.DataFrame) -> pd.DataFrame:
        s = self.spec["items"]
        df = items.copy()
        pool = list(self.rng.permutation(len(df)))

        for pos in self._take(pool, s["missing_title"]):
            df.iat[pos, df.columns.get_loc("title")] = ""
            self._record("items", "missing_title", "impute", item_id=df.iat[pos, 0])
        for pos in self._take(pool, s["messy_genres"]):
            original = df.iat[pos, df.columns.get_loc("genres")]
            style = self.rng.integers(0, 3)
            if style == 0:
                g = original.lower()
            elif style == 1:
                g = " | ".join(original.split("|"))
            else:
                g = original.upper() + "|"
            if g == original:  # padding the separator does nothing to a one-genre movie
                g = original.lower()
            df.iat[pos, df.columns.get_loc("genres")] = g
            self._record("items", "messy_genres", "standardise", item_id=df.iat[pos, 0])
        for pos in self._take(pool, s["missing_genres"]):
            df.iat[pos, df.columns.get_loc("genres")] = ""
            self._record("items", "missing_genres", "impute", item_id=df.iat[pos, 0])

        dup = df.iloc[self.rng.choice(len(df), s["exact_duplicates"], replace=False)].copy()
        for iid in dup["item_id"]:
            self._record("items", "exact_duplicate", "quarantine", item_id=iid)
        broken = df.iloc[self.rng.choice(len(df), s["missing_item_id"], replace=False)].copy()
        broken["item_id"] = ""
        for _ in range(len(broken)):
            self._record("items", "missing_item_id", "quarantine", item_id="")
        return pd.concat([df, dup, broken], ignore_index=True)

    # ----------------------------------------------------------- interactions
    def interactions(self, inter: pd.DataFrame) -> pd.DataFrame:
        s = self.spec["interactions"]
        df = inter.copy()
        cols = {c: df.columns.get_loc(c) for c in df.columns}
        pool = list(self.rng.permutation(len(df)))

        def keys(pos: int) -> dict:
            return {"user_id": df.iat[pos, cols["user_id"]], "item_id": df.iat[pos, cols["item_id"]]}

        for i, pos in enumerate(self._take(pool, s["missing_keys"])):
            k = keys(pos)
            df.iat[pos, cols["user_id" if i % 2 == 0 else "item_id"]] = ""
            self._record("interactions", "missing_key", "quarantine", **k)
        for pos in self._take(pool, s["missing_rating"]):
            df.iat[pos, cols["rating"]] = ""
            self._record("interactions", "missing_rating", "quarantine", **keys(pos))
        for pos in self._take(pool, s["non_numeric_rating"]):
            df.iat[pos, cols["rating"]] = str(self.rng.choice(["five", "4 stars", "N/A", "?", "3,5"]))
            self._record("interactions", "non_numeric_rating", "quarantine", **keys(pos))
        for pos in self._take(pool, s["out_of_range_rating"]):
            df.iat[pos, cols["rating"]] = str(self.rng.choice(["0", "6", "7", "10", "-1"]))
            self._record("interactions", "out_of_range_rating", "quarantine", **keys(pos))
        for pos in self._take(pool, s["unparseable_timestamp"]):
            df.iat[pos, cols["timestamp"]] = str(self.rng.choice(["not_a_date", "null", "TBD", "2000-13-45 25:61"]))
            self._record("interactions", "unparseable_timestamp", "quarantine", **keys(pos))
        for pos in self._take(pool, s["future_timestamp"]):
            df.iat[pos, cols["timestamp"]] = _epoch(int(self.rng.integers(2031, 2036)), self.rng)
            self._record("interactions", "future_timestamp", "quarantine", **keys(pos))
        for i, pos in enumerate(self._take(pool, s["prehistoric_timestamp"])):
            # half are the classic "epoch zero" default, half are just too early
            df.iat[pos, cols["timestamp"]] = "0" if i % 2 == 0 else _epoch(1985, self.rng)
            self._record("interactions", "prehistoric_timestamp", "quarantine", **keys(pos))

        clean_rows = df.iloc[pool]  # rows that no defect touched
        extra = []

        dup = clean_rows.iloc[self.rng.choice(len(clean_rows), s["exact_duplicates"], replace=False)].copy()
        for u, it in zip(dup["user_id"], dup["item_id"]):
            self._record("interactions", "exact_duplicate", "quarantine", user_id=u, item_id=it)
        extra.append(dup)

        # a user changes their mind: same pair, later timestamp, new rating
        rer = clean_rows.iloc[self.rng.choice(len(clean_rows), s["rerated_pairs"], replace=False)].copy()
        max_ts = int(inter["timestamp"].astype("int64").max())
        new_ts = rer["timestamp"].astype("int64") + self.rng.integers(3600, 30 * 86400, len(rer))
        rer["timestamp"] = np.minimum(new_ts, max_ts).astype(str)
        # rotate around the 1..5 scale; k in {1,2,3} can never land back on r
        rer["rating"] = [str(1 + (int(r) + int(self.rng.integers(1, 4))) % 5) for r in rer["rating"]]
        for u, it in zip(rer["user_id"], rer["item_id"]):
            self._record("interactions", "rerated_pair", "keep_latest", user_id=u, item_id=it)
        extra.append(rer)

        base = clean_rows.iloc[self.rng.choice(len(clean_rows), s["orphan_user"], replace=False)].copy()
        base["user_id"] = [str(x) for x in self.rng.integers(10_001, 20_000, len(base))]
        for u, it in zip(base["user_id"], base["item_id"]):
            self._record("interactions", "orphan_user", "quarantine", user_id=u, item_id=it)
        extra.append(base)

        base = clean_rows.iloc[self.rng.choice(len(clean_rows), s["orphan_item"], replace=False)].copy()
        base["item_id"] = [str(x) for x in self.rng.integers(5_001, 9_000, len(base))]
        for u, it in zip(base["user_id"], base["item_id"]):
            self._record("interactions", "orphan_item", "quarantine", user_id=u, item_id=it)
        extra.append(base)

        out = pd.concat([df, *extra], ignore_index=True)
        # a real feed does not arrive grouped by defect type
        return out.sample(frac=1.0, random_state=int(self.rng.integers(1 << 31))).reset_index(drop=True)

    def ground_truth(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)
