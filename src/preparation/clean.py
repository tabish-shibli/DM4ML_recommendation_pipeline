"""Stage 3 - Preparation (cleaning and standardisation).

Validation only decides whether a record is usable. This stage fixes the
usable-but-untidy ones, and every fix is counted:

  users        gender spellings -> M/F/U, age and occupation codes -> labels,
               blank zip -> "unknown"
  items        release year parsed from the title, trailing articles moved to
               the front ("Matrix, The" -> "The Matrix"), genre spelling and
               separators normalised, blank titles filled
  interactions typed columns, epoch -> UTC datetime, re-rated pairs collapsed
               to the most recent rating

Output: data/processed/{users,items,interactions}_clean.parquet
"""
from __future__ import annotations

import re

import pandas as pd

from src.common.config import Config, load_config
from src.common.io import write_json
from src.common.logging_setup import get_logger
from src.validation.schemas import CANONICAL_GENRES

log = get_logger(__name__)

AGE_LABELS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATIONS = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student",
    5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired",
    14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}
GENRE_LOOKUP = {g.lower(): g for g in CANONICAL_GENRES}
# "Matrix, The" and "Promise, The (La Promesse)" both put the article first
ARTICLE_RE = re.compile(
    r"^(?P<body>.+?), (?P<art>The|A|An|Les|Le|La|Il|El|Das|Der|Die|Det|Den|Lo)(?P<rest> \(.*\))?$")
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def clean_users(df: pd.DataFrame, fixes: dict) -> pd.DataFrame:
    out = df.copy()
    g = out["gender"].astype(str).str.strip().str.upper()
    g = g.replace({"MALE": "M", "FEMALE": "F"})
    std = g.where(g.isin(["M", "F"]), "U")
    fixes["users.gender_standardised"] = int(((out["gender"] != std) & std.isin(["M", "F"])).sum())
    fixes["users.gender_unknown"] = int((std == "U").sum())
    out["gender"] = std

    age = out["age"].astype("Int64")
    valid_age = age.isin(list(AGE_LABELS))
    fixes["users.age_unknown"] = int((~valid_age).sum())
    out["age_code"] = age.where(valid_age)
    out["age_group"] = age.map(AGE_LABELS).where(valid_age, "unknown").astype(str)

    occ = out["occupation"].astype("Int64")
    fixes["users.occupation_unknown"] = int((~occ.isin(list(OCCUPATIONS))).sum())
    out["occupation"] = occ.map(OCCUPATIONS).fillna("unknown").astype(str)

    z = out["zip_code"].astype(str).str.strip()
    fixes["users.zip_unknown"] = int((z == "").sum())
    out["zip_code"] = z.where(z != "", "unknown")
    out["zip_prefix"] = out["zip_code"].str.extract(r"^(\d{3})", expand=False).fillna("unknown")

    out["user_id"] = out["user_id"].astype("int32")
    return out[["user_id", "gender", "age_code", "age_group", "occupation", "zip_code", "zip_prefix"]]


def _normalise_genres(raw: str) -> tuple[str, int]:
    parts = [p.strip() for p in str(raw).split("|")]
    canon, dropped = [], 0
    for p in parts:
        if not p:
            continue
        g = GENRE_LOOKUP.get(p.lower())
        if g is None:
            dropped += 1
        elif g not in canon:
            canon.append(g)
    return "|".join(canon), dropped


def _move_article(title: str) -> str:
    m = ARTICLE_RE.match(title)
    return f"{m['art']} {m['body']}{m['rest'] or ''}" if m else title


def clean_items(df: pd.DataFrame, fixes: dict) -> pd.DataFrame:
    out = df.copy()
    out["item_id"] = out["item_id"].astype("int32")
    title = out["title"].astype(str).str.strip()
    blank = title == ""
    fixes["items.title_filled"] = int(blank.sum())

    out["release_year"] = title.str.extract(YEAR_RE, expand=False).astype("Int64")
    missing_year = out["release_year"].isna()
    median_year = int(out["release_year"].median())
    out["release_year_imputed"] = missing_year
    out["release_year"] = out["release_year"].fillna(median_year).astype("int16")
    fixes["items.release_year_imputed"] = int(missing_year.sum())

    base = title.str.replace(YEAR_RE, "", regex=True).str.strip()
    moved = base.map(_move_article)
    fixes["items.article_moved"] = int((moved != base).sum())
    out["title"] = moved.where(~blank, "Unknown title (item " + out["item_id"].astype(str) + ")")
    out["title_original"] = title

    norm = out["genres"].map(_normalise_genres)
    new_genres = norm.map(lambda t: t[0])
    fixes["items.genres_reformatted"] = int(((new_genres != out["genres"]) & (out["genres"] != "")).sum())
    fixes["items.genre_tokens_dropped"] = int(norm.map(lambda t: t[1]).sum())
    fixes["items.genres_missing"] = int((new_genres == "").sum())
    out["genres"] = new_genres
    out["n_genres"] = out["genres"].map(lambda g: len(g.split("|")) if g else 0).astype("int8")
    return out[["item_id", "title", "title_original", "release_year", "release_year_imputed", "genres", "n_genres"]]


def clean_interactions(df: pd.DataFrame, fixes: dict) -> pd.DataFrame:
    out = df.copy()
    out["user_id"] = out["user_id"].astype("int32")
    out["item_id"] = out["item_id"].astype("int32")
    out["rating"] = out["rating"].astype("int8")
    out["timestamp"] = out["timestamp"].astype("int64")
    out["event_time"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    out["_order"] = range(len(out))

    # stable sort so that, for identical timestamps, the row that arrived later wins
    out = out.sort_values(["timestamp", "_order"], kind="stable")
    before = len(out)
    out = out.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    fixes["interactions.rerates_collapsed"] = before - len(out)
    out = out.drop(columns="_order").sort_values(["event_time", "user_id", "item_id"]).reset_index(drop=True)
    out["event_type"] = "rating"
    return out[["user_id", "item_id", "rating", "timestamp", "event_time", "event_type"]]


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    vdir, pdir = cfg.path("validated"), cfg.path("processed")
    fixes: dict[str, int] = {}
    counts = {}

    for name, fn in (("users", clean_users), ("items", clean_items), ("interactions", clean_interactions)):
        raw = pd.read_parquet(vdir / f"{name}.parquet")
        clean = fn(raw, fixes)
        clean.to_parquet(pdir / f"{name}_clean.parquet", index=False)
        counts[name] = {"rows_in": len(raw), "rows_out": len(clean)}
        log.info("%-12s %d -> %d rows", name, len(raw), len(clean))

    # Interactions on movies or users dropped upstream are already gone (validation FK
    # checks). Items nobody rated are kept: they are part of the catalogue.
    summary = {"stage": "preparation", "row_counts": counts, "fixes": fixes}
    out_dir = cfg.path("reports", "preparation")
    write_json(summary, out_dir / "preparation_summary.json")
    lines = ["# Preparation Summary", "", "| table | rows in | rows out |", "|---|---|---|"]
    lines += [f"| {k} | {v['rows_in']:,} | {v['rows_out']:,} |" for k, v in counts.items()]
    lines += ["", "| fix | rows affected |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in fixes.items()]
    (out_dir / "preparation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in fixes.items():
        log.info("fix %-34s %d", k, v)
    return summary


if __name__ == "__main__":
    run()
