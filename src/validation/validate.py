"""Stage 2 - Validation.

Reads the landing CSVs as plain text, parses types, runs the Pandera
contracts, and splits every table into
  data/validated/<table>.parquet       rows that passed all critical checks
  data/errors/<table>_rejected.csv     rows that failed one or more, with reasons
  data/errors/<table>_warnings.csv     rows kept but flagged for the cleaning stage

Then it writes the validation report and applies the quality gate. If too
much of a table is rejected the stage raises, which stops the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from pandera.errors import SchemaErrors

from src.common.config import Config, load_config
from src.common.io import read_json, utc_now
from src.common.logging_setup import get_logger
from src.validation import schemas
from src.validation.report import write_validation_report

log = get_logger(__name__)


class DataQualityError(RuntimeError):
    """Raised when the quality gate fails."""


@dataclass
class TableResult:
    name: str
    raw: pd.DataFrame
    typed: pd.DataFrame
    flags: dict[str, pd.Series] = field(default_factory=dict)  # check name -> failed mask

    def flag(self, check: str, mask: pd.Series) -> None:
        mask = mask.reindex(self.raw.index, fill_value=False).astype(bool)
        self.flags[check] = self.flags.get(check, pd.Series(False, index=self.raw.index)) | mask

    def mask(self, severity: str) -> pd.Series:
        out = pd.Series(False, index=self.raw.index)
        for check, m in self.flags.items():
            if schemas.CHECK_CATALOG[check][0] == severity:
                out |= m
        return out

    def reasons(self, severity: str, rows: pd.Series) -> pd.Series:
        """Semicolon-joined failed checks, computed only for the rows asked for."""
        idx = rows[rows].index
        if len(idx) == 0:
            return pd.Series([], index=idx, dtype=str)
        names = [c for c in self.flags if schemas.CHECK_CATALOG[c][0] == severity]
        flagged = pd.DataFrame({c: self.flags[c].loc[idx] for c in names}, index=idx)
        return flagged.apply(lambda r: ";".join(c for c in names if r[c]), axis=1).astype(str)


# ---------------------------------------------------------------- parsing
def _parse_int(raw: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    s = raw.fillna("").astype(str).str.strip()
    missing = s.eq("")
    ok = s.str.fullmatch(r"-?\d+")
    bad = ~missing & ~ok
    parsed = pd.to_numeric(s.where(ok), errors="coerce").astype("Int64")
    return parsed, missing, bad


def _parse_float(raw: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    s = raw.fillna("").astype(str).str.strip()
    missing = s.eq("")
    parsed = pd.to_numeric(s.where(~missing), errors="coerce").astype("Float64")
    bad = ~missing & parsed.isna()
    return parsed, missing, bad


def _apply_schema(result: TableResult, schema) -> None:
    """Run a Pandera schema lazily and turn its failure cases into row flags."""
    try:
        schema.validate(result.typed, lazy=True)
    except SchemaErrors as err:
        fc = err.failure_cases
        unknown = fc[fc["index"].isna()]
        if len(unknown):
            # schema-level problems (wrong dtype, missing column) are bugs, not bad rows
            raise RuntimeError(f"{schema.name}: schema-level failure\n{unknown.to_string()}")
        # Pandera reports built-in checks by their repr ("isin([...])"), so the
        # check object is looked up by column + position to recover its name.
        fc = fc.assign(column=fc["column"].astype(str))
        for (context, column, number), grp in fc.groupby(["schema_context", "column", "check_number"]):
            if context == "DataFrameSchema":
                check = schema.checks[int(number)]
            else:
                check = schema.columns[column].checks[int(number)]
            if check.name not in schemas.CHECK_CATALOG:
                raise RuntimeError(f"Unregistered check '{check.name}' in {schema.name}")
            idx = pd.Index(grp["index"].astype(int).unique())
            result.flag(check.name, pd.Series(True, index=idx))


# ----------------------------------------------------------------- tables
def validate_users(raw: pd.DataFrame, vcfg: dict) -> TableResult:
    typed = pd.DataFrame(index=raw.index)
    res = TableResult("users", raw, typed)
    typed["user_id"], missing, bad = _parse_int(raw["user_id"])
    res.flag("missing_value", missing)
    res.flag("invalid_type", bad)
    typed["gender"] = raw["gender"].fillna("").astype(str)
    typed["age"], _, bad_age = _parse_int(raw["age"])
    typed["occupation"], _, bad_occ = _parse_int(raw["occupation"])
    typed["zip_code"] = raw["zip_code"].fillna("").astype(str)
    res.flag("age_code_valid", bad_age)
    res.flag("occupation_valid", bad_occ)

    _apply_schema(res, schemas.users_critical())
    _apply_schema(res, schemas.users_warning(vcfg["allowed_age_codes"], tuple(vcfg["occupation_range"])))
    return res


def validate_items(raw: pd.DataFrame) -> TableResult:
    typed = pd.DataFrame(index=raw.index)
    res = TableResult("items", raw, typed)
    typed["item_id"], missing, bad = _parse_int(raw["item_id"])
    res.flag("missing_value", missing)
    res.flag("invalid_type", bad)
    typed["title"] = raw["title"].fillna("").astype(str)
    typed["genres"] = raw["genres"].fillna("").astype(str)

    _apply_schema(res, schemas.items_critical())
    _apply_schema(res, schemas.items_warning())
    return res


def validate_interactions(raw: pd.DataFrame, valid_users: set[int], valid_items: set[int],
                          vcfg: dict, ingested_at: datetime) -> TableResult:
    typed = pd.DataFrame(index=raw.index)
    res = TableResult("interactions", raw, typed)
    for col in ("user_id", "item_id", "timestamp"):
        typed[col], missing, bad = _parse_int(raw[col])
        res.flag("missing_value", missing)
        res.flag("invalid_type", bad)
    typed["rating"], missing, bad = _parse_float(raw["rating"])
    res.flag("missing_value", missing)
    res.flag("invalid_type", bad)
    typed = typed[["user_id", "item_id", "rating", "timestamp"]]
    res.typed = typed

    ts_min = int(pd.Timestamp(vcfg["platform_launch"], tz="UTC").timestamp())
    ts_max = int(ingested_at.timestamp())
    _apply_schema(res, schemas.interactions_critical(valid_users, valid_items, ts_min, ts_max))

    # Re-ratings are only meaningful among rows that are otherwise valid
    ok = ~res.mask("critical")
    pair_dupes = typed[ok].duplicated(subset=["user_id", "item_id"], keep=False)
    res.flag("unique_user_item_pair", pair_dupes)
    return res


# ------------------------------------------------------------------- run
def _split_and_save(res: TableResult, cfg: Config, keep_cols: list[str]) -> dict:
    validated_dir, errors_dir = cfg.path("validated"), cfg.path("errors")
    critical = res.mask("critical")
    warning = res.mask("warning") & ~critical

    rejected = res.raw[critical].copy()
    rejected.insert(0, "landing_row", rejected.index + 2)  # +1 header, +1 to make it 1-based
    rejected["error_reasons"] = res.reasons("critical", critical)
    rejected["validated_at"] = utc_now()
    rejected.to_csv(errors_dir / f"{res.name}_rejected.csv", index=False)

    warned = res.raw[warning].copy()
    warned.insert(0, "landing_row", warned.index + 2)
    warned["warning_reasons"] = res.reasons("warning", warning)
    warned.to_csv(errors_dir / f"{res.name}_warnings.csv", index=False)

    valid = res.typed.loc[~critical, keep_cols].reset_index(drop=True)
    valid.to_parquet(validated_dir / f"{res.name}.parquet", index=False)

    log.info("%-12s total=%d valid=%d rejected=%d warned=%d", res.name, len(res.raw), len(valid),
             int(critical.sum()), int(warning.sum()))
    return {"total": len(res.raw), "valid": len(valid), "rejected": int(critical.sum()),
            "warned": int(warning.sum()), "rejected_share": float(critical.mean())}


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    vcfg = cfg["validation"]
    landing = cfg.path("raw") / "landing"
    manifest = read_json(landing / "_manifest.json")
    ingested_at = datetime.fromisoformat(manifest["ingested_at"]).astimezone(timezone.utc)

    def read(name):
        # keep_default_na=False: "NA", "null" etc. stay as text so they are reported, not silently dropped
        return pd.read_csv(landing / f"{name}.csv", dtype=str, keep_default_na=False)

    users = validate_users(read("users"), vcfg)
    items = validate_items(read("items"))
    valid_users = set(users.typed.loc[~users.mask("critical"), "user_id"].dropna().astype(int))
    valid_items = set(items.typed.loc[~items.mask("critical"), "item_id"].dropna().astype(int))
    inter = validate_interactions(read("interactions"), valid_users, valid_items, vcfg, ingested_at)

    summary = {
        "users": _split_and_save(users, cfg, ["user_id", "gender", "age", "occupation", "zip_code"]),
        "items": _split_and_save(items, cfg, ["item_id", "title", "genres"]),
        "interactions": _split_and_save(inter, cfg, ["user_id", "item_id", "rating", "timestamp"]),
    }

    gate_failures = []
    for name, s in summary.items():
        if s["rejected_share"] > vcfg["max_error_rate"]:
            gate_failures.append(f"{name}: {s['rejected_share']:.2%} rejected > {vcfg['max_error_rate']:.0%} limit")
        if s["valid"] < vcfg["min_rows"][name]:
            gate_failures.append(f"{name}: only {s['valid']} valid rows < {vcfg['min_rows'][name]}")
    gate = {"passed": not gate_failures, "failures": gate_failures,
            "max_error_rate": vcfg["max_error_rate"], "min_rows": vcfg["min_rows"]}

    report = write_validation_report(cfg, [users, items, inter], summary, gate, manifest)
    if not gate["passed"]:
        raise DataQualityError("Quality gate failed: " + "; ".join(gate_failures))
    log.info("Quality gate passed")
    return report


if __name__ == "__main__":
    run()
