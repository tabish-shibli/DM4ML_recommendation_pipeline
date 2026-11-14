"""Writes reports/validation/validation_report.{md,json} and check_results.csv."""
from __future__ import annotations

import pandas as pd

from src.common.io import df_to_markdown, utc_now, write_json
from src.validation.schemas import CHECK_CATALOG

# Which check each simulated defect should trip (see src/ingestion/dirty_feed.py)
EXPECTED_CHECK = {
    ("interactions", "missing_key"): "missing_value",
    ("interactions", "missing_rating"): "missing_value",
    ("interactions", "non_numeric_rating"): "invalid_type",
    ("interactions", "unparseable_timestamp"): "invalid_type",
    ("interactions", "out_of_range_rating"): "rating_valid_value",
    ("interactions", "future_timestamp"): "timestamp_in_window",
    ("interactions", "prehistoric_timestamp"): "timestamp_in_window",
    ("interactions", "orphan_user"): "user_exists",
    ("interactions", "orphan_item"): "item_exists",
    ("interactions", "exact_duplicate"): "no_exact_duplicates",
    ("interactions", "rerated_pair"): "unique_user_item_pair",
    ("users", "exact_duplicate"): "unique_user_id",
    ("users", "missing_user_id"): "missing_value",
    ("users", "missing_gender"): "gender_valid",
    ("users", "messy_gender"): "gender_valid",
    ("users", "invalid_age"): "age_code_valid",
    ("users", "missing_zip"): "zip_present",
    ("items", "exact_duplicate"): "unique_item_id",
    ("items", "missing_item_id"): "missing_value",
    ("items", "missing_title"): "title_present",
    ("items", "messy_genres"): "genres_canonical",
    ("items", "missing_genres"): "genres_present",
}


def _check_table(results) -> pd.DataFrame:
    rows = []
    for res in results:
        critical = res.mask("critical")
        for check, m in res.flags.items():
            severity, desc = CHECK_CATALOG[check]
            # a warning on a row that is rejected anyway is noise, so it is not counted
            failed_mask = m if severity == "critical" else (m & ~critical)
            failed = int(failed_mask.sum())
            if check == "unique_user_item_pair":
                detail = f"{failed} rows in {res.typed[failed_mask].drop_duplicates(['user_id', 'item_id']).shape[0]} pairs"
            else:
                detail = ""
            rows.append({"table": res.name, "check": check, "severity": severity, "description": desc,
                         "rows_checked": len(res.raw), "rows_failed": failed,
                         "pass_rate": 1 - failed / len(res.raw), "detail": detail})
    order = {"critical": 0, "warning": 1}
    df = pd.DataFrame(rows)
    df["_o"] = df["severity"].map(order)
    return df.sort_values(["table", "_o", "check"]).drop(columns="_o").reset_index(drop=True)


def _injection_vs_caught(cfg, checks: pd.DataFrame) -> pd.DataFrame | None:
    path = cfg.path("raw") / "landing" / "_injected_anomalies.csv"
    if not path.exists():
        return None
    inj = pd.read_csv(path, dtype=str, keep_default_na=False)
    inj["check"] = [EXPECTED_CHECK[(t, a)] for t, a in zip(inj["table"], inj["anomaly"])]
    expected = inj.groupby(["table", "check"]).agg(
        defects_injected=("anomaly", "size"),
        defect_types=("anomaly", lambda s: ", ".join(sorted(set(s))))).reset_index()
    caught = checks[["table", "check", "rows_failed", "detail"]]
    out = expected.merge(caught, on=["table", "check"], how="left").fillna({"rows_failed": 0})
    # re-rated pairs are counted per pair, not per row
    pair = out["check"] == "unique_user_item_pair"
    out.loc[pair, "rows_failed"] = out.loc[pair, "detail"].str.extract(r"in (\d+) pairs")[0].astype(float).values
    out = out.rename(columns={"rows_failed": "caught"}).drop(columns="detail")
    out["caught"] = out["caught"].astype(int)
    out["match"] = (out["caught"] == out["defects_injected"]).map({True: "yes", False: "see note"})
    return out[["table", "defect_types", "check", "defects_injected", "caught", "match"]]


def _profile(results) -> dict:
    res = {r.name: r for r in results}
    inter = res["interactions"]
    ok = ~inter.mask("critical")
    v = inter.typed[ok]
    ts = pd.to_datetime(v["timestamp"].astype("int64"), unit="s", utc=True)
    n_u, n_i = v["user_id"].nunique(), v["item_id"].nunique()
    return {
        "interactions": int(len(v)),
        "distinct_users": int(n_u),
        "distinct_items": int(n_i),
        "matrix_density": float(len(v) / (n_u * n_i)),
        "rating_distribution": {int(k): int(c) for k, c in v["rating"].value_counts().sort_index().items()},
        "mean_rating": float(v["rating"].mean()),
        "first_event": ts.min().isoformat(),
        "last_event": ts.max().isoformat(),
        "ratings_per_user_median": float(v.groupby("user_id").size().median()),
        "ratings_per_item_median": float(v.groupby("item_id").size().median()),
    }


def write_validation_report(cfg, results, summary: dict, gate: dict, manifest: dict) -> dict:
    out_dir = cfg.path("reports", "validation")
    checks = _check_table(results)
    checks.to_csv(out_dir / "check_results.csv", index=False)
    injected = _injection_vs_caught(cfg, checks)
    profile = _profile(results)

    report = {
        "generated_at": utc_now(),
        "source": manifest["source"],
        "archive_md5": manifest["archive_md5"],
        "dirty_feed_simulated": manifest["dirty_feed_simulated"],
        "quality_gate": gate,
        "tables": summary,
        "checks": checks.to_dict("records"),
        "injected_vs_caught": None if injected is None else injected.to_dict("records"),
        "profile_after_validation": profile,
    }
    write_json(report, out_dir / "validation_report.json")

    tbl = pd.DataFrame([{"table": k, **v} for k, v in summary.items()])
    tbl["rejected_share"] = tbl["rejected_share"].map(lambda x: f"{x:.3%}")
    md = [
        "# Data Validation Report",
        "",
        f"Generated {report['generated_at']} from {manifest['source']} (archive md5 `{manifest['archive_md5']}`).",
        f"Dirty-feed simulation: {'on' if manifest['dirty_feed_simulated'] else 'off'}.",
        "",
        f"## Quality gate: {'PASSED' if gate['passed'] else 'FAILED'}",
        "",
        f"A table fails the gate if more than {gate['max_error_rate']:.0%} of its rows are rejected "
        "or if fewer valid rows remain than the configured minimum.",
        "",
    ]
    md += [f"- {f}" for f in gate["failures"]] or []
    md += ["", "## Rows per table", "", df_to_markdown(tbl), "",
           "Rejected rows are in `data/errors/<table>_rejected.csv` with an `error_reasons` column. "
           "Rows that were kept but need cleaning are in `data/errors/<table>_warnings.csv`.", "",
           "## Check results", "",
           df_to_markdown(checks.drop(columns=["rows_checked"]), floatfmt="{:.4%}"), ""]
    if injected is not None:
        md += ["## Simulated defects versus what validation caught", "",
               "Every simulated defect is logged at ingestion. This table checks the validator against that log.", "",
               df_to_markdown(injected), ""]
    md += ["## Profile of the validated interactions", ""]
    for k, v in profile.items():
        md.append(f"- **{k}**: {v}")
    md += ["", "## Sample of rejected interaction rows", ""]
    sample = pd.read_csv(cfg.path("errors") / "interactions_rejected.csv", dtype=str, keep_default_na=False)
    if len(sample):
        sample = sample.groupby("error_reasons", group_keys=False).head(2).head(16)
        md.append(df_to_markdown(sample.drop(columns=["validated_at"])))
    (out_dir / "validation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
