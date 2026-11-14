"""The stage list, shared by both runners (run_pipeline.py and the Prefect flow).

Each stage reads only what earlier stages wrote to disk, so the order below is
the whole dependency graph:

    ingest -> validate -> prepare -> transform -> features -> train -> evaluate

The run summary at the end reads from validate, transform and evaluate.
"""
from __future__ import annotations

from src.evaluation import evaluate
from src.features import materialize
from src.ingestion import ingest
from src.preparation import clean
from src.training import train
from src.transformation import transform
from src.validation import validate
from src.common.io import df_to_markdown, utc_now, write_json

STAGES = [
    ("ingest", ingest.run),
    ("validate", validate.run),
    ("prepare", clean.run),
    ("transform", transform.run),
    ("features", materialize.run),
    ("train", train.run),
    ("evaluate", evaluate.run),
]
STAGE_NAMES = [name for name, _ in STAGES]


def write_run_summary(cfg, results: dict, timings: dict, runner: str, run_id: str | None = None) -> str:
    """One page that ties a run together: what ran, how long, gate status, headline metrics."""
    import pandas as pd

    lines = ["# Pipeline Run Summary", "", f"- Finished: {utc_now()}", f"- Runner: {runner}"]
    if run_id:
        lines.append(f"- Run id: {run_id}")
    lines += ["", "## Stages", "", "| stage | seconds |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in timings.items()]

    v = results.get("validate")
    if v:
        gate = v["quality_gate"]
        lines += ["", f"## Data quality gate: {'PASSED' if gate['passed'] else 'FAILED'}", ""]
        lines += [f"- {t}: {s['valid']:,} valid of {s['total']:,} ({s['rejected']:,} rejected)"
                  for t, s in v["tables"].items()]
    t = results.get("transform")
    if t:
        s = t["test_split"]
        lines += ["", "## Split", "", f"- Cutoff {s['cutoff']}: {s['train_rows']:,} train / {s['test_rows']:,} test rows",
                  f"- Warm test users {s['warm_test_users']:,}, cold-start {s['cold_start_test_users']:,}"]
    e = results.get("evaluate")
    if e:
        cols = ["model", "precision_at_10", "recall_at_10", "ndcg_at_10", "coverage_at_10", "diversity_at_10", "rmse", "mae"]
        tbl = pd.DataFrame(e["metrics"])
        lines += ["", "## Test metrics (K = 10)", "", df_to_markdown(tbl[[c for c in cols if c in tbl.columns]])]
    text = "\n".join(lines) + "\n"
    out = cfg.path("reports") / "pipeline_run_summary.md"
    out.write_text(text, encoding="utf-8")
    write_json({"runner": runner, "run_id": run_id, "timings": timings, "finished": utc_now()},
               cfg.path("reports") / "pipeline_run_summary.json")
    return text
