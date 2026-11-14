"""Prefect flow for the whole pipeline.

    python dags/recommendation_flow.py            # run once
    python dags/recommendation_flow.py --serve    # register a weekly schedule and wait for runs

Each stage is a Prefect task. A task only starts when the tasks it depends on
have finished successfully; if validation's quality gate fails, nothing
downstream runs and the flow is marked Failed. Ingestion gets retries because
it is the only stage that talks to the network.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
# forward the pipeline's own log lines into the Prefect UI
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "src")

from prefect import flow, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.runtime import flow_run  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import materialize  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.pipeline import write_run_summary  # noqa: E402
from src.preparation import clean  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.validation import validate  # noqa: E402


@task(name="ingest", retries=2, retry_delay_seconds=30)
def ingest_task(config_path: str) -> dict:
    return ingest.run(load_config(config_path))


@task(name="validate")
def validate_task(config_path: str, ingestion: dict) -> dict:
    return validate.run(load_config(config_path))


@task(name="prepare")
def prepare_task(config_path: str, validation: dict) -> dict:
    return clean.run(load_config(config_path))


@task(name="transform")
def transform_task(config_path: str, preparation: dict) -> dict:
    return transform.run(load_config(config_path))


@task(name="build-features")
def features_task(config_path: str, transformation: dict) -> dict:
    return materialize.run(load_config(config_path))


@task(name="train-models")
def train_task(config_path: str, features: dict) -> dict:
    return train.run(load_config(config_path))


@task(name="evaluate-models")
def evaluate_task(config_path: str, training: dict) -> dict:
    return evaluate.run(load_config(config_path))


@task(name="publish-run-summary")
def summary_task(config_path: str, validation: dict, transformation: dict, evaluation: dict, timings: dict) -> str:
    cfg = load_config(config_path)
    text = write_run_summary(cfg, {"validate": validation, "transform": transformation, "evaluate": evaluation},
                             timings, runner="prefect", run_id=str(flow_run.get_id()))
    create_markdown_artifact(key="recsys-run-summary", markdown=text,
                             description="Data quality gate, split and test metrics for this run")
    return text


@flow(name="movielens-recsys-pipeline", description="Ingest -> validate -> prepare -> transform -> features -> train -> evaluate")
def recommendation_pipeline(config_path: str = "configs/pipeline.yaml") -> dict:
    timings = {}

    def step(name, task_fn, *upstream):
        # passing the upstream result in is what tells Prefect about the dependency
        start = time.perf_counter()
        result = task_fn(config_path, *upstream)
        timings[name] = time.perf_counter() - start
        return result

    ingestion = step("ingest", ingest_task)
    validation = step("validate", validate_task, ingestion)
    preparation = step("prepare", prepare_task, validation)
    transformation = step("transform", transform_task, preparation)
    features = step("features", features_task, transformation)
    training = step("train", train_task, features)
    evaluation = step("evaluate", evaluate_task, training)
    summary_task(config_path, validation, transformation, evaluation, timings)
    return {"best_params": training["best_params"], "metrics": evaluation["metrics"]}


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # Weekly retrain, Monday 02:00. Keeps running and waits for scheduled runs.
        recommendation_pipeline.serve(name="weekly-retrain", cron="0 2 * * 1")
    else:
        recommendation_pipeline()
