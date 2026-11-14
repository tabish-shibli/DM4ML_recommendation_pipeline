"""MLflow setup kept in one place so every stage logs to the same experiment."""
from __future__ import annotations

import subprocess

import mlflow

from src.common.config import PROJECT_ROOT, Config


def setup_mlflow(cfg: Config) -> str:
    mcfg = cfg["mlflow"]
    uri = mcfg["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////") and ":" not in uri[10:]:
        # make the sqlite path absolute so it does not depend on the working directory
        uri = "sqlite:///" + (PROJECT_ROOT / uri[len("sqlite:///"):]).as_posix()
    mlflow.set_tracking_uri(uri)
    name = mcfg["experiment_name"]
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        artifacts = (PROJECT_ROOT / mcfg["artifact_location"]).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        exp_id = mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "not-a-git-repo"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
