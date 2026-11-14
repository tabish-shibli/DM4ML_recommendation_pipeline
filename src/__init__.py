import os

# MLflow 3 prints an "agent hint" banner on import; it is noise in pipeline logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
