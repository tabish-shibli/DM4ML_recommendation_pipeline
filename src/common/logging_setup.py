"""One logger setup for every stage: console plus logs/pipeline.log."""
from __future__ import annotations

import logging
import sys

from src.common.config import PROJECT_ROOT

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        root = logging.getLogger("src")
        root.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(file_handler)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        root.addHandler(console)
        root.propagate = False
        _configured = True
    # modules run with `python -m` are called __main__; keep them under "src"
    if not name.startswith("src"):
        name = f"src.{name.strip('_')}"
    return logging.getLogger(name)
