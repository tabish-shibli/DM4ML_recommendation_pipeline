"""Plain-Python runner. Same stages, same order as the Prefect flow, no server needed.

    python run_pipeline.py                     # everything
    python run_pipeline.py --from-stage train  # resume from a stage
    python run_pipeline.py --only validate     # one stage
"""
from __future__ import annotations

import argparse
import sys
import time

from src.common.config import load_config
from src.common.logging_setup import get_logger
from src.pipeline import STAGE_NAMES, STAGES, write_run_summary

log = get_logger("src.run_pipeline")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--from-stage", choices=STAGE_NAMES)
    group.add_argument("--only", choices=STAGE_NAMES)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    elif args.from_stage:
        selected = STAGES[STAGE_NAMES.index(args.from_stage):]
    else:
        selected = STAGES

    results, timings = {}, {}
    for name, fn in selected:
        log.info("=== stage: %s ===", name)
        t0 = time.perf_counter()
        try:
            results[name] = fn(cfg)
        except Exception:
            log.exception("Stage '%s' failed; later stages were not run", name)
            return 1
        timings[name] = time.perf_counter() - t0
        log.info("=== %s done in %.1fs ===", name, timings[name])

    if len(selected) == len(STAGES):
        write_run_summary(cfg, results, timings, runner="run_pipeline.py")
    log.info("Pipeline finished in %.1fs", sum(timings.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
