"""Run the reproducible patient-wise CV for one experiment config.

    python scripts/run_cv.py --config configs/experiments/E12.yaml [--jobs 4]

Every (seed, fold) job runs in its own process with a fixed thread count, so
results do not depend on scheduling. Finished jobs are cached as
reports/cv/{exp}/part_s{seed}_f{fold}.npz and skipped on re-run. When all
jobs are done, OOF outputs are merged into reports/oof/oof_{exp}.npz and the
summary into reports/cv/{exp}/summary.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.cv_protocol import load_config, merge, run_job  # noqa: E402
from src.training.experiment_report import format_summary  # noqa: E402


def _job(args):
    cfg, seed, fold, out_dir = args
    run_job(cfg, seed, fold, Path(out_dir))
    return seed, fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel processes (default: 1 on GPU, 4 on CPU)")
    ap.add_argument("--max_epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.jobs is None:
        import torch
        args.jobs = 1 if torch.cuda.is_available() else 4
    if args.max_epochs is not None:
        cfg["optim"]["max_epochs"] = args.max_epochs
    if args.seeds:
        cfg["seeds"] = args.seeds
    out_dir = Path("reports") / "cv" / cfg["exp_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    jobs = [(cfg, int(s), k, str(out_dir)) for s in cfg["seeds"] for k in range(int(cfg["cv"]["n_splits"]))]
    if args.jobs <= 1:
        # GPU (or debugging): run jobs sequentially in this process
        for j in jobs:
            seed, fold = _job(j)
            print(f"[{cfg['exp_id']}] done seed={seed} fold={fold}", flush=True)
    else:
        ctx = multiprocessing.get_context("spawn")  # CUDA-safe
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
            for seed, fold in ex.map(_job, jobs):
                print(f"[{cfg['exp_id']}] done seed={seed} fold={fold}", flush=True)

    summary = merge(cfg, out_dir, Path("reports") / "oof" / f"oof_{cfg['exp_id']}.npz")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(format_summary(summary))


if __name__ == "__main__":
    main()
