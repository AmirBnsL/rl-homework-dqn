#!/usr/bin/env python
"""Run one (environment, variant, seed) triplet or a whole suite, then save the artifacts.

Examples
--------
    python scripts/run_experiment.py --env cartpole --suite ex1
    python scripts/run_experiment.py --env pong --variant dueling --seeds 0,1,2 --budget steps:300000
    python scripts/run_experiment.py --env pong --suite ex2 --ckpt-every 50000

Artifacts: runs/<env>/<variant>_s<seed>.npz (curves and metrics), .json (metadata) and .pth (the
online network of the trained agent).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqnlib.envs import ENV_SPECS
from dqnlib.experiments import (EX1_VARIANTS, EX2_VARIANTS, SEEDS, run_matrix)


def parse_budget(value: str | None, env_name: str):
    if value is None:
        return None
    if ":" in value:
        unit, amount = value.split(":", 1)
        if unit not in ("steps", "episodes"):
            raise SystemExit(f"--budget unit must be steps or episodes, got {unit!r}")
        return (unit, int(amount))
    return ("steps", int(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=sorted(ENV_SPECS))
    parser.add_argument("--suite", default=None, choices=["ex1", "ex2", "all"])
    parser.add_argument("--variant", default=None,
                        help="single variant, or a comma-separated list")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--budget", default=None,
                        help="'steps:300000', 'episodes:200' or a bare number (= steps)")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--ckpt-every", type=int, default=None,
                        help="save a checkpoint every N environment steps")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--force", action="store_true", help="re-run even if the artifact exists")
    args = parser.parse_args()

    if args.variant:
        variants = tuple(v.strip() for v in args.variant.split(",") if v.strip())
    elif args.suite == "ex1":
        variants = EX1_VARIANTS
    elif args.suite == "ex2":
        variants = EX2_VARIANTS
    elif args.suite == "all":
        variants = EX1_VARIANTS + tuple(v for v in EX2_VARIANTS if v not in EX1_VARIANTS)
    else:
        parser.error("give either --variant or --suite")

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    budget = parse_budget(args.budget, args.env)

    import torch
    device = None
    if args.device:
        device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    print(f"env={args.env} variants={variants} seeds={seeds} "
          f"budget={budget or ENV_SPECS[args.env]['budget']} out={args.out}", flush=True)

    run_matrix(args.env, variants, seeds=seeds, budget=budget, out_dir=args.out,
               device=device, verbose=True, ckpt_every_steps=args.ckpt_every, force=args.force)
    print("done", flush=True)


if __name__ == "__main__":
    main()
