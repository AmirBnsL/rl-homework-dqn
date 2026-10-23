#!/usr/bin/env python
"""Measure how fast a training step is on this machine, so the Pong budget can be chosen sensibly.

    python scripts/calibrate.py --env pong --steps 3000

Reports environment steps per second, gradient steps per second and the projected duration of the
usual budgets, plus the RAM footprint of the replay buffer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqnlib.envs import ENV_SPECS, make_env
from dqnlib.experiments import build_agent
from dqnlib.utils import format_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="pong", choices=sorted(ENV_SPECS))
    parser.add_argument("--variant", default="hard")
    parser.add_argument("--warmup", type=int, default=300, help="steps to fill the buffer")
    parser.add_argument("--steps", type=int, default=2000, help="timed steps")
    parser.add_argument("--budgets", default="100000,250000,300000,500000,1000000")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = ENV_SPECS[args.env]
    env = make_env(args.env)
    agent = build_agent(env, args.env, args.variant, 0, device=device)
    # a short warm-up so that the buffer is full and `learn` runs on every step
    agent.learning_starts = min(agent.learning_starts, 64)
    budget = ("steps", args.warmup)
    print(f"device={device} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | "
          f"{agent.n_parameters:,} parameters | buffer {agent.memory.capacity:,} transitions", flush=True)

    import dqnlib.train as train_module
    train_module.train(agent, env, dict(spec), seed=0, budget=budget, progress=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    done, steps = False, 0
    state, _ = env.reset(seed=1)
    while steps < args.steps:
        action = agent.act(state, epsilon=0.05)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.step(state, action, reward, next_state, terminated)
        state = next_state
        steps += 1
        if terminated or truncated:
            state, _ = env.reset()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    sps = steps / elapsed

    buffer_bytes = (agent.memory.states.nbytes + agent.memory.next_states.nbytes
                    if hasattr(agent.memory, "states") else 0)
    print(f"{steps:,} steps in {elapsed:.1f}s -> {sps:,.0f} env steps/s "
          f"({agent.updates:,} gradient steps, {agent.updates / elapsed:,.0f} updates/s)")
    print(f"replay buffer: {buffer_bytes / 2**30:.2f} GiB")
    print("\nprojected duration of one run:")
    for budget_steps in [int(b) for b in args.budgets.split(",")]:
        print(f"  {budget_steps:>9,} steps -> {format_seconds(budget_steps / sps)}")
    print("\nprojected wall-clock for the 21 Pong runs (3 seeds x 7 variants):")
    for budget_steps in [int(b) for b in args.budgets.split(",")]:
        total = 21 * budget_steps / sps
        print(f"  {budget_steps:>9,} steps -> {total / 3600:5.1f} h sequential, "
              f"{total / 3600 / 3:5.1f} h with 3 sessions in parallel")
    env.close()


if __name__ == "__main__":
    main()
