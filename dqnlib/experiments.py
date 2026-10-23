"""The seven configurations of the study, and one function per run.

Only one thing changes between two variants; everything else comes from `ENV_SPECS`, so the
comparison is honest by construction.

Exercise 1 (ablation)              Exercise 2 (improvements)
    hard       reference DQN            hard             reference DQN (shared with Exercise 1)
    soft       tau = 0.01               double           Double DQN target
    no_replay  memory = one batch       dueling          dueling head
    no_target  targets from theta       dueling_double   both
"""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np

from dqnlib.agent import DQNAgent
from dqnlib.envs import ENV_SPECS, make_env, make_eval_env
from dqnlib.train import evaluate, train
from dqnlib.utils import atomic_save_json, atomic_savez, format_seconds

VARIANT_SWITCHES = {
    "hard": dict(target_update="hard"),
    "soft": dict(target_update="soft", tau=0.01),
    "no_replay": dict(use_replay=False),
    "no_target": dict(use_target_network=False),
    "double": dict(double_dqn=True),
    "dueling": dict(dueling=True),
    "dueling_double": dict(dueling=True, double_dqn=True),
}

EX1_VARIANTS = ("hard", "soft", "no_replay", "no_target")
EX2_VARIANTS = ("hard", "double", "dueling", "dueling_double")
SEEDS = (0, 1, 2)

VARIANT_LABELS = {
    "hard": "DQN hard update (reference)",
    "soft": "DQN soft update (tau=0.01)",
    "no_replay": "DQN without experience replay",
    "no_target": "DQN without target network",
    "double": "Double DQN",
    "dueling": "Dueling DQN",
    "dueling_double": "Dueling Double DQN",
}


def variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def agent_kwargs(env_name: str, variant: str) -> dict:
    spec = ENV_SPECS[env_name]
    kwargs = dict(spec["agent"])
    if variant not in VARIANT_SWITCHES:
        raise ValueError(f"unknown variant {variant!r}")
    kwargs.update(VARIANT_SWITCHES[variant])
    if variant == "no_replay":
        # the memory must hold exactly one minibatch of consecutive transitions
        kwargs["buffer_size"] = kwargs["batch_size"]
    return kwargs


def run_paths(env_name: str, variant: str, seed: int, out_dir: str = "runs") -> tuple[str, str]:
    stem = os.path.join(out_dir, env_name, f"{variant}_s{seed}")
    return stem + ".npz", stem + ".pth"


def _decimate(values, max_points: int = 20_000) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64)
    stride = max(1, len(values) // max_points)
    return values[::stride], stride


def build_agent(env, env_name: str, variant: str, seed: int, device=None,
                budget=None, buffer_size: int | None = None) -> DQNAgent:
    """Instantiate the agent for a run, scaling the epsilon schedule to a reduced step budget.

    `buffer_size` overrides the replay capacity: probe agents in the notebooks use a small one, so
    that opening the Pong notebook does not allocate 5.6 GB per agent.
    """
    spec = ENV_SPECS[env_name]
    kwargs = agent_kwargs(env_name, variant)
    if buffer_size is not None:
        kwargs["buffer_size"] = int(buffer_size)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, seed=seed,
                     device=device, **kwargs)
    unit, amount = budget if budget is not None else spec["budget"]
    agent.eps = dict(spec["eps"])
    if unit == "steps" and amount < agent.eps["eps_decay_steps"]:
        # a reduced uniform budget must not leave epsilon close to 1 at the end of training
        agent.eps["eps_decay_steps"] = int(amount)
    return agent


def is_complete(npz_path: str) -> bool:
    """A checkpoint written every N steps is a valid artifact but the run is not finished yet."""
    if not os.path.exists(npz_path):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return not bool(data["partial"]) if "partial" in data else True
    except Exception:
        return False


def run_one(env_name: str, variant: str, seed: int, budget=None, out_dir: str = "runs",
            device=None, verbose: bool = True, ckpt_every_steps: int | None = None,
            force: bool = False, gamma_eval: float = 0.99) -> dict:
    """Train one (environment, variant, seed) triplet, evaluate it greedily and save the artifacts."""
    npz_path, pth_path = run_paths(env_name, variant, seed, out_dir)
    if is_complete(npz_path) and not force:
        if verbose:
            print(f"[skip] {env_name}/{variant} seed {seed}: {npz_path} already exists")
        return load_run(env_name, variant, seed, out_dir)

    spec = ENV_SPECS[env_name]
    budget = budget if budget is not None else spec["budget"]
    t0 = time.time()

    env = make_env(env_name)
    agent = build_agent(env, env_name, variant, seed, device=device, budget=budget)
    spec = dict(spec)
    spec["eps"] = agent.eps

    if verbose:
        print(f"[run ] {env_name}/{variant} seed {seed} | {agent.n_parameters:,} parameters | "
              f"budget {budget[1]:,} {budget[0]} | device {agent.device}", flush=True)

    def checkpoint(result, agent):
        _save_result(env_name, variant, seed, result, None, budget, out_dir, partial=True)
        agent.save(pth_path)
        if verbose:
            print(f"  [ckpt] {result.env_steps:,} steps | {result.n_episodes} episodes | "
                  f"score(last {spec['metric_window']}) "
                  f"{result.training_score(spec['metric_window']):.1f}", flush=True)

    result = train(agent, env, spec, seed=seed, budget=budget, progress=verbose,
                   ckpt_every_steps=ckpt_every_steps, ckpt_fn=checkpoint)
    env.close()

    eval_env = make_eval_env(env_name)
    metrics = evaluate(agent, eval_env, n_episodes=spec["eval_episodes"],
                       seed=spec["eval_seed"], gamma=gamma_eval,
                       max_steps=spec["max_episode_steps"])
    eval_env.close()

    agent.save(pth_path)
    meta = _save_result(env_name, variant, seed, result, metrics, budget, out_dir, agent=agent)
    meta["wall_time"] = time.time() - t0

    if verbose:
        print(f"[done] {env_name}/{variant} seed {seed} | train score "
              f"{meta['training_score']:.1f} | greedy {metrics['returns'].mean():.1f} "
              f"+- {metrics['returns'].std():.1f} | bias {meta['bias']:.1f} | "
              f"{format_seconds(meta['wall_time'])}", flush=True)
    return load_run(env_name, variant, seed, out_dir)


def _save_result(env_name, variant, seed, result, metrics, budget, out_dir, agent=None,
                 partial=False) -> dict:
    npz_path, _ = run_paths(env_name, variant, seed, out_dir)
    json_path = npz_path.replace(".npz", ".json")
    spec = ENV_SPECS[env_name]
    losses, loss_stride = _decimate(result.losses)
    q_means, q_stride = _decimate(result.q_means)

    payload = dict(
        episode_returns=np.asarray(result.episode_returns, dtype=np.float64),
        episode_lengths=np.asarray(result.episode_lengths, dtype=np.int64),
        episode_steps=np.asarray(result.episode_steps, dtype=np.int64),
        episode_terminated=np.asarray(result.episode_terminated, dtype=bool),
        episode_epsilon=np.asarray(result.episode_epsilon, dtype=np.float64),
        losses=losses, loss_stride=np.asarray(loss_stride),
        q_means=q_means, q_stride=np.asarray(q_stride),
        env_steps=np.asarray(result.env_steps),
        updates=np.asarray(result.updates),
        n_episodes=np.asarray(result.n_episodes),
        wall_time=np.asarray(result.wall_time),
        cut=np.asarray(result.cut),
        partial=np.asarray(partial),
        training_score=np.asarray(result.training_score(spec["metric_window"])),
    )
    if metrics is not None:
        payload.update(
            eval_returns=np.asarray(metrics["returns"], dtype=np.float64),
            eval_lengths=np.asarray(metrics["lengths"], dtype=np.int64),
            eval_q_pred=np.asarray(metrics["q_pred"], dtype=np.float64),
            eval_mc_return=np.asarray(metrics["mc_return"], dtype=np.float64),
            eval_success=np.asarray(metrics.get("success", np.full(len(metrics["returns"]), np.nan))),
            eval_optimal=np.asarray(metrics.get("optimal", np.full(len(metrics["returns"]), np.nan))),
            bias=np.asarray(metrics["bias"]),
        )

    target = spec["target_score"]
    meta = dict(
        env=env_name, env_id=spec["env_id"], variant=variant, variant_label=variant_label(variant),
        seed=seed, budget_unit=budget[0], budget=int(budget[1]),
        eps={k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in spec["eps"].items()},
        training_score=float(payload["training_score"]),
        greedy_score=float(metrics["returns"].mean()) if metrics is not None else None,
        greedy_std=float(metrics["returns"].std()) if metrics is not None else None,
        solved=(bool(metrics["returns"].mean() >= target)
                if (metrics is not None and target is not None) else None),
        success_rate=(float(np.nanmean(metrics["success"]))
                      if metrics is not None and np.isfinite(metrics.get("success", np.nan)).any()
                      else None),
        bias=float(metrics["bias"]) if metrics is not None else None,
        env_steps=int(result.env_steps), updates=int(result.updates),
        n_episodes=int(result.n_episodes), wall_time=float(result.wall_time),
        agent=agent.config() if agent is not None else None,
        device=str(agent.device) if agent is not None else None,
        python=platform.python_version(),
        label=spec["label"], target_score=spec["target_score"],
        metric_window=spec["metric_window"],
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    atomic_savez(npz_path, **payload)
    atomic_save_json(json_path, meta)
    return meta


def load_run(env_name: str, variant: str, seed: int, out_dir: str = "runs") -> dict:
    npz_path, pth_path = run_paths(env_name, variant, seed, out_dir)
    data = dict(np.load(npz_path, allow_pickle=False))
    json_path = npz_path.replace(".npz", ".json")
    if os.path.exists(json_path):
        with open(json_path) as fh:
            data["meta_json"] = json.load(fh)
    data["env"], data["variant"], data["seed"] = env_name, variant, seed
    data["pth_path"] = pth_path
    return data


def load_matrix(env_name: str, variants, seeds=SEEDS, out_dir: str = "runs") -> dict:
    """{variant: [per-seed run dicts]}; runs still in progress (partial checkpoints) are skipped."""
    out = {}
    for variant in variants:
        runs = []
        for seed in seeds:
            npz_path, _ = run_paths(env_name, variant, seed, out_dir)
            if is_complete(npz_path):
                runs.append(load_run(env_name, variant, seed, out_dir))
        if runs:
            out[variant] = runs
    return out


def run_matrix(env_name: str, variants, seeds=SEEDS, budget=None, out_dir: str = "runs",
               device=None, verbose: bool = True, ckpt_every_steps: int | None = None,
               force: bool = False) -> dict:
    for variant in variants:
        for seed in seeds:
            run_one(env_name, variant, seed, budget=budget, out_dir=out_dir, device=device,
                    verbose=verbose, ckpt_every_steps=ckpt_every_steps, force=force)
    return load_matrix(env_name, variants, seeds, out_dir)
