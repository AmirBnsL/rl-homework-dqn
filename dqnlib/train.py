"""Training loop and greedy evaluation.

Metrics used everywhere in the report:

* **training score** — average return of the last 100 episodes (CartPole, LunarLander) or of the
  last 20 games (Pong);
* **greedy score** — average return of the policy with epsilon = 0 on 20 fresh episodes, whose
  seeds (100 000 + i) are never used during training;
* **Q-value bias** — mean predicted Q(s, a) minus the mean empirical discounted return of the same
  states, both measured on the greedy evaluation episodes.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from dqnlib.utils import epsilon_at


@dataclasses.dataclass
class TrainResult:
    episode_returns: list[float] = dataclasses.field(default_factory=list)
    episode_lengths: list[int] = dataclasses.field(default_factory=list)
    episode_steps: list[int] = dataclasses.field(default_factory=list)   # cumulative env steps
    episode_terminated: list[bool] = dataclasses.field(default_factory=list)
    episode_epsilon: list[float] = dataclasses.field(default_factory=list)
    losses: list[float] = dataclasses.field(default_factory=list)
    q_means: list[float] = dataclasses.field(default_factory=list)
    env_steps: int = 0
    updates: int = 0
    wall_time: float = 0.0
    cut: bool = False  # the last episode was interrupted by the step budget

    def training_score(self, window: int) -> float:
        if not self.episode_returns:
            return float("nan")
        return float(np.mean(self.episode_returns[-window:]))

    @property
    def n_episodes(self) -> int:
        return len(self.episode_returns)


def train(agent, env, spec: dict, seed: int = 0, budget=None, progress: bool = False,
          ckpt_every_steps: int | None = None, ckpt_fn=None) -> TrainResult:
    """Train `agent` on `env`.

    `budget` is either ("episodes", n) or ("steps", n); the step budget may cut the last episode.
    `ckpt_fn(result, agent)` is called every `ckpt_every_steps` environment steps so a killed
    Colab session only loses the current segment.
    """
    unit, amount = budget if budget is not None else spec["budget"]
    eps_cfg = spec["eps"]
    eps_kwargs = dict(eps_start=eps_cfg["eps_start"], eps_end=eps_cfg["eps_end"],
                      decay_steps=eps_cfg["eps_decay_steps"])
    window = spec["metric_window"]

    result = TrainResult()
    t0 = time.time()
    next_ckpt = ckpt_every_steps
    episode = 0
    iterator = iter(range(1, amount + 1)) if unit == "episodes" else None
    if unit == "episodes" and progress:
        from tqdm import tqdm
        iterator = iter(tqdm(range(1, amount + 1), desc=f"train seed {seed}"))

    while True:
        if unit == "episodes":
            try:
                episode = next(iterator)
            except StopIteration:
                break
        else:
            if agent.env_steps >= amount:
                break
            episode += 1

        state, _ = env.reset(seed=seed * 10_000 + episode)
        score, done, length, terminated = 0.0, False, 0, False
        eps = epsilon_at(agent.env_steps, **eps_kwargs)

        while not done:
            eps = epsilon_at(agent.env_steps, **eps_kwargs)
            action = agent.act(state, epsilon=eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, terminated)  # terminated, not truncated
            score += float(reward)
            length += 1
            state = next_state
            done = terminated or truncated

            if unit == "steps" and agent.env_steps >= amount:
                result.cut = not done
                done = True

        result.episode_returns.append(score)
        result.episode_lengths.append(length)
        result.episode_steps.append(agent.env_steps)
        result.episode_terminated.append(bool(terminated))
        result.episode_epsilon.append(eps)

        if ckpt_every_steps and ckpt_fn is not None and agent.env_steps >= next_ckpt:
            result.env_steps, result.updates = agent.env_steps, agent.updates
            result.losses, result.q_means = list(agent.losses), list(agent.q_means)
            result.wall_time = time.time() - t0
            ckpt_fn(result, agent)
            next_ckpt += ckpt_every_steps

        if progress and unit == "steps" and episode % 10 == 0:
            print(f"  step {agent.env_steps:>9,} | episodes {episode:>4} | "
                  f"mean return (last {window}) {result.training_score(window):7.2f} | eps {eps:.2f}",
                  flush=True)

    result.env_steps, result.updates = agent.env_steps, agent.updates
    result.losses, result.q_means = list(agent.losses), list(agent.q_means)
    result.wall_time = time.time() - t0
    return result


def evaluate(agent, env, n_episodes: int = 20, seed: int = 100_000, gamma: float = 0.99,
             max_steps: int | None = None, collect_q: bool = True) -> dict:
    """Greedy policy (epsilon = 0) on `n_episodes` fresh episodes.

    Returns the undiscounted returns, the lengths, and (if `collect_q`) the predicted Q(s, a) of the
    greedy action next to the empirical discounted return of the same state, which gives the
    overestimation bias of the agent.
    """
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    success = np.full(n_episodes, np.nan)
    optimal = np.full(n_episodes, np.nan)
    q_pred, mc_return = [], []
    was_training = agent.q_net.training
    agent.q_net.eval()

    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        episode_rewards, episode_q = [], []
        done, steps, info = False, 0, {}
        while not done:
            if collect_q:
                q_row = agent.q_values(state)
                episode_q.append(float(q_row.max()))
            action = agent.greedy_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(float(reward))
            steps += 1
            done = terminated or truncated
            if max_steps is not None and steps >= max_steps:
                done = True
        returns[i] = float(np.sum(episode_rewards))
        lengths[i] = steps
        if "success" in info:
            success[i] = float(bool(info["success"]))
        if "optimal_distance" in info:
            optimal[i] = float(info["optimal_distance"])

        if collect_q:
            rewards = np.asarray(episode_rewards, dtype=np.float64)
            n = len(rewards)
            discounts = gamma ** np.arange(n)
            returns_to_go = np.cumsum((rewards * discounts)[::-1])[::-1] / discounts
            q_pred.extend(episode_q)
            mc_return.extend(returns_to_go.tolist())

    if was_training:
        agent.q_net.train()

    out = dict(returns=returns, lengths=lengths, n_episodes=n_episodes,
               success=success, optimal=optimal)
    if collect_q:
        q_pred = np.asarray(q_pred, dtype=np.float64)
        mc_return = np.asarray(mc_return, dtype=np.float64)
        out.update(q_pred=q_pred, mc_return=mc_return,
                   bias=float(q_pred.mean() - mc_return.mean()) if len(q_pred) else float("nan"))
    return out
