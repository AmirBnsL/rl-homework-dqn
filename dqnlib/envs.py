"""The three environments of Part 0, their preprocessing and their hyperparameters.

Every hyperparameter of the reference DQN (hard update) lives in `ENV_SPECS`, so a notebook can
build an agent without hard-coding anything:

    spec = ENV_SPECS["cartpole"]
    env = make_env("cartpole")
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, **spec["agent"])

Atari preprocessing follows Mnih et al. (2015) and is built from the Gymnasium wrappers:
frame skip 4 with max-pooling of the last two frames, grayscale 84x84, 30 random no-ops at reset,
stack of the last 4 frames, no sticky actions, observations kept as uint8.
"""
from __future__ import annotations

import gymnasium as gym

ENV_SPECS = {
    "cartpole": dict(
        label="CartPole-v1",
        env_id="CartPole-v1",
        description="Classic control, state in R^4, 2 actions, +1 per step, truncated at 500 steps.",
        obs_shape=(4,),
        n_actions=2,
        agent=dict(net="mlp", hidden_dim=64, lr=5e-4, gamma=0.99, buffer_size=100_000,
                   batch_size=64, learn_every=1, learning_starts=64,
                   target_update_every=100, target_update="hard", loss="mse", grad_clip=None, clip_reward=False),
        eps=dict(eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000),
        budget=("episodes", 300),
        max_episode_steps=500,
        target_score=475.0,
        metric_window=100,
        eval_episodes=20,
        eval_seed=100_000,
        q_reference=100.0,  # 1 / (1 - gamma)
    ),
    "lunarlander": dict(
        label="LunarLander-v3",
        env_id="LunarLander-v3",
        description="Box2D, state in R^8, 4 actions, shaped reward, truncated at 1000 steps.",
        obs_shape=(8,),
        n_actions=4,
        agent=dict(net="mlp", hidden_dim=128, lr=5e-4, gamma=0.99, buffer_size=100_000,
                   batch_size=64, learn_every=4, learning_starts=64,
                   target_update_every=250, target_update="hard", loss="mse", grad_clip=None, clip_reward=False),
        eps=dict(eps_start=1.0, eps_end=0.05, eps_decay_steps=25_000),
        budget=("episodes", 800),
        max_episode_steps=1_000,
        target_score=200.0,
        metric_window=100,
        eval_episodes=20,
        eval_seed=100_000,
        q_reference=None,
    ),
    "pong": dict(
        label="ALE/Pong-v5",
        env_id="ALE/Pong-v5",
        description="Atari from pixels, 4 stacked 84x84 grayscale frames, 6 actions, reward in {-1, 0, +1}.",
        obs_shape=(4, 84, 84),
        n_actions=6,
        agent=dict(net="cnn", hidden_dim=0, lr=1e-4, gamma=0.99, buffer_size=100_000,
                   batch_size=32, learn_every=1, learning_starts=10_000,
                   target_update_every=1_000, target_update="hard", loss="huber", grad_clip=10.0, clip_reward=True),
        eps=dict(eps_start=1.0, eps_end=0.01, eps_decay_steps=1_000_000),
        budget=("steps", 1_000_000),
        max_episode_steps=None,
        target_score=18.0,
        metric_window=20,
        eval_episodes=20,
        eval_seed=100_000,
        q_reference=None,
    ),
    "treasure_v0": dict(
        label="TreasureGrid-v0",
        env_id="TreasureGrid-v0",
        description="Grid world, one-hot grid planes, sparse reward, static traps.",
        obs_shape=None,  # read from the environment: it depends on the version
        n_actions=4,
        agent=dict(net="mlp", hidden_dim=64, lr=5e-4, gamma=0.90, buffer_size=100_000,
                   batch_size=64, learn_every=1, learning_starts=64,
                   target_update_every=100, target_update="hard", loss="mse",
                   grad_clip=None, clip_reward=False),
        eps=dict(eps_start=1.0, eps_end=0.10, eps_decay_steps=10_000),
        budget=("episodes", 800),
        max_episode_steps=144,
        target_score=None,
        metric_window=20,
        eval_episodes=20,
        eval_seed=100_000,
        q_reference=None,
    ),
    "treasure_v1": dict(
        label="TreasureGrid-v1",
        env_id="TreasureGrid-v1",
        description="Grid world, coordinates, potential-based shaping, static traps.",
        obs_shape=None,
        n_actions=4,
        agent=dict(net="mlp", hidden_dim=64, lr=5e-4, gamma=0.90, buffer_size=100_000,
                   batch_size=64, learn_every=1, learning_starts=64,
                   target_update_every=100, target_update="hard", loss="mse",
                   grad_clip=None, clip_reward=False),
        eps=dict(eps_start=1.0, eps_end=0.10, eps_decay_steps=10_000),
        budget=("episodes", 800),
        max_episode_steps=144,
        target_score=None,
        metric_window=20,
        eval_episodes=20,
        eval_seed=100_000,
        q_reference=None,
    ),
    "treasure_v2": dict(
        label="TreasureGrid-v2",
        env_id="TreasureGrid-v2",
        description="Grid world, egocentric patch, sparse reward, moving traps.",
        obs_shape=None,
        n_actions=4,
        agent=dict(net="mlp", hidden_dim=64, lr=5e-4, gamma=0.90, buffer_size=100_000,
                   batch_size=64, learn_every=1, learning_starts=64,
                   target_update_every=100, target_update="hard", loss="mse",
                   grad_clip=None, clip_reward=False),
        eps=dict(eps_start=1.0, eps_end=0.10, eps_decay_steps=10_000),
        budget=("episodes", 800),
        max_episode_steps=144,
        target_score=None,
        metric_window=20,
        eval_episodes=20,
        eval_seed=100_000,
        q_reference=None,
    ),
}


def make_env(name: str, render_mode: str | None = None) -> gym.Env:
    """Create the environment, with the Atari preprocessing applied for Pong."""
    if name == "cartpole":
        return gym.make("CartPole-v1", render_mode=render_mode)
    if name == "lunarlander":
        return gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False,
                        turbulence_power=1.5, render_mode=render_mode)
    if name == "pong":
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0,
                       full_action_space=False, render_mode=render_mode)
        env = gym.wrappers.AtariPreprocessing(
            env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False,
            grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        return env
    if name.startswith("treasure_"):
        try:
            import treasure_grid  # noqa: F401  (registers TreasureGrid-v0/v1/v2)
        except ImportError:
            pass  # in the notebooks the environment is already defined in this namespace
        suffix = {"treasure_v0": "TreasureGrid-v0", "treasure_v1": "TreasureGrid-v1",
                  "treasure_v2": "TreasureGrid-v2"}[name]
        return gym.make(suffix, render_mode=render_mode)
    raise ValueError(f"unknown environment: {name!r}")


def make_eval_env(name: str) -> gym.Env:
    """A fresh environment for the greedy evaluation (never the one used for training)."""
    return make_env(name)


def spec_for(name: str) -> dict:
    return ENV_SPECS[name]
