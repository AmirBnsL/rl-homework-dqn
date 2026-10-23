"""Checks for TreasureGrid: Gymnasium API conformance, observation shapes and the two baselines."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import treasure_grid
from treasure_grid import TreasureGridEnv, bfs_distance, bfs_policy_episode, make_walls, random_policy_episode


def test_registered_versions_are_gymnasium_compliant():
    for version in ("TreasureGrid-v0", "TreasureGrid-v1", "TreasureGrid-v2"):
        env = gym.make(version)
        check_env(env.unwrapped, skip_render_check=True)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert env.action_space.n == 4
        assert info["optimal_distance"] >= 0
        env.close()


def test_observation_shapes():
    expected = {"v0": 4 * 36, "v1": 2 + 2 + 2 * 2 + 1, "v2": 5 * 5 * 5}
    for variant, dim in expected.items():
        env = TreasureGridEnv(variant=variant)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (dim,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)


def test_walls_leave_the_grid_connected():
    walls = make_walls(6, layout_seed=0)
    size = 6
    free = [(r, c) for r in range(size) for c in range(size) if (r, c) not in walls]
    assert len(free) == size * size - len(walls)
    assert all(bfs_distance(free[0], cell, walls, size) >= 0 for cell in free)


def test_step_into_wall_keeps_the_agent_in_place():
    env = TreasureGridEnv(variant="v0")
    env.reset(seed=0)
    env.agent = next(cell for cell in env.free_cells if (cell[0], cell[1] - 1) in env.walls)
    env.step(2)  # left, into the wall
    assert env.agent == next(cell for cell in env.free_cells if (cell[0], cell[1] - 1) in env.walls)


def _origin_and_action(env, target):
    """A free cell and the action that steps from it onto `target`."""
    for cell in env.free_cells:
        if cell in env.traps or cell == target:
            continue
        for action, (dr, dc) in treasure_grid.MOVES.items():
            if (cell[0] + dr, cell[1] + dc) == target:
                return cell, action
    raise AssertionError(f"no cell steps into {target}")


def test_trap_and_treasure_end_the_episode():
    env = TreasureGridEnv(variant="v0")
    env.reset(seed=0)

    trap = env.traps[0]
    env.agent, action = _origin_and_action(env, trap)
    _, reward, terminated, truncated, info = env.step(action)
    assert env.agent == trap and terminated and not truncated
    assert reward == env.trap_reward and not info["success"]

    env.reset(seed=0)
    env.agent, action = _origin_and_action(env, env._treasure)
    _, reward, terminated, _, info = env.step(action)
    assert terminated and reward == env.treasure_reward and info["success"]


def test_truncation_after_4_n_squared_steps():
    env = TreasureGridEnv(variant="v0")
    env.reset(seed=0)
    env._treasure = (-5, -5)  # unreachable target: the episode can only end by truncation
    for _ in range(env.max_steps - 1):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


def test_reward_shaping_is_potential_based():
    """v1 adds F(s, s') = gamma*phi(s') - phi(s) on every non-terminal step."""
    env = TreasureGridEnv(variant="v1")
    env.reset(seed=3)
    before = env.agent
    previous_potential = env._potential(before)
    action = next(
        a for a, (dr, dc) in treasure_grid.MOVES.items()
        if (before[0] + dr, before[1] + dc) in env.free_cells
        and (before[0] + dr, before[1] + dc) not in env.traps
        and (before[0] + dr, before[1] + dc) != env._treasure)
    _, reward, terminated, _, _ = env.step(action)
    assert not terminated
    expected = env.step_penalty + env.gamma_shaping * env._potential(env.agent) - previous_potential
    assert abs(reward - expected) < 1e-6
    assert env._potential(env._treasure) == 0.0


def test_baselines():
    env = TreasureGridEnv(variant="v0")
    random_runs = [random_policy_episode(env, seed=100 + i) for i in range(20)]
    optimal_runs = [bfs_policy_episode(env, seed=100 + i) for i in range(20)]
    assert np.mean([r["success"] for r in random_runs]) < 0.9
    assert all(r["success"] for r in optimal_runs), "the BFS agent always finds the treasure"
    for r in optimal_runs:
        assert r["length"] == r["optimal"], "and in the optimal number of steps"


def test_dqn_runs_on_treasure_grid(tmp_path=None):
    import tempfile
    from dqnlib.experiments import run_one

    out_dir = str(tmp_path or tempfile.mkdtemp())
    run = run_one("treasure_v1", "double", 0, budget=("episodes", 3), out_dir=out_dir, verbose=False)
    assert run["episode_returns"].shape == (3,)
    assert np.isfinite(run["eval_success"]).all()
    assert run["meta_json"]["success_rate"] is not None


if __name__ == "__main__":
    import tempfile
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        fn(tempfile.mkdtemp()) if name.startswith("test_dqn") else fn()
        print(f"ok   {name}")
    print("all treasure checks passed")
