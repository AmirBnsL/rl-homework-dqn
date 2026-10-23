"""Sanity checks for the library: shapes, parameter counts, replay semantics, target formulas.

Run with `pytest -q tests/` or `python tests/test_sanity.py`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqnlib.agent import DQNAgent
from dqnlib.envs import ENV_SPECS, make_env, make_eval_env
from dqnlib.experiments import EX1_VARIANTS, EX2_VARIANTS, agent_kwargs, run_one, load_run
from dqnlib.networks import CNN_FEATURES, DuelingNatureCNN, DuelingQNetwork, NatureCNN, QNetwork
from dqnlib.replay import ReplayMemory
from dqnlib.train import evaluate, train
from dqnlib.utils import epsilon_at, set_seed


def test_mlp_shapes():
    net = QNetwork(4, 2, 64)
    assert net(torch.zeros(3, 4)).shape == (3, 2)


def test_nature_cnn_matches_the_paper_count():
    net = NatureCNN(6)
    assert CNN_FEATURES == 3_136
    assert net(torch.zeros(2, 4, 84, 84, dtype=torch.uint8)).shape == (2, 6)
    assert sum(p.numel() for p in net.parameters()) == 1_687_206


def test_dueling_head_is_a_centred_decomposition():
    for net, shape in ((DuelingQNetwork(8, 4, 128), (5, 8)),
                       (DuelingNatureCNN(6), (2, 4, 84, 84))):
        x = torch.zeros(*shape, dtype=torch.uint8) if len(shape) == 4 else torch.randn(*shape)
        q = net(x)
        assert q.shape == (shape[0], net.advantage[-1].out_features)
        h = net.body(x) if hasattr(net, "body") else net.encoder(x)
        v = net.value(h)
        assert torch.allclose(q.mean(dim=1, keepdim=True), v, atol=1e-5)


def test_replay_uniform_sampling():
    mem = ReplayMemory(10, state_shape=(4,), device=torch.device("cpu"))
    for i in range(10):
        mem.push(np.full(4, i, dtype=np.float32), i % 2, float(i), np.full(4, i + 1), i == 9)
    assert len(mem) == 10
    states, actions, rewards, next_states, terminateds = mem.sample(4)
    assert states.dtype == torch.float32 and states.shape == (4, 4)
    assert actions.dtype == torch.int64 and rewards.dtype == torch.float32
    assert terminateds.shape == (4,)
    mem.push(np.zeros(4, dtype=np.float32), 0, 0.0, np.zeros(4, dtype=np.float32), False)
    assert len(mem) == 10, "ring buffer must drop the oldest transition"


def test_replay_chronological_order():
    mem = ReplayMemory(10, state_shape=(1,), device=torch.device("cpu"), chronological=True)
    for i in range(10):
        mem.push(np.array([i], dtype=np.float32), i, 0.0, np.array([i + 1], dtype=np.float32), False)
    _, actions, _, next_states, _ = mem.sample(4)
    assert actions.tolist() == [6, 7, 8, 9], "no-replay mode samples the B most recent, in order"
    assert next_states.flatten().tolist() == [7.0, 8.0, 9.0, 10.0]


def test_atari_memory_is_uint8():
    mem = ReplayMemory(16, state_shape=(4, 84, 84), state_dtype=np.uint8)
    assert mem.states.dtype == np.uint8
    assert mem.states.nbytes / 16 == 4 * 84 * 84, "56 KB per transition"


def test_epsilon_schedule():
    assert epsilon_at(0, 1.0, 0.05, 10_000) == 1.0
    assert abs(epsilon_at(5_000, 1.0, 0.05, 10_000) - (1.0 + 0.5 * (0.05 - 1.0))) < 1e-9
    assert abs(epsilon_at(10_000, 1.0, 0.05, 10_000) - 0.05) < 1e-9
    assert abs(epsilon_at(50_000, 1.0, 0.05, 10_000) - 0.05) < 1e-9


def _agent(**kwargs):
    defaults = dict(state_shape=(4,), action_dim=2, net="mlp", hidden_dim=16, batch_size=8,
                    buffer_size=100, learning_starts=8, device=torch.device("cpu"), seed=0)
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def test_hard_target_formula_and_update():
    agent = _agent(target_update="hard", target_update_every=5)
    rew, next_states, term = torch.randn(8), torch.randn(8, 4), torch.zeros(8)
    with torch.no_grad():
        expected = rew + agent.gamma * agent.target_net(next_states).max(dim=1).values
    assert torch.allclose(agent.compute_targets(rew, next_states, term), expected)

    with torch.no_grad():  # break the copy, then check the hard update restores it every C steps
        for p in agent.target_net.parameters():
            p.add_(1.0)
    for _ in range(4):
        agent.updates += 1
        agent.update_target_network()
    assert not torch.allclose(next(agent.target_net.parameters()), next(agent.q_net.parameters()))
    agent.updates += 1
    agent.update_target_network()
    for p_t, p_q in zip(agent.target_net.parameters(), agent.q_net.parameters()):
        assert torch.allclose(p_t, p_q)


def test_soft_update_formula():
    agent = _agent(target_update="soft", tau=0.25)
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.zero_()
    old = [p.detach().clone() for p in agent.target_net.parameters()]
    agent.update_target_network()
    for p_old, p_t, p_q in zip(old, agent.target_net.parameters(), agent.q_net.parameters()):
        assert torch.allclose(p_t, 0.75 * p_old + 0.25 * p_q.data)


def test_double_dqn_target_formula():
    agent = _agent(double_dqn=True)
    rew, next_states, term = torch.randn(8), torch.randn(8, 4), torch.zeros(8)
    with torch.no_grad():
        a_star = agent.q_net(next_states).argmax(dim=1, keepdim=True)
        expected = rew + agent.gamma * agent.target_net(next_states).gather(1, a_star).squeeze(1)
    assert torch.allclose(agent.compute_targets(rew, next_states, term), expected)


def test_no_target_network_uses_online_network():
    agent = _agent(use_target_network=False)
    rew, next_states, term = torch.randn(8), torch.randn(8, 4), torch.zeros(8)
    with torch.no_grad():
        expected = rew + agent.gamma * agent.q_net(next_states).max(dim=1).values
    assert torch.allclose(agent.compute_targets(rew, next_states, term), expected)


def test_learn_records_loss_and_q_mean():
    set_seed(0)
    agent = _agent()
    for i in range(8):
        agent.step(np.random.randn(4).astype(np.float32), i % 2, 1.0,
                   np.random.randn(4).astype(np.float32), False)
    assert agent.updates >= 1 and len(agent.losses) == agent.updates == len(agent.q_means)
    assert np.isfinite(agent.losses).all()


def test_variant_switches_are_isolated():
    """Only one setting may change between a variant and the reference."""
    ref = agent_kwargs("cartpole", "hard")
    for variant in EX1_VARIANTS[1:] + EX2_VARIANTS[1:]:
        kw = agent_kwargs("cartpole", variant)
        diff = {k for k in set(ref) | set(kw) if ref.get(k) != kw.get(k)}
        assert len(diff) <= 2, f"{variant} changes {diff}"
    for env_name in ENV_SPECS:
        assert agent_kwargs(env_name, "no_replay")["buffer_size"] == \
            agent_kwargs(env_name, "no_replay")["batch_size"]


def test_pong_pipeline_end_to_end():
    env = make_env("pong")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert env.action_space.n == 6
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, **agent_kwargs("pong", "hard"),
                     seed=0, device=torch.device("cpu"))
    agent.memory = ReplayMemory(64, state_shape=(4, 84, 84), state_dtype=np.uint8,
                                device=torch.device("cpu"))
    agent.learning_starts = 4
    agent.batch_size = 4
    for _ in range(6):
        action = agent.act(obs, epsilon=1.0)
        nxt, reward, terminated, truncated, _ = env.step(action)
        agent.step(obs, action, reward, nxt, terminated)
        obs = nxt
        if terminated or truncated:
            obs, _ = env.reset()
    assert agent.updates >= 1 and np.isfinite(agent.losses).all()
    env.close()


def test_tiny_cartpole_run_and_artifacts(tmp_path=None):
    import tempfile
    out_dir = tmp_path or tempfile.mkdtemp()
    env = make_env("cartpole")
    spec = dict(ENV_SPECS["cartpole"])
    agent = _agent()
    spec["eps"] = dict(eps_start=1.0, eps_end=0.05, eps_decay_steps=100)
    spec["metric_window"] = 5
    result = train(agent, env, spec, seed=0, budget=("episodes", 3))
    env.close()
    assert result.n_episodes == 3 and result.env_steps > 0
    metrics = evaluate(agent, make_eval_env("cartpole"), n_episodes=4, seed=123)
    assert metrics["returns"].shape == (4,) and np.isfinite(metrics["bias"])


def test_run_one_roundtrip(tmp_path=None):
    import tempfile
    out_dir = str(tmp_path or tempfile.mkdtemp())
    run = run_one("cartpole", "double", 0, budget=("episodes", 2), out_dir=out_dir, verbose=False)
    assert run["episode_returns"].shape == (2,)
    assert run["eval_returns"].shape == (20,)
    again = load_run("cartpole", "double", 0, out_dir)
    assert np.allclose(run["episode_returns"], again["episode_returns"])
    assert again["meta_json"]["variant"] == "double"
    assert os.path.exists(again["pth_path"])


if __name__ == "__main__":
    import tempfile
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if name == "test_tiny_cartpole_run_and_artifacts":
            fn(tempfile.mkdtemp())
        elif name == "test_run_one_roundtrip":
            fn(tempfile.mkdtemp())
        else:
            fn()
        print(f"ok   {name}")
    print("all checks passed")
