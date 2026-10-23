"""The DQN agent, with every switch used by the ablation study and by Double/Dueling DQN.

Reference DQN (Mnih et al., 2015): replay memory + target network with a hard update.

Switches (Exercise 1)
    target_update="hard" | "soft"   hard: theta- <- theta every C gradient steps
                                    soft: theta- <- tau*theta + (1-tau)*theta- after every step
    use_replay=False                memory keeps only the last batch_size transitions
    use_target_network=False        targets computed with the online network theta

Switches (Exercise 2)
    double_dqn=True                 a* = argmax_a Q(s', a; theta), y = r + gamma Q(s', a*; theta-)
    dueling=True                    dueling head instead of the standard head
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dqnlib.networks import make_q_network
from dqnlib.replay import ReplayMemory
from dqnlib.utils import get_device, set_seed


class DQNAgent:
    """Online network theta, target network theta-, replay memory D."""

    def __init__(self, state_shape, action_dim: int, net: str = "mlp", hidden_dim: int = 64,
                 lr: float = 5e-4, gamma: float = 0.99, buffer_size: int = 100_000,
                 batch_size: int = 64, learn_every: int = 1, learning_starts: int | None = None,
                 target_update: str = "hard", target_update_every: int = 100, tau: float = 0.01,
                 use_replay: bool = True, use_target_network: bool = True,
                 double_dqn: bool = False, dueling: bool = False, loss: str = "mse",
                 grad_clip: float | None = None, clip_reward: bool = False,
                 device=None, seed: int = 0):
        if target_update not in ("hard", "soft"):
            raise ValueError(f"target_update must be 'hard' or 'soft', got {target_update!r}")
        if loss not in ("mse", "huber"):
            raise ValueError(f"loss must be 'mse' or 'huber', got {loss!r}")

        set_seed(seed)
        self.seed = seed
        self.device = device if device is not None else get_device()
        self.state_shape = tuple(state_shape)
        self.action_dim = int(action_dim)
        self.net = net
        self.hidden_dim = hidden_dim
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learn_every = int(learn_every)
        self.learning_starts = int(batch_size if learning_starts is None else learning_starts)
        self.target_update = target_update
        self.target_update_every = int(target_update_every)
        self.tau = float(tau)
        self.use_replay = bool(use_replay)
        self.use_target_network = bool(use_target_network)
        self.double_dqn = bool(double_dqn)
        self.dueling = bool(dueling)
        self.loss_name = loss
        self.grad_clip = grad_clip
        self.clip_reward = bool(clip_reward)

        # Atari observations are uint8 (56 KB per transition in the buffer), others float32.
        self.state_dtype = np.uint8 if net == "cnn" else np.float32

        self.q_net = make_q_network(net, self.state_shape, self.action_dim,
                                    hidden_dim, dueling=dueling).to(self.device)
        self.target_net = make_q_network(net, self.state_shape, self.action_dim,
                                         hidden_dim, dueling=dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayMemory(
            capacity=buffer_size if self.use_replay else self.batch_size,
            state_shape=self.state_shape,
            state_dtype=self.state_dtype,
            device=self.device,
            chronological=not self.use_replay,
        )

        self.env_steps = 0    # environment steps seen
        self.updates = 0      # gradient steps done
        self.losses: list[float] = []
        self.q_means: list[float] = []  # mean of Q(s_i, a_i; theta) over the minibatch

    # ------------------------------------------------------------------ acting
    def _to_tensor(self, state) -> torch.Tensor:
        if self.state_dtype == np.uint8:
            arr = np.asarray(state, dtype=np.uint8)
        else:
            arr = np.asarray(state, dtype=np.float32)
        return torch.as_tensor(arr, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def q_values(self, state) -> np.ndarray:
        return self.q_net(self._to_tensor(state)).squeeze(0).cpu().numpy()

    def greedy_action(self, state) -> int:
        return int(self.q_values(state).argmax())

    def act(self, state, epsilon: float = 0.0) -> int:
        """epsilon-greedy: a uniformly random action with probability epsilon."""
        if epsilon > 0.0 and random.random() < epsilon:
            return random.randrange(self.action_dim)
        return self.greedy_action(state)

    # ------------------------------------------------------------------ learning
    def step(self, state, action, reward, next_state, terminated) -> None:
        """Store the transition; every `learn_every` steps, take one gradient step."""
        reward = float(np.sign(reward)) if self.clip_reward else float(reward)
        self.memory.push(state, action, reward, next_state, terminated)
        self.env_steps += 1
        if (self.env_steps >= self.learning_starts
                and self.env_steps % self.learn_every == 0
                and len(self.memory) >= self.batch_size):
            self.learn(self.memory.sample(self.batch_size))

    def compute_targets(self, rewards, next_states, terminateds) -> torch.Tensor:
        """y = r + gamma (1 - terminated) * (max_a' Q(s', a'; theta-) or Q(s', a*; theta-))."""
        eval_net = self.target_net if self.use_target_network else self.q_net
        if self.double_dqn:
            # theta selects the next action, theta- evaluates it (one extra forward pass of theta)
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q = eval_net(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q = eval_net(next_states).max(dim=1).values
        return rewards + self.gamma * (1.0 - terminateds) * next_q

    def learn(self, batch) -> None:
        states, actions, rewards, next_states, terminateds = batch

        with torch.no_grad():  # semi-gradient: the targets are constants
            targets = self.compute_targets(rewards, next_states, terminateds)

        q_sa = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        if self.loss_name == "huber":
            loss = F.smooth_l1_loss(q_sa, targets)
        else:
            loss = F.mse_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.losses.append(float(loss.item()))
        self.q_means.append(float(q_sa.mean().item()))

        self.updates += 1
        self.update_target_network()

    def update_target_network(self) -> None:
        if not self.use_target_network:
            return
        if self.target_update == "soft":
            with torch.no_grad():
                for target_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                    target_p.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
        elif self.updates % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------ io
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.q_net.parameters())

    def config(self) -> dict:
        """Everything needed to rebuild the agent (stored in the run metadata)."""
        return dict(state_shape=list(self.state_shape), action_dim=self.action_dim, net=self.net,
                    hidden_dim=self.hidden_dim, gamma=self.gamma, batch_size=self.batch_size,
                    learn_every=self.learn_every, learning_starts=self.learning_starts,
                    target_update=self.target_update,
                    target_update_every=self.target_update_every, tau=self.tau,
                    use_replay=self.use_replay, use_target_network=self.use_target_network,
                    double_dqn=self.double_dqn, dueling=self.dueling, loss=self.loss_name,
                    grad_clip=self.grad_clip, clip_reward=self.clip_reward, seed=self.seed,
                    lr=self.optimizer.param_groups[0]["lr"], buffer_size=self.memory.capacity)

    def save(self, path: str) -> None:
        from dqnlib.utils import atomic_save_torch
        atomic_save_torch(path, self.q_net.state_dict())

    def load(self, path: str) -> None:
        self.q_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
