"""Experience replay.

Two storage modes:

* **uniform replay** (the reference DQN): a FIFO ring buffer of `capacity` transitions from which
  minibatches are drawn uniformly at random, which decorrelates the samples of a gradient step;
* **no replay**: only the `batch_size` most recent transitions are kept and `sample()` returns them
  *in chronological order*. The agent then performs SGD on a single consecutive trajectory.

For Atari the states are stored in pre-allocated `np.uint8` arrays (56 KB per transition instead of
226 KB for float32) and divided by 255 inside the network.
"""
from __future__ import annotations

from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))


class ReplayMemory:
    """Fixed-capacity buffer of transitions, array-backed when `state_shape` is given."""

    def __init__(self, capacity: int, state_shape=None, state_dtype=np.float32, device=None,
                 chronological: bool = False):
        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape) if state_shape is not None else None
        self.state_dtype = state_dtype
        self.device = device
        self.chronological = chronological

        if self.state_shape is None:  # generic fallback (used for ad-hoc objects)
            self.memory = deque(maxlen=self.capacity)
            self.size = 0
            return

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=state_dtype)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=state_dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminateds = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    # ------------------------------------------------------------------ writing
    def push(self, state, action, reward, next_state, terminated) -> None:
        if self.state_shape is None:
            self.memory.append(Transition(state, action, reward, next_state, terminated))
            self.size = len(self.memory)
            return
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.terminateds[self.pos] = float(bool(terminated))
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------ reading
    def _indices(self, batch_size: int) -> np.ndarray:
        if self.chronological:  # the `batch_size` most recent transitions, oldest first
            n = min(batch_size, self.size)
            return (self.pos - np.arange(n, 0, -1)) % self.capacity
        return np.random.randint(0, self.size, size=batch_size)

    def sample(self, batch_size: int):
        """Minibatch already converted into tensors on `device`."""
        batch_size = int(batch_size)
        if self.state_shape is None:
            batch = (list(self.memory) if self.chronological else
                     [self.memory[i] for i in np.random.choice(self.size, batch_size, replace=False)])
            states, actions, rewards, next_states, terminateds = zip(*batch)
            to_state = lambda arr: torch.as_tensor(np.asarray(arr, dtype=np.float32),
                                                   device=self.device)
            return (to_state(states),
                    torch.as_tensor(actions, dtype=torch.int64, device=self.device),
                    torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                    to_state(next_states),
                    torch.as_tensor(terminateds, dtype=torch.float32, device=self.device))

        idx = self._indices(batch_size)
        return (
            torch.as_tensor(self.states[idx], device=self.device),
            torch.as_tensor(self.actions[idx], dtype=torch.int64, device=self.device),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_states[idx], device=self.device),
            torch.as_tensor(self.terminateds[idx], dtype=torch.float32, device=self.device),
        )

    def __len__(self) -> int:
        return self.size
