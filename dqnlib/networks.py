"""Q-networks: the MLP of CartPole/LunarLander, the Nature CNN of Pong, and their dueling heads.

Every network takes the state as input and returns **one Q-value per action**, so the greedy
action is simply the argmax of a single forward pass.

Dueling heads (Wang et al., 2016) split the output into a state value V(s) and an advantage
A(s, a), recombined as

    Q(s, a) = V(s) + A(s, a) - (1 / |A|) sum_{a'} A(s, a')

which removes the unidentifiability of the (V, A) decomposition.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CNN_FEATURES = 64 * 7 * 7  # 3 136: size of the Nature encoder output for an 84x84 input


def _prep_pixels(x: torch.Tensor) -> torch.Tensor:
    """Replay memory stores uint8 frames: convert and scale to [0, 1] inside the network."""
    if x.dtype == torch.uint8:
        return x.float().div_(255.0)
    return x.div(255.0)


class QNetwork(nn.Module):
    """MLP state_dim -> hidden_dim -> hidden_dim -> action_dim, ReLU, linear output."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)


class NatureEncoder(nn.Module):
    """The three convolutions of Mnih et al. (2015): (C, 84, 84) -> 3 136 features."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = _prep_pixels(state)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return torch.flatten(x, 1)


class NatureCNN(nn.Module):
    """Nature CNN: encoder -> FC 512 -> linear 6 Q-values (1 687 206 parameters)."""

    def __init__(self, action_dim: int, in_channels: int = 4):
        super().__init__()
        self.encoder = NatureEncoder(in_channels)
        self.fc = nn.Linear(CNN_FEATURES, 512)
        self.out = nn.Linear(512, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(self.encoder(state)))
        return self.out(x)


class DuelingQNetwork(nn.Module):
    """MLP with the dueling head: shared body, then a value stream and an advantage stream."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.body(state)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


class DuelingNatureCNN(nn.Module):
    """Nature CNN with the dueling head: 3 136 -> 512 streams for V and A."""

    def __init__(self, action_dim: int, in_channels: int = 4):
        super().__init__()
        self.encoder = NatureEncoder(in_channels)
        self.value = nn.Sequential(nn.Linear(CNN_FEATURES, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage = nn.Sequential(nn.Linear(CNN_FEATURES, 512), nn.ReLU(),
                                       nn.Linear(512, action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.encoder(state)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


def make_q_network(kind: str, state_shape, action_dim: int, hidden_dim: int = 64,
                   dueling: bool = False) -> nn.Module:
    """Build the network named by the environment spec ("mlp" or "cnn"), plain or dueling."""
    if kind == "mlp":
        state_dim = int(np.prod(state_shape))
        cls = DuelingQNetwork if dueling else QNetwork
        return cls(state_dim, action_dim, hidden_dim)
    if kind == "cnn":
        channels = int(state_shape[0]) if len(state_shape) == 3 else 4
        cls = DuelingNatureCNN if dueling else NatureCNN
        return cls(action_dim, in_channels=channels)
    raise ValueError(f"unknown network kind: {kind!r}")
