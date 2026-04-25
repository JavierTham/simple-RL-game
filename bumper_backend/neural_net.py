"""
DQN (Deep Q-Network) for the bumper bot.
Architecture: Dueling DQN — separates state value V(s) from advantage A(s,a).
This is much more sample-efficient for games where many states have similar
values but the best action varies.

15 → 128 → V(s) stream + A(s,a) stream → Q(s,a)
"""
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('cpu')


class QNetwork(nn.Module):
    """Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A)."""

    def __init__(self, input_size=15, hidden1=128, hidden2=64, output_size=9, lr=0.001):
        super().__init__()
        self.sizes = (input_size, hidden1, hidden2, output_size)
        self.lr = lr

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
        )

        # Init
        for module in [self.features, self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(DEVICE)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
        features = self.features(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, 9)
        # Dueling: Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def copy_weights_from(self, other: 'QNetwork'):
        self.load_state_dict(other.state_dict())

    def get_weights(self) -> dict:
        """Export weights as JSON-serializable dict."""
        sd = self.state_dict()
        result = {}
        for key, tensor in sd.items():
            result[key] = tensor.numpy().tolist()
        result['config'] = {
            'input_size': self.sizes[0], 'hidden1': self.sizes[1],
            'hidden2': self.sizes[2], 'output_size': self.sizes[3],
            'lr': self.lr, 'arch': 'dueling',
        }
        return result

    def set_weights(self, data: dict):
        """Load weights from JSON-serializable dict."""
        sd = {}
        for key, value in data.items():
            if key == 'config':
                continue
            sd[key] = torch.tensor(value, dtype=torch.float32)

        # Handle legacy format (w1/b1/w2/b2/w3/b3)
        if 'w1' in data and 'features.0.weight' not in data:
            # Old 3-layer format — load into features + advantage stream
            sd = {
                'features.0.weight': torch.tensor(data['w1'], dtype=torch.float32).T,
                'features.0.bias': torch.tensor(data['b1'], dtype=torch.float32),
                'advantage_stream.0.weight': torch.tensor(data['w2'], dtype=torch.float32).T,
                'advantage_stream.0.bias': torch.tensor(data['b2'], dtype=torch.float32),
                'advantage_stream.2.weight': torch.tensor(data['w3'], dtype=torch.float32).T,
                'advantage_stream.2.bias': torch.tensor(data['b3'], dtype=torch.float32),
            }
            # Initialize value stream with small values (legacy bots didn't have it)
            for key in ['value_stream.0.weight', 'value_stream.0.bias',
                        'value_stream.2.weight', 'value_stream.2.bias']:
                sd[key] = self.state_dict()[key]
            self.load_state_dict(sd)
            return

        self.load_state_dict(sd)


class ReplayBuffer:
    """Fixed-size circular replay buffer with priority sampling."""

    def __init__(self, capacity: int = 20000, obs_dim: int = 15):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        # High-reward transitions get higher priority
        self.priorities[i] = 1.0 + abs(reward) * 2.0
        self.pos = (i + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int):
        n = min(batch_size, self.size)
        # Proportional priority sampling
        probs = self.priorities[:self.size]
        probs = probs / probs.sum()
        idx = np.random.choice(self.size, size=n, p=probs, replace=False)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size
