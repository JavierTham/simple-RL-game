"""
DQN (Deep Q-Network) for the bumper bot.
Architecture: 15 → 192 → 96 → 9
Uses PyTorch with Adam optimizer for stable, fast convergence.
"""
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

# Use CPU — network is tiny, GPU overhead would slow things down
DEVICE = torch.device('cpu')


class QNetwork(nn.Module):
    """Feedforward Q-network using PyTorch."""

    def __init__(self, input_size=15, hidden1=192, hidden2=96, output_size=9, lr=0.001):
        super().__init__()
        self.sizes = (input_size, hidden1, hidden2, output_size)
        self.lr = lr

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
        )

        # Xavier init (same as before)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(DEVICE)

    def forward(self, x):
        """Forward pass. Accepts numpy array or torch tensor."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
        return self.net(x)

    def copy_weights_from(self, other: 'QNetwork'):
        """Copy weights from another network (for target network)."""
        self.load_state_dict(other.state_dict())

    def get_weights(self) -> dict:
        """Export weights as JSON-serializable dict (numpy lists)."""
        sd = self.state_dict()
        return {
            'w1': sd['net.0.weight'].T.numpy().tolist(),  # transpose to match old format
            'b1': sd['net.0.bias'].numpy().tolist(),
            'w2': sd['net.2.weight'].T.numpy().tolist(),
            'b2': sd['net.2.bias'].numpy().tolist(),
            'w3': sd['net.4.weight'].T.numpy().tolist(),
            'b3': sd['net.4.bias'].numpy().tolist(),
            'config': {
                'input_size': self.sizes[0], 'hidden1': self.sizes[1],
                'hidden2': self.sizes[2], 'output_size': self.sizes[3],
                'lr': self.lr,
            },
        }

    def set_weights(self, data: dict):
        """Load weights from JSON-serializable dict."""
        sd = {
            'net.0.weight': torch.tensor(data['w1'], dtype=torch.float32).T,
            'net.0.bias': torch.tensor(data['b1'], dtype=torch.float32),
            'net.2.weight': torch.tensor(data['w2'], dtype=torch.float32).T,
            'net.2.bias': torch.tensor(data['b2'], dtype=torch.float32),
            'net.4.weight': torch.tensor(data['w3'], dtype=torch.float32).T,
            'net.4.bias': torch.tensor(data['b3'], dtype=torch.float32),
        }
        self.load_state_dict(sd)


class ReplayBuffer:
    """Fixed-size circular replay buffer backed by numpy arrays."""

    def __init__(self, capacity: int = 50000, obs_dim: int = 15):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.pos = (i + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size
