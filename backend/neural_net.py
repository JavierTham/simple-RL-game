"""
DQN (Deep Q-Network) for the bumper bot.
Architecture: 8 → 128 → 64 → 9
Uses experience replay and a target network for stable training.
"""
import numpy as np
from collections import deque
import random


class QNetwork:
    """Simple feedforward Q-network with manual forward/backward."""

    def __init__(self, input_size=8, hidden1=128, hidden2=64, output_size=9, lr=0.001):
        self.lr = lr
        self.sizes = (input_size, hidden1, hidden2, output_size)

        # Xavier init
        self.w1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / (input_size + hidden1))
        self.b1 = np.zeros(hidden1)
        self.w2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / (hidden1 + hidden2))
        self.b2 = np.zeros(hidden2)
        self.w3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0 / (hidden2 + output_size))
        self.b3 = np.zeros(output_size)

        # Cached activations
        self._x = self._z1 = self._a1 = None
        self._z2 = self._a2 = self._z3 = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x can be (8,) or (batch, 8)."""
        self._x = x
        self._z1 = x @ self.w1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        self._z2 = self._a1 @ self.w2 + self.b2
        self._a2 = np.maximum(0, self._z2)
        self._z3 = self._a2 @ self.w3 + self.b3
        return self._z3  # raw Q-values (no activation on output)

    def backward_batch(self, dq: np.ndarray):
        """Backprop from output gradient dq, shape (batch, 9). Updates weights."""
        batch = dq.shape[0]
        dq = np.clip(dq, -1.0, 1.0)  # gradient clipping (Huber-like)

        # Layer 3
        dw3 = self._a2.T @ dq / batch
        db3 = dq.mean(axis=0)
        da2 = dq @ self.w3.T

        # Layer 2
        dz2 = da2 * (self._z2 > 0)
        dw2 = self._a1.T @ dz2 / batch
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.w2.T

        # Layer 1
        dz1 = da1 * (self._z1 > 0)
        dw1 = self._x.T @ dz1 / batch
        db1 = dz1.mean(axis=0)

        # SGD update
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3

    def copy_weights_from(self, other: 'QNetwork'):
        """Copy weights from another network (for target network)."""
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()
        self.w3 = other.w3.copy()
        self.b3 = other.b3.copy()

    def get_weights(self) -> dict:
        return {
            'w1': self.w1.tolist(), 'b1': self.b1.tolist(),
            'w2': self.w2.tolist(), 'b2': self.b2.tolist(),
            'w3': self.w3.tolist(), 'b3': self.b3.tolist(),
            'config': {
                'input_size': self.sizes[0], 'hidden1': self.sizes[1],
                'hidden2': self.sizes[2], 'output_size': self.sizes[3],
                'lr': self.lr,
            },
        }

    def set_weights(self, data: dict):
        self.w1 = np.array(data['w1'], dtype=np.float64)
        self.b1 = np.array(data['b1'], dtype=np.float64)
        self.w2 = np.array(data['w2'], dtype=np.float64)
        self.b2 = np.array(data['b2'], dtype=np.float64)
        self.w3 = np.array(data['w3'], dtype=np.float64)
        self.b3 = np.array(data['b3'], dtype=np.float64)


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
