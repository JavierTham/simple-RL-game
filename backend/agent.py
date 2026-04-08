"""
DQN Agent with epsilon-greedy exploration and target network.
Also includes a heuristic DefaultAgent for training opponent.
"""
import math
import random
import numpy as np
from neural_net import QNetwork, ReplayBuffer


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(self, lr=0.001, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=10):
        self.q_net = QNetwork(lr=lr)
        self.target_net = QNetwork(lr=lr)
        self.target_net.copy_weights_from(self.q_net)

        self.replay = ReplayBuffer(capacity=50000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0
        self.episodes_done = 0

    def get_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 9)
        q_values = self.q_net.forward(state.reshape(1, -1))
        return int(np.argmax(q_values[0]))

    def get_action_greedy(self, state: np.ndarray) -> int:
        """Purely greedy (no exploration) for test/PvP."""
        q_values = self.q_net.forward(state.reshape(1, -1))
        return int(np.argmax(q_values[0]))

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self):
        """One batch training step from replay buffer."""
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # Current Q-values
        q_current = self.q_net.forward(states)  # (batch, 9)

        # Target Q-values (from target network)
        q_next = self.target_net.forward(next_states)  # (batch, 9)
        max_q_next = np.max(q_next, axis=1)

        # TD targets
        targets = rewards + self.gamma * max_q_next * (1 - dones.astype(float))

        # Compute gradient: dQ/doutput for the chosen actions only
        dq = np.zeros_like(q_current)
        for i in range(len(actions)):
            # MSE gradient: 2 * (q_predicted - target) for the taken action
            dq[i, actions[i]] = q_current[i, actions[i]] - targets[i]

        # Backprop and update
        self.q_net.forward(states)  # re-forward to set cached activations
        self.q_net.backward_batch(dq)

        self.train_steps += 1

        # Update target network periodically
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.q_net)

        return float(np.mean(np.abs(dq[np.arange(len(actions)), actions])))

    def end_episode(self):
        """Called at end of each episode. Decays epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1

    def get_weights(self) -> dict:
        return self.q_net.get_weights()

    def set_weights(self, data: dict):
        self.q_net.set_weights(data)
        self.target_net.copy_weights_from(self.q_net)


class DefaultAgent:
    """Heuristic bot: chases opponent, avoids edge.
    strength: 0=random, 1=full heuristic.
    """
    _PI = 3.141592653589793
    _DIR = [
        (-_PI/2, 1), (-_PI/4, 2), (0, 3), (_PI/4, 4),
        (_PI/2, 5), (3*_PI/4, 6), (_PI, 7), (-3*_PI/4, 8),
    ]

    def __init__(self, aggression: float = 0.7, strength: float = 1.0):
        self.aggression = aggression
        self.strength = strength

    def get_action(self, state: np.ndarray) -> int:
        if random.random() > self.strength:
            return random.randint(0, 8)

        opp_rx, opp_ry = state[4], state[5]
        my_x, my_y = state[0], state[1]
        angle = math.atan2(opp_ry, opp_rx)
        dist = math.sqrt(my_x * my_x + my_y * my_y)
        if dist > 0.5:
            ca = math.atan2(-my_y, -my_x)
            blend = min(1.0, (dist - 0.5) * 4)
            angle = angle * (1 - blend * 0.5) + ca * blend * 0.5

        best, best_diff = 0, 999.0
        for da, act in self._DIR:
            diff = abs(angle - da)
            if diff > self._PI:
                diff = 6.283185307 - diff
            if diff < best_diff:
                best_diff, best = diff, act
        if random.random() < 0.05:
            best = random.randint(0, 8)
        return best

    def store(self, *_a): pass
