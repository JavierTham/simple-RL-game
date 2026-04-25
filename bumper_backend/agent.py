"""
DQN Agent with n-step returns, Double DQN, and soft target updates.
Also includes a heuristic DefaultAgent for training opponent.

Key design: n-step returns (n=10) solve the temporal credit assignment
problem for the charge→dash→hit sequence. The dash-hit reward at step
N+10 directly updates Q(charge) at step N.
"""
import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from neural_net import QNetwork, ReplayBuffer


class DQNAgent:
    """DQN agent with n-step returns, Double DQN, and soft target updates."""

    def __init__(self, lr=0.001, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=10, tau=0.005,
                 n_steps=10):
        self.q_net = QNetwork(lr=lr)
        self.target_net = QNetwork(lr=lr)
        self.target_net.copy_weights_from(self.q_net)
        self.target_net.eval()

        self.replay = ReplayBuffer(capacity=20000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.n_steps = n_steps
        self.train_steps = 0
        self.episodes_done = 0

        # N-step buffer: holds transitions until we can compute n-step return
        self._n_step_buffer = deque(maxlen=n_steps)

    def get_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy exploration that respects charge-only movement."""
        if random.random() < self.epsilon:
            my_charge = state[12]
            my_cooldown = state[14]

            # No charge = must charge (movement does nothing without charge)
            if my_charge < 0.05 or my_cooldown > 0.05:
                return 0  # charge

            # Have charge: decide whether to charge more or dash
            # Higher charge → more likely to release
            release_prob = 0.15 + 0.6 * my_charge
            if random.random() < release_prob:
                # Dash toward opponent with some noise
                opp_rx, opp_ry = state[4], state[5]
                angle = math.atan2(opp_ry, opp_rx) + random.gauss(0, 0.4)
                return self._angle_to_action(angle)
            return 0  # keep charging

        with torch.no_grad():
            q_values = self.q_net(state.reshape(1, -1))
            return int(q_values.argmax(dim=1).item())

    @staticmethod
    def _angle_to_action(angle: float) -> int:
        _PI = 3.141592653589793
        dirs = [
            (-_PI/2, 1), (-_PI/4, 2), (0, 3), (_PI/4, 4),
            (_PI/2, 5), (3*_PI/4, 6), (_PI, 7), (-3*_PI/4, 8),
        ]
        best, best_diff = 1, 999.0
        for da, act in dirs:
            diff = abs(angle - da)
            if diff > _PI:
                diff = 6.283185307 - diff
            if diff < best_diff:
                best_diff, best = diff, act
        return best

    def get_action_greedy(self, state: np.ndarray) -> int:
        """Purely greedy (no exploration) for test/PvP."""
        with torch.no_grad():
            q_values = self.q_net(state.reshape(1, -1))
            return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        """Store transition using n-step returns.

        Instead of immediately storing (s, a, r, s'), we buffer N
        transitions and compute the n-step discounted return:
            R_n = r_0 + γ*r_1 + γ²*r_2 + ... + γ^(n-1)*r_(n-1)
        Then store (s_0, a_0, R_n, s_n, done_n).

        This directly propagates the dash-hit reward back to the
        charge action N steps earlier.
        """
        self._n_step_buffer.append((state, action, reward, next_state, done))

        # If the episode ended, flush all remaining transitions
        if done:
            while len(self._n_step_buffer) > 0:
                self._flush_n_step()
            return

        # If we have enough transitions, compute n-step return
        if len(self._n_step_buffer) == self.n_steps:
            self._flush_n_step()

    def _flush_n_step(self):
        """Compute n-step return from buffer and store in replay."""
        n = len(self._n_step_buffer)
        if n == 0:
            return

        # Compute discounted return: R = r_0 + γ*r_1 + ... + γ^(n-1)*r_(n-1)
        R = 0.0
        for i in range(n - 1, -1, -1):
            R = self._n_step_buffer[i][2] + self.gamma * R

        s0, a0 = self._n_step_buffer[0][0], self._n_step_buffer[0][1]
        s_n = self._n_step_buffer[-1][3]
        done_n = self._n_step_buffer[-1][4]

        # Store: (initial_state, initial_action, n_step_return, final_next_state, final_done)
        # The TD target will be: R + γ^n * V(s_n) if not done
        self.replay.push(s0, a0, R, s_n, done_n)
        self._n_step_buffer.popleft()

    def train_step(self):
        """One batch training step with n-step returns."""
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, returns, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.from_numpy(states).float()
        actions_t = torch.from_numpy(actions).long()
        returns_t = torch.from_numpy(returns).float()
        next_states_t = torch.from_numpy(next_states).float()
        dones_t = torch.from_numpy(dones)

        # Current Q-values for chosen actions
        q_current = self.q_net(states_t)
        q_taken = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN with n-step bootstrap
        with torch.no_grad():
            q_next_online = self.q_net(next_states_t)
            best_actions = q_next_online.argmax(dim=1, keepdim=True)
            q_next_target = self.target_net(next_states_t)
            max_q_next = q_next_target.gather(1, best_actions).squeeze(1)
            # n-step TD target: R_n + γ^n * V(s_n)
            gamma_n = self.gamma ** self.n_steps
            td_targets = returns_t + gamma_n * max_q_next * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_taken, td_targets)

        self.q_net.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_net.optimizer.step()

        self.train_steps += 1

        # Soft target update
        if self.target_update_freq > 0:
            if self.train_steps % self.target_update_freq == 0:
                self.target_net.copy_weights_from(self.q_net)
        else:
            for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(op.data * self.tau)

        return loss.item()

    def end_episode(self):
        """Called at end of each episode. Decays epsilon."""
        # Flush any remaining transitions in n-step buffer
        while len(self._n_step_buffer) > 0:
            self._flush_n_step()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1

    def get_weights(self) -> dict:
        return self.q_net.get_weights()

    def set_weights(self, data: dict):
        self.q_net.set_weights(data)
        self.target_net.copy_weights_from(self.q_net)


class DefaultAgent:
    """Heuristic bot for charge-only movement.

    ALL movement is charge→dash. The bot decides:
    1. How long to charge (short for repositioning, long for attacks)
    2. Which direction to dash

    Tactical modes:
    - ATTACK: Charge toward opponent, release at high charge
    - REPOSITION: Short charge + dash to improve position
    - EVADE_EDGE: Short charge + dash toward center
    - DODGE: Short charge + dash perpendicular when opponent charges

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
        self._circle_dir = 1 if random.random() > 0.5 else -1
        self._step = 0
        self._charge_target = 0   # how many steps to charge before releasing
        self._dash_angle = 0.0    # direction to dash when releasing
        self._mode = 'idle'       # current tactical mode

    def get_action(self, state: np.ndarray) -> int:
        opp_rx, opp_ry = state[4], state[5]
        my_x, my_y = state[0], state[1]
        my_edge = state[8]
        dist_norm = state[11]
        my_charge = state[12]
        opp_charge = state[13]
        my_cooldown = state[14]

        angle_to_opp = math.atan2(opp_ry, opp_rx)
        self._step += 1

        # Strength check: at low strength, play a sloppy version
        # (still charges & dashes, but with poor aim — unlike pure random
        #  which wastes turns on uncharged move actions that do nothing)
        if random.random() > self.strength:
            if my_cooldown > 0.05:
                return 0
            if my_charge < 0.15:
                return 0  # at least charge up
            # Dash with poor aim (~46° deviation)
            noise = random.gauss(0, 0.8)
            return self._angle_to_action(angle_to_opp + noise)

        # On cooldown: can't do anything, just wait
        if my_cooldown > 0.05:
            return 0  # charge (will be ignored during cooldown, but that's fine)

        # If we have charge, decide whether to release or keep charging
        if my_charge > 0.02:
            # URGENT: near edge — dash toward center immediately
            if my_edge > 0.70:
                angle = math.atan2(-my_y, -my_x)
                return self._angle_to_action(angle)

            # URGENT: opponent about to dash us and we're close
            if self.strength > 0.75 and opp_charge > 0.4 and dist_norm < 0.30:
                # Dodge: dash perpendicular
                perp = angle_to_opp + self._circle_dir * self._PI / 2
                return self._angle_to_action(perp)

            # If mode is set, follow it
            if self._mode == 'attack':
                self._charge_target -= 1
                if self._charge_target <= 0 or my_charge > 0.85:
                    # Release the attack dash
                    self._mode = 'idle'
                    aim_noise = 0.12 + 0.4 * (1.0 - self.strength)
                    angle = angle_to_opp + random.gauss(0, aim_noise)
                    return self._angle_to_action(angle)
                return 0  # keep charging

            if self._mode == 'reposition':
                self._charge_target -= 1
                if self._charge_target <= 0:
                    self._mode = 'idle'
                    return self._angle_to_action(self._dash_angle)
                return 0  # keep charging

            # No mode set but have charge — release toward opponent
            if my_charge > 0.15:
                aim_noise = 0.2 + 0.4 * (1.0 - self.strength)
                return self._angle_to_action(angle_to_opp + random.gauss(0, aim_noise))

            # Low charge, keep going
            return 0

        # No charge — decide what to do next (pick a mode)
        time_pressure = min(1.0, self._step / 200)

        # ATTACK: charge up and dash at opponent
        if dist_norm < 0.40:
            attack_chance = 0.4 + 0.4 * time_pressure
            if random.random() < attack_chance:
                self._mode = 'attack'
                # Charge longer when far, shorter when close
                base = int(8 + 20 * dist_norm / 0.40)
                self._charge_target = base + random.randint(-3, 5)
                return 0

        # REPOSITION: short dash to improve position
        if dist_norm > 0.35 or my_edge > 0.55:
            self._mode = 'reposition'
            self._charge_target = random.randint(3, 8)  # short charge

            if my_edge > 0.55:
                # Dash toward center
                self._dash_angle = math.atan2(-my_y, -my_x)
            else:
                # Dash toward opponent with flanking offset
                offset = self._circle_dir * self._PI / 4
                self._dash_angle = angle_to_opp + offset
                if self._step % 40 == 0 and random.random() < 0.3:
                    self._circle_dir *= -1
            return 0

        # Default: start an attack charge
        self._mode = 'attack'
        self._charge_target = random.randint(10, 25)
        return 0

    def _angle_to_action(self, angle: float) -> int:
        best, best_diff = 1, 999.0
        for da, act in self._DIR:
            diff = abs(angle - da)
            if diff > self._PI:
                diff = 6.283185307 - diff
            if diff < best_diff:
                best_diff, best = diff, act
        return best

    def store(self, *_a): pass

