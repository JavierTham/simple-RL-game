"""
Training manager: runs DQN training with experience replay.
"""
import math
import numpy as np
from physics import PhysicsWorld, ARENA_RADIUS, MAX_STEPS
from agent import DQNAgent, DefaultAgent

TRAIN_EVERY = 2      # train every N environment steps (2x speedup vs every step)
TRAIN_MAX_STEPS = 300 # shorter episodes during training for speed
_ARENA_DIAM = 2 * ARENA_RADIUS


class Trainer:
    def __init__(self):
        self.agent: DQNAgent | None = None
        self.world = PhysicsWorld()
        self.is_training = False
        self.training_stats: list[dict] = []

    # ── reward shaping ──────────────────────────────────────
    @staticmethod
    def compute_reward(world: PhysicsWorld, bot_idx: int,
                       done: bool, winner, collision: bool,
                       w: dict) -> float:
        bot = world.bot1 if bot_idx == 0 else world.bot2
        opp = world.bot2 if bot_idx == 0 else world.bot1
        r = 0.0

        # Terminal
        if done:
            if winner == bot_idx:
                r += w.get('win_bonus', 1.0) * 10.0
            elif winner not in (None, -1):
                r -= 10.0

        # Survival
        if not done:
            r += w.get('survival', 0.1) * 0.3

        # Aggression
        dx = opp.x - bot.x
        dy = opp.y - bot.y
        dist_opp = math.sqrt(dx * dx + dy * dy)
        r += w.get('aggression', 0.3) * (1.0 - dist_opp / _ARENA_DIAM) * 0.3

        # Center control
        dc = math.sqrt(bot.x * bot.x + bot.y * bot.y)
        edge = dc / ARENA_RADIUS
        r += w.get('center_control', 0.2) * (1.0 - edge) * 0.3

        # Hit reward
        if collision:
            r += w.get('hit_reward', 0.5) * 1.5

        # Edge penalty (quadratic)
        if edge > 0.5:
            r -= w.get('edge_penalty', 0.3) * (edge - 0.5) * (edge - 0.5) * 6.0

        # Hard boundary repulsion
        if edge > 0.85:
            r -= 1.5 * (edge - 0.85) / 0.15

        return r

    # ── single episode (with DQN updates) ───────────────────
    def run_episode(self, agent: DQNAgent, opponent, reward_weights: dict,
                    train: bool = True, record_frames: bool = False,
                    max_steps: int | None = None):
        self.world.reset()
        frames = [] if record_frames else None
        done = False
        winner = None
        total_reward = 0.0
        step_count = 0
        step_limit = max_steps or (TRAIN_MAX_STEPS if train else MAX_STEPS)

        while not done:
            obs1 = self.world.get_observation(0)
            obs2 = self.world.get_observation(1)

            if train:
                a1 = agent.get_action(obs1)
            else:
                a1 = agent.get_action_greedy(obs1)
            a2 = opponent.get_action(obs2)

            done, winner = self.world.step(a1, a2)
            step_count += 1
            if step_count >= step_limit and not done:
                done = True
                d1 = self.world.bot1.dist_from_center()
                d2 = self.world.bot2.dist_from_center()
                winner = 0 if d1 < d2 else (1 if d2 < d1 else -1)

            r = self.compute_reward(self.world, 0, done, winner,
                                    self.world.collision_occurred, reward_weights)
            next_obs1 = self.world.get_observation(0)

            if train:
                agent.store(obs1, a1, r, next_obs1, done)
                if step_count % TRAIN_EVERY == 0:
                    agent.train_step()

            total_reward += r

            if record_frames:
                fd = self.world.get_frame_data()
                fd['actions'] = [a1, a2]
                frames.append(fd)

        if train:
            agent.end_episode()

        return {'winner': winner, 'steps': step_count,
                'total_reward': total_reward, 'frames': frames}

    # ── training loop ────────────────────────────────────────
    def train(self, config: dict, progress_callback=None) -> dict | None:
        num_episodes = config.get('num_episodes', 1000)
        lr = config.get('learning_rate', 0.001)
        rw = config.get('reward_weights', {})

        self.agent = DQNAgent(
            lr=lr,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            batch_size=64,
            target_update_freq=20,
        )
        self.is_training = True
        self.training_stats.clear()

        wins, recent = 0, []
        for ep in range(num_episodes):
            if not self.is_training:
                break

            # Curriculum: ramp opponent strength
            progress = ep / max(num_episodes - 1, 1)
            opp_strength = 0.3 + 0.7 * min(1.0, progress * 2.0)
            opponent = DefaultAgent(strength=opp_strength)

            result = self.run_episode(self.agent, opponent, rw, train=True)

            won = result['winner'] == 0
            if won:
                wins += 1
            recent.append(int(won))
            if len(recent) > 100:
                recent.pop(0)

            stats = {
                'episode': ep + 1,
                'total_episodes': num_episodes,
                'win_rate': round(sum(recent) / len(recent), 4),
                'total_wins': wins,
                'avg_reward': round(result['total_reward'], 4),
                'steps': result['steps'],
                'won': won,
                'epsilon': round(self.agent.epsilon, 4),
            }
            self.training_stats.append(stats)
            if progress_callback and (ep % 5 == 0 or ep == num_episodes - 1):
                progress_callback(stats)

        self.is_training = False
        return self.agent.get_weights() if self.agent else None

    # ── test match ───────────────────────────────────────────
    def test_match(self, reward_weights: dict | None = None):
        if self.agent is None:
            return None
        result = self.run_episode(self.agent, DefaultAgent(),
                                  reward_weights or {},
                                  train=False, record_frames=True)
        return result

    # ── PvP match ────────────────────────────────────────────
    def run_match(self, w1: dict | None, w2: dict | None):
        a1, a2 = DQNAgent(), DQNAgent()
        if w1:
            a1.set_weights(w1)
        if w2:
            a2.set_weights(w2)
        self.world.reset()
        frames = []
        done, winner = False, None
        while not done:
            o1 = self.world.get_observation(0)
            o2 = self.world.get_observation(1)
            act1 = a1.get_action_greedy(o1)
            act2 = a2.get_action_greedy(o2)
            done, winner = self.world.step(act1, act2)
            fd = self.world.get_frame_data()
            fd['actions'] = [act1, act2]
            frames.append(fd)
        return {'winner': winner, 'steps': self.world.step_count, 'frames': frames}
