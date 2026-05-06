"""
Training manager: runs DQN training with experience replay.
Reward shaping designed for the charge-only dash mechanic.

Optimized for speed and effectiveness:
- Dueling DQN with prioritized replay
- n-step returns (n=5) for fast temporal credit
- Train every 4 steps for more gradient updates
- Tight episode limit (150 steps) to force decisive play
- Self-play league: late training mixes in past snapshots + current
  self so the agent doesn't overfit to the scripted heuristic.
"""
import math
import random
import numpy as np
from physics import (PhysicsWorld, ARENA_RADIUS, MAX_SPEED, MAX_STEPS,
                     DASH_MAX_SPEED, DASH_COOLDOWN_STEPS)
from agent import DQNAgent, DefaultAgent

TRAIN_EVERY = 8       # train every 8 env steps (good speed/learning balance)
TRAIN_MAX_STEPS = 150  # tight limit forces decisive play
SNAPSHOT_EVERY = 200   # episodes between weight snapshots for past-self pool
SNAPSHOT_KEEP = 5      # FIFO depth of past-self snapshots
_ARENA_DIAM = 2 * ARENA_RADIUS


class _GreedyDQNOpponent:
    """Wraps a frozen DQNAgent so it satisfies the opponent interface
    (get_action(obs) → int). Used for past-self snapshots; the wrapped
    agent never trains and always plays greedily."""
    __slots__ = ('agent',)

    def __init__(self, weights: dict):
        self.agent = DQNAgent()
        self.agent.set_weights(weights)

    def get_action(self, state):
        return self.agent.get_action_greedy(state)


class _CurrentSelfOpponent:
    """Live mirror of the training agent; opponent's actions track the
    current network so self-play exploits new policy improvements
    immediately. Plays greedily — no extra exploration noise from the
    opponent side."""
    __slots__ = ('agent',)

    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def get_action(self, state):
        return self.agent.get_action_greedy(state)


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
        my_knock = world.bot1_knock if bot_idx == 0 else world.bot2_knock
        r = 0.0

        # ── Terminal signals (DOMINANT) ──
        if done:
            if winner == bot_idx:
                r += w.get('win_bonus', 1.0) * 15.0
            elif winner == -1:
                r -= 6.0              # strong draw penalty
            elif world.self_ko == bot_idx:
                # Self-KO: chose to dash off (or dashed during knockback
                # bypass window). Worse than a normal loss so the agent
                # doesn't substitute "dash off" for "outplayed loss" when
                # cornered.
                r -= 20.0
            else:
                r -= 12.0             # loss penalty

        # ── Spatial calculations ──
        dx = opp.x - bot.x
        dy = opp.y - bot.y
        dist_opp = math.sqrt(dx * dx + dy * dy)
        dc = math.sqrt(bot.x * bot.x + bot.y * bot.y)
        edge = dc / ARENA_RADIUS
        dist_ratio = dist_opp / _ARENA_DIAM
        time_frac = world.step_count / TRAIN_MAX_STEPS

        # ── Time pressure (flat) ──
        # Constraint: total per-episode time penalty must stay below
        # (loss_penalty - draw_penalty) = 6.0, otherwise timeout becomes
        # *worse* than a fast loss and the agent prefers running off the
        # edge to stalling. At 0.02/step over 150 training steps that's 3.0.
        # Escalation removed: the agent has no step-count input, so the
        # ramp couldn't translate into actionable behavior anyway.
        if not done:
            r -= 0.02

        # ── CHARGE reward: incentivize charging near opponent, but scale
        # down when opponent is highly charged (they could dash you mid-charge).
        if bot.charge_level > 0:
            closeness = max(0, 1.0 - dist_ratio / 0.45)
            # pressure_factor: 1.0 when opp is uncharged, 0.5 when fully charged
            pressure_factor = 1.0 - 0.5 * opp.charge_level
            r += (w.get('charge_reward', 1.0) * 0.15 * bot.charge_level
                  * (0.3 + 0.7 * closeness) * pressure_factor)

        # ── DASH toward opponent: reward dashing in the right direction ──
        if bot.is_dashing and dist_opp > 1e-6:
            approach = (bot.vx * dx + bot.vy * dy) / dist_opp
            speed = math.sqrt(bot.vx * bot.vx + bot.vy * bot.vy)
            if approach > 0 and speed > 0:
                aim_quality = approach / speed  # 1.0 = perfect aim
                r += 0.3 * aim_quality * bot.last_dash_charge

        # ── DASH HIT: the big payoff ──
        if collision:
            if bot.is_dashing and bot.last_dash_charge > 0:
                charge_q = bot.last_dash_charge
                impulse = min(my_knock / DASH_MAX_SPEED, 2.0)
                r += w.get('hit_reward', 1.0) * (2.5 + 3.0 * charge_q) * (0.5 + 0.5 * impulse)
            elif opp.is_dashing and opp.last_dash_charge > 0:
                # Hit-received penalty scales with the attacker's charge AND
                # the actual knockback impulse — a glancing hit from a
                # half-charged dash hurts less than a clean full-charge hit.
                charge_q = opp.last_dash_charge
                impulse = min(my_knock / DASH_MAX_SPEED, 2.0)
                r -= (w.get('hit_reward', 1.0) * (0.5 + 1.0 * charge_q)
                      * (0.5 + 0.5 * impulse))

        # ── Engagement bonus: small per-step reward for being in fight range
        # while at least one bot is meaningfully charged. Encourages staying
        # close enough to threaten/dodge instead of running away.
        if not done and dist_opp < 100.0 and (bot.charge_level > 0.2 or opp.charge_level > 0.2):
            r += 0.03

        # ── Dodge bonus: opponent's dash flight just ended without a hit.
        # Only counts when the agent was in plausible target range (otherwise
        # the dash wasn't really "at" them).
        opp_missed = (world.bot2_missed_dash if bot_idx == 0 else world.bot1_missed_dash)
        opp_missed_charge = (world.bot2_missed_charge if bot_idx == 0
                             else world.bot1_missed_charge)
        if opp_missed and opp_missed_charge > 0 and dist_opp < 120.0:
            r += 0.6 * opp_missed_charge

        # ── Opponent near edge ──
        opp_dc = math.sqrt(opp.x * opp.x + opp.y * opp.y)
        opp_edge = opp_dc / ARENA_RADIUS
        if opp_edge > 0.5:
            excess = opp_edge - 0.5
            r += w.get('opp_edge', 1.0) * 0.5 * excess * excess

        # ── Self near edge: penalty ──
        if edge > 0.6:
            r -= w.get('edge_penalty', 1.0) * 0.5 * (edge - 0.6) * (edge - 0.6)
        if edge > 0.85:
            r -= 1.5 * (edge - 0.85) / 0.15

        # ── Center control ──
        r += 0.01 * (1.0 - edge)

        return r

    # ── opponent sampling (curriculum + self-play league) ───
    def _pick_opponent(self, progress: float, snapshots: list[dict]):
        """Sample an opponent based on the configured curriculum preset.
        Mix tuples are (easy_heuristic, hard_heuristic, past_self_dqn,
        current_self_dqn) probabilities.

        Presets:
          easy_only    — always weak DefaultAgent (strength=0.3). Bot
                         crushes the dummy but fails on real opponents.
          easy_to_hard — gentle ramp: starts mostly easy, ends mixed
                         with self-play. Often the strongest setting.
          default      — original 3-phase schedule (preserved verbatim).
          hard_only    — always max-strength heuristic. Often plateaus
                         because the bot can't bootstrap basic skills.
          self_play    — past + current snapshots only (folded to
                         current-self if no snapshots yet).

        If no snapshots are available, the past-self mass is folded
        into the current-self bucket (any self-play is better than none).
        """
        curriculum = getattr(self, 'curriculum', 'default')

        if curriculum == 'easy_only':
            return DefaultAgent(strength=0.3)
        if curriculum == 'hard_only':
            return DefaultAgent(strength=1.0)

        if curriculum == 'self_play':
            mix = (0.0, 0.0, 0.4, 0.6)
        elif curriculum == 'easy_to_hard':
            # Linear interp (0.8, 0.2, 0, 0) → (0.1, 0.4, 0.3, 0.2)
            a, b = (0.8, 0.2, 0.0, 0.0), (0.1, 0.4, 0.3, 0.2)
            mix = tuple(a[i] + (b[i] - a[i]) * progress for i in range(4))
        else:  # 'default' — preserve original 3-phase logic
            if progress < 0.3:
                mix = (0.55, 0.30, 0.05, 0.10)
            elif progress < 0.7:
                mix = (0.20, 0.30, 0.30, 0.20)
            else:
                mix = (0.05, 0.20, 0.45, 0.30)

        if not snapshots:
            mix = (mix[0], mix[1], 0.0, mix[2] + mix[3])

        kind = random.choices(['easy', 'hard', 'past', 'current'], weights=mix, k=1)[0]
        if kind == 'easy':
            # Strength ramps from sloppy to mid as training progresses.
            return DefaultAgent(strength=min(1.0, 0.2 + 0.4 * progress))
        if kind == 'hard':
            return DefaultAgent(strength=1.0)
        if kind == 'past':
            return _GreedyDQNOpponent(random.choice(snapshots))
        # current self
        return _CurrentSelfOpponent(self.agent)

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
                winner = -1  # Timeout = draw

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

        # Lab Mode hyperparameters — defaults match the prior hardcoded values
        # so behavior is unchanged when these keys are absent.
        gamma = config.get('gamma', 0.95)
        epsilon_decay = config.get('epsilon_decay', 0.994)
        n_steps = config.get('n_steps', 6)
        self.curriculum = config.get('curriculum', 'default')

        self.agent = DQNAgent(
            lr=lr,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=epsilon_decay,
            batch_size=128,
            target_update_freq=0,
            tau=0.01,
            n_steps=n_steps,
        )
        self.is_training = True
        self.training_stats.clear()

        snapshots: list[dict] = []  # FIFO of past-self weights for self-play

        wins, recent_wins, recent_losses = 0, [], []
        for ep in range(num_episodes):
            if not self.is_training:
                break

            progress = ep / max(num_episodes - 1, 1)

            # Snapshot past-self every SNAPSHOT_EVERY episodes (skip ep 0 —
            # weights would be raw init, useless as an opponent).
            if ep > 0 and ep % SNAPSHOT_EVERY == 0:
                snapshots.append(self.agent.get_weights())
                if len(snapshots) > SNAPSHOT_KEEP:
                    snapshots.pop(0)

            opponent = self._pick_opponent(progress, snapshots)

            result = self.run_episode(self.agent, opponent, rw, train=True)

            won = result['winner'] == 0
            lost = result['winner'] == 1
            if won:
                wins += 1
            recent_wins.append(int(won))
            recent_losses.append(int(lost))
            if len(recent_wins) > 100:
                recent_wins.pop(0)
            if len(recent_losses) > 100:
                recent_losses.pop(0)

            stats = {
                'episode': ep + 1,
                'total_episodes': num_episodes,
                'win_rate': round(sum(recent_wins) / len(recent_wins), 4),
                'lose_rate': round(sum(recent_losses) / len(recent_losses), 4),
                'total_wins': wins,
                'avg_reward': round(result['total_reward'], 4),
                'steps': result['steps'],
                'won': won,
                'epsilon': round(self.agent.epsilon, 4),
            }
            self.training_stats.append(stats)
            if progress_callback and (ep % 10 == 0 or ep == num_episodes - 1):
                progress_callback(stats)

        self.is_training = False
        return self.agent.get_weights() if self.agent else None

    # ── gauntlet (held-out evaluation against 3 difficulties) ─
    GAUNTLET = (('Easy', 0.3), ('Medium', 0.6), ('Hard', 1.0))

    def gauntlet_match(self, reward_weights: dict | None = None):
        """Run the trained agent against three fixed-strength heuristics
        in sequence. Wins/3 is a held-out generalization score: an agent
        trained on `easy_only` will ace Easy but drop Hard, while one
        trained on `default` or `easy_to_hard` should sweep."""
        if self.agent is None:
            return None
        rw = reward_weights or {}
        results = []
        wins = 0
        for label, strength in self.GAUNTLET:
            ep = self.run_episode(self.agent,
                                  DefaultAgent(strength=strength),
                                  rw, train=False, record_frames=True)
            won = ep['winner'] == 0
            if won:
                wins += 1
            results.append({
                'opponent': label,
                'strength': strength,
                'winner': ep['winner'],
                'steps': ep['steps'],
                'frames': ep['frames'],
                'won': won,
            })
        return {'results': results, 'wins': wins,
                'total': len(self.GAUNTLET)}

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
