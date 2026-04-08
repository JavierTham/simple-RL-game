"""
Physics engine for the Bumper Bot arena.
Handles arena bounds, bot movement, collisions, and win conditions.
Uses 9 discrete actions (stay + 8 compass directions).

Performance: uses math module for scalar ops (10x faster than numpy scalars).
"""
import math
import random
import numpy as np

ARENA_RADIUS = 200
BOT_RADIUS = 15
MAX_SPEED = 5.0
FORCE_MAGNITUDE = 1.0
FRICTION = 0.92
MAX_STEPS = 500
_AR2 = (ARENA_RADIUS - BOT_RADIUS) ** 2   # squared out-of-bounds threshold

_d = 0.7071
ACTION_FORCES = [
    (0.0, 0.0),    # 0: stay
    (0.0, -1.0),   # 1: up
    (_d, -_d),     # 2: up-right
    (1.0, 0.0),    # 3: right
    (_d, _d),      # 4: down-right
    (0.0, 1.0),    # 5: down
    (-_d, _d),     # 6: down-left
    (-1.0, 0.0),   # 7: left
    (-_d, -_d),    # 8: up-left
]


class Bot:
    __slots__ = ('x', 'y', 'vx', 'vy', 'radius')

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.radius = BOT_RADIUS

    def apply_action(self, action: int):
        fx, fy = ACTION_FORCES[action]
        self.vx += fx * FORCE_MAGNITUDE
        self.vy += fy * FORCE_MAGNITUDE

    def update(self):
        self.vx *= FRICTION
        self.vy *= FRICTION
        sp2 = self.vx * self.vx + self.vy * self.vy
        if sp2 > MAX_SPEED * MAX_SPEED:
            scale = MAX_SPEED / math.sqrt(sp2)
            self.vx *= scale
            self.vy *= scale
        self.x += self.vx
        self.y += self.vy

    def is_out(self) -> bool:
        return self.x * self.x + self.y * self.y > _AR2

    def dist_from_center_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def dist_from_center(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)


class PhysicsWorld:
    def __init__(self):
        self.bot1 = Bot(0, 0)
        self.bot2 = Bot(0, 0)
        self.step_count = 0
        self.collision_occurred = False
        self.reset()

    def reset(self):
        angle = random.random() * 6.283185307
        spawn_dist = 70
        c, s = math.cos(angle), math.sin(angle)
        self.bot1 = Bot(-c * spawn_dist, -s * spawn_dist)
        self.bot2 = Bot(c * spawn_dist, s * spawn_dist)
        self.step_count = 0
        self.collision_occurred = False

    def get_observation(self, bot_idx: int) -> np.ndarray:
        if bot_idx == 0:
            bot, opp = self.bot1, self.bot2
        else:
            bot, opp = self.bot2, self.bot1
        ar = ARENA_RADIUS
        ms = MAX_SPEED
        return np.array([
            bot.x / ar, bot.y / ar,
            bot.vx / ms, bot.vy / ms,
            (opp.x - bot.x) / ar, (opp.y - bot.y) / ar,
            (opp.vx - bot.vx) / ms, (opp.vy - bot.vy) / ms,
        ], dtype=np.float64)

    def step(self, action1: int, action2: int):
        # ── Gravitational attraction ────────────────────
        # Constant gentle pull toward opponent (0.15 vs 1.0 from actions)
        # Ensures bots always converge; "stay" = drift toward opponent
        gx = self.bot2.x - self.bot1.x
        gy = self.bot2.y - self.bot1.y
        gd2 = gx * gx + gy * gy
        if gd2 > 1.0:
            gd = math.sqrt(gd2)
            gf = 0.15  # gravity strength
            gnx, gny = gx / gd, gy / gd
            self.bot1.vx += gnx * gf
            self.bot1.vy += gny * gf
            self.bot2.vx -= gnx * gf
            self.bot2.vy -= gny * gf

        self.bot1.apply_action(action1)
        self.bot2.apply_action(action2)
        self.bot1.update()
        self.bot2.update()
        self.collision_occurred = self._handle_collision()
        self.step_count += 1
        bot1_out = self.bot1.is_out()
        bot2_out = self.bot2.is_out()
        done = bot1_out or bot2_out or self.step_count >= MAX_STEPS
        winner = None
        if done:
            if bot1_out and not bot2_out:
                winner = 1
            elif bot2_out and not bot1_out:
                winner = 0
            elif bot1_out and bot2_out:
                winner = -1
            else:
                d1 = self.bot1.dist_from_center_sq()
                d2 = self.bot2.dist_from_center_sq()
                winner = 0 if d1 < d2 else (1 if d2 < d1 else -1)
        return done, winner

    def _handle_collision(self) -> bool:
        dx = self.bot2.x - self.bot1.x
        dy = self.bot2.y - self.bot1.y
        dist_sq = dx * dx + dy * dy
        min_dist = self.bot1.radius + self.bot2.radius
        if dist_sq < min_dist * min_dist and dist_sq > 1e-12:
            dist = math.sqrt(dist_sq)
            nx, ny = dx / dist, dy / dist  # normal: bot1 → bot2

            dvx = self.bot1.vx - self.bot2.vx
            dvy = self.bot1.vy - self.bot2.vy
            dvn = dvx * nx + dvy * ny      # relative closing speed

            if dvn > 0:
                # ── Charging momentum bonus ─────────────────
                # Base boost from closing speed (unchanged from before)
                base_boost = 1.0 + 0.5 * min(dvn / MAX_SPEED, 1.0)

                # Charge bonus: attacker's own speed toward opponent (quadratic)
                # Slow approach = modest bonus, full-speed charge = devastating
                v1_approach = max(0.0, self.bot1.vx * nx + self.bot1.vy * ny)
                v2_approach = max(0.0, -(self.bot2.vx * nx + self.bot2.vy * ny))
                attacker_speed = max(v1_approach, v2_approach)
                charge_ratio = min(attacker_speed / MAX_SPEED, 1.0)
                charge_bonus = 0.8 * charge_ratio * charge_ratio  # quadratic ramp

                boost = base_boost + charge_bonus
                impulse_x = dvn * nx * boost
                impulse_y = dvn * ny * boost

                # ── Asymmetric knockback ────────────────────
                # Reuses v1_approach / v2_approach from charge calculation
                total_approach = v1_approach + v2_approach

                if total_approach > 0.1:
                    r = v1_approach / total_approach  # 0→1, higher = bot1 is attacker
                else:
                    r = 0.5

                # Attacker barely bounces (0.3×), defender gets launched (1.7×)
                KEEP = 0.3   # attacker's knockback multiplier
                AMP  = 1.7   # defender's knockback multiplier
                bot1_kb = KEEP * r + AMP * (1.0 - r)
                bot2_kb = AMP  * r + KEEP * (1.0 - r)

                self.bot1.vx -= impulse_x * bot1_kb
                self.bot1.vy -= impulse_y * bot1_kb
                self.bot2.vx += impulse_x * bot2_kb
                self.bot2.vy += impulse_y * bot2_kb

            # Separate overlapping bots
            overlap = min_dist - dist
            half = overlap * 0.5
            self.bot1.x -= half * nx
            self.bot1.y -= half * ny
            self.bot2.x += half * nx
            self.bot2.y += half * ny
            return True
        return False

    def get_frame_data(self) -> dict:
        return {
            'bots': [
                {'x': self.bot1.x, 'y': self.bot1.y,
                 'vx': self.bot1.vx, 'vy': self.bot1.vy,
                 'radius': self.bot1.radius},
                {'x': self.bot2.x, 'y': self.bot2.y,
                 'vx': self.bot2.vx, 'vy': self.bot2.vy,
                 'radius': self.bot2.radius},
            ],
            'arena_radius': ARENA_RADIUS,
            'step': self.step_count,
            'collision': self.collision_occurred,
        }
