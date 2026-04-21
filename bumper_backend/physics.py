"""
Physics engine for the Bumper Bot arena (pymunk backend).
Handles arena bounds, bot movement, collisions, and win conditions.

Uses 9 discrete actions:
  0 = CHARGE (stand still, accumulate power)
  1-8 = MOVE in 8 compass directions (if charged → releases a high-speed dash)

Charge mechanic:
  - Holding action 0 accumulates charge_level (0 → 1.0 over ~40 steps)
  - Taking a move action (1-8) while charged releases a dash:
      dash_speed = DASH_MIN_SPEED + charge_level * (DASH_MAX_SPEED - DASH_MIN_SPEED)
  - After dashing, a cooldown prevents immediate re-charging
  - Charging while near opponent is risky (sitting duck) but enables devastating hits

Uses pymunk for realistic physics: proper momentum transfer, elastic
collisions, and separation — no hand-rolled collision math.
"""
import math
import random
import numpy as np
import pymunk

ARENA_RADIUS = 180
BOT_RADIUS = 15
MAX_SPEED = 6.0
DASH_MIN_SPEED = 10.0       # dash at minimum viable charge (~10%)
DASH_MAX_SPEED = 22.0       # dash at 100% charge
KNOCKBACK_MAX_SPEED = 28.0  # post-collision speed cap (decays via friction)
FORCE_MAGNITUDE = 1.2       # impulse per move action
FRICTION = 0.90             # per-step velocity damping
MAX_STEPS = 400
CHARGE_RATE = 0.025          # charge per step when holding action 0 (~40 steps to full)
MIN_CHARGE_TO_DASH = 0.10   # minimum charge to trigger a dash
DASH_COOLDOWN_STEPS = 15    # cooldown after dashing before can charge again
DASH_FLIGHT_FRAMES = 8      # frames of friction immunity during dash flight
_AR2 = (ARENA_RADIUS - BOT_RADIUS) ** 2   # squared out-of-bounds threshold

_d = 0.7071
ACTION_FORCES = [
    (0.0, 0.0),    # 0: charge (no movement)
    (0.0, -1.0),   # 1: up
    (_d, -_d),     # 2: up-right
    (1.0, 0.0),    # 3: right
    (_d, _d),      # 4: down-right
    (0.0, 1.0),    # 5: down
    (-_d, _d),     # 6: down-left
    (-1.0, 0.0),   # 7: left
    (-_d, -_d),    # 8: up-left
]

BOT_COLLISION_TYPE = 1
BOT_MASS = 1.0
BOT_ELASTICITY = 1.5    # slightly super-elastic for dramatic bumper hits
BOT_FRICTION = 0.0      # no surface friction between bots


def _clamp_velocity(body, gravity, damping, dt):
    """Custom velocity function: applies damping, then caps speed.
    Dashing bots skip damping during flight frames to preserve dash velocity.
    Uses KNOCKBACK_MAX_SPEED as the hard ceiling."""
    # Check if this body's bot is in dash flight (skip damping if so)
    bot = getattr(body, '_bot_ref', None)
    if bot and bot.dash_frames_left > 0:
        # No damping during dash flight — preserve momentum
        pymunk.Body.update_velocity(body, gravity, 1.0, dt)  # damping=1.0 = no damping
    else:
        pymunk.Body.update_velocity(body, gravity, damping, dt)
    vx, vy = body.velocity
    sp2 = vx * vx + vy * vy
    if sp2 > KNOCKBACK_MAX_SPEED * KNOCKBACK_MAX_SPEED:
        scale = KNOCKBACK_MAX_SPEED / math.sqrt(sp2)
        body.velocity = (vx * scale, vy * scale)


class Bot:
    """Pymunk-backed bot with charge/dash mechanic.
    Exposes x, y, vx, vy properties for interface compatibility."""
    __slots__ = ('body', 'shape', 'radius',
                 'charge_level', 'is_dashing', 'dash_cooldown',
                 'last_dash_charge', 'dash_frames_left')

    def __init__(self, space: pymunk.Space, x: float, y: float):
        self.radius = BOT_RADIUS
        self.charge_level = 0.0
        self.is_dashing = False
        self.dash_cooldown = 0
        self.last_dash_charge = 0.0  # charge level of most recent dash (for rewards)
        self.dash_frames_left = 0   # frames of friction immunity remaining

        # Infinite moment prevents rotation (irrelevant for circles)
        self.body = pymunk.Body(BOT_MASS, float('inf'))
        self.body.position = (x, y)
        self.body.velocity = (0.0, 0.0)
        self.body.velocity_func = _clamp_velocity
        self.body._bot_ref = self  # back-reference for velocity_func
        self.shape = pymunk.Circle(self.body, BOT_RADIUS)
        self.shape.collision_type = BOT_COLLISION_TYPE
        self.shape.elasticity = BOT_ELASTICITY
        self.shape.friction = BOT_FRICTION
        space.add(self.body, self.shape)

    # ── Position/velocity properties for interface compatibility ──
    @property
    def x(self) -> float:
        return self.body.position.x

    @property
    def y(self) -> float:
        return self.body.position.y

    @property
    def vx(self) -> float:
        return self.body.velocity.x

    @property
    def vy(self) -> float:
        return self.body.velocity.y

    def apply_action(self, action: int):
        self.is_dashing = False
        self.last_dash_charge = 0.0

        # Tick dash flight frames
        if self.dash_frames_left > 0:
            self.dash_frames_left -= 1

        # Tick cooldown
        if self.dash_cooldown > 0:
            self.dash_cooldown -= 1

        if action == 0:
            # CHARGE: accumulate power while standing still
            if self.dash_cooldown <= 0:
                self.charge_level = min(1.0, self.charge_level + CHARGE_RATE)
            # Apply braking when charging (slow to a stop)
            vx, vy = self.body.velocity
            self.body.velocity = (vx * 0.85, vy * 0.85)
            return

        # MOVE action (1-8)
        fx, fy = ACTION_FORCES[action]

        if self.charge_level >= MIN_CHARGE_TO_DASH and self.dash_cooldown <= 0:
            # DASH: release stored charge as high-speed burst
            charge = self.charge_level
            dash_speed = DASH_MIN_SPEED + charge * (DASH_MAX_SPEED - DASH_MIN_SPEED)
            self.body.velocity = (fx * dash_speed, fy * dash_speed)
            self.is_dashing = True
            self.last_dash_charge = charge
            self.charge_level = 0.0
            self.dash_cooldown = DASH_COOLDOWN_STEPS
            self.dash_frames_left = DASH_FLIGHT_FRAMES
            return

        # Normal movement (no charge or on cooldown)
        self.charge_level = 0.0  # moving cancels any partial charge
        vx0, vy0 = self.body.velocity
        speed_before = math.sqrt(vx0 * vx0 + vy0 * vy0)
        # Apply the action impulse
        self.body.apply_impulse_at_local_point(
            (fx * FORCE_MAGNITUDE, fy * FORCE_MAGNITUDE)
        )
        # Only clamp if the action INCREASED speed beyond MAX_SPEED.
        # This preserves knockback velocity (which decays via friction)
        # while preventing infinite acceleration from holding a direction.
        vx, vy = self.body.velocity
        sp2 = vx * vx + vy * vy
        max_allowed = max(speed_before, MAX_SPEED)
        if sp2 > max_allowed * max_allowed:
            scale = max_allowed / math.sqrt(sp2)
            self.body.velocity = (vx * scale, vy * scale)

    def is_out(self) -> bool:
        x, y = self.body.position
        return x * x + y * y > _AR2

    def dist_from_center_sq(self) -> float:
        x, y = self.body.position
        return x * x + y * y

    def dist_from_center(self) -> float:
        x, y = self.body.position
        return math.sqrt(x * x + y * y)

    def remove_from(self, space: pymunk.Space):
        """Remove this bot's body and shape from the pymunk space."""
        space.remove(self.body, self.shape)


class PhysicsWorld:
    def __init__(self):
        self.bot1: Bot | None = None
        self.bot2: Bot | None = None
        self.step_count = 0
        self.collision_occurred = False
        self.bot1_knock = 0.0   # velocity-change magnitude from collision
        self.bot2_knock = 0.0
        self._collision_this_step = False
        self._collision_impulse_mag = 0.0
        # Track dash-collision (at least one bot was dashing during collision)
        self.dash_collision = False
        self._init_space()
        self.reset()

    def _init_space(self):
        """Create pymunk space — pymunk handles collision response natively."""
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = FRICTION
        self.space.iterations = 3   # bumped slightly for high-speed collisions
        # Track collisions via post_solve to get impulse magnitude for rewards
        self.space.on_collision(
            BOT_COLLISION_TYPE, BOT_COLLISION_TYPE,
            post_solve=self._on_collision,
        )

    def _on_collision(self, arbiter, space, data):
        """Record collision impulse for reward calculation."""
        self._collision_this_step = True
        impulse = arbiter.total_impulse
        self._collision_impulse_mag = math.sqrt(
            impulse.x * impulse.x + impulse.y * impulse.y
        ) / BOT_MASS

    def reset(self):
        # Remove old bots from space
        if self.bot1 is not None:
            self.bot1.remove_from(self.space)
        if self.bot2 is not None:
            self.bot2.remove_from(self.space)

        angle = random.random() * 6.283185307
        spawn_dist = 60
        c, s = math.cos(angle), math.sin(angle)
        self.bot1 = Bot(self.space, -c * spawn_dist, -s * spawn_dist)
        self.bot2 = Bot(self.space, c * spawn_dist, s * spawn_dist)
        self.step_count = 0
        self.collision_occurred = False
        self.dash_collision = False
        self.bot1_knock = 0.0
        self.bot2_knock = 0.0

    def get_observation(self, bot_idx: int) -> np.ndarray:
        if bot_idx == 0:
            bot, opp = self.bot1, self.bot2
        else:
            bot, opp = self.bot2, self.bot1
        ar = ARENA_RADIUS
        ms = MAX_SPEED
        bx, by = bot.body.position
        bvx, bvy = bot.body.velocity
        ox, oy = opp.body.position
        ovx, ovy = opp.body.velocity
        dx = ox - bx
        dy = oy - by
        # Distance of each bot from center (0=center, 1=edge)
        my_edge = math.sqrt(bx * bx + by * by) / ar
        opp_edge = math.sqrt(ox * ox + oy * oy) / ar
        # Angle between bot's velocity and direction to opponent (-1 to 1)
        speed_sq = bvx * bvx + bvy * bvy
        dist_opp = math.sqrt(dx * dx + dy * dy)
        if speed_sq > 1e-6 and dist_opp > 1e-6:
            speed = math.sqrt(speed_sq)
            vel_dot_dir = (bvx * dx + bvy * dy) / (speed * dist_opp)
        else:
            vel_dot_dir = 0.0
        return np.array([
            bx / ar, by / ar,
            bvx / ms, bvy / ms,
            dx / ar, dy / ar,
            (ovx - bvx) / ms, (ovy - bvy) / ms,
            my_edge,              # own distance from center (normalized)
            opp_edge,             # opponent distance from center (normalized)
            vel_dot_dir,          # alignment: +1=charging, 0=flanking, -1=fleeing
            dist_opp / (2 * ar),  # inter-bot distance (normalized)
            # ── New charge/dash features ──
            bot.charge_level,                          # own charge (0-1)
            opp.charge_level,                          # opponent charge (0-1)
            bot.dash_cooldown / DASH_COOLDOWN_STEPS,   # own cooldown (0-1)
        ], dtype=np.float64)

    def step(self, action1: int, action2: int):
        # Apply actions as impulses (instant velocity change)
        self.bot1.apply_action(action1)
        self.bot2.apply_action(action2)

        # Reset collision tracking
        self._collision_this_step = False
        self._collision_impulse_mag = 0.0

        # Pymunk step: velocity_func (damping + clamp) → collision → position
        self.space.step(1.0)

        # Record collision data for reward system
        if self._collision_this_step:
            self.collision_occurred = True
            self.dash_collision = self.bot1.is_dashing or self.bot2.is_dashing
            # With equal masses, both bots receive equal impulse (Newton's 3rd law)
            self.bot1_knock = self._collision_impulse_mag
            self.bot2_knock = self._collision_impulse_mag
        else:
            self.collision_occurred = False
            self.dash_collision = False
            self.bot1_knock = 0.0
            self.bot2_knock = 0.0

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
                # Timeout = draw — only knockouts count as wins
                winner = -1
        return done, winner

    def get_frame_data(self) -> dict:
        return {
            'bots': [
                {'x': self.bot1.x, 'y': self.bot1.y,
                 'vx': self.bot1.vx, 'vy': self.bot1.vy,
                 'radius': self.bot1.radius,
                 'charge': self.bot1.charge_level,
                 'dashing': self.bot1.is_dashing,
                 'cooldown': self.bot1.dash_cooldown},
                {'x': self.bot2.x, 'y': self.bot2.y,
                 'vx': self.bot2.vx, 'vy': self.bot2.vy,
                 'radius': self.bot2.radius,
                 'charge': self.bot2.charge_level,
                 'dashing': self.bot2.is_dashing,
                 'cooldown': self.bot2.dash_cooldown},
            ],
            'arena_radius': ARENA_RADIUS,
            'step': self.step_count,
            'collision': self.collision_occurred,
            'dash_collision': self.dash_collision,
        }
