"""
╔═══════════════════════════════════════════════════════════════╗
║                    ARCANE ARENA — Backend                     ║
║   A Reinforcement Learning Sumo Game for Education            ║
║   Built with FastAPI + Tabular Q-Learning                     ║
╚═══════════════════════════════════════════════════════════════╝

Students train "Magical Golems" via Q-learning to compete in a
sumo-style ring on a 7×7 grid with a circular boundary (r ≤ 3.2).
"""

import math
import json
import random
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ─────────────────────────────────────────────────────────────
# 1. ENVIRONMENT  —  SumoEnv
# ─────────────────────────────────────────────────────────────

# The grid is 7×7 (coordinates 0-6). The centre is (3, 3).
GRID_SIZE = 7
CENTER = GRID_SIZE // 2          # 3
RING_RADIUS = 3.2                # tiles from centre

# Actions: Up, Down, Left, Right
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
NUM_ACTIONS = len(ACTIONS)


def _is_inside_ring(x: int, y: int) -> bool:
    """Check whether (x, y) lies inside the circular arena.

    The ring is centred on (CENTER, CENTER) with RING_RADIUS.
    We use Euclidean distance so the boundary is a true circle,
    not a diamond or a square.
    """
    dx = x - CENTER
    dy = y - CENTER
    return math.sqrt(dx * dx + dy * dy) <= RING_RADIUS


class SumoEnv:
    """Simulates a sumo-wrestling match between two golems.

    State representation (for tabular Q-learning):
        (agent_x, agent_y, opponent_x, opponent_y)

    The state space is bounded:  7^4 = 2 401 possible states
    (many unreachable), which is small enough for a Q-table.

    Movement rules:
        • Both golems move on integer grid cells.
        • If the agent moves INTO the opponent's cell, the
          opponent is "pushed" 1 tile in that same direction.
        • If a golem exits the ring it is eliminated.
    """

    def __init__(
        self,
        push_reward: float = 10.0,
        fall_penalty: float = -50.0,
        center_reward: float = 1.0,
        step_penalty: float = -0.1,
    ):
        # ── Reward weights (tunable from the frontend) ──
        self.push_reward = push_reward
        self.fall_penalty = fall_penalty
        self.center_reward = center_reward
        self.step_penalty = step_penalty

        self.reset()

    # ── Reset ────────────────────────────────────────────
    def reset(self, agent_pos=None, opp_pos=None):
        """Place both golems. Defaults to symmetric positions near centre."""
        self.agent_x, self.agent_y = agent_pos or (CENTER - 1, CENTER)
        self.opp_x, self.opp_y = opp_pos or (CENTER + 1, CENTER)
        self.done = False
        self.winner = None  # "agent" | "opponent" | None
        return self._state()

    def _state(self):
        return (self.agent_x, self.agent_y, self.opp_x, self.opp_y)

    # ── Step ─────────────────────────────────────────────
    def step(self, action_idx: int):
        """Execute one action for the agent; the opponent stands still
        (or uses its own Q-table if provided externally).

        Returns: (next_state, reward, done, info)
        """
        if self.done:
            return self._state(), 0.0, True, {"event": "already_done"}

        dx, dy = ACTIONS[action_idx]
        new_ax = self.agent_x + dx
        new_ay = self.agent_y + dy

        reward = self.step_penalty       # small cost per timestep
        info = {"event": "move"}

        # ── Pushing logic ────────────────────────────────
        # If agent moves into the opponent's tile, push opponent
        if new_ax == self.opp_x and new_ay == self.opp_y:
            push_ox = self.opp_x + dx
            push_oy = self.opp_y + dy
            self.opp_x = push_ox
            self.opp_y = push_oy
            reward += self.push_reward
            info["event"] = "push"

            # Did we push the opponent out of the ring?
            if not _is_inside_ring(self.opp_x, self.opp_y):
                self.done = True
                self.winner = "agent"
                reward += 100.0          # massive win bonus
                info["event"] = "win"

        # Update agent position
        self.agent_x = new_ax
        self.agent_y = new_ay

        # ── Self-elimination check ───────────────────────
        if not _is_inside_ring(self.agent_x, self.agent_y):
            self.done = True
            self.winner = "opponent"
            reward += self.fall_penalty  # massive penalty for falling
            info["event"] = "fall"

        # ── Centre-seeking shaped reward ─────────────────
        # Encourage the agent to stay near the centre of the ring.
        if not self.done:
            dist_to_center = math.sqrt(
                (self.agent_x - CENTER) ** 2 + (self.agent_y - CENTER) ** 2
            )
            # Reward decreases linearly with distance from centre
            reward += self.center_reward * max(0, 1.0 - dist_to_center / RING_RADIUS)

        return self._state(), reward, self.done, info

    # ── Step for a second agent (PvP) ────────────────────
    def step_opponent(self, action_idx: int):
        """Move the opponent using the same physics, mirrored."""
        if self.done:
            return self._state(), 0.0, True, {"event": "already_done"}

        dx, dy = ACTIONS[action_idx]
        new_ox = self.opp_x + dx
        new_oy = self.opp_y + dy
        info = {"event": "move"}

        # Push agent if opponent lands on agent
        if new_ox == self.agent_x and new_oy == self.agent_y:
            self.agent_x += dx
            self.agent_y += dy
            info["event"] = "push"

            if not _is_inside_ring(self.agent_x, self.agent_y):
                self.done = True
                self.winner = "opponent"
                info["event"] = "agent_pushed_out"

        self.opp_x = new_ox
        self.opp_y = new_oy

        if not _is_inside_ring(self.opp_x, self.opp_y):
            self.done = True
            self.winner = "agent"
            info["event"] = "opponent_fell"

        return self._state(), 0.0, self.done, info


# ─────────────────────────────────────────────────────────────
# 2. Q-LEARNING  —  The Bellman Update
# ─────────────────────────────────────────────────────────────

def train_agent(
    push_reward: float = 10.0,
    fall_penalty: float = -50.0,
    center_reward: float = 1.0,
    step_penalty: float = -0.1,
    episodes: int = 3000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.997,
    max_steps: int = 80,
) -> tuple[dict, list]:
    """
    Train a golem with Tabular Q-Learning.

    ═══════════════════════════════════════════════════════════
    ▌ THE BELLMAN EQUATION (core of Q-learning):             ▐
    ▌                                                        ▐
    ▌   Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a')     ▐
    ▌                              − Q(s, a) ]               ▐
    ▌                                                        ▐
    ▌ Where:                                                 ▐
    ▌   s   = current state  (agent_x, agent_y, opp_x, opp_y)
    ▌   a   = action taken   (Up / Down / Left / Right)      ▐
    ▌   r   = reward received after taking action a in state s
    ▌   s'  = next state the environment transitions to      ▐
    ▌   α   = learning rate  (how fast we update beliefs)    ▐
    ▌   γ   = discount factor (how much we value future      ▐
    ▌         rewards vs. immediate ones)                    ▐
    ▌   max_a' Q(s', a') = the BEST expected future reward   ▐
    ▌         the agent can get from the next state          ▐
    ═══════════════════════════════════════════════════════════

    The update nudges Q(s, a) toward the "target":
        target = r + γ max_a' Q(s', a')

    If the target is higher than our current estimate, Q goes up.
    If lower, Q goes down.  Over many episodes this converges to
    the optimal action-value function Q*.
    """

    env = SumoEnv(push_reward, fall_penalty, center_reward, step_penalty)

    # ── Initialise Q-table as a dictionary ───────────────
    # Key:   state tuple  (ax, ay, ox, oy)
    # Value: numpy array of shape (NUM_ACTIONS,)
    # Defaulting to zeros means the agent starts with no
    # preference — it must EXPLORE to discover good actions.
    q_table: dict[tuple, np.ndarray] = {}

    def _get_q(state):
        """Lazily initialise Q-values for unseen states."""
        if state not in q_table:
            q_table[state] = np.zeros(NUM_ACTIONS)
        return q_table[state]

    # ── Training history for the frontend charts ─────────
    history = []
    epsilon = epsilon_start
    wins = 0

    # ── Some interesting starting positions for variety ──
    start_positions = [
        ((CENTER - 1, CENTER), (CENTER + 1, CENTER)),
        ((CENTER, CENTER - 1), (CENTER, CENTER + 1)),
        ((CENTER - 1, CENTER - 1), (CENTER + 1, CENTER + 1)),
        ((CENTER + 1, CENTER - 1), (CENTER - 1, CENTER + 1)),
        ((CENTER, CENTER), (CENTER + 1, CENTER)),
        ((CENTER - 1, CENTER + 1), (CENTER + 1, CENTER - 1)),
        ((CENTER, CENTER), (CENTER, CENTER + 1)),
        ((CENTER - 2, CENTER), (CENTER + 2, CENTER)),
        ((CENTER, CENTER - 2), (CENTER, CENTER + 2)),
    ]

    for ep in range(episodes):
        # Pick a random starting configuration for curriculum variety
        agent_start, opp_start = random.choice(start_positions)
        state = env.reset(agent_pos=agent_start, opp_pos=opp_start)

        total_reward = 0.0

        for step in range(max_steps):
            q_vals = _get_q(state)

            # ── ε-greedy action selection ────────────────
            # With probability ε we EXPLORE (random action),
            # otherwise we EXPLOIT (pick the best known action).
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = int(np.argmax(q_vals))

            # ── Take action, observe result ──────────────
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # ── THE BELLMAN UPDATE ───────────────────────
            # This is the heart of Q-learning.
            #
            # Step 1: Look up our current estimate Q(s, a).
            old_q = q_vals[action]
            #
            # Step 2: Compute the "target" — what we THINK
            #         Q(s, a) should be based on new info.
            #         If the episode ended (done=True), there
            #         is no future, so the target is just r.
            if done:
                target = reward
            else:
                # The best future value from the next state
                best_future_q = np.max(_get_q(next_state))
                target = reward + gamma * best_future_q
            #
            # Step 3: Nudge Q(s, a) toward the target.
            #         α controls the step size.
            #         Δ = α * (target − old_q)
            q_vals[action] = old_q + alpha * (target - old_q)
            # ─────────────────────────────────────────────

            state = next_state
            if done:
                if info.get("event") == "win":
                    wins += 1
                break

        # ── Decay exploration rate ───────────────────────
        # Over time the agent shifts from exploring to exploiting.
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Record history every 10 episodes
        if ep % 10 == 0:
            history.append({
                "episode": ep,
                "reward": round(total_reward, 2),
                "epsilon": round(epsilon, 4),
                "win_rate": round(wins / (ep + 1), 4),
            })

    # ── Serialise Q-table to plain dict of lists ─────────
    serialised_q = {}
    for state_key, q_vals in q_table.items():
        str_key = str(state_key)      # "(ax, ay, ox, oy)"
        serialised_q[str_key] = q_vals.tolist()

    return serialised_q, history


# ─────────────────────────────────────────────────────────────
# 3. SIMULATION  —  Replay a trained agent
# ─────────────────────────────────────────────────────────────

def simulate_episode(
    q_table_dict: dict,
    agent_start: tuple = (CENTER - 1, CENTER),
    opp_start: tuple = (CENTER + 1, CENTER),
    max_steps: int = 80,
) -> list[dict]:
    """Play one episode using a trained Q-table and return frames."""
    env = SumoEnv()
    state = env.reset(agent_pos=agent_start, opp_pos=opp_start)
    frames = [{"ax": env.agent_x, "ay": env.agent_y,
               "ox": env.opp_x, "oy": env.opp_y, "event": "start"}]

    for _ in range(max_steps):
        str_key = str(state)
        if str_key in q_table_dict:
            q_vals = q_table_dict[str_key]
            action = int(np.argmax(q_vals))
        else:
            action = random.randint(0, NUM_ACTIONS - 1)

        state, _, done, info = env.step(action)
        frames.append({
            "ax": env.agent_x, "ay": env.agent_y,
            "ox": env.opp_x, "oy": env.opp_y,
            "event": info["event"],
            "action": ACTION_NAMES[action],
        })
        if done:
            break

    return frames


def simulate_pvp(
    q_table_a: dict,
    q_table_b: dict,
    max_steps: int = 80,
) -> list[dict]:
    """Two trained agents battle. Agent A is 'agent', Agent B is 'opponent'."""
    env = SumoEnv()
    state = env.reset(agent_pos=(CENTER - 1, CENTER), opp_pos=(CENTER + 1, CENTER))
    frames = [{"ax": env.agent_x, "ay": env.agent_y,
               "ox": env.opp_x, "oy": env.opp_y, "event": "start"}]

    for _ in range(max_steps):
        if env.done:
            break

        # Agent A picks action
        str_key_a = str(state)
        if str_key_a in q_table_a:
            action_a = int(np.argmax(q_table_a[str_key_a]))
        else:
            action_a = random.randint(0, NUM_ACTIONS - 1)
        state, _, done, info = env.step(action_a)
        frames.append({
            "ax": env.agent_x, "ay": env.agent_y,
            "ox": env.opp_x, "oy": env.opp_y,
            "event": info["event"],
            "action_a": ACTION_NAMES[action_a],
        })
        if done:
            break

        # Agent B picks action (from opponent's perspective: swap roles)
        opp_state = (env.opp_x, env.opp_y, env.agent_x, env.agent_y)
        str_key_b = str(opp_state)
        if str_key_b in q_table_b:
            action_b = int(np.argmax(q_table_b[str_key_b]))
        else:
            action_b = random.randint(0, NUM_ACTIONS - 1)
        state, _, done, info = env.step_opponent(action_b)
        frames.append({
            "ax": env.agent_x, "ay": env.agent_y,
            "ox": env.opp_x, "oy": env.opp_y,
            "event": info["event"],
            "action_b": ACTION_NAMES[action_b],
        })
        if done:
            break

    frames.append({"winner": env.winner or "draw"})
    return frames


# ─────────────────────────────────────────────────────────────
# 4. FASTAPI  —  Endpoints
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Arcane Arena API")


class TrainRequest(BaseModel):
    push_reward: float = 10.0
    fall_penalty: float = -50.0
    center_reward: float = 1.0
    step_penalty: float = -0.1
    episodes: int = 3000
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    max_steps: int = 80


class SimulateRequest(BaseModel):
    q_table: dict
    scenarios: Optional[list] = None  # list of {agent: [x,y], opp: [x,y]}


class PvPRequest(BaseModel):
    q_table_a: dict
    q_table_b: dict


@app.post("/api/train")
async def api_train(req: TrainRequest):
    q_table, history = train_agent(
        push_reward=req.push_reward,
        fall_penalty=req.fall_penalty,
        center_reward=req.center_reward,
        step_penalty=req.step_penalty,
        episodes=req.episodes,
        alpha=req.alpha,
        gamma=req.gamma,
        epsilon_start=req.epsilon_start,
        epsilon_end=req.epsilon_end,
        epsilon_decay=req.epsilon_decay,
        max_steps=req.max_steps,
    )
    return JSONResponse({"q_table": q_table, "history": history})


@app.post("/api/simulate")
async def api_simulate(req: SimulateRequest):
    """Run 9 simulations with different starting positions."""
    default_scenarios = [
        {"agent": [CENTER - 1, CENTER], "opp": [CENTER + 1, CENTER]},
        {"agent": [CENTER, CENTER - 1], "opp": [CENTER, CENTER + 1]},
        {"agent": [CENTER - 1, CENTER - 1], "opp": [CENTER + 1, CENTER + 1]},
        {"agent": [CENTER + 1, CENTER - 1], "opp": [CENTER - 1, CENTER + 1]},
        {"agent": [CENTER, CENTER], "opp": [CENTER + 1, CENTER]},
        {"agent": [CENTER - 1, CENTER + 1], "opp": [CENTER + 1, CENTER - 1]},
        {"agent": [CENTER, CENTER], "opp": [CENTER, CENTER + 1]},
        {"agent": [CENTER - 2, CENTER], "opp": [CENTER + 2, CENTER]},
        {"agent": [CENTER, CENTER - 2], "opp": [CENTER, CENTER + 2]},
    ]
    scenarios = req.scenarios or default_scenarios

    results = []
    for sc in scenarios[:9]:
        frames = simulate_episode(
            req.q_table,
            agent_start=tuple(sc["agent"]),
            opp_start=tuple(sc["opp"]),
        )
        results.append(frames)

    return JSONResponse({"simulations": results})


@app.post("/api/pvp")
async def api_pvp(req: PvPRequest):
    frames = simulate_pvp(req.q_table_a, req.q_table_b)
    return JSONResponse({"frames": frames})


# ── Serve the frontend ───────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("index.html")


# ─────────────────────────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
