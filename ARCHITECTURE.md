# Bumper Bot — RL & Physics Architecture

A walkthrough of how the four backend modules compose into a working DQN training loop. Frontend is intentionally out of scope.

---

## 1. Physics (`physics.py`)

The arena is a circle of radius 110 with two unit-mass disc bots (radius 15). All collision response is delegated to **pymunk** — the code never hand-rolls normals or impulses. The only custom physics is what sits *around* pymunk's solver: charge/dash state, a per-step velocity hook, and a soft-edge brake.

### Action space (9 discrete)

```
0     = CHARGE (build power, stand still)
1..8  = MOVE in 8 compass directions
```

But move actions are gated: they only do anything if `charge_level >= MIN_CHARGE_TO_DASH (0.08)` and `dash_cooldown == 0`. Otherwise they're wasted (just a small braking multiplier on velocity). This means the agent literally cannot move except by spending charge — every motion is a dash.

### The charge → dash mechanic (`Bot.apply_action`, physics.py:157-227)

- **Action 0** ramps `charge_level` by `CHARGE_RATE = 0.04` per step (≈25 steps to full) and applies `vel *= 0.85` braking.
- **Action 1–8** with charge releases an instantaneous velocity:
  ```
  dash_speed = 10 + charge * (22 - 10)
  velocity   = unit_dir * dash_speed
  ```
  This sets `is_dashing = True`, stores `last_dash_charge`, zeroes charge, starts a **10-step cooldown**, and starts a **3-frame "dash flight" window** (`dash_frames_left`).

### Velocity hook (`_clamp_velocity`, physics.py:93-109)

Pymunk lets bodies override their integration step. The hook does two things:

1. If the bot is in **dash flight** (`dash_frames_left > 0`), it calls `update_velocity` with `damping=1.0` — i.e. *no* damping that frame. This is critical: pymunk's space-wide damping (0.90) would otherwise eat the dash velocity instantly.
2. After damping, hard-clamp speed to `KNOCKBACK_MAX_SPEED = 28` so post-collision rebounds don't blow up.

The body holds `_bot_ref` back to the Python `Bot` so the global hook can read dash state — see physics.py:133.

### Soft-edge brake (physics.py:208-227)

If a bot's center is past `SOFT_EDGE_INNER` (= rim − 25 px), the brake **linearly damps the outward-radial component of velocity** to zero by the rim. Inward and tangential motion are untouched, so the bot can still escape and circle. This is what stops self-KOs from a miss-aimed dash near the edge.

The brake is **bypassed for `KNOCKED_FRAMES = 10` after a collision** (`knocked_frames` countdown). The numeric reasoning is in the comments: with FRICTION=0.9, a 22-speed knockback decays geometrically and travels ≈143 px in 10 frames — just enough to push the loser off the 135-px rim from a center hit. So legitimate hits *can* KO; tactical mistakes can't.

Important subtlety in physics.py:97-99: braking is applied **before** `space.step`, not inside the velocity hook. Pymunk integrates position with the pre-step velocity, so braking inside the hook would only take effect *next* frame — one full dash-frame too late.

### Collision callback (physics.py:281-302)

A `post_solve` collision handler:
- Records `total_impulse / mass` for the reward function.
- Ends both bots' dash flight (`dash_frames_left = 0`) so the rebound velocity damps normally — without this, the dasher's rebound was preserved while the target's push damped, and the dasher actually traveled further than the bot it hit.
- Starts the 10-frame brake-bypass on both bots.

`is_dashing` and `last_dash_charge` are deliberately **not** cleared here — they need to survive until the *next* `apply_action` so the reward function on this same step can attribute the hit to a dash. See the comment block at physics.py:286-289.

### Material constants (physics.py:80, 90)

- `BOT_ELASTICITY = 1.0` → effective restitution `e_eff = 1·1 = 1`, perfectly elastic. Higher injected energy and made dashers rebound nearly as far as their target.
- `BOT_FRICTION = 0.7` → tangential momentum transfer on glancing hits. `μ=0` left the dasher faster than the target on offset hits; `μ=1` dissipated too much KE.

### Observation vector (physics.py:327-387)

`get_observation(bot_idx)` returns an **18-dim float64** vector. Layout is positional and consumed by index in `agent.py`, so any change has to be coordinated:

| idx | meaning |
|---|---|
| 0,1 | own pos `(bx, by) / ARENA_RADIUS` |
| 2,3 | own vel `/ KNOCKBACK_MAX_SPEED` (28, not MAX_SPEED — dashes hit 22) |
| 4,5 | relative opponent pos `(dx, dy) / ARENA_RADIUS` |
| 6,7 | relative opponent velocity `/ KNOCKBACK_MAX_SPEED` |
| 8 | own distance from center / arena radius (edge proximity) |
| 9 | opp distance from center |
| 10 | `vel · (opp − me) / (|vel|·|d|)` — alignment, +1 = charging at, −1 = fleeing |
| 11 | inter-bot distance `/ (2·ARENA_RADIUS)` |
| 12 | own `charge_level` |
| 13 | opp `charge_level` |
| 14 | own `dash_cooldown / DASH_COOLDOWN_STEPS` |
| 15 | **dash_safety**: how far you can dash toward opp before hitting the rim (quadratic ray-circle intersect, normalized by max dash range 66 px). >1 = full dash safe |
| 16 | binary: opp committed mid-dash |
| 17 | opp cooldown |

Indices 12 and 14 are special-cased in `agent.get_action` to enforce charge-only movement.

### Step dynamics (physics.py:389-463)

Per step, in order:
1. Snapshot pre-action `dash_frames_left` and `last_dash_charge` so missed-dash detection can recover the charge value before `apply_action` clears it.
2. `apply_action` for both bots → updates charge/cooldown, may inject dash velocity, applies soft-edge brake.
3. `space.step(1.0)` → pymunk's velocity hook runs (damping + speed cap), then collision solve, then position integration.
4. Record collision impulse (Newton's 3rd law: equal masses → equal impulse on both bots).
5. **Missed-dash detection**: a dash flight that ended this step *without* a collision flips `bot{1,2}_missed_dash`, with the saved pre-charge stashed for the dodge bonus.
6. **Self-KO detection**: a bot that's `is_out` AND was mid-own-dash AND no collision happened → `self_ko = bot_idx`. This makes the reward function able to distinguish "got knocked off" (bad) from "dashed off the rim by your own action" (much worse).
7. Episode ends on out-of-bounds OR `step_count >= MAX_STEPS (400)`. Timeout = draw (`winner = -1`).

---

## 2. Q-network (`neural_net.py`)

### Dueling DQN

```
state (18) ─► Linear(18→128) ─► ReLU ─► features
                                         ├─► Linear(128→64)→ReLU→Linear(64→1)  = V(s)
                                         └─► Linear(128→64)→ReLU→Linear(64→9)  = A(s,a)

Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))
```

The `mean(A)` subtraction is the standard dueling identifiability trick — without it V and A can shift by an additive constant without changing Q. Implementation at neural_net.py:64.

Why dueling here: many states are "fine, just keep charging" where the *value* is similar across actions but the optimal action varies. Decoupling V from A makes V learn faster from any action's outcome.

Xavier init on all linears, bias = 0. Adam optimizer, lr=1e-3.

`get_weights / set_weights` JSON-serialize the full state dict and also accept a **legacy 3-layer format** (`w1/b1/w2/b2/w3/b3`) — those load into features + advantage, with value-stream init left at default. So old saved bots remain loadable.

### Prioritized replay buffer (neural_net.py:112-149)

Fixed-size circular buffer of preallocated numpy arrays (capacity 20k). On `push`, the priority is `1 + 2·|reward|` — high-magnitude transitions sampled more often. `sample` does proportional priority sampling with `np.random.choice(replace=False)`.

Two non-standard aspects flagged in CLAUDE.md:
- **`replace=False`**, where canonical PER uses replacement.
- No importance-sampling correction on the loss (which would be needed to debias prioritized sampling).

So it's "PER-flavored" rather than canonical PER.

---

## 3. Agent (`agent.py`)

### `DQNAgent`: DQN + Double DQN + n-step + soft target

Constructor defaults: `gamma=0.99, batch=64, n_steps=10`, but the **trainer overrides** to `gamma=0.95, batch=128, n_steps=6, tau=0.01, target_update_freq=0` (trainer.py:264-274). `target_update_freq=0` flips the agent into Polyak/soft-update mode (agent.py:181-186).

### Action selection (agent.py:45-68): not uniform random ε-greedy

This is the most non-obvious piece. Standard DQN uses uniform random over actions during exploration; here:

```python
my_charge   = state[12]
my_cooldown = state[14]

# If we can't dash, we MUST charge — movement does nothing
if my_charge < 0.05 or my_cooldown > 0.05:
    return 0

if random.random() < epsilon:
    release_prob = 0.15 + 0.6 * my_charge       # higher charge → more likely to release
    if random.random() < release_prob:
        angle = atan2(opp_dy, opp_dx) + N(0, 0.4)  # toward opp + Gaussian noise
        return _angle_to_action(angle)
    return 0  # keep charging

# greedy: argmax Q(s,·)
```

Rationale: pure ε-greedy would waste the vast majority of exploration steps on move actions that the physics layer simply ignores (no charge → no movement, no learning signal). The scripted explorer charges→dashes-toward-opponent-with-noise, generating useful experience even at high ε.

`get_action_greedy` is the inference-time variant — same charge/cooldown gate but no exploration (used by test matches, PvP, and self-play opponents).

### N-step returns (agent.py:101-142) — central to credit assignment

The whole training challenge here is: the reward signal lives at the moment of collision, but the *decision* that mattered ("start charging now") happened ~10 steps earlier. 1-step TD targets would only propagate that signal one step per gradient update.

Implementation:
- A `deque(maxlen=n_steps)` buffers transitions.
- When full (or on episode end), it computes
  `R_n = r_0 + γ·r_1 + γ²·r_2 + … + γ^(n-1)·r_(n-1)`
- Stores `(s_0, a_0, R_n, s_n, done_n)` in replay and pops the oldest.
- TD target in `train_step` (agent.py:168-169):
  `y = R_n + γ^n · Q_target(s_n, argmax_a Q_online(s_n, a)) · (1 − done)`
  i.e. **Double DQN** (online net picks the action, target net evaluates it) over the n-step bootstrap.

The CLAUDE.md notes a critical invariant here: `train_step` always uses `gamma ** n_steps` regardless of how many transitions actually went into `R_n`. This is correct *only* because partial-length flushes (when the deque has fewer than n entries) happen exclusively when `done=True`, and the `(1 - dones_t)` mask zeros the bootstrap term. Subtle, easy to break.

### Soft target update (Polyak)

Every `train_step` (no frequency gating, since `target_update_freq == 0`):

```
target ← (1 − τ)·target + τ·online        # τ = 0.01
```

Smoother than periodic hard copies; reduces target oscillation.

### Loss

`F.smooth_l1_loss` (Huber). Gradient norm clipped to 1.0 to keep large-reward transitions from blowing up the update.

### `DefaultAgent` (heuristic opponent)

Mode-machine over four tactical states: `attack`, `reposition`, `evade_edge`, `dodge`. Picks a `_charge_target` (number of frames to charge), then dashes when reached. `strength ∈ [0,1]` parameterizes aim noise and disables the more advanced rules at low values. Crucially even at `strength=0` it doesn't pick uniformly random actions — it still charge→dashes with poor aim, because random move-action spam would waste turns on the no-op gate (agent.py:251-258). Same reasoning as the `DQNAgent`'s exploration policy.

---

## 4. Trainer (`trainer.py`)

### Episode loop (trainer.py:209-256)

```
reset world
loop:
    obs1 = world.get_observation(0)
    obs2 = world.get_observation(1)
    a1 = agent.get_action(obs1)         # ε-greedy if training
    a2 = opponent.get_action(obs2)
    done, winner = world.step(a1, a2)
    r = compute_reward(world, 0, done, winner, ...)
    next_obs1 = world.get_observation(0)
    agent.store(obs1, a1, r, next_obs1, done)
    if step % TRAIN_EVERY == 0: agent.train_step()
agent.end_episode()    # decays epsilon, flushes n-step buffer
```

Training episodes are capped at `TRAIN_MAX_STEPS = 150` (vs 400 for free play) to force decisive engagement. `TRAIN_EVERY = 8` — one gradient update per 8 env steps.

### Reward shaping (`compute_reward`, trainer.py:63-169)

This is the densest piece of the design. Layered components, terminal signals dominant:

| Component | Sign | Magnitude |
|---|---|---|
| Win | + | `15 · w_win` |
| Loss | − | 12 |
| Self-KO | − | **20** (worse than a normal loss — discourages "dash off when cornered") |
| Draw / timeout | − | 6 |
| Time pressure (per step, while not done) | − | 0.02 (capped at 3.0/episode < draw–loss gap of 6) |
| Charging while close to a low-charged opponent | + | `0.15 · charge · closeness · (1 − 0.5·opp_charge) · w_charge` |
| Dashing toward opponent (aim quality × charge) | + | `0.3 · cos(θ) · last_dash_charge` |
| **Dash-hit landed** | + | `(2.5 + 3·charge) · (0.5 + 0.5·impulse_norm) · w_hit` |
| Dash-hit received | − | `(0.5 + charge) · (0.5 + 0.5·impulse) · w_hit` |
| Engagement (close + somebody charged) | + | 0.03 |
| Dodge bonus (opp dash missed, you in range) | + | `0.6 · opp_charge_at_miss` |
| Opponent near edge | + | `0.5 · (opp_edge − 0.5)² · w_opp_edge` |
| Self near edge (>0.6) | − | `0.5 · (edge − 0.6)² · w_edge` |
| Self deep edge (>0.85) | − | `1.5 · (edge − 0.85) / 0.15` (hardcoded — no UI weight) |
| Center control | + | `0.01 · (1 − edge)` |

Key tuning choices baked in:
- **Total per-episode time penalty < (loss − draw) gap.** 0.02 × 150 = 3.0 < 6, so timing-out is still better than running off the edge.
- **Self-KO is much worse than a regular loss.** Without this the agent can substitute "dash off" for "outplayed loss" when cornered.
- **Charge reward downweighted when opponent is highly charged.** Standing still while a fully-charged opp stares you down is dangerous, not virtuous.
- **Hit reward scales with both charge and observed impulse.** A glancing partial-charge bonk pays less than a clean center-mass full-charge hit.
- **CLAUDE.md flag**: only some terms multiply by the UI's reward weights (`charge`, `hit`, `opp_edge`, `edge_penalty`, `win`). Dash-aim, deep-edge, time pressure, engagement, dodge, center-control are hardcoded — slider effects are partial.

### Curriculum + self-play league (`_pick_opponent`, trainer.py:172-206)

Per episode, sample an opponent from a 4-way mix `(easy, hard, past_self, current_self)` whose weights shift with `progress = ep / num_episodes`:

| Phase | Mix |
|---|---|
| < 30% | (0.55, 0.30, 0.05, 0.10) — mostly easy heuristic, just learn the loop |
| 30–70% | (0.20, 0.30, 0.30, 0.20) — even, self-play kicks in |
| > 70% | (0.05, 0.20, 0.45, 0.30) — mostly self-play |

- **`easy`**: `DefaultAgent` whose `strength` ramps `0.2 → 0.6` with progress.
- **`hard`**: `DefaultAgent(strength=1.0)`.
- **`past_self`**: a `_GreedyDQNOpponent` wrapping a frozen snapshot. Snapshots are taken **every 200 episodes** into a FIFO of depth 5 (trainer.py:289-292). If no snapshots exist yet, that probability mass folds into `current`.
- **`current_self`**: `_CurrentSelfOpponent` that calls *the live training agent's* `get_action_greedy` — opponent always tracks the latest weights.

Why the league: training only against the scripted heuristic produces a policy that exploits the heuristic's quirks. Mixing past self stops policy collapse; current self stops the agent from being able to "memorize" frozen snapshots.

### Top-level `train` (trainer.py:259-325)

Builds a fresh `DQNAgent` with the trainer's overrides (`gamma=0.95, n_steps=6, tau=0.01, soft updates`), then loops `num_episodes`:

1. Maybe snapshot weights into the past-self pool.
2. Pick opponent.
3. Run episode (training mode, no frame recording).
4. Track 100-episode rolling win/loss rate.
5. Push stats through `progress_callback` every 10 episodes.
6. Honor `is_training` flag for cooperative cancellation from the WS layer.

Returns the final agent's weights.

---

## How the pieces compose

Putting it together, one full step of training:

1. `world.get_observation(0/1)` — pymunk body state + bot-internal charge/cooldown/dash → 18-dim obs.
2. Agent chooses action via the **charge-aware ε-greedy** (forced charge if uncharged or cooling down; otherwise scripted dash-with-noise during exploration, argmax during exploitation). Opponent does the same via its policy.
3. `world.step(a1, a2)`:
   - `Bot.apply_action` updates charge/cooldown, possibly injects dash velocity, applies soft-edge brake.
   - `space.step` runs the velocity hook (no-damp during dash flight, then 28-px/step speed cap), then pymunk's collision solver, then position integration.
   - Collision callback records impulse and ends dash flight.
   - Missed-dash and self-KO flags get computed from pre/post snapshots.
4. `compute_reward` reads world state — collision impulse, dash flags, charge levels, edge proximity, terminal/self-KO/draw flags — and produces a scalar.
5. `agent.store` adds the transition to a 6-step deque. When the deque is full (or done), the n-step return is computed and pushed to the replay buffer with priority `1 + 2·|reward|`.
6. Every 8 env steps, `train_step` samples a priority-weighted batch, computes a **Double-DQN n-step TD target** through the target net, and does a Huber-loss SGD step with grad-clip. Target net is Polyak-blended toward online with τ=0.01.
7. At episode end, ε decays (× 0.994 per episode, floor 0.05), and any partial n-step transitions are flushed (safe because `done=True` zeros their bootstrap).

The end result is a 9-action policy where the agent has to learn three couplings simultaneously: **how long to charge**, **when to release**, and **what direction to release**, with credit for the eventual hit propagated back to the charge decision via the n-step bootstrap.
