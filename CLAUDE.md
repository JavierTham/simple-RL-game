# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

- After any sensible, self-contained change, commit it to git with a clear message that describes the *why*, not just the *what*. Don't batch unrelated edits into one commit. Don't leave the tree dirty across sessions.

## Running the project

Install (once):
```
pip install -r requirements.txt
pip install pymunk    # not in requirements.txt but imported by physics.py
```

Run the dev server (serves frontend + WebSocket at http://127.0.0.1:8000):
```
cd bumper_backend && python server.py
# or: uvicorn server:app --reload --app-dir bumper_backend
```

Run the physics/training smoke tests (assert-script, not pytest):
```
cd bumper_backend && python test_changes.py
```
The script uses `sys.path.insert(0, '.')` so it must be run from `bumper_backend/`. There is no test runner; failures surface as `AssertionError`.

## Architecture

### Backend (`bumper_backend/`)
Four layers, each importable on its own:

- **`physics.py`** — pymunk-backed 2D arena. `PhysicsWorld` owns a `pymunk.Space` and two `Bot`s. `Bot` stores RL-relevant state (`charge_level`, `is_dashing`, `dash_cooldown`, `last_dash_charge`, `dash_frames_left`) alongside the pymunk body. The pymunk body carries a `body._bot_ref` back-pointer so the global `_clamp_velocity` callback can check dash state to skip damping during dash flight. `get_observation()` returns a 15-dim float vector; the layout is positional and consumed by index in `agent.py` (e.g., `state[12]` = own charge), so changing it requires updating both files.
- **`neural_net.py`** — Dueling DQN (`QNetwork`) + a numpy-array `ReplayBuffer` with proportional priority sampling (note: uses `replace=False`, which is non-standard PER). Weights are JSON-serializable via `get_weights/set_weights`; a legacy `w1/b1/w2/b2/w3/b3` format is also accepted.
- **`agent.py`** — `DQNAgent` (n-step + Double DQN + soft target updates) and `DefaultAgent` (heuristic opponent). The agent's epsilon-greedy explorer is *not* uniform random — it scripts a charge→dash policy with Gaussian aim noise to avoid wasting moves on uncharged actions.
- **`trainer.py`** — Owns the training loop, reward shaping (`compute_reward`), and a 3-phase opponent curriculum keyed off episode index. PvP/test matches run synchronously here too.

### Frontend (`bumper_frontend/`)
Plain HTML + vanilla JS, no build step. `app.js` is an IIFE with module-level mutable state (training/PvP/tournament flags). `renderer.js` does idle ambient animation, per-frame match rendering, and particle effects in one class. `connection.js` wraps a single WebSocket. CSS is one file (~930 lines).

### Server / message flow (`server.py`)
A single global `Trainer()` instance is shared across all WebSocket clients (no multi-tenant isolation). Training runs in `loop.run_in_executor(None, ...)` and pushes progress through an `asyncio.Queue` back to the WebSocket; matches are recorded fully server-side then streamed at ~60 fps via `_stream_frames`.

WebSocket message types (client → server): `start_training`, `stop_training`, `test_match`, `pvp_match`, `save_bot`, `list_bots`, `load_bot`, `delete_bot`. Server → client: `training_progress`, `training_complete`, `match_frame`, `match_result`, `bot_saved`, `bot_list`, `bot_loaded`, `bot_deleted`, `error`.

Saved bots live in `bumper_saved_bots/<name>.json` (one file per bot, contains `{name, weights}`).

## Non-obvious behaviors / gotchas

- **n-step buffer invariant** (`agent.py`): `train_step` always uses `gamma ** n_steps` for the bootstrap discount, regardless of how many transitions were actually accumulated. This is correct *only* because partial-length flushes happen exclusively when `done=True` (the `(1 - dones_t)` mask zeros the bootstrap). Refactoring `store()`/`_flush_n_step()` without preserving this invariant will silently break the math.
- **Reward weights from the UI apply unevenly.** `compute_reward` multiplies *some* terms by `w.get(...)` (charge, hit, opp_edge, edge_penalty, win) but leaves dash-aim, deep-edge penalty, time pressure, and center-control hardcoded. Slider effects are partial.
- **Naming collision in physics.py:** `FRICTION` (module constant, 0.90) is actually the pymunk *space damping* coefficient, not surface friction; `BOT_FRICTION` (0.0) is real surface friction. Don't conflate them.
- **Stale `D` files in git status** (e.g., `backend/`, `frontend/`, old `saved_bots/`, `test_training.py`) are leftovers from a rename to the `bumper_*` prefix that was never committed. Treat them as deleted; commit the deletion when convenient.
- **Single shared trainer:** two clients connecting simultaneously will clobber each other's `is_training` flag and `world` state. Acceptable for solo dev, dangerous otherwise.
