"""Test charge/dash mechanics + training convergence."""
import sys
import time
import math
sys.path.insert(0, '.')
from physics import (PhysicsWorld, MAX_SPEED, ARENA_RADIUS, DASH_MAX_SPEED,
                     KNOCKBACK_MAX_SPEED, FRICTION, CHARGE_RATE, MIN_CHARGE_TO_DASH)
from trainer import Trainer
from agent import DQNAgent, DefaultAgent

print("="*60)
print("BUMPER BOT v2 - CHARGE/DASH MECHANIC TESTS")
print("="*60)

# -- Test 1: Charge accumulation --
print("\n=== Test 1: Charge accumulation ===")
w = PhysicsWorld()
w.reset()
for i in range(50):
    w.step(0, 3)  # bot1 charges, bot2 moves right
    if i % 10 == 0:
        print(f"  Step {i}: bot1 charge={w.bot1.charge_level:.2f}, bot2 charge={w.bot2.charge_level:.2f}")
print(f"  Final: bot1 charge={w.bot1.charge_level:.2f} (expected ~1.0)")
assert w.bot1.charge_level >= 0.99, f"Charge didn't reach max: {w.bot1.charge_level}"
print("  PASS")

# -- Test 2: Dash release --
print("\n=== Test 2: Dash release ===")
w.reset()
for _ in range(20):
    w.step(0, 0)
charge_at_release = w.bot1.charge_level
print(f"  Charge before dash: {charge_at_release:.2f}")
w.step(3, 0)
v1 = math.sqrt(w.bot1.vx**2 + w.bot1.vy**2)
print(f"  Speed after dash: {v1:.2f}")
print(f"  bot1.is_dashing: {w.bot1.is_dashing}")
assert v1 > MAX_SPEED * 1.5, f"Dash speed too low: {v1}"
assert w.bot1.charge_level == 0.0, "Charge should be 0 after dash"
print("  PASS")

# -- Test 3: DefaultAgent decisive rate --
print("\n=== Test 3: DefaultAgent vs DefaultAgent (100 games) ===")
outcomes = {'bot0': 0, 'bot1': 0, 'draw': 0}
total_steps = 0
for g in range(100):
    t = Trainer()
    t.world.reset()
    opp1, opp2 = DefaultAgent(strength=1.0), DefaultAgent(strength=1.0)
    done, steps = False, 0
    while not done and steps < 200:
        o1, o2 = t.world.get_observation(0), t.world.get_observation(1)
        done, winner = t.world.step(opp1.get_action(o1), opp2.get_action(o2))
        steps += 1
        if steps >= 200 and not done:
            done, winner = True, -1
    total_steps += steps
    if winner == 0: outcomes['bot0'] += 1
    elif winner == 1: outcomes['bot1'] += 1
    else: outcomes['draw'] += 1
decisive = outcomes['bot0'] + outcomes['bot1']
print(f"  Decisive: {decisive}%, Draws: {outcomes['draw']}%, Avg steps: {total_steps/100:.0f}")
assert decisive >= 50, f"Only {decisive}% decisive - needs >50%"
print("  PASS")

# -- Test 4: Training convergence + timing --
print("\n=== Test 4: 1000-episode DQN training ===")
t = Trainer()
t.agent = DQNAgent(
    batch_size=256, epsilon_decay=0.997, lr=0.001,
    gamma=0.995, epsilon_end=0.15,
    target_update_freq=0, tau=0.005,
)
wins, recent = 0, []
start_time = time.time()
for ep in range(1000):
    progress = ep / 999
    opp_str = 0.2 + 0.8 * min(1.0, progress * 2.5)
    result = t.run_episode(t.agent, DefaultAgent(strength=opp_str), {}, train=True)
    won = result['winner'] == 0
    if won: wins += 1
    recent.append(int(won))
    if len(recent) > 100: recent.pop(0)
    if (ep + 1) % 200 == 0:
        wr = sum(recent) / len(recent)
        elapsed = time.time() - start_time
        print(f"  Ep {ep+1:4d}: wins={wins}, last-100 WR={wr:.0%}, eps={t.agent.epsilon:.3f}, time={elapsed:.1f}s")

elapsed = time.time() - start_time
print(f"\n  Total: {wins}/1000 wins ({wins/1000:.0%})")
print(f"  Training time: {elapsed:.1f}s (target <90s)")

# Greedy evaluation
print("\n=== Test 5: Greedy evaluation vs full-strength DefaultAgent ===")
test_wins, test_draws = 0, 0
for _ in range(30):
    result = t.run_episode(t.agent, DefaultAgent(strength=1.0), {}, train=False, max_steps=400)
    if result['winner'] == 0: test_wins += 1
    elif result['winner'] == -1: test_draws += 1
print(f"  Greedy: {test_wins}/30 wins ({test_wins/30:.0%}), {test_draws} draws")

print("\n" + "="*60)
results = {
    'decisive_rate': decisive,
    'training_time': elapsed,
    'training_wr': wins/1000,
    'greedy_wr': test_wins/30,
    'greedy_draws': test_draws,
}
all_pass = (
    decisive >= 50 and
    elapsed < 90 and
    test_wins >= 6  # at least 20% greedy WR
)
for k, v in results.items():
    print(f"  {k}: {v}")
print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
print("="*60)
