"""Full training test with momentum collisions."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from trainer import Trainer

t = Trainer()
config = {
    'num_episodes': 1500,
    'learning_rate': 0.001,
    'reward_weights': {
        'win_bonus': 1.0, 'survival': 0.1, 'aggression': 0.3,
        'center_control': 0.2, 'hit_reward': 0.5, 'edge_penalty': 0.3,
    },
}

def cb(stats):
    if stats['episode'] % 100 == 0 or stats['episode'] <= 20 and stats['episode'] % 5 == 0:
        print(f"  ep {stats['episode']:4d}  win_rate={stats['win_rate']:.2f}  eps={stats.get('epsilon',0):.3f}  steps={stats['steps']}")

start = time.time()
print("Training 1500 episodes...")
t.train(config, cb)
elapsed = time.time() - start
print(f"\nDone in {elapsed:.1f}s ({elapsed/1500*1000:.0f}ms/episode)")

test_wins = 0
for _ in range(20):
    result = t.test_match()
    if result and result['winner'] == 0:
        test_wins += 1
print(f"Test: {test_wins}/20 wins vs full-strength default")
