# Lab Mode — Answer Key

Teacher reference for the Training tab's Lab Mode knobs and reward sliders.
Lists the defensible "right zone" for each knob, the obviously wrong settings,
and the RL concept each mistake teaches.

## Per-knob guidance

| Knob | Right zone | Obviously wrong | Why |
|---|---|---|---|
| **Foresight (γ)** | 0.93–0.99 | < 0.7 | Episodes are 150 steps; the charge→dash→hit chain is 6+ steps of credit propagation. Low γ makes the bot myopic — fires uncharged dashes for instant aim-reward, never connects charge to win. |
| **Curiosity decay (ε)** | 0.994–0.997 | 0.985 | The explorer is *scripted* charge-dash with aim noise (not uniform random), so exploration time is genuinely useful experience. Decaying too fast (ε floor by ep 200) starves the bot of structured trials. 0.999 is also bad for short (<300 ep) runs — never commits. |
| **Credit window (n-step)** | 6 or 10 | 1 | n=1 cannot propagate hit-reward back to the charge action that earned it. Bot stalls around 30% win rate even with everything else right. |
| **Sparring partners** | `default` or `easy_to_hard` | `easy_only`, `hard_only` | `easy_only` overfits to a weak opponent (sweeps Easy in gauntlet, drops Hard). `hard_only` can't bootstrap basic skills — plateaus. `self_play` works late but cold-starts poorly because there are no snapshots until ep 200. |
| **Reward preset** | Balanced or Aggressive | Camper / Bumper | Both are *deliberately* hackable: Camper has no win bonus + huge edge penalty (bot freezes mid-arena); Bumper has near-zero win bonus (bot dribbles for hit reward but never finishes). Any preset with low `win_bonus` is a tell. |
| **Episodes** | 300–500 | 100 | 100 episodes barely covers the early curriculum phase; the bot is still mostly exploring. Quality plateaus around 300–400. |
| **Learning rate** | 0.001–0.003 | 0.05+, 0.0001 | High LR thrashes weights — win rate oscillates instead of climbing. Very low LR can't make progress in 500 episodes. |

## Strongest config (target: sweep the gauntlet)

- **Foresight** = 0.97
- **Curiosity decay** = 0.995
- **Credit window** = 10
- **Sparring partners** = `easy_to_hard`
- **Reward preset** = Balanced
- **Episodes** = 500
- **Learning rate** = 0.001

## Pedagogical "wrong" configs (set as challenges)

| Config | Outcome | Concept taught |
|---|---|---|
| `easy_only` curriculum, everything else default | Trains a "bully" that sweeps Easy but drops Hard | Distribution shift / overfitting to training opponents |
| n-step = 1 | Win rate flatlines around 30% | Credit assignment and temporal-difference depth |
| Camper reward preset | Bot freezes mid-arena, refuses to engage | Reward hacking — bot optimizes proxy rewards over the real objective |
| γ = 0.5 | Twitchy bot fires uncharged dashes constantly | Discount factor / planning horizon |
| Learning rate = 0.05 | Win rate oscillates instead of climbing | Optimization stability |

## How to use this with students

1. Have them train once with defaults. Run the gauntlet — typically 2/3 wins.
2. Show them a "wrong" config from the table above. Have them predict the failure mode *before* training. Run the gauntlet to confirm.
3. Have them iterate toward the strongest config and aim for 3/3.
