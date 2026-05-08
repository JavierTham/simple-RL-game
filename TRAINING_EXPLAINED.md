# How the bot learns — in plain English

## The setup

Picture a child learning to play a video game. Nobody hands them a manual. They press buttons, see what happens, and slowly figure out which buttons help them win. That's all reinforcement learning is. The bot is the child. The game is the teacher.

## The score is the only feedback

The bot has no idea what "winning" means at first. We give it a **scoreboard**:

- Land a hit on the opponent → **+ points**
- Get hit yourself → **− points**
- Knock the opponent off the edge → **big + points**
- Fall off yourself → **big − points** (and even bigger if you did it to yourself by mistake — we punish stupidity harder than bad luck)

That's it. No instructions. The bot just wants its score to go up.

## The brain

Inside the bot is a small **neural network** — think of it as a lookup table with 9 entries, one per possible action (charge, or dash in 8 directions). Given the current situation (where am I, where's my opponent, how charged am I, how close to the edge), it spits out a number for each action: *how good do I think this move is right now?* It picks the highest one.

At the start, those numbers are basically random. The bot flails. It charges when it should dash, dashes off the edge, sits still while the opponent batters it. **It loses, a lot.**

## Learning = nudging the numbers

After every match, we look back: which moves led to good outcomes? Which led to disasters? We nudge the network's internal weights so that next time it sees a similar situation, the move that *worked* gets a slightly higher score.

Repeat this **hundreds of times**. The numbers slowly drift toward something sensible. The bot starts charging up before fights. It starts aiming. It starts noticing the edge. By episode 200 it can win. By 400 it can win convincingly.

## The clever bits that make it actually work

A naive version of this would take days to learn. Three tricks make it train in 30 seconds:

**1. Credit where it's due.** The hit happens *now*, but the *decision* that earned it ("start charging") was made 6 steps ago. Reinforcement learning has to push the reward signal backwards through time so the bot connects "charging earlier" to "hitting later." We do this with something called **n-step returns** — basically, when scoring a move, we look 6 steps ahead instead of just 1.

**2. Smart curiosity.** Early on, the bot has to try random things to discover what works. But *truly* random doesn't help here — most random moves do nothing because the bot isn't charged. So we bias the exploration: when curious, charge first, then dash roughly toward the opponent with a wobble. It explores the *interesting* part of the action space, not the empty bits.

**3. A graduated curriculum.** A complete beginner can't learn from a grandmaster — they just lose every match and have nothing to learn from. So the bot first practises against a **dumb hand-coded opponent**, then a smarter one, and finally against **older copies of itself** (this is "self-play," the same idea AlphaGo used). Each phase is the right level of difficulty for where the bot currently is.

## What it actually feels like, watching it train

The win-rate graph starts near 0%. For about 50 episodes it's flat — the bot is mostly just learning that running off the edge is bad. Then it picks up that charging matters. Win rate climbs to 30%. Around episode 150 it discovers aiming. 50%. Then it figures out that you can hit *harder* if you charge longer, and that you can dodge a charged opponent. 70%. By the end it's playing real, recognisable strategy.

## The whole thing in one sentence

We let a tiny brain make millions of small decisions, score every one, and nudge it after each match toward decisions that scored well — and out the other end falls a bot that genuinely plays the game.
