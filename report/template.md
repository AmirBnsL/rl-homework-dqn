---
title: "Homework Assignment 1 — Deep Q-Network"
subtitle: "Ablation of DQN, Double/Dueling DQN, and a custom Gymnasium environment"
author: "__AUTHOR__"
date: "__DATE__"
geometry: margin=1.9cm
fontsize: 9pt
---

# 1. What this report measures

DQN adds two ingredients to deep Q-learning — experience replay and a target network — and two
improvements have become standard since: the Double DQN target and the dueling architecture. We
measure each of them separately, on three tasks of increasing difficulty, and then apply the same
agents, unchanged, to an environment written from scratch.

Every experiment follows the same protocol:

* **seeds 0, 1, 2** for every configuration, and only **one setting changes at a time** between two
  variants;
* the **training score** is the mean return of the last 100 episodes (CartPole, LunarLander) or the
  last 20 games (Pong);
* the **greedy score** is the mean return of 20 new episodes with ε = 0, using seeds 100 000 + i
  that never appear during training;
* a difference is called meaningful only when it exceeds the **min–max band across the three
  seeds** — RL curves are noisy and a single seed is not evidence.

All artifacts (curves, metrics, networks) come from one command per run, and every notebook in the
archive re-plots them from the saved `.npz` files, so the numbers below are reproducible without
re-training.

**Results at a glance.**

{{glance}}

# 2. Part 0 — the three environments

| | CartPole-v1 | LunarLander-v3 | ALE/Pong-v5 |
|---|---|---|---|
| State | $\mathbb{R}^4$ | $\mathbb{R}^8$ | 4 stacked `84 x 84` grayscale frames (`uint8`) |
| Actions | 2 | 4 | 6 |
| Reward | +1 per step | shaped (distance, speed, legs, fuel, crash) | +1 / −1 per point |
| Terminated | pole falls | crash / rest / out of screen | 21 points |
| Truncated | 500 steps | 1000 steps | end of the game |
| Solved | 475 | 200 | 18 |
| Network | MLP 4-64-64-2 | MLP 8-128-128-4 | Nature CNN, 1 687 206 parameters |
| lr / γ | 5e-4 / 0.99 | 5e-4 / 0.99 | 1e-4 / 0.99 |
| buffer / batch / learn every | 100 000 / 64 / 1 | 100 000 / 64 / 4 | 100 000 (uint8) / 32 / 1 |
| warm-up | 64 | 64 | 10 000 |
| target update C | 100 | 250 | 1 000 |
| ε | 1.0 → 0.05 / 10 000 | 1.0 → 0.05 / 25 000 | 1.0 → 0.01 / 1 000 000 |
| loss | MSE | MSE | Huber, grad norm clipped to 10 |
| budget | 300 episodes | 800 episodes | __PONG_BUDGET__ |

Atari preprocessing is the one of Mnih et al. (2015): frame skip 4 with max-pooling of the last two
frames, grayscale 84 x 84, up to 30 random no-ops at reset, stack of the last 4 frames, no sticky
actions. Transition storage is `uint8` (56 KB instead of 226 KB per transition) and the frames are
divided by 255 inside the network; the reward fed to the loss is `sign(r)` while the reported score
is the true game score.

# 3. Exercise 1 — which ingredient of DQN matters?

Four variants share everything except one switch:

| Variant | What changes |
|---|---|
| `hard` (reference) | replay + target network, $\theta^- \leftarrow \theta$ every C gradient steps |
| `soft` | $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ after every gradient step, $\tau = 0.01$ |
| `no_replay` | memory keeps only the last B transitions, sampled in chronological order |
| `no_target` | targets computed with the online network $\theta$ |

The notebooks first check the implementation itself: the Nature CNN has exactly 1 687 206
parameters, the dueling head satisfies `Q = V + A − mean(A)`, the two replay modes return the
documented minibatches, and `compute_targets` matches the formulas of the assignment numerically.

## 3.1 CartPole-v1

{{figure:ex1_cartpole:CartPole-v1 — 20-episode moving average (top) and predicted Q-values (bottom), mean of 3 seeds with the min–max band.}}

{{table:ex1_cartpole}}

{{verdicts:cartpole:ex1}}

On CartPole all four variants end up with a usable policy and **no difference clears the band across
seeds**: at 300 episodes the task is too short and too seed-sensitive, and we say so rather than read
a ranking into the noise (the soft target is the only variant whose *worst* seed clears 475). What
does separate them is the *value* scale: the reference plateaus at $Q \approx 124$, already above the
$1/(1-\gamma) = 100$ bound, the no-target agent reaches 400–600 while its policy keeps working, and
the no-replay agent never builds a consistent estimate. That is the first hint of what the target
network really controls — not the return, but the calibration of the values behind it.

## 3.2 LunarLander-v3

{{figure:ex1_lunarlander:LunarLander-v3 — 20-episode moving average (top) and predicted Q-values (bottom).}}

{{table:ex1_lunarlander}}

{{verdicts:lunarlander:ex1}}

LunarLander is where the ablation separates the variants, and with a large margin. **Without replay
the agent never learns**: the curve stalls around −180 and the greedy policy crashes immediately, a
323-point gap against the reference that is seven times the min–max band. The soft target beats the
hard one by 47 points (band 28) and is the only variant that clears the 200 threshold with margin on
every seed. Removing the target network costs nothing measurable — 233.5 against 196.7 is inside the
band, and slightly *higher* — which is the honest reminder that a target network is not free
insurance: on episodes of a few hundred steps with a dense, shaped reward the online network's
targets are already fresh enough. The episodes last up to 1000 steps here, so a bootstrap error has
much more time to accumulate than on CartPole.

## 3.3 ALE/Pong-v5

{{figure:ex1_pong:Pong — 20-game moving average and predicted Q-values for the four variants, same budget for every variant and seed.}}

{{table:ex1_pong}}

{{verdicts:pong:ex1}}

Pong is played from pixels with a Nature CNN and 6 actions; a random agent scores about −20 and the
convention for "solved" is 18. At the uniform budget of __PONG_BUDGET__ the task sits at the edge of
what DQN learns in a single run, and the ablation separates *learning* from *not learning* rather
than ranking the variants: the reference scores −4.1, −11.2 and −21.0 on its three seeds (one seed
never leaves the random-agent level), the soft target is the only variant whose mean is positive
(+5.1, −9.5, −3.9), and **all three no-replay seeds sit at exactly −21.0** with a flat curve — the
same signature as the −180 collapse on LunarLander, now without the possibility of mistaking it for
noise. The bands across seeds (17–20 points) are wider than the gaps between the remaining variants,
so no finer claim is made: at 10^6 steps the ordering would have room to emerge, and §6 explains why
that budget was out of reach here.

# 4. Exercise 2 — Double DQN and Dueling DQN

| Agent | Target | Head |
|---|---|---|
| `hard` | $r + \gamma(1-d)\max_{a'} Q(s', a'; \theta^-)$ | standard |
| `double` | $r + \gamma(1-d)\,Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$ | standard |
| `dueling` | DQN target | $Q = V + A - \overline{A}$ |
| `dueling_double` | Double DQN target | dueling |

The max operator in the DQN target is positively biased when the Q-values are noisy — it is a
maximum of estimates, not an estimate of the maximum. Double DQN removes the bias by letting the
online network select the action and the target network evaluate it. The dueling head attacks a
different problem: in many states the value of the state dominates the choice of action, and
learning `V(s)` separately from `A(s, a)` makes the shared representation easier to train.

**Measuring the bias.** For every state visited by the greedy policy we compare the value the
network predicts with the discounted return actually obtained from that state, over the 20
evaluation episodes:

$$\text{bias} = \overline{Q(s, a)} - \overline{G_t}, \qquad G_t = \sum_{k \geq 0} \gamma^k r_{t+k}$$

The last column of the tables below is the fraction of evaluation episodes that ended by
**truncation** rather than termination, and it matters: `Q` bootstraps through a truncation while
`G_t` stops at the horizon, so an agent whose episodes are cut short is systematically penalised by
this metric. On LunarLander only 3–18 % of the episodes are truncated and the bias ordering is clean;
on CartPole, where 33–90 % are truncated, the ordering follows episode length more than value error,
so the bias claim is made on LunarLander only.

{{figure:ex2_cartpole:CartPole-v1 — curves, Q-values, overestimation bias and greedy returns for the four agents.}}

{{table:ex2_cartpole}}

{{verdicts:cartpole:ex2}}

On CartPole the two improvements do not separate either: all four agents land inside each other's
bands and the ordering of the bias (47, 59, 47, 41) does not follow the ordering of the returns, so
we do not claim a winner. CartPole is simply too easy at this budget.

{{figure:ex2_lunarlander:LunarLander-v3 — the same four panels.}}

{{table:ex2_lunarlander}}

{{verdicts:lunarlander:ex2}}

Here the ordering is monotone on both axes: **197 → 222 → 250 → 261** for the greedy return and
**+22 → +12 → +4 → −1** for the overestimation bias. Double DQN halves the bias; the dueling head
removes most of what remains; together they land at a bias indistinguishable from zero and gain 63
points over the reference, three times the band across seeds. The Q-value panel shows the same thing
from the other side: the reference's values keep climbing past what the returns justify, while
dueling-double stays the flattest of the four, and the greedy-return bars order the agents exactly
like the bias bars.

{{figure:ex2_pong:Pong — the same four panels, with the same step budget as in Exercise 1.}}

{{table:ex2_pong}}

{{verdicts:pong:ex2}}

# 5. Exercise 3 — TreasureGrid, an environment written from scratch

`treasure_grid.py` implements a Gymnasium environment from scratch: an `N x N` grid (N = 6) with
walls, two traps and one treasure. Moving into a wall keeps the agent in place; reaching the
treasure ends the episode with +1, stepping on a trap with −1; the episode is truncated after
`4N² = 144` steps. The start cell is random, the walls and the treasure are fixed so that every
agent faces the same map. The environment passes `gymnasium.utils.env_checker.check_env`, and two
baselines are provided: a uniform random agent and an optimal agent following a BFS shortest path.

Three registered versions change the three things the assignment asks for:

| Version | State representation | Reward | Dynamics |
|---|---|---|---|
| `TreasureGrid-v0` | 4 one-hot planes of the whole grid, 144 floats | sparse: +1 treasure, −1 trap | 2 static traps |
| `TreasureGrid-v1` | 9 coordinates (agent, treasure, traps, distance) | potential-based shaping `F = Φ(s′) − Φ(s)`, `Φ = −dist/N` | 2 static traps |
| `TreasureGrid-v2` | egocentric `5 x 5` patch, 5 channels, 125 floats | sparse: +1 treasure, −1 trap | traps move after every action, treasure random at reset |

{{baselines}}

Three agents (`hard`, `double`, `dueling`) are trained on every version, starting from the CartPole
column of Part 0 (MLP 64, Adam 5e-4, buffer 100 000, batch 64, learn every step, target update every
100 gradient steps). Two settings are changed and then kept fixed for all 27 runs: `γ = 0.90`
instead of 0.99 — the episode lasts 144 steps but the optimal path is about 6, and with 0.99 the
Q-values saturate and the greedy policy collapses to standing still — and `ε_end = 0.10` instead of
0.05, because the reward is sparse. There is no per-step penalty: over a 144-step horizon it would
make dying on a trap cheaper than finishing the episode. The budget is 800 episodes. The agents
themselves run without any modification: only `observation_space` and `action_space` differ from
CartPole.

{{figure:ex3_curves:TreasureGrid — learning curves of the three agents on the three versions (3 seeds).}}

{{table:ex3}}

{{verdicts:treasure:ex3}}

Three observations. First, **the representation matters**: the egocentric patch (`v2`) gives the
highest success rate and by far the shortest episodes (12–22 steps against a BFS optimum of 5.3),
while the coordinate version (`v1`) gives the lowest. Second, **the potential-based shaping of `v1`
does not help** — its success rate is below the sparse `v0` for every agent (40 % against 65 % for the
reference), and its episodes are 70–83 steps long against 40–56: with a potential that is linear in
the Manhattan distance, shaping adds a gradient that the small MLP follows without ever committing
to the terminal action. That is a useful negative result, and it is the reason the streaming
"shaping always helps" intuition is worth testing rather than assuming. Third, Double DQN is the
best agent on `v0` and `v1`, while on `v2` the three agents are within noise of each other — which is
what one expects when the state representation already carries the information: there is less room
for the target correction to help. Finally, every agent remains far from the BFS optimum in *length*: they succeed 60–83 % of the time
but wander 2–13 times longer than the shortest path. That is not an artifact of the reward: we
re-ran the three versions with a per-step penalty of −0.005 and of −0.01 and the episodes did **not**
get shorter (56 and 60 steps against 51 for `v0` without penalty, 72 and 83 against 79 for `v1`),
they only earned less. The greedy policy is a local optimum that never discovers the short route —
success and optimality are different things, and only the second one requires search.

# 6. Compute budget and reproducibility

{{compute}}

The Pong budget was fixed after a calibration run on the target GPU: `scripts/calibrate.py` times a
short training segment of the exact loop used for the real runs and extrapolates. The same budget is
used for the 21 Pong runs (3 seeds × 7 variants); when it is below the 10^6 steps of the reference
table, the ε decay is scaled to the same horizon so that ε reaches its final value at the end of
training, and this is stated in the results.

Every run seeds Python, NumPy and PyTorch, and the environment seeds are derived from the run seed
(`seed * 10 000 + episode` for training, `100 000 + i` for evaluation). Runs are resumable: a
checkpoint (metrics + network) is written every 50 000 environment steps, so a recycled Colab VM
loses at most one segment. The `.npz` artifacts store the episode returns, the losses and the mean
Q-value per gradient step, the 20 evaluation episodes and the full hyperparameter dictionary of the
run, which is what every notebook re-plots.

## 6.1 Limitations of this protocol

* **Three seeds is the protocol's floor.** The min–max band across three seeds is a conservative
  criterion: one outlier seed can make the band exceed the effect we are looking for, which is
  exactly what happens on CartPole and why no claim here rests on a single environment.
* **Pong is measured below its reference budget.** At __PONG_BUDGET__ the task is at the edge of what
  DQN can learn in one run: a random agent scores −20 and one of our reference seeds finishes at
  −21, i.e. it did not learn at all. The Pong comparison is therefore a comparison *in this regime*;
  it is uniform across variants and seeds, but it is not the 10^6-step regime of the reference paper.
* **The bias is measured on each agent's own states**, so it mixes value error with a different state
  distribution per agent; the *ordering* is the meaningful part, and the training-time Q-value curves
  (same minibatch distribution for everyone) corroborate it independently.
* **One map, one grid size.** TreasureGrid uses a single 6 x 6 layout; the versions change the
  representation and the dynamics, not the difficulty of the map.

# 7. Conclusion

The three experiments agree on some points and contradict the textbook on others, which is what
three seeds and a band were for:

* **Experience replay is not optional.** Removing it costs nothing on CartPole, but on LunarLander
  it is a 323-point collapse (band 42) and on Pong all three no-replay seeds stay at exactly −21.0,
  the random-agent score, while the reference reaches −4.1 on its best seed. The CartPole result is
  a warning about the protocol: an easy task hides a broken algorithm.
* **The target network does not buy return at this budget — it buys calibration.** On LunarLander
  the no-target variant is *better* (233 against 197, inside the band), on CartPole it is slightly
  worse, and on neither does it cost what the theory suggests. What it does control is the value
  scale: without it the CartPole Q-values reach 400–600 (bias +237) against 124 (+47) for the
  reference. The claim "the target network is necessary" is therefore too strong; "the target
  network keeps the values usable" is what the measurements support.
* **The soft target is the cheapest real improvement of the ablation**: +47 points on LunarLander
  with a band of 28, a smoother Q-value curve, and no extra cost per step.
* **Double DQN and dueling DQN work as advertised where the task is hard enough to show it.**
  On LunarLander both quantities move monotonically — return 197 → 222 → 250 → 261, bias
  +22 → +12 → +4 → −1 — and the pair gains 63 points, three times the band. On CartPole all four
  agents are inside each other's bands: the task is solved at this budget by everyone, so it cannot
  rank them.
* **The custom environment transfers without a single line of agent code**, and it teaches two
  lessons. The state representation dominates. The egocentric patch (`v2`)
  beats the one-hot planes (`v0`) and the raw coordinates (`v1`) on success rate and on episode
  length; the potential-based shaping of `v1` *hurts* (40 % against 65 %), which is a reminder that
  shaping changes what is easy to learn, not only how fast. And every learned policy stays 2–13
  times longer than the BFS optimum even when it succeeds — a per-step penalty does not fix that
  (we tried −0.005 and −0.01), because the greedy policy is simply a local optimum.

The deliverables contain the seven executed notebooks, `treasure_grid.py`, the best network of every
agent, and the artifacts needed to regenerate every figure of this report.
