# DQN homework — ablation, Double/Dueling DQN and a custom environment

Homework Assignment 1 (Deep Q-Network): the DQN of Lab 3 extended with every switch needed by the
assignment, evaluated on CartPole-v1, LunarLander-v3 and ALE/Pong-v5, plus a Gymnasium environment
written from scratch (TreasureGrid).

## Layout

```
dqnlib/              the library (single source of truth, imported by scripts and embedded in notebooks)
  utils.py           seeding, epsilon schedule, moving average, atomic writes
  networks.py        QNetwork (MLP), NatureCNN, DuelingQNetwork, DuelingNatureCNN
  replay.py          replay memory (uint8 ring buffer for Atari, "last B" mode for no-replay)
  agent.py           DQNAgent with every switch (hard/soft target, replay, target net, double, dueling)
  envs.py            the three environments, Atari preprocessing and the hyperparameter table
  train.py           training loop, greedy evaluation, Q-value bias measurement
  experiments.py     the 7 variants, run_one / run_matrix, artifact (de)serialisation
  plotting.py        learning curves with min-max bands, Q-value curves, summary tables
treasure_grid.py     TreasureGrid-v0/v1/v2 (Exercise 3)
scripts/
  run_experiment.py  CLI to run one or several (env, variant, seed) triplets
  calibrate.py       steps/s benchmark used to pick the Pong budget
  aggregate.py       figures + tables for the report
  build_notebooks.py generates the seven deliverable notebooks (library source embedded)
tests/               sanity checks (param counts, target formulas, replay semantics, env API)
runs/                artifacts: <env>/<variant>_s<seed>.{npz,json,pth}
notebooks/           the generated, executed notebooks
report/              figures, tables and report.pdf
```

## Installing

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

On Colab the GPU build of PyTorch is already installed; `uv venv --system-site-packages .venv`
avoids downloading it again.

## The variants

| Variant | What changes |
|---|---|
| `hard` | reference DQN: replay + target network, theta- <- theta every C gradient steps |
| `soft` | theta- <- tau*theta + (1-tau)*theta- after every gradient step, tau = 0.01 |
| `no_replay` | memory keeps only the last `batch_size` transitions, returned in chronological order |
| `no_target` | targets computed with the online network |
| `double` | Double DQN target: theta selects the action, theta- evaluates it |
| `dueling` | dueling head: Q = V + A - mean(A) |
| `dueling_double` | both |

Exercise 1 compares `hard` vs `soft`, `hard` vs `no_replay`, `hard` vs `no_target`.
Exercise 2 compares `hard`, `double`, `dueling`, `dueling_double` (the reference run is shared).

## Running

```bash
# one triplet
python scripts/run_experiment.py --env cartpole --variant dueling_double --seeds 0,1,2
# a whole exercise on one environment
python scripts/run_experiment.py --env lunarlander --suite ex1 --ckpt-every 50000
# the Pong budget decided by the benchmark
python scripts/calibrate.py --env pong --steps 3000
python scripts/run_experiment.py --env pong --suite all --budget steps:1000000 --ckpt-every 50000
# figures and tables for the report
python scripts/aggregate.py
# regenerate the deliverable notebooks (library source is embedded in each of them)
python scripts/build_notebooks.py
```

Existing artifacts are skipped unless `--force` is passed, so a killed session resumes where it
stopped; `--ckpt-every` writes a partial `.npz` + `.pth` every N environment steps.

Every run stores: episode returns/lengths, loss and mean Q(s, a) per gradient step (decimated),
env steps, gradient steps, wall time, the 20 greedy evaluation episodes, the Q-value bias, and the
full agent configuration.

## Protocol

* seeds 0, 1, 2 for every variant; only one setting changes at a time;
* training score = mean return of the last 100 episodes (CartPole, LunarLander) or last 20 games
  (Pong); greedy score = 20 episodes with epsilon = 0 and seeds 100 000 + i, never used in training;
* two variants differ meaningfully only if the gap between their means exceeds the min-max band
  across the three seeds;
* Pong uses the same budget for every variant (see `report/` for the exact value and the
  calibration that justified it).
