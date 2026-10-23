#!/usr/bin/env python
"""Generate the seven deliverable notebooks.

The library is the single source of truth, so each notebook embeds its source (minus the internal
imports, since everything ends up in one namespace). That way a notebook stays readable end-to-end
and runnable on its own, while a bug is still fixed in exactly one place.

    python scripts/build_notebooks.py            # writes notebooks/*.ipynb
"""
from __future__ import annotations

import os
import sys
import textwrap

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "notebooks")
REPO_URL = "https://github.com/AmirBnsL/rl-homework-dqn.git"

MODULES = ["dqnlib/utils.py", "dqnlib/networks.py", "dqnlib/replay.py", "dqnlib/agent.py",
           "dqnlib/envs.py", "dqnlib/train.py", "dqnlib/experiments.py", "dqnlib/plotting.py"]

ENV_MD = {
    "cartpole": """
| Element | CartPole-v1 |
|---|---|
| State | `s ∈ R^4` — cart position, cart velocity, pole angle, pole angular velocity |
| Actions | `a ∈ {0, 1}` — push left (0) or right (1) |
| Reward | +1 for every step the pole stays upright |
| Terminated | the pole falls or the cart leaves the track |
| Truncated | after 500 steps |
| Solved | average return >= 475 over 100 consecutive episodes |
""",
    "lunarlander": """
| Element | LunarLander-v3 |
|---|---|
| State | `s ∈ R^8` — x, y, vx, vy, angle, angular velocity, left / right leg contact |
| Actions | `a ∈ {0, 1, 2, 3}` — do nothing, fire left engine, fire main engine, fire right engine |
| Reward | shaped: closer / slower / more level is better, +10 per leg on the ground, -0.3 per main-engine frame, -0.03 per side-engine frame, -100 crash, +100 safe landing |
| Terminated | the lander crashes, leaves the screen or comes to rest |
| Truncated | after 1000 steps |
| Solved | average return >= 200 over 100 episodes |
""",
    "pong": """
| Element | ALE/Pong-v5 |
|---|---|
| Raw observation | RGB frame 210 x 160 x 3 at 60 fps |
| State | stack of the last 4 preprocessed frames, `uint8`, shape `(4, 84, 84)` |
| Actions | 6 — NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE |
| Reward | +1 when the agent scores, -1 when the opponent scores |
| Terminated | one player reaches 21 points |
| Solved (our convention) | average return >= 18 over the last 20 games |

Preprocessing (Mnih et al., 2015), built from the Gymnasium wrappers:
frame skip 4 with max-pooling of the last two frames, grayscale 84 x 84, up to 30 random no-ops at
reset, stack of the last 4 frames, no sticky actions. Frames stay `uint8` in the replay memory
(56 KB per transition instead of 226 KB) and are divided by 255 inside the network.
""",
}

TREASURE_MD = """
| Element | TreasureGrid |
|---|---|
| World | `N x N` grid (N = 6) with walls, 2 traps and 1 treasure |
| Actions | `a ∈ {0, 1, 2, 3}` — up, down, left, right (moving into a wall keeps the agent in place) |
| Terminated | the agent reaches the treasure (+1) or steps on a trap (-1) |
| Truncated | after `4 N^2 = 144` steps |
| Reset | random free start cell; fixed treasure (v0, v1) or random treasure (v2) |
| Baselines | uniform random agent and optimal agent (BFS shortest path) |

The walls and the treasure are generated once (`layout_seed=0`), so every agent, variant and seed
faces the same map and only the start cell changes between episodes.
"""

VARIANTS_MD = {
    "ex1": """
| Variant | What changes compared with the reference |
|---|---|
| `hard` (reference) | replay memory (100 000) + target network `θ⁻ ← θ` every `C` gradient steps |
| `soft` | `θ⁻ ← τθ + (1 − τ)θ⁻` after every gradient step, `τ = 0.01` |
| `no_replay` | the memory keeps only the last `B` transitions, sampled in chronological order |
| `no_target` | targets computed with the online network: `y = r + γ(1−terminated)·maxₐ′Q(s′,a′;θ)` |
""",
    "ex2": """
| Agent | Target `y` | Network |
|---|---|---|
| `hard` (reference) | `r + γ(1−d)·maxₐ′ Q(s′, a′; θ⁻)` | standard head |
| `double` | `r + γ(1−d)·Q(s′, argmaxₐ′ Q(s′, a′; θ); θ⁻)` | standard head |
| `dueling` | DQN target | dueling head |
| `dueling_double` | Double DQN target | dueling head |
""",
}

BIAS_MD = """
**Bias measurement.** The greedy evaluation episodes give, for every visited state `s`, both the
value predicted by the network (the Q-value of the action it actually takes) and the empirical
discounted return `G_t = Σₖ γᵏ r_{t+k}` obtained from that same state. Overestimation is then

```
bias = mean over evaluation states of Q(s, a) − mean over the same states of G_t
```

A positive bias means the agent promises itself more than it actually gets — the pathology that
Double DQN was introduced to remove.
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str, title: str | None = None):
    source = textwrap.dedent(text).strip("\n")
    if title:
        source = f"# @title {title}\n{source}"
    return nbf.v4.new_code_cell(source)


def _strip_local_imports(source: str) -> str:
    kept = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("from dqnlib", "from .", "import dqnlib",
                                "from __future__ import annotations")):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def library_source(include_treasure: bool = False) -> str:
    parts = ["# ============================================================================",
             "# Library — identical to dqnlib/ in the repository, inlined so that this",
             "# notebook is self-contained. Order matters: helpers, networks, replay, agent,",
             "# environments, training, experiments, plots.",
             "# ============================================================================",
             ""]
    files = list(MODULES)
    if include_treasure:
        files.append("treasure_grid.py")
    for relative in files:
        with open(os.path.join(ROOT, relative)) as handle:
            source = _strip_local_imports(handle.read())
        parts += [f"# ----------------------------------------------------------------------------",
                  f"# {relative}", "# ----------------------------------------------------------------------------",
                  source, ""]
    return "\n".join(parts)


def setup_cell(env_name: str | None):
    return code(f'''
        # @title 1. Setup — run this cell first
        import os, sys, json, shutil, subprocess

        REPO_URL = "{REPO_URL}"
        ENV_NAME = {env_name!r}

        # In Colab, fetch the repository (it holds `runs/`); in the delivered archive, nothing is cloned.
        root = os.path.abspath(".")
        while not os.path.isdir(os.path.join(root, "runs")) and os.path.dirname(root) != root:
            root = os.path.dirname(root)
        if not os.path.isdir(os.path.join(root, "runs")):
            if not os.path.isdir("rl-homework-dqn"):
                subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
            root = os.path.abspath("rl-homework-dqn")
        os.chdir(root)
        os.makedirs("runs", exist_ok=True)
        os.makedirs("deliverables", exist_ok=True)

        import numpy as np
        import pandas as pd
        import torch
        import gymnasium
        import matplotlib.pyplot as plt
        from IPython.display import display

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"working directory : {{os.getcwd()}}")
        print(f"python {{sys.version.split()[0]}} | torch {{torch.__version__}} | "
              f"gymnasium {{gymnasium.__version__}}")
        print(f"device            : {{DEVICE}}"
              + (f" ({{torch.cuda.get_device_name(0)}})" if DEVICE.type == "cuda" else ""))

        pd.set_option("display.width", 160)
        pd.set_option("display.max_columns", 30)
    ''')


def title_cells(exercise: str, env_name: str, objective: str, extra_md: str = ""):
    spec = _spec(env_name)
    return [
        md(f"""
        # {exercise} — {spec['label']}

        {objective}

        **Protocol.** Every comparison uses the same hyperparameters and the same seeds (0, 1, 2);
        only one setting changes at a time. For each run we report the *training score* (mean return
        of the last {spec['metric_window']} {'games' if env_name == 'pong' else 'episodes'}) and the
        *greedy score* (ε = 0, 20 fresh evaluation episodes with seeds 100 000 + i, never used during
        training). A difference between two variants is only meaningful if it is larger than the
        min–max band across the three seeds.
        {extra_md}
        """),
        md("""
        ## 0. Environment, preprocessing and hyperparameters
        """),
        md(ENV_MD[env_name]),
    ]


def _spec(env_name: str) -> dict:
    spec_file = os.path.join(ROOT, "dqnlib", "envs.py")
    namespace: dict = {}
    with open(spec_file) as handle:
        exec(compile(handle.read(), spec_file, "exec"), namespace)
    return namespace["ENV_SPECS"][env_name]


def env_cells(env_name: str, variants, pairs):
    spec = _spec(env_name)
    x_key = "steps" if env_name == "pong" else "episode"
    return [
        code(f'''
        # @title 2. The environment and the reference hyperparameters
        spec = ENV_SPECS[ENV_NAME]
        env = make_env(ENV_NAME)
        state, info = env.reset(seed=0)

        print(f"{{spec['label']}}: observation {{env.observation_space}}")
        print(f"  first state      : shape {{np.asarray(state).shape}}, dtype {{np.asarray(state).dtype}}")
        print(f"  action space     : {{env.action_space}}")
        print(f"  truncation       : {{spec['max_episode_steps']}} steps")
        env.close()

        agent0 = build_agent(make_env(ENV_NAME), ENV_NAME, "hard", seed=0, device=DEVICE)
        cfg = agent_kwargs(ENV_NAME, "hard")
        architecture = ("MLP " + " -> ".join([str(np.prod(spec['obs_shape'])), str(cfg['hidden_dim']),
                                              str(cfg['hidden_dim']), str(spec['n_actions'])])
                        if cfg["net"] == "mlp" else "Nature CNN (3 conv + FC 512)")
        rows = [
            ("network", architecture),
            ("parameters", f"{{agent0.n_parameters:,}}"),
            ("optimiser / lr", f"Adam, {{cfg['lr']}}"),
            ("gamma", cfg["gamma"]),
            ("replay buffer", f"{{cfg['buffer_size']:,}}"),
            ("batch size", cfg["batch_size"]),
            ("learn every", f"{{cfg['learn_every']}} env step(s)"),
            ("learning_starts", cfg["learning_starts"]),
            ("target update", f"hard, every C = {{cfg['target_update_every']}} gradient steps"),
            ("epsilon", f"{{spec['eps']['eps_start']}} -> {{spec['eps']['eps_end']}} "
                        f"over {{spec['eps']['eps_decay_steps']:,}} env steps"),
            ("loss", cfg["loss"] + (" (grad norm clipped to 10)" if cfg["grad_clip"] else "")),
            ("training budget", f"{{spec['budget'][1]:,}} {{spec['budget'][0]}}"),
            ("training score", f"mean of the last {{spec['metric_window']}} episodes"),
            ("target score", spec["target_score"]),
        ]
        display(pd.DataFrame(rows, columns=["hyperparameter", "value"]).set_index("hyperparameter"))

        X_KEY = "{x_key}"   # x-axis of the learning curves: episodes or environment steps
        WINDOW = 20         # moving average used in every curve
        VARIANTS = {variants!r}
        PAIRS = {pairs!r}
    '''),
    ]


def sanity_cells(env_name: str, variants: tuple[str, ...]):
    return [
        md("""
        ## 1. Sanity checks

        Four checks before spending compute: the network returns one Q-value per action and has the
        expected number of parameters; the switches change exactly one setting each; the two replay
        modes behave as specified; the target functions match the formulas of the assignment.
        """),
        code('''
        # @title 3. Network shape and parameter count
        env = make_env(ENV_NAME)
        state, _ = env.reset(seed=0)
        batch = np.repeat(np.asarray(state)[None], 8, axis=0)
        if batch.dtype != np.uint8:  # the MLP expects float32; the CNN takes uint8 frames
            batch = batch.astype(np.float32)
        batch = torch.as_tensor(batch, device=DEVICE)
        with torch.no_grad():
            q = agent0.q_net(batch)
        assert q.shape == (8, env.action_space.n), q.shape
        print(f"Q(s, .) shape {tuple(q.shape)} | {agent0.n_parameters:,} parameters")
        env.close()
        '''),
        code('''
        # @title 4. Only one setting changes between two variants
        reference = agent_kwargs(ENV_NAME, "hard")
        for variant in VARIANTS:
            changed = {k: (reference.get(k), agent_kwargs(ENV_NAME, variant).get(k))
                       for k in set(reference) | set(agent_kwargs(ENV_NAME, variant))
                       if reference.get(k) != agent_kwargs(ENV_NAME, variant).get(k)}
            print(f"{variant:15s} {changed if changed else 'identical to the reference'}")
        '''),
        code('''
        # @title 5. Replay: uniform sampling vs the last B transitions in order
        obs_shape = agent0.state_shape
        uniform = ReplayMemory(1_000, state_shape=obs_shape, state_dtype=agent0.state_dtype,
                               device=torch.device("cpu"))
        recent = ReplayMemory(agent0.batch_size, state_shape=obs_shape,
                              state_dtype=agent0.state_dtype, device=torch.device("cpu"),
                              chronological=True)
        for i in range(200):
            fake = np.zeros(obs_shape, dtype=agent0.state_dtype)
            uniform.push(fake, i % agent0.action_dim, 1.0, fake, False)
            if len(recent) < recent.capacity:
                recent.push(fake, i % agent0.action_dim, 1.0, fake, False)
        s, a, r, s2, d = uniform.sample(agent0.batch_size)
        print(f"uniform : states {tuple(s.shape)} {s.dtype}, actions {tuple(a.shape)} {a.dtype}")
        _, actions, _, _, _ = recent.sample(agent0.batch_size)
        print(f"no replay: the last {agent0.batch_size} transitions, in order -> "
              f"actions {actions.tolist()[:10]} ...")
        assert len(uniform) == uniform.capacity and len(recent) == agent0.batch_size
        '''),
        code('''
        # @title 6. Target formulas of the assignment, checked numerically
        batch = uniform.sample(agent0.batch_size)
        states, actions, rewards, next_states, terminateds = batch
        with torch.no_grad():
            hard_target = (rewards + agent0.gamma * (1 - terminateds)
                           * agent0.target_net(next_states).max(dim=1).values)
            assert torch.allclose(agent0.compute_targets(rewards, next_states, terminateds), hard_target)
            print("hard DQN target      : y = r + gamma (1 - terminated) max_a' Q(s', a'; theta-)  OK")

            double_agent = build_agent(make_env(ENV_NAME), ENV_NAME, "double", seed=0, device=DEVICE)
            a_star = double_agent.q_net(next_states).argmax(dim=1, keepdim=True)
            double_target = (rewards + double_agent.gamma * (1 - terminateds)
                             * double_agent.target_net(next_states).gather(1, a_star).squeeze(1))
            assert torch.allclose(double_agent.compute_targets(rewards, next_states, terminateds),
                                  double_target)
            print("Double DQN target    : theta picks a*, theta- evaluates it                   OK")

            dueling_agent = build_agent(make_env(ENV_NAME), ENV_NAME, "dueling", seed=0, device=DEVICE)
            q_dueling = dueling_agent.q_net(states)
            print(f"dueling head         : {q_dueling.shape[1]} Q-values from V + A - mean(A)     OK")
            print(f"parameters: standard {agent0.n_parameters:,} -> dueling "
                  f"{dueling_agent.n_parameters:,}")
        '''),
        code('''
        # @title 7. Exploration schedule
        steps = np.linspace(0, spec["eps"]["eps_decay_steps"] * 1.15, 200)
        eps = [epsilon_at(int(s), **spec["eps"]) for s in steps]
        plt.figure(figsize=(6, 2.6))
        plt.plot(steps, eps)
        plt.xlabel("environment steps"); plt.ylabel("epsilon")
        plt.title("linear epsilon schedule")
        plt.tight_layout(); plt.show()
        print(f"epsilon: {eps[0]:.2f} at step 0 -> {eps[-1]:.2f} after "
              f"{spec['eps']['eps_decay_steps']:,} steps")
        '''),
    ]


def training_cell(variants: tuple[str, ...], env_name: str, budget_expr: str = "None"):
    return code(f'''
        # @title 8. Train {len(variants)} variants x 3 seeds
        RUN_MODE = "auto"    # "auto": reuse runs/ and train only what is missing | "train": redo everything
        BUDGET = {budget_expr}  # None means the reference budget of Part 0

        RESULTS = run_matrix(ENV_NAME, VARIANTS, seeds=SEEDS, budget=BUDGET, out_dir="runs",
                             device=DEVICE, verbose=True, force=(RUN_MODE == "train"))

        summary = pd.DataFrame([
            dict(variant=v, seed=r["seed"], episodes=int(r["n_episodes"]),
                 env_steps=int(r["env_steps"]), updates=int(r["updates"]),
                 train_score=float(np.mean(r["episode_returns"][-spec["metric_window"]:])),
                 greedy=float(np.mean(r["eval_returns"])),
                 minutes=float(r["wall_time"]) / 60)
            for v, runs in RESULTS.items() for r in runs])
        display(summary.round(2))
    ''')


def curves_cell(pairs, env_name: str, title: str):
    return code(f'''
        # @title 9. Learning curves — mean over 3 seeds, band = min–max
        pairs = PAIRS
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (a, b) in zip(axes, pairs):
            runs = {{}}
            for name in (a, b):
                if name in RESULTS:
                    runs[variant_label(name)] = RESULTS[name]
            plot_curves(ax, runs, window=WINDOW, x_key=X_KEY,
                        threshold=spec["target_score"],
                        title=f"{{b}} vs reference" if b != "hard" else "reference")
        axes[0].set_ylabel("episode return")
        fig.suptitle("{title}: " + spec["label"] + " (20-episode moving average)", y=1.02)
        plt.tight_layout(); plt.show()
    ''')


def q_cell(pairs, env_name: str):
    reference = _spec(env_name)["q_reference"]
    ref_txt = (f'ax.axhline({reference}, ls="--", c="k", lw=1, label="1/(1-gamma)")'
               if reference else "pass")
    return code(f'''
        # @title 10. Predicted Q-values during training
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (a, b) in zip(axes, pairs):
            runs = {{}}
            for name in (a, b):
                if name in RESULTS:
                    runs[variant_label(name)] = RESULTS[name]
            plot_q_curves(ax, runs, title=f"{{b}} vs reference")
            {ref_txt}
        axes[0].set_ylabel("mean of Q(s, a; theta) over the minibatch")
        fig.suptitle(spec["label"] + " — predicted Q-values (mean of 3 seeds, band = min–max)", y=1.02)
        plt.tight_layout(); plt.show()
    ''')


def eval_cells(env_name: str):
    return [
        md("""
        ## 6. Greedy evaluation

        Every saved network plays 20 new episodes with ε = 0 and seeds 100 000 + i, which were never
        used during training.
        """),
        code('''
        # @title 11. Summary table (training score, greedy score, band across seeds)
        table = summary_table(RESULTS, ENV_NAME)
        display(table.round(2))
        per_seed = pd.DataFrame([
            dict(variant=v, seed=r["seed"], greedy=float(np.mean(r["eval_returns"])),
                 std=float(np.std(r["eval_returns"])))
            for v, runs in RESULTS.items() for r in runs])
        display(per_seed.pivot(index="variant", columns="seed", values="greedy").round(1))
        '''),
        code('''
        # @title 12. Is a difference larger than the min–max band across seeds?
        rows = []
        for a, b in PAIRS:
            if a in RESULTS and b in RESULTS:
                c = compare(RESULTS[a], RESULTS[b])
                rows.append(dict(comparison=f"{a} vs {b}", mean_a=c["mean_a"], mean_b=c["mean_b"],
                                 difference=c["difference"], min_max_band=c["band"],
                                 significant=c["significant"]))
        display(pd.DataFrame(rows).round(2))
        '''),
        code('''
        # @title 13. Strip plot of the 20 greedy evaluation episodes
        fig, axes = plt.subplots(len(VARIANTS), 1, figsize=(9, 1.5 * len(VARIANTS)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, variant in zip(axes, VARIANTS):
            returns = np.concatenate([r["eval_returns"] for r in RESULTS[variant]])
            strip_plot(ax, returns, f"{variant} ({len(RESULTS[variant])} seeds x 20 episodes)",
                       spec["target_score"], color=f"C{VARIANTS.index(variant)}")
        axes[0].set_title(f"{spec['label']} — greedy evaluation, epsilon = 0")
        plt.tight_layout(); plt.show()
        '''),
    ]


def save_cells(exercise: str, env_name: str):
    return [
        md("""
        ## 7. Saved networks

        The deliverable requires the network of the best seed of every agent.
        """),
        code(f'''
        # @title 14. Save the best seed of each variant
        saved = []
        for variant, runs in RESULTS.items():
            best = max(runs, key=lambda r: float(np.mean(r["eval_returns"])))
            destination = (f"deliverables/{exercise}_{{ENV_NAME}}_{{variant}}_seed{{best['seed']}}.pth")
            shutil.copy(best["pth_path"], destination)
            saved.append(dict(variant=variant, seed=best["seed"],
                              greedy=float(np.mean(best["eval_returns"])),
                              train_score=float(np.mean(best["episode_returns"][-spec["metric_window"]:])),
                              file=destination))
        display(pd.DataFrame(saved).round(2))
        ''' if exercise != "EX3" else '''
        # @title 14. Save the best seed of each (variant, agent)
        saved = []
        for env_variant, runs_by_agent in ALL_RESULTS.items():
            for agent, runs in runs_by_agent.items():
                best = max(runs, key=lambda r: float(np.mean(r["eval_returns"])))
                destination = f"deliverables/EX3_{env_variant}_{agent}_seed{best['seed']}.pth"
                shutil.copy(best["pth_path"], destination)
                saved.append(dict(env_variant=env_variant, agent=agent, seed=best["seed"],
                                  greedy=float(np.mean(best["eval_returns"])),
                                  success=float(np.nanmean(best["eval_success"])), file=destination))
        display(pd.DataFrame(saved).round(2))
        '''),
    ]


def conclusion_cell(env_name: str, exercise: str):
    spec = _spec(env_name)
    return [
        md(f"""
        ## 8. Conclusion — {spec['label']}

        (Filled in from the numbers above; the text below is generated when the cell runs.)
        """),
        code(f'''
        # @title 15. Data-driven conclusion
        rows = []
        for variant, runs in RESULTS.items():
            greedy = [float(np.mean(r["eval_returns"])) for r in runs]
            train = [float(np.mean(r["episode_returns"][-spec["metric_window"]:])) for r in runs]
            rows.append((variant, np.mean(train), np.mean(greedy), np.min(greedy), np.max(greedy)))
        frame = pd.DataFrame(rows, columns=["variant", "train", "greedy", "min", "max"]).set_index("variant")
        reference = frame.loc["hard"] if "hard" in frame.index else frame.iloc[0]
        print(f"{{spec['label']}} — training score / greedy score (mean of 3 seeds)")
        for variant, row in frame.iterrows():
            delta = row.greedy - reference.greedy
            print(f"  {{variant:15s}} {{row.train:7.1f}} / {{row.greedy:7.1f}}  "
                  f"({{'+' if delta >= 0 else ''}}{{delta:6.1f}} vs reference, band "
                  f"{{row['max'] - row['min']:.1f}})")
        best = frame.greedy.idxmax()
        print(f"\\nbest greedy policy: {{best}} ({{frame.loc[best, 'greedy']:.1f}}); "
              f"target {{spec['target_score']}}")
        '''),
    ]


# --------------------------------------------------------------------------- Exercise 1
def ex1_notebook(env_name: str):
    spec = _spec(env_name)
    pairs = [("hard", "soft"), ("hard", "no_replay"), ("hard", "no_target")]
    cells = title_cells("Exercise 1 — Ablation study of DQN", env_name,
                        "Which ingredient of DQN actually matters? We train the reference DQN, a soft-target "
                        "variant, a no-replay variant and a no-target-network variant, and compare their "
                        "learning curves and predicted Q-values.")
    cells += [setup_cell(env_name), code(library_source(), "3. Library (inlined)")]
    cells += env_cells(env_name, ("hard", "soft", "no_replay", "no_target"), pairs)
    cells += sanity_cells(env_name, ("hard", "soft", "no_replay", "no_target"))
    cells += [md("## 2. The four variants\n" + VARIANTS_MD["ex1"])]
    cells += [training_cell(("hard", "soft", "no_replay", "no_target"), env_name)]
    cells += [md("""
        ## 3. Learning curves

        Each curve is the 20-episode moving average of the return, averaged over the three seeds; the
        shaded band is the min–max envelope across seeds. Two variants differ meaningfully only when
        the gap between their means exceeds that band.
        """)]
    cells += [curves_cell(pairs, env_name, "DQN ablation")]
    cells += [md("""
        ## 4. Predicted Q-values

        `q_means` stores, at every gradient step, the mean of `Q(sᵢ, aᵢ; θ)` over the minibatch. For
        CartPole the true value of a good policy is bounded by `1/(1−γ) = 100`, drawn as a dashed
        line: a curve that shoots above it is overestimating.
        """)]
    cells += [q_cell(pairs, env_name)]
    cells += eval_cells(env_name)
    cells += save_cells("EX1", env_name)
    cells += conclusion_cell(env_name, "ex1")
    return cells, f"EX1_Ablation_{env_name}"


# --------------------------------------------------------------------------- Exercise 2
def ex2_notebook(env_name: str):
    pairs = [("hard", "double"), ("hard", "dueling"), ("hard", "dueling_double")]
    cells = title_cells("Exercise 2 — Double DQN and Dueling DQN", env_name,
                        "Two classic improvements, evaluated one at a time: the Double DQN target removes "
                        "the max-operator bias, the dueling head separates the value of the state from the "
                        "advantage of each action. Four agents share everything except the target and the "
                        "head.")
    cells += [setup_cell(env_name), code(library_source(), "3. Library (inlined)")]
    cells += env_cells(env_name, ("hard", "double", "dueling", "dueling_double"), pairs)
    cells += sanity_cells(env_name, ("hard", "double", "dueling", "dueling_double"))
    cells += [md("## 2. The four agents\n" + VARIANTS_MD["ex2"] + BIAS_MD)]
    cells += [training_cell(("hard", "double", "dueling", "dueling_double"), env_name)]
    cells += [md("## 3. Learning curves")]
    cells += [curves_cell(pairs, env_name, "Double / Dueling DQN")]
    cells += [md("## 4. Predicted Q-values during training")]
    cells += [q_cell(pairs, env_name)]
    cells += [md("""
        ## 5. Overestimation bias

        For every state visited by the greedy policy we compare the value predicted by the network with
        the discounted return actually obtained from that state (20 evaluation episodes, seeds
        100 000 + i).
        """),
        code('''
        # @title 11. Bias: predicted Q(s, a) minus the empirical discounted return
        rows = []
        for variant, runs in RESULTS.items():
            for run in runs:
                rows.append(dict(variant=variant, seed=run["seed"],
                                 q_pred=float(np.mean(run["eval_q_pred"])),
                                 mc_return=float(np.mean(run["eval_mc_return"])),
                                 bias=float(run["bias"])))
        bias = pd.DataFrame(rows)
        display(bias.round(2))
        aggregated = bias.groupby("variant").agg(q_pred=("q_pred", "mean"),
                                                 mc_return=("mc_return", "mean"),
                                                 bias=("bias", "mean"),
                                                 bias_min=("bias", "min"),
                                                 bias_max=("bias", "max"))
        display(aggregated.round(2))
        '''),
        code('''
        # @title 12. Bias per agent (bar = mean over seeds, whiskers = min–max)
        order = [v for v in VARIANTS if v in aggregated.index]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        means = aggregated.loc[order, "bias"]
        lower = means - aggregated.loc[order, "bias_min"]
        upper = aggregated.loc[order, "bias_max"] - means
        ax.bar(range(len(order)), means, yerr=[lower, upper], capsize=4,
               color=[f"C{VARIANTS.index(v)}" for v in order])
        ax.axhline(0, c="k", lw=1)
        ax.set_xticks(range(len(order)), order)
        ax.set(ylabel="Q(s, a) - discounted return", title="overestimation bias on the greedy evaluation")
        plt.tight_layout(); plt.show()
        '''),
    ]
    cells += eval_cells(env_name)
    cells += save_cells("EX2", env_name)
    cells += conclusion_cell(env_name, "ex2")
    return cells, f"EX2_DDQN_Dueling_{env_name}"


# --------------------------------------------------------------------------- Exercise 3
def ex3_notebook():
    treasure_spec = _spec("treasure_v0")
    cells = [
        md("""
        # Exercise 3 — TreasureGrid, an environment written from scratch

        The three tasks of Exercises 1–2 come from Gymnasium. Here we design our own: a small
        `N x N` grid world with walls, two traps and a treasure, registered as a Gymnasium
        environment with three versions that change the **state representation**, the **reward
        function** and the **dynamics**.

        The DQN, Double DQN and Dueling DQN agents of Exercises 1–2 are used **without any
        modification**: only `env.observation_space` and `env.action_space` change.
        """),
        md("## 0. Rules of the world\n" + TREASURE_MD),
        setup_cell(None),
        code(library_source(include_treasure=True), "3. Library + treasure_grid.py (inlined)"),
        md("""
        ## 1. The three versions

        | Version | State representation | Reward | Dynamics |
        |---|---|---|---|
        | `TreasureGrid-v0` | 4 one-hot planes of the whole grid (agent, treasure, traps, walls), 144 floats | sparse: +1 treasure, −1 trap, −0.01 per step | two static traps |
        | `TreasureGrid-v1` | 7 coordinates: agent, treasure, each trap, normalised distance | potential-based shaping `F = γΦ(s′) − Φ(s)` with `Φ(s) = −dist(s, treasure)/N`, plus −0.01 per step | two static traps |
        | `TreasureGrid-v2` | egocentric 5 × 5 patch, 5 channels (agent, treasure, trap, wall, outside), 125 floats | sparse: +1 treasure, −1 trap, no step penalty | two traps that move by one random cell after every action, treasure position random at each reset |

        The shaping of `v1` is potential-based, so the optimal policy is unchanged: only the
        difficulty of *learning* it changes.
        """),
        code('''
        # @title 4. Gymnasium API check + observation spaces
        from gymnasium.utils.env_checker import check_env
        for version in ("TreasureGrid-v0", "TreasureGrid-v1", "TreasureGrid-v2"):
            env = gym.make(version)
            check_env(env.unwrapped, skip_render_check=True)
            obs, info = env.reset(seed=0)
            print(f"{version}: obs {env.observation_space.shape} {env.observation_space.dtype} | "
                  f"actions {env.action_space.n} | optimal BFS distance {info['optimal_distance']}")
            env.close()
        print("check_env passed for the three versions")
        '''),
        code('''
        # @title 5. The map (same walls and treasure for every agent and every seed)
        env = TreasureGridEnv(size=6, variant="v0")
        obs, info = env.reset(seed=0)
        img = env.render()
        plt.figure(figsize=(3.2, 3.2))
        plt.imshow(img); plt.axis("off")
        plt.title("TreasureGrid: walls, 2 traps (red), treasure (yellow), agent (blue)", fontsize=8)
        plt.tight_layout(); plt.show()
        print("free cells:", len(env.free_cells), "| walls:", len(env.walls),
              "| optimal distance from this start:", info["optimal_distance"])
        '''),
        code('''
        # @title 6. Baselines: uniform random agent vs optimal BFS agent
        rows = []
        for variant in ("v0", "v1", "v2"):
            env = TreasureGridEnv(variant=variant)
            random_runs = [random_policy_episode(env, seed=500 + i) for i in range(50)]
            optimal_runs = [bfs_policy_episode(env, seed=500 + i) for i in range(50)]
            rows.append(dict(version=variant,
                             random_success=np.mean([r["success"] for r in random_runs]),
                             random_length=np.mean([r["length"] for r in random_runs]),
                             bfs_success=np.mean([r["success"] for r in optimal_runs]),
                             bfs_length=np.mean([r["length"] for r in optimal_runs]),
                             bfs_optimal=np.mean([r["optimal"] for r in optimal_runs])))
        display(pd.DataFrame(rows).round(2))
        '''),
        md("""
        ## 2. Experimental protocol

        Three versions × three agents (`hard`, `double`, `dueling`) × three seeds = 27 runs, with the
        CartPole hyperparameters of Part 0 (MLP 64, Adam 5e-4, buffer 100 000, batch 64, learn every
        step, target update every 100 gradient steps, ε 1.0 → 0.05 over 10 000 steps) and a budget of
        400 episodes. The greedy evaluation uses 20 episodes with seeds 100 000 + i.
        """),
        code('''
        # @title 7. Train 3 versions x 3 agents x 3 seeds
        RUN_MODE = "auto"
        ALL_RESULTS = {}
        for version, env_key in (("v0", "treasure_v0"), ("v1", "treasure_v1"), ("v2", "treasure_v2")):
            ENV_NAME = env_key
            spec = ENV_SPECS[ENV_NAME]
            print(f"\\n===== {version} ({spec['label']}) =====", flush=True)
            RESULTS = run_matrix(ENV_NAME, ("hard", "double", "dueling"), seeds=SEEDS,
                                 out_dir="runs", device=DEVICE, verbose=True,
                                 force=(RUN_MODE == "train"))
            ALL_RESULTS[version] = RESULTS
        '''),
        md("""
        ## 3. Learning curves per version

        One panel per version; the three lines are the three agents (mean over seeds, band = min–max).
        """),
        code('''
        # @title 8. Learning curves
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
        for ax, (version, runs) in zip(axes, ALL_RESULTS.items()):
            plot_curves(ax, {variant_label(v): r for v, r in runs.items()},
                        window=20, x_key="episode", title=f"TreasureGrid-{version}")
        axes[0].set_ylabel("episode return")
        fig.suptitle("TreasureGrid — DQN vs Double DQN vs Dueling DQN (3 seeds)", y=1.02)
        plt.tight_layout(); plt.show()
        '''),
        code('''
        # @title 9. Greedy return, success rate and episode length vs the BFS optimum
        rows = []
        for version, runs in ALL_RESULTS.items():
            for agent, seeds in runs.items():
                rows.append(dict(
                    version=version, agent=agent,
                    train_score=np.mean([np.mean(r["episode_returns"][-20:]) for r in seeds]),
                    greedy=np.mean([np.mean(r["eval_returns"]) for r in seeds]),
                    success=np.nanmean([np.nanmean(r["eval_success"]) for r in seeds]),
                    length=np.mean([np.mean(r["eval_lengths"]) for r in seeds]),
                    bfs_optimal=np.nanmean([np.nanmean(r["eval_optimal"]) for r in seeds])))
        table = pd.DataFrame(rows)
        table["length_over_bfs"] = table["length"] / table["bfs_optimal"]
        display(table.round(2))
        print("\\nBFS optimum (mean shortest path from the random start cells):")
        print(table.groupby("version")["bfs_optimal"].mean().round(2))
        '''),
        md("""
        ## 4. Qualitative check

        A greedy episode of the best agent of each version, rendered from the same start cell
        (`seed=100000`, the first evaluation episode).
        """),
        code('''
        # @title 10. One greedy episode with the best agent of each version
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, (version, runs) in zip(axes, ALL_RESULTS.items()):
            best_runs = max(runs.values(), key=lambda seeds: np.mean([np.mean(r["eval_returns"]) for r in seeds]))
            best = max(best_runs, key=lambda r: float(np.mean(r["eval_returns"])))
            env_key = f"treasure_{version}"
            env = make_env(env_key)
            agent = build_agent(env, env_key, best["variant"], seed=best["seed"], device=DEVICE)
            agent.load(best["pth_path"])
            state, info = env.reset(seed=100_000)
            frames = [env.render()]
            done, total = False, 0.0
            while not done:
                state, reward, terminated, truncated, info = env.step(agent.greedy_action(state))
                total += reward
                frames.append(env.render())
                done = terminated or truncated
            ax.imshow(frames[len(frames) // 2]); ax.axis("off")
            ax.set_title(f"{version}: return {total:.2f}\\nsuccess={info['success']}, "
                         f"{info['steps']} steps (BFS {info['optimal_distance']})", fontsize=8)
            env.close()
        plt.tight_layout(); plt.show()
        '''),
    ]
    cells += save_cells("EX3", None)
    cells += [
        md("## 5. Conclusion"),
        code('''
        # @title 12. Data-driven conclusion
        for version, runs in ALL_RESULTS.items():
            print(f"TreasureGrid-{version}")
            for agent, seeds in runs.items():
                success = np.nanmean([np.nanmean(r["eval_success"]) for r in seeds])
                greedy = np.mean([np.mean(r["eval_returns"]) for r in seeds])
                length = np.mean([np.mean(r["eval_lengths"]) for r in seeds])
                optimal = np.nanmean([np.nanmean(r["eval_optimal"]) for r in seeds])
                print(f"  {agent:10s} success {success:5.1%} | greedy {greedy:6.2f} | "
                      f"length {length:5.1f} vs BFS {optimal:4.1f}")
            print()
        '''),
    ]
    return cells, "EX3_CustomEnv"


def write_notebook(cells, filename: str) -> str:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{filename}.ipynb")
    with open(path, "w") as handle:
        nbf.write(notebook, handle)
    return path


def main() -> None:
    written = []
    for env_name in ("cartpole", "lunarlander", "pong"):
        cells, name = ex1_notebook(env_name)
        written.append(write_notebook(cells, name))
    for env_name in ("cartpole", "lunarlander", "pong"):
        cells, name = ex2_notebook(env_name)
        written.append(write_notebook(cells, name))
    cells, name = ex3_notebook()
    written.append(write_notebook(cells, name))
    for path in written:
        print(os.path.relpath(path, ROOT), f"({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
