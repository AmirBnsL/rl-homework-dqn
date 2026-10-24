#!/usr/bin/env python
"""Build the figures and tables of the report from the artifacts in runs/.

    python scripts/aggregate.py                 # everything that has been run
    python scripts/aggregate.py --env cartpole  # one environment

Figures go to report/figures/, tables to report/tables/ (markdown, ready to paste in the report).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqnlib.envs import ENV_SPECS
from dqnlib.experiments import EX1_VARIANTS, EX2_VARIANTS, SEEDS, load_matrix, variant_label
from dqnlib.plotting import compare, plot_curves, plot_q_curves, summary_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "report", "figures")
TABLES = os.path.join(ROOT, "report", "tables")


def ex1_figure(env_name: str, runs_dir: str) -> str | None:
    results = load_matrix(env_name, EX1_VARIANTS, SEEDS, runs_dir)
    if not complete(results, EX1_VARIANTS):
        return None
    spec = ENV_SPECS[env_name]
    x_key = "steps" if env_name == "pong" else "episode"
    pairs = [("hard", "soft"), ("hard", "no_replay"), ("hard", "no_target")]

    if env_name == "pong":
        # all three comparisons share the same reference: one panel per row reads better and
        # keeps the report within ten pages
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        plot_curves(axes[0], {variant_label(v): r for v, r in results.items()}, window=20,
                    x_key=x_key, threshold=spec["target_score"],
                    title="moving average over the last 20 games")
        plot_q_curves(axes[1], {variant_label(v): r for v, r in results.items()},
                      title="predicted Q-values")
        axes[0].set_ylabel("game score")
        fig.suptitle(f"{spec['label']} — ablation of DQN (mean of 3 seeds, band = min–max)", y=1.02)
        fig.tight_layout()
        path = os.path.join(FIGURES, f"ex1_{env_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    fig, axes = plt.subplots(2, 3, figsize=(16, 6.2))  # rows are scaled independently
    axes[0][1].sharey(axes[0][0])
    axes[0][2].sharey(axes[0][0])
    for ax, (a, b) in zip(axes[0], pairs):
        curves = {variant_label(a): results[a]}
        if b in results:
            curves[variant_label(b)] = results[b]
        plot_curves(ax, curves, window=20, x_key=x_key, threshold=spec["target_score"],
                    title=f"{b} vs reference")
        ax.set_ylabel("episode return")
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    for ax, (a, b) in zip(axes[1], pairs):
        curves = {variant_label(a): results[a]}
        if b in results:
            curves[variant_label(b)] = results[b]
        plot_q_curves(ax, curves, title=f"{b} vs reference")
        if spec["q_reference"]:
            ax.axhline(spec["q_reference"], ls="--", c="k", lw=1, label="1/(1-gamma)")
            ax.legend(fontsize=8)
    for ax in axes[1][1:]:
        ax.set_ylabel("")
    fig.suptitle(f"{spec['label']} — ablation of DQN (mean of 3 seeds, band = min–max)", y=1.0)
    fig.tight_layout()
    path = os.path.join(FIGURES, f"ex1_{env_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


REPORT_COLUMNS = ["train_score", "greedy_mean", "greedy_min", "greedy_max", "bias"]


def complete(results: dict, variants) -> bool:
    """True only when every requested variant ran on every seed (a missing variant is not a result)."""
    return all(variant in results and len(results[variant]) == len(SEEDS) for variant in variants)


def compact(table: pd.DataFrame) -> str:
    """Keep the columns that fit in the report, with readable headers."""
    columns = [c for c in REPORT_COLUMNS if c in table.columns]
    frame = table[columns].round(1)
    frame.columns = [{"train_score": "training score", "greedy_mean": "greedy",
                      "greedy_min": "min seed", "greedy_max": "max seed",
                      "bias": "bias"}[c] for c in columns]
    return frame.to_markdown()


def ex1_table(env_name: str, runs_dir: str) -> str | None:
    results = load_matrix(env_name, EX1_VARIANTS, SEEDS, runs_dir)
    if not complete(results, EX1_VARIANTS):
        return None
    table = summary_table(results, env_name)
    comparisons = []
    for b in ("soft", "no_replay", "no_target"):
        if b in results:
            c = compare(results["hard"], results[b])
            c["difference"] = -c["difference"]  # variant minus reference
            comparisons.append(dict(comparison=f"hard vs {b}", **c))
    lines = [f"## Exercise 1 — {ENV_SPECS[env_name]['label']}", "",
             compact(table), "",
             "Training score = mean of the last episodes, greedy = 20 evaluation episodes "
             "(ε = 0); min/max = band across the three seeds; bias = predicted Q minus the "
             "empirical discounted return.", ""]
    if comparisons:
        frame = pd.DataFrame(comparisons)[["comparison", "difference", "band", "significant"]]
        lines += ["### Is the gap larger than the min–max band across seeds?", "",
                  frame.round(1).to_markdown(index=False), ""]
    path = os.path.join(TABLES, f"ex1_{env_name}.md")
    os.makedirs(TABLES, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


def ex2_figure(env_name: str, runs_dir: str) -> str | None:
    results = load_matrix(env_name, EX2_VARIANTS, SEEDS, runs_dir)
    if not complete(results, EX2_VARIANTS):
        return None
    spec = ENV_SPECS[env_name]
    x_key = "steps" if env_name == "pong" else "episode"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    plot_curves(axes[0][0], {variant_label(v): r for v, r in results.items()}, window=20,
                x_key=x_key, threshold=spec["target_score"], title="learning curves")
    axes[0][0].set_ylabel("episode return")
    plot_q_curves(axes[0][1], {variant_label(v): r for v, r in results.items()},
                  title="predicted Q-values")
    if spec["q_reference"]:
        axes[0][1].axhline(spec["q_reference"], ls="--", c="k", lw=1, label="1/(1-gamma)")
        axes[0][1].legend(fontsize=8)

    rows = [dict(variant=v, seed=r["seed"], bias=float(r["bias"]))
            for v, runs in results.items() for r in runs]
    bias = pd.DataFrame(rows)
    grouped = bias.groupby("variant")["bias"]
    order = [v for v in results if v in grouped.groups]
    means = grouped.mean().loc[order]
    axes[1][0].bar(range(len(order)), means,
                   yerr=[means - grouped.min().loc[order], grouped.max().loc[order] - means],
                   capsize=4, color=[f"C{EX2_VARIANTS.index(v)}" for v in order])
    axes[1][0].axhline(0, c="k", lw=1)
    axes[1][0].set_xticks(range(len(order)), order)
    axes[1][0].set(ylabel="Q(s, a) - discounted return", title="overestimation bias")

    for i, variant in enumerate(order):
        axes[1][1].bar(i, summary_table({variant: results[variant]}, env_name)["greedy_mean"].iloc[0],
                       color=f"C{i}")
    axes[1][1].set_xticks(range(len(order)), order)
    axes[1][1].set(ylabel="greedy return", title="greedy evaluation (20 episodes)")

    fig.suptitle(f"{spec['label']} — Double DQN and Dueling DQN", y=1.0)
    fig.tight_layout()
    path = os.path.join(FIGURES, f"ex2_{env_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def ex2_table(env_name: str, runs_dir: str) -> str | None:
    results = load_matrix(env_name, EX2_VARIANTS, SEEDS, runs_dir)
    if not complete(results, EX2_VARIANTS):
        return None
    table = summary_table(results, env_name)
    limit = ENV_SPECS[env_name]["max_episode_steps"]
    bias = pd.DataFrame([dict(variant=v, seed=r["seed"],
                              q_pred=float(np.mean(r["eval_q_pred"])),
                              mc_return=float(np.mean(r["eval_mc_return"])),
                              bias=float(r["bias"]),
                              truncated=float(np.mean(np.asarray(r["eval_lengths"]) >= limit))
                              if limit else np.nan)
                         for v, runs in results.items() for r in runs])
    lines = [f"## Exercise 2 — {ENV_SPECS[env_name]['label']}", "",
             compact(table), "", "### Overestimation bias", "",
             bias.groupby("variant")[["q_pred", "mc_return", "bias", "truncated"]].mean()
             .reindex([v for v in EX2_VARIANTS if v in bias["variant"].unique()])
             .dropna(axis=1, how="all").round(1).to_markdown(), ""]
    path = os.path.join(TABLES, f"ex2_{env_name}.md")
    os.makedirs(TABLES, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


def ex3_figure_and_table(runs_dir: str) -> list[str]:
    paths = []
    versions = {"v0": "treasure_v0", "v1": "treasure_v1", "v2": "treasure_v2"}
    collected = {}
    for version, env_key in versions.items():
        results = load_matrix(env_key, ("hard", "double", "dueling"), SEEDS, runs_dir)
        if results:
            collected[version] = (env_key, results)
    if not collected:
        return paths

    fig, axes = plt.subplots(1, len(collected), figsize=(5.4 * len(collected), 4))
    axes = np.atleast_1d(axes)
    for ax, (version, (env_key, results)) in zip(axes, collected.items()):
        plot_curves(ax, {variant_label(v): r for v, r in results.items()}, window=20,
                    x_key="episode", title=f"TreasureGrid-{version}")
    axes[0].set_ylabel("episode return")
    fig.suptitle("TreasureGrid — DQN vs Double DQN vs Dueling DQN (3 seeds)", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES, "ex3_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    rows = []
    for version, (env_key, results) in collected.items():
        for agent, runs in results.items():
            rows.append(dict(
                version=version, agent=agent,
                train_score=np.mean([np.mean(r["episode_returns"][-20:]) for r in runs]),
                greedy=np.mean([np.mean(r["eval_returns"]) for r in runs]),
                success=np.nanmean([np.nanmean(r["eval_success"]) for r in runs
                                 if np.isfinite(r["eval_success"]).any()]) if any(
                    np.isfinite(r["eval_success"]).any() for r in runs) else np.nan,
                length=np.mean([np.mean(r["eval_lengths"]) for r in runs]),
                bfs_optimal=np.nanmean([np.nanmean(r["eval_optimal"]) for r in runs])))
    table = pd.DataFrame(rows)
    table["length_over_bfs"] = table["length"] / table["bfs_optimal"]
    path = os.path.join(TABLES, "ex3.md")
    os.makedirs(TABLES, exist_ok=True)
    keep = ["version", "agent", "train_score", "greedy", "success", "length",
            "bfs_optimal", "length_over_bfs"]
    with open(path, "w") as handle:
        handle.write("## Exercise 3 — TreasureGrid\n\n"
                     + table[keep].round(2).to_markdown(index=False) + "\n\n"
                     + "Success = fraction of evaluation episodes reaching the treasure; "
                       "length / BFS = mean episode length divided by the optimal path length.\n")
    paths.append(path)
    return paths


def _drop(path: str) -> None:
    """Remove a stale artifact so an incomplete environment cannot leak into the report."""
    if os.path.exists(path):
        os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None, choices=sorted(ENV_SPECS))
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    args = parser.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    envs = [args.env] if args.env else ["cartpole", "lunarlander", "pong"]
    for env_name in envs:
        for fn in (ex1_figure, ex1_table, ex2_figure, ex2_table):
            path = fn(env_name, args.runs)
            if path:
                print("wrote " + os.path.relpath(path, ROOT))
            else:
                stem = fn.__name__.replace("_figure", "").replace("_table", "")
                _drop(os.path.join(FIGURES if "figure" in fn.__name__ else TABLES,
                                   f"{stem}_{env_name}.png" if "figure" in fn.__name__
                                   else f"{stem}_{env_name}.md"))
                print(f"skip  {env_name} ({fn.__name__})")
    if not args.env:
        for path in ex3_figure_and_table(args.runs):
            print("wrote " + os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
