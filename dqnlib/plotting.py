"""Plot helpers shared by the seven notebooks: learning curves with a min-max band across seeds,
Q-value curves, greedy-return strip plots, bias bars and the summary tables of the report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dqnlib.envs import ENV_SPECS
from dqnlib.utils import moving_average


def _common_grid(xs, n_points: int = 400) -> np.ndarray:
    return np.linspace(0.0, min(float(np.max(x)) for x in xs), n_points)


def curve_stats(runs, key: str = "episode_returns", window: int = 20, x_key: str = "episode",
                n_points: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean / min / max over seeds of a moving-average curve, on a grid shared by every seed."""
    curves, xs = [], []
    for run in runs:
        values = moving_average(np.asarray(run[key], dtype=np.float64), window)
        if len(values) == 0:
            continue
        x = (np.arange(1, len(values) + 1) if x_key == "episode"
             else np.asarray(run["episode_steps"][:len(values)], dtype=np.float64))
        curves.append(values)
        xs.append(x)

    grid = _common_grid(xs, n_points)
    aligned = np.stack([np.interp(grid, x, y) for x, y in zip(xs, curves)])
    return grid, aligned.mean(axis=0), aligned


def plot_curves(ax, runs_by_label: dict, window: int = 20, x_key: str = "episode",
                threshold: float | None = None, title: str = "", xlabel: str = "episode",
                ylabel: str = "return") -> None:
    for label, runs in runs_by_label.items():
        grid, mean, aligned = curve_stats(runs, window=window, x_key=x_key)
        ax.plot(grid, mean, lw=1.8, label=label)
        ax.fill_between(grid, aligned.min(axis=0), aligned.max(axis=0), alpha=0.18)
    if threshold is not None:
        ax.axhline(threshold, ls="--", c="k", lw=1, label=f"solved ({threshold:g})")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=8)


def plot_q_curves(ax, runs_by_label: dict, reference: float | None = None, title: str = "") -> None:
    for label, runs in runs_by_label.items():
        strides = [int(run["q_stride"]) for run in runs]
        xs = [np.arange(len(run["q_means"])) * stride for run, stride in zip(runs, strides)]
        grid = _common_grid(xs)
        aligned = np.stack([np.interp(grid, x, np.asarray(run["q_means"], dtype=np.float64))
                            for x, run in zip(xs, runs)])
        ax.plot(grid, aligned.mean(axis=0), lw=1.8, label=label)
        ax.fill_between(grid, aligned.min(axis=0), aligned.max(axis=0), alpha=0.18)
    if reference is not None:
        ax.axhline(reference, ls="--", c="k", lw=1, label="1/(1-gamma)")
    ax.set(xlabel="gradient step", ylabel="mean max_a Q(s, a)", title=title)
    ax.legend(fontsize=8)


def strip_plot(ax, returns, label: str, threshold: float | None = None, color: str = "C0") -> None:
    """One dot per evaluation episode: shaded band = middle 50 %, black bar = median, diamond = mean."""
    returns = np.asarray(returns, dtype=np.float64)
    rng = np.random.default_rng(0)
    q1, median, q3 = np.percentile(returns, [25, 50, 75])
    ax.hlines(0, q1, q3, lw=22, color=color, alpha=0.25)
    ax.scatter(returns, rng.uniform(-0.15, 0.15, len(returns)), s=18, color=color, alpha=0.7)
    ax.plot(median, 0, "|", ms=30, mew=3, c="k")
    ax.plot(returns.mean(), 0, "D", ms=8, mfc="white", mec="k",
            label=f"mean {returns.mean():.1f}")
    if threshold is not None:
        ax.axvline(threshold, c="k", ls="--", lw=1, label=f"solved ({threshold:g})")
    ax.set_yticks([0], [label])
    ax.set(ylim=(-0.5, 0.5), xlabel="episode return")


def summary_table(runs_by_variant: dict, env_name: str) -> pd.DataFrame:
    """One row per variant: training score, greedy score (mean + min-max band across seeds), bias."""
    spec = ENV_SPECS[env_name]
    window, target = spec["metric_window"], spec["target_score"]
    rows = []
    for variant, runs in runs_by_variant.items():
        train_scores = [float(np.mean(run["episode_returns"][-window:])) for run in runs]
        greedy = [float(np.mean(run["eval_returns"])) for run in runs]
        biases = [float(run["bias"]) for run in runs if "bias" in run]
        successes = [float(np.nanmean(run["eval_success"])) for run in runs if "eval_success" in run]
        rows.append(dict(
            variant=variant,
            train_score=np.mean(train_scores),
            greedy_mean=np.mean(greedy),
            greedy_min=np.min(greedy),
            greedy_max=np.max(greedy),
            greedy_std=np.std(greedy),
            solved=bool(target is not None and np.mean(greedy) >= target),
            success_rate=np.mean(successes) if successes else np.nan,
            eval_length=float(np.mean([np.mean(run["eval_lengths"]) for run in runs])),
            bias=np.mean(biases) if biases else np.nan,
            env_steps=float(np.mean([run["env_steps"] for run in runs])),
            minutes=float(np.mean([run["wall_time"] for run in runs]) / 60),
            seeds=len(runs),
        ))
    return pd.DataFrame(rows).set_index("variant")


def compare(runs_a, runs_b, key: str = "eval_returns") -> dict:
    """Difference of the two variants' greedy means compared with the min-max band across seeds."""
    mean_a = float(np.mean([np.mean(run[key]) for run in runs_a]))
    mean_b = float(np.mean([np.mean(run[key]) for run in runs_b]))
    band = max(float(np.ptp([np.mean(run[key]) for run in runs_a])),
               float(np.ptp([np.mean(run[key]) for run in runs_b])))
    return dict(mean_a=mean_a, mean_b=mean_b, difference=mean_a - mean_b, band=band,
                significant=bool(abs(mean_a - mean_b) > band))
