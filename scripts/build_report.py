#!/usr/bin/env python
"""Substitute the measured results into report/template.md and build report/report.pdf.

    python scripts/build_report.py --author "NOM Prénom"
    python scripts/build_report.py --author "NOM Prénom" --no-pdf

Placeholders understood by the template:
    {{figure:name:caption}}     report/figures/name.png
    {{table:name}}              report/tables/name.md
    {{verdicts:env:suite}}      bullets comparing every variant with the reference, with the band
    {{baselines}}               random vs BFS on the three TreasureGrid versions
    {{compute}}                 wall-clock and env steps per environment
    {{PONG_BUDGET}}
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqnlib.envs import ENV_SPECS
from dqnlib.experiments import (EX1_VARIANTS, EX2_VARIANTS, SEEDS, load_matrix)
from dqnlib.plotting import compare, summary_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")
FIGURES = os.path.join(REPORT, "figures")
TABLES = os.path.join(REPORT, "tables")
RUNS = os.path.join(ROOT, "runs")


def treasure_verdicts() -> str:
    """Bullets for Exercise 3: per version, which agent wins and by how much."""
    lines = []
    summary = {}
    for version, env_key in (("v0", "treasure_v0"), ("v1", "treasure_v1"), ("v2", "treasure_v2")):
        results = load_matrix(env_key, ("hard", "double", "dueling"), SEEDS, RUNS)
        if not results or any(len(runs) < len(SEEDS) for runs in results.values()):
            continue
        success = {agent: float(np.nanmean([np.nanmean(r["eval_success"]) for r in runs]))
                   for agent, runs in results.items()}
        greedy = {agent: float(np.mean([np.mean(r["eval_returns"]) for r in runs]))
                  for agent, runs in results.items()}
        length = {agent: float(np.mean([np.mean(r["eval_lengths"]) for r in runs]))
                  for agent, runs in results.items()}
        best = max(success, key=success.get)
        summary[version] = (success, greedy, length, best)
        lines.append(
            f"* **TreasureGrid-{version}** — best success rate for `{best}` "
            f"({success[best]:.0%}), greedy {greedy[best]:+.2f}, {length[best]:.0f} steps per episode "
            f"(the three agents: " + ", ".join(f"`{a}` {success[a]:.0%}" for a in success) + ").")
    if summary:
        best_version = max(summary, key=lambda v: max(summary[v][0].values()))
        worst_version = min(summary, key=lambda v: max(summary[v][0].values()))
        lines.append(f"* Across versions, `{best_version}` gives the best agent "
                     f"({max(summary[best_version][0].values()):.0%} success) and `{worst_version}` "
                     f"the worst ({max(summary[worst_version][0].values()):.0%}).")
    return "\n".join(lines) if lines else "_(no run available yet)_"


def verdict_bullets(env_name: str, suite: str) -> str:
    if suite == "ex3":
        return treasure_verdicts()
    variants = EX1_VARIANTS if suite == "ex1" else EX2_VARIANTS
    results = load_matrix(env_name, variants, SEEDS, RUNS)
    if "hard" not in results:
        return "_(no run available yet)_"
    table = summary_table(results, env_name)
    reference = table.loc["hard", "greedy_mean"]
    lines = []
    for variant in variants[1:]:
        if variant not in results:
            continue
        stats = compare(results["hard"], results[variant])
        delta = -stats["difference"]  # variant minus reference
        direction = "higher" if delta > 0 else "lower"
        relative = ("gains" if delta > 0 else "loses") + f" {abs(delta):.1f} points"
        verdict = ("a gap larger than the band across seeds" if stats["significant"]
                   else "a gap inside the band across seeds, so not conclusive")
        lines.append(
            f"* **`{variant}`** — greedy {table.loc[variant, 'greedy_mean']:.1f} against "
            f"{reference:.1f} for the reference: it {relative} ({direction}) and this is "
            f"{verdict} (band {stats['band']:.1f}).")
    if suite == "ex2":
        bias = {v: float(np.mean([r["bias"] for r in runs])) for v, runs in results.items()}
        best = min(bias, key=bias.get)
        lines.append("* **Overestimation bias** — " + ", ".join(
            f"`{v}` {b:+.1f}" for v, b in bias.items()) +
            f"; the least biased agent is `{best}`.")
    return "\n".join(lines)


def glance() -> str:
    """One bullet per environment: best agent of each exercise, greedy score, solved or not."""
    lines = []
    for env_name, spec in ENV_SPECS.items():
        if env_name.startswith("treasure_"):
            continue
        results = load_matrix(env_name, tuple(dict.fromkeys(EX1_VARIANTS + EX2_VARIANTS)), SEEDS, RUNS)
        if not results or any(len(runs) < len(SEEDS) for runs in results.values()):
            continue  # still training: a two-seed glance would be misleading
        table = summary_table(results, env_name)
        best = table["greedy_mean"].idxmax()
        solved = [v for v in table.index if table.loc[v, "solved"]]
        lines.append(
            f"* **{spec['label']}** — best agent `{best}` with a greedy score of "
            f"{table.loc[best, 'greedy_mean']:.1f} (target {spec['target_score']:g}); "
            + (f"solvable variants: {', '.join('`' + v + '`' for v in solved)}."
               if solved else "no variant reaches the target on the mean of three seeds."))
    return "\n".join(lines) if lines else "_(no run available yet)_"


def baselines_table() -> str:
    import treasure_grid  # registers the environments

    rows = []
    for variant in ("v0", "v1", "v2"):
        env = treasure_grid.TreasureGridEnv(variant=variant)
        random_runs = [treasure_grid.random_policy_episode(env, seed=500 + i) for i in range(50)]
        optimal_runs = [treasure_grid.bfs_policy_episode(env, seed=500 + i) for i in range(50)]
        rows.append(dict(version=f"`v{variant[1]}`",
                         random_success=np.mean([r["success"] for r in random_runs]),
                         random_length=np.mean([r["length"] for r in random_runs]),
                         bfs_success=np.mean([r["success"] for r in optimal_runs]),
                         bfs_length=np.mean([r["length"] for r in optimal_runs]),
                         bfs_optimal=np.mean([r["optimal"] for r in optimal_runs])))
    frame = pd.DataFrame(rows)
    markdown = frame.round(2).to_markdown(index=False)
    return ("Baselines over 50 episodes per version (success rate and episode length, compared with"
            " the BFS optimum):\n\n" + markdown)


def compute_table() -> str:
    rows = []
    for env_name in ENV_SPECS:
        results = load_matrix(env_name, EX1_VARIANTS + tuple(v for v in EX2_VARIANTS
                                                             if v not in EX1_VARIANTS), SEEDS, RUNS)
        runs = [run for seeds in results.values() for run in seeds]
        if not runs:
            continue
        rows.append(dict(environment=ENV_SPECS[env_name]["label"], runs=len(runs),
                         mean_minutes=np.mean([r["wall_time"] for r in runs]) / 60,
                         mean_env_steps=np.mean([r["env_steps"] for r in runs]),
                         total_hours=np.sum([r["wall_time"] for r in runs]) / 3600))
    if not rows:
        return "_no run available yet_"
    table = pd.DataFrame(rows).round(2).to_markdown(index=False)
    total = sum(r["total_hours"] for r in rows)
    return f"{table}\n\nTotal training time in this study: {total:.1f} core-hours."


def pong_budget() -> str:
    results = load_matrix("pong", EX1_VARIANTS, SEEDS, RUNS)
    for runs in results.values():
        for run in runs:
            meta = run.get("meta_json", {})
            if meta:
                steps = meta.get("budget") or int(run["env_steps"])
                decay = meta.get("eps", {}).get("eps_decay_steps")
                scaled = ("" if decay == 1_000_000
                          else f" (ε decay scaled to {decay:,} steps to match the reduced horizon)")
                return f"{steps:,} env steps{scaled}"
    return "1 000 000 (reference budget)"


def include_table(name: str) -> str:
    path = os.path.join(TABLES, f"{name}.md")
    if not os.path.exists(path):
        return f"_(table {name} not available yet — run scripts/aggregate.py)_"
    with open(path) as handle:
        text = handle.read().strip()
    lines = [line for line in text.splitlines() if not line.startswith("## ")]
    return "\n".join(lines).strip()


def include_figure(name: str, caption: str) -> str:
    path = os.path.join(FIGURES, f"{name}.png")
    if not os.path.exists(path):
        return f"_(figure {name} not available yet)_"
    return f"![{caption}](figures/{name}.png){{width=74%}}\n"


TEXT_SYMBOLS = {
    "θ⁻": r"$\theta^-$", "θ": r"$\theta$", "γ": r"$\gamma$", "ε": r"$\varepsilon$",
    "τ": r"$\tau$", "Φ": r"$\Phi$", "Δ": r"$\Delta$", "Σ": r"$\Sigma$", "∈": r"$\in$",
    "≈": r"$\approx$", "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\to$", "×": r"$\times$",
    "−": "-", "²": r"$^2$", "·": r"$\cdot$", "ₐ": r"$_a$", "′": r"$'$", "⁻": "",
    "—": "---", "–": "--", "…": "...",
}
CODE_SYMBOLS = {
    "θ⁻": "theta-", "θ": "theta", "γ": "gamma", "ε": "epsilon", "τ": "tau", "Φ": "Phi",
    "Σ": "Sigma", "Δ": "Delta", "∈": "in", "≈": "~", "−": "-", "→": "->", "×": "x",
    "²": "^2", "·": "*", "ₐ": "_a", "′": "'", "⁻": "", "≥": ">=", "≤": "<=",
    "—": "--", "–": "--", "…": "...",
}


def latex_safe(text: str) -> str:
    """XeLaTeX is unavailable on this machine: turn the Unicode of the text into pdflatex-safe LaTeX.

    Inside code (fenced blocks and inline spans) the symbols become ASCII, because a `$...$` inside
    a code span would be printed literally.
    """
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    for i, part in enumerate(parts):
        symbols = CODE_SYMBOLS if part.startswith("`") else TEXT_SYMBOLS
        for source, target in symbols.items():
            part = part.replace(source, target)
        parts[i] = part
    return "".join(parts)


def build(author: str, date: str) -> str:
    with open(os.path.join(REPORT, "template.md")) as handle:
        text = handle.read()

    text = text.replace("__AUTHOR__", author).replace("__DATE__", date)
    text = text.replace("__PONG_BUDGET__", pong_budget())
    text = re.sub(r"\{\{figure:([^:}]+):([^}]*)\}\}",
                  lambda m: include_figure(m.group(1), m.group(2).strip()), text)
    text = re.sub(r"\{\{table:([^}]+)\}\}", lambda m: include_table(m.group(1).strip()), text)
    text = re.sub(r"\{\{verdicts:([^:}]+):([^}]+)\}\}",
                  lambda m: verdict_bullets(m.group(1).strip(), m.group(2).strip()), text)
    text = text.replace("{{baselines}}", baselines_table())
    text = text.replace("{{glance}}", glance())
    text = text.replace("{{compute}}", compute_table())

    text = latex_safe(text)
    out = os.path.join(REPORT, "report.md")
    with open(out, "w") as handle:
        handle.write(text)
    return out


def to_pdf(markdown_path: str) -> str:
    pdf_path = os.path.join(REPORT, "report.pdf")
    command = ["pandoc", os.path.basename(markdown_path), "-o", os.path.basename(pdf_path),
               "--pdf-engine=pdflatex", "-V", "geometry:margin=1.5cm", "-V", "fontsize=10pt",
               "-V", "colorlinks=true", "--resource-path", "."]
    subprocess.run(command, cwd=REPORT, check=True)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default="NOM Prénom")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    markdown_path = build(args.author, args.date)
    print("wrote", os.path.relpath(markdown_path, ROOT))
    if not args.no_pdf:
        print("wrote", os.path.relpath(to_pdf(markdown_path), ROOT))


if __name__ == "__main__":
    main()
