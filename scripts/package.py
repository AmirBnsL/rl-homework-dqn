#!/usr/bin/env python
"""Assemble the deliverable archive NOM_Prenom_HW_DQN.zip.

    python scripts/package.py --name "BENSLIMANE_Amir"

Contents: the seven executed notebooks, treasure_grid.py, the best network of every agent, the
report, the optional video, the artifacts (runs/) so that every figure can be regenerated without
training, and the library and scripts used to produce them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTEBOOKS = [
    "notebooks/EX1_Ablation_cartpole.ipynb",
    "notebooks/EX1_Ablation_lunarlander.ipynb",
    "notebooks/EX1_Ablation_pong.ipynb",
    "notebooks/EX2_DDQN_Dueling_cartpole.ipynb",
    "notebooks/EX2_DDQN_Dueling_lunarlander.ipynb",
    "notebooks/EX2_DDQN_Dueling_pong.ipynb",
    "notebooks/EX3_CustomEnv.ipynb",
]
ALWAYS = ["treasure_grid.py", "README.md", "requirements.txt", "report/report.pdf"]
DIRECTORIES = ["dqnlib", "scripts", "tests", "runs", "deliverables"]


def notebook_outputs(path: str) -> int:
    with open(path) as handle:
        notebook = json.load(handle)
    return sum(len(cell.get("outputs", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")


def collect() -> list[str]:
    files, missing = [], []
    for path in NOTEBOOKS:
        absolute = os.path.join(ROOT, path)
        if not os.path.exists(absolute):
            missing.append(path)
            continue
        if notebook_outputs(absolute) == 0:
            print(f"WARNING: {path} has no outputs — execute it before packaging")
        files.append(path)
    for path in ALWAYS:
        if os.path.exists(os.path.join(ROOT, path)):
            files.append(path)
        else:
            missing.append(path)
    for directory in DIRECTORIES:
        base = os.path.join(ROOT, directory)
        if not os.path.isdir(base):
            continue
        for current, _, names in os.walk(base):
            if "__pycache__" in current:
                continue
            for name in names:
                if name.endswith((".pyc", ".tmp", ".tmp.npz")):
                    continue
                files.append(os.path.relpath(os.path.join(current, name), ROOT))
    for optional in ("video.mp4", "video_pong.mp4", "video_treasure_v0.mp4"):
        if os.path.exists(os.path.join(ROOT, optional)):
            files.append(optional)
    if missing:
        print("missing required files:")
        for path in missing:
            print("  ", path)
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="NOM_Prenom", help="NOM_Prenom for the archive name")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    archive = args.out or os.path.join(ROOT, f"{args.name}_HW_DQN.zip")
    files = collect()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(os.path.join(ROOT, path), arcname=path)
    size = os.path.getsize(archive) / 2**20
    print(f"wrote {os.path.relpath(archive, ROOT)} ({size:.1f} MiB, {len(files)} files)")
    deliverables = [f for f in files if f.startswith(("deliverables/", "notebooks/"))]
    print(f"{len(deliverables)} deliverable files (notebooks + networks)")


if __name__ == "__main__":
    main()
