#!/usr/bin/env python
"""Execute notebooks in place, keeping their outputs (the deliverable wants executed notebooks).

    python scripts/execute_notebooks.py notebooks/EX1_Ablation_cartpole.ipynb
    python scripts/execute_notebooks.py notebooks/*.ipynb --timeout 7200

The kernel runs with the notebook's own directory as working directory; every notebook's setup cell
walks up to the repository root, so `runs/` is found from anywhere.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import nbformat
from nbclient import NotebookClient


def execute(path: str, timeout: int, kernel: str) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout, kernel_name=kernel,
                            resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}},
                            allow_errors=False)
    started = time.time()
    client.execute()
    nbformat.write(notebook, path)
    print(f"executed {path} in {time.time() - started:.0f}s "
          f"({len(notebook.cells)} cells, "
          f"{sum(len(c.get('outputs', [])) for c in notebook.cells if c.cell_type == 'code')} outputs)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+")
    parser.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    for path in args.notebooks:
        if not os.path.exists(path):
            sys.exit(f"no such notebook: {path}")
        execute(path, args.timeout, args.kernel)


if __name__ == "__main__":
    main()
