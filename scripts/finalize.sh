#!/usr/bin/env bash
# One command to produce the deliverables from the artifacts in runs/.
#
#   scripts/finalize.sh "NOM_Prenom"
#
# Order matters: build_notebooks.py rewrites the .ipynb files (which drops any output), so every
# notebook is executed afterwards, and the report is rebuilt once the figures are up to date.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-NOM_Prenom}"
PY="${PY:-.venv/bin/python}"

echo "== 1/6 tests =="
$PY tests/test_sanity.py | tail -1
$PY tests/test_treasure.py | tail -1

echo "== 2/6 notebooks =="
$PY scripts/build_notebooks.py

echo "== 3/6 execute =="
$PY scripts/execute_notebooks.py notebooks/*.ipynb

echo "== 4/6 figures and tables =="
$PY scripts/aggregate.py

echo "== 5/6 report and video =="
$PY scripts/build_report.py --author "$NAME"
$PY scripts/make_video.py --game both --scale 2 --fps 20 --hold 3 || echo "video skipped"

echo "== 6/6 archive =="
$PY scripts/package.py --name "$NAME"
