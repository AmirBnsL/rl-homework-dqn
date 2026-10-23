#!/usr/bin/env bash
# Re-run the 27 TreasureGrid runs with the corrected hyperparameters (parallel on the local CPU).
set -u
cd "$(dirname "$0")/.."
rm -rf runs/treasure_v0 runs/treasure_v1 runs/treasure_v2
mkdir -p /tmp/opencode
: > /tmp/opencode/treasure_run.log

for env in treasure_v0 treasure_v1 treasure_v2; do
  for variant in hard double dueling; do
    for seed in 0 1 2; do
      echo "$env $variant $seed"
    done
  done
done | OMP_NUM_THREADS=1 xargs -P 6 -n 3 bash -c '
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_experiment.py \
      --env "$0" --variant "$1" --seeds "$2" --out runs --device cpu \
      >> /tmp/opencode/treasure_run.log 2>&1
  echo "done $0 $1 seed $2"
' 
echo "all treasure runs finished"
