#!/usr/bin/env bash
# Pull the run artifacts off a Colab session into the local repository.
#
# Colab recycles sessions without warning; this script is meant to be run in the background of the
# machine driving the experiment, so at most the last interval of training is lost.
#
#   scripts/sync_from_colab.sh t4 600      # session name, interval in seconds
#
set -u
SESSION="${1:-t4}"
INTERVAL="${2:-600}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="/tmp/opencode/sync_${SESSION}"
mkdir -p "$STAGE"

while true; do
  if colab status -s "$SESSION" >/dev/null 2>&1; then
    cat > "$STAGE/pack.py" <<'PY'
import subprocess
commands = [
    "cd /content/rl-homework-dqn && tar czf /content/sync_runs.tgz runs deliverables logs 2>/dev/null",
]
for command in commands:
    subprocess.run(["bash", "-lc", command], capture_output=True)
print("packed", flush=True)
PY
    if colab exec -s "$SESSION" -f "$STAGE/pack.py" --timeout 300 >/dev/null 2>&1; then
      if colab download -s "$SESSION" /content/sync_runs.tgz "$STAGE/sync_runs.tgz" >/dev/null 2>&1; then
        tar xzf "$STAGE/sync_runs.tgz" -C "$ROOT" 2>/dev/null
        echo "$(date +%H:%M:%S) synced $SESSION: $(find "$ROOT/runs" -name '*.npz' | wc -l) run files"
      else
        echo "$(date +%H:%M:%S) download failed for $SESSION"
      fi
    else
      echo "$(date +%H:%M:%S) exec failed for $SESSION"
    fi
  else
    echo "$(date +%H:%M:%S) session $SESSION not reachable"
  fi
  sleep "$INTERVAL"
done
