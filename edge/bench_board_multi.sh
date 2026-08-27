#!/bin/sh
# Measure every swept model family on the UNO Q itself, stdlib-only runner.
# One JSON object per model, newline separated.
cd "$(dirname "$0")" || exit 1
REPEAT="${1:-300}"
for f in sweep/*.json; do
  [ -e "$f" ] || continue
  python3 edge_runner_multi.py --model "$f" --samples replay_samples.json --repeat "$REPEAT" 2>&1
done
# the two already-deployed models, through the same runner, for a fair comparison
python3 edge_runner_multi.py --model edge_model.json --samples replay_samples.json --repeat "$REPEAT" --label "deployed_persistence" 2>&1
python3 edge_runner_multi.py --model challenger_edge_model.json --samples replay_samples.json --repeat "$REPEAT" --label "deployed_challenger_mlp" 2>&1
