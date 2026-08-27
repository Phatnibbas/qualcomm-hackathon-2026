#!/bin/sh
# Measure baseline vs optimized inference for both models on the UNO Q itself.
# Emits one JSON object per combination, newline separated.
cd "$(dirname "$0")" || exit 1
REPEAT="${1:-500}"
for runner in edge_runner.py edge_runner_fast.py; do
  for model in edge_model.json challenger_edge_model.json; do
    echo "##### runner=$runner model=$model"
    python3 "$runner" --model "$model" --samples replay_samples.json --repeat "$REPEAT"
  done
done
