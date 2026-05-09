#!/bin/bash
# Experiment 4: Personality Crystallization Curve Measurement
# Pure observation — no training, just probing Pythia checkpoints
set -e

echo "=== Experiment 4: Personality Crystallization Curves ==="

python -m src.main experiment=experiment4 \
    model.name="EleutherAI/pythia-160m" \
    checkpoint.step_interval=5000 \
    output.results_dir="data/results/experiment4"

echo "Done. See data/results/experiment4/ for plots."
