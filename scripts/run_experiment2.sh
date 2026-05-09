#!/bin/bash
# Experiment 2: Self-Consistency RL (DPO)
set -e

echo "=== Experiment 2: Self-Consistency RL ==="

python -m src.main experiment=experiment2 \
    model.name="EleutherAI/pythia-160m" \
    training.learning_rate=1e-4 \
    training.max_steps=500 \
    dpo.beta=0.1 \
    output.results_dir="data/results/experiment2"

echo "Done. See data/results/experiment2/"
