#!/bin/bash
# Experiment 3: Drive-Embedded Training
set -e

echo "=== Experiment 3: Drive-Embedded Training ==="

python -m src.main experiment=experiment3 \
    model.name="EleutherAI/pythia-160m" \
    training.learning_rate=1e-4 \
    training.max_steps=1000 \
    drives.curiosity.weight=0.01 \
    drives.coherence.weight=0.01 \
    drives.empowerment.weight=0.005 \
    output.results_dir="data/results/experiment3"

echo "Done. See data/results/experiment3/"
