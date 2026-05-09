#!/bin/bash
# Experiment 1: Identity Direction Anchoring (IDA)
# Extracts v_id from seed narratives, trains with dual loss
set -e

echo "=== Experiment 1: Identity Direction Anchoring ==="

# Sweep lambda values
for lambda in 0.01 0.05 0.1 0.2 0.5 1.0; do
    echo "--- Training with lambda=$lambda ---"
    python -m src.main experiment=experiment1 \
        training.learning_rate=2e-4 \
        training.max_steps=500 \
        config.lambda=$lambda \
        training.output_dir="data/results/experiment1/lambda_${lambda}"
done

echo "Done. See data/results/experiment1/"
