#!/bin/bash
# Standalone execution wrapper for reproducibility pipeline
PYTHON_EXE="/Users/saptarshisadhu/miniconda3/envs/EDA/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Executing fit_statistical_distributions.py..."
$PYTHON_EXE "fit_statistical_distributions.py"

echo "Executing evaluate_ml_transferability.py..."
$PYTHON_EXE "evaluate_ml_transferability.py"

echo "Execution finished with status: $?"
