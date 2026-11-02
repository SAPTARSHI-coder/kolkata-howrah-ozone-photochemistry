#!/bin/bash
# Standalone execution wrapper for Paper_2 proof script
PYTHON_EXE="/Users/saptarshisadhu/miniconda3/envs/EDA/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Executing recompute_paper2_distributions.py..."
$PYTHON_EXE "recompute_paper2_distributions.py"
echo "Execution finished with status: $?"
