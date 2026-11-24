#!/usr/bin/env bash
# Reproduce the reference workflow with the configured dataset.
# Usage: bash reproduce.sh [path/to/dataset.yaml]
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${1:-config/datasets/2411.yaml}"

run() {
    echo "==> $*"
    "$PYTHON" -m wind10m --config "$CONFIG" "$@"
}

run estimate-cache
run qc
# Skipped by the entry point when cache.mode is none.
run cache
run train
run verify
run mi
run plot
