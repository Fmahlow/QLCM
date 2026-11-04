#!/bin/bash
# Phase 2 (paper/TODO.md items 2-7). Starts after phase 1 (campaign + MNIST replication) has finished.
set -u
cd "$(dirname "$0")/.."
P=.venv-v2/bin/python
while pgrep -f run_phase1.sh >/dev/null; do sleep 60; done
echo "START endpoint $(date -Is)"
$P v2/diagnostics/endpoint_regression.py --campaign results/v2-campaign-datasets-teachers --workers 6
echo "EXIT endpoint $? $(date -Is)"
echo "START review_eval (background, GPU) $(date -Is)"
$P v2/diagnostics/review_eval.py --campaign results/v2-campaign-datasets-teachers --output results/v2-review-eval > results/logs/review_eval.log 2>&1 &
EVAL=$!
echo "START e9 $(date -Is)"
$P v2/diagnostics/capacity_sweep.py --grid e9 --output results/v2-e9-correlators --workers 6
echo "EXIT e9 $? $(date -Is)"
echo "START e10 $(date -Is)"
$P v2/diagnostics/capacity_sweep.py --grid e10 --seeds 3,4,5 --output results/v2-e10-capacity-seeds --ae-sources results/v2-mnist-replication --workers 6
echo "EXIT e10 $? $(date -Is)"
wait $EVAL; echo "EXIT review_eval $? $(date -Is)"
$P paper/analysis/make_sample_grid.py; echo "EXIT sample_grid $? $(date -Is)"
echo "PHASE2 DONE $(date -Is)"
