#!/bin/bash
# Phase 1 (paper/TODO.md items 1 and campaign reproduction). Run from repo root.
set -u
cd "$(dirname "$0")/.."
P=.venv-v2/bin/python
echo "START campaign $(date -Is)"
$P v2/diagnostics/dataset_teacher_campaign.py --output results/v2-campaign-datasets-teachers \
  --datasets mnist,fashion_mnist,kmnist,pneumoniamnist --seeds 0,1,2 --workers 7
echo "EXIT campaign $? $(date -Is)"
echo "START replication $(date -Is)"
$P v2/diagnostics/dataset_teacher_campaign.py --output results/v2-mnist-replication \
  --datasets mnist --seeds 3,4,5,6,7 --workers 7
echo "EXIT replication $? $(date -Is)"
