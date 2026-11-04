#!/bin/bash
# Review round 1 (R1, R2, R4, R5 of the round-1 preregistration in v2/PROTOCOL.md).
set -u
cd "$(dirname "$0")/.."
P=.venv-v2/bin/python
echo "START r5-endpoint-100k (background, CPU) $(date -Is)"
$P v2/diagnostics/endpoint_regression.py --pairs 100000 --val-pairs 2000 --name student_endpoint_100k --workers 6 > results/logs/r5.log 2>&1 &
R5=$!
echo "START r1-e9 $(date -Is)"; $P v2/diagnostics/sweep_rescore.py --sweep results/v2-e9-correlators --seeds 0,1,2 --output results/v2-r1-e9; echo "EXIT r1-e9 $? $(date -Is)"
echo "START r1-e10 $(date -Is)"; $P v2/diagnostics/sweep_rescore.py --sweep results/v2-e10-capacity-seeds --seeds 3,4,5 --output results/v2-r1-e10 --perturb; echo "EXIT r1-e10 $? $(date -Is)"
wait $R5; echo "EXIT r5 $? $(date -Is)"
mv results/v2-review-eval results/v2-review-eval.before-r4
echo "START review_eval (R4) $(date -Is)"; $P v2/diagnostics/review_eval.py > results/logs/review_eval.log 2>&1; echo "EXIT review_eval $? $(date -Is)"
echo "ROUND1 DONE $(date -Is)"
