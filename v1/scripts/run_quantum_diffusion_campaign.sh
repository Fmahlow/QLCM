#!/bin/bash
# Quantum diffusion baseline: same quantum module as QLCM-R (4 qubits, depth 2),
# trained as a naive multi-step noise predictor instead of a consistency student,
# then evaluated across an NFE sweep. Purpose: give the QLCM few-step speed claim
# a same-architecture comparison, instead of relying on cross-paper NFE counts.
set -uo pipefail
cd /home/mahlow/QLCM
source .venv/bin/activate

log() { echo "[$(date +%H:%M:%S)] $*"; }

MAE=results/mnist/autoencoder.pt
STEPS_SWEEP=(1 2 4 8 16 32 50)

log "=== Training quantum diffusion baseline (4 qubits, depth 2), seeds 0/1/2 ==="
for seed in 0 1 2; do
  runid="quantum_teacher_4_2_seed${seed}"
  log "TRAIN $runid"
  if python scripts/train_quantum_teacher.py --config configs/mnist.yaml \
      --autoencoder "$MAE" --qubits 4 --depth 2 --seed "$seed" --run-id "$runid" \
      > "results/_log_train_${runid}.txt" 2>&1; then
    log "OK $runid"
  else
    log "FAIL $runid -- see results/_log_train_${runid}.txt"
  fi
done

log "=== Evaluating seed 0 across NFE sweep: ${STEPS_SWEEP[*]} ==="
for steps in "${STEPS_SWEEP[@]}"; do
  runid="QDIFF-4-2-seed0-${steps}step"
  log "EVAL $runid"
  if python scripts/evaluate_quantum_diffusion.py --config configs/mnist.yaml \
      --autoencoder "$MAE" --quantum-teacher results/mnist/quantum_teacher_4_2_seed0.pt \
      --qubits 4 --depth 2 --steps "$steps" --run-id "$runid" --seed 0 \
      > "results/_log_eval_${runid}.txt" 2>&1; then
    log "OK eval $runid"
  else
    log "FAIL eval $runid -- see results/_log_eval_${runid}.txt"
  fi
done

log "=== Evaluating seeds 1/2 at the two sweep endpoints (1-step, 50-step) ==="
for seed in 1 2; do
  for steps in 1 50; do
    runid="QDIFF-4-2-seed${seed}-${steps}step"
    log "EVAL $runid"
    if python scripts/evaluate_quantum_diffusion.py --config configs/mnist.yaml \
        --autoencoder "$MAE" --quantum-teacher "results/mnist/quantum_teacher_4_2_seed${seed}.pt" \
        --qubits 4 --depth 2 --steps "$steps" --run-id "$runid" --seed "$seed" \
        > "results/_log_eval_${runid}.txt" 2>&1; then
      log "OK eval $runid"
    else
      log "FAIL eval $runid -- see results/_log_eval_${runid}.txt"
    fi
  done
done

log "=== Aggregating ==="
python scripts/aggregate_results.py
log "Done."
