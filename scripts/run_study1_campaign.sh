#!/bin/bash
# Study 1 full campaign (docs/research_design.md): parameter-matching curve
# for QLCM-R (sub/equal/over), QLCM-H/T replication at the equal point, and
# the qubit/depth secondary ablation for residual at the equal point.
# MNIST only (Fashion-MNIST dropped -- EMA did not resolve its mode
# collapse, see docs/research_design.md). 5 seeds per cell. Reuses the 6
# EMA-validation runs already trained (classical/residual seeds 0-2 at the
# reference hidden_dim) instead of retraining them.
set -uo pipefail
cd /home/mahlow/QLCM

MAX_PARALLEL=6
log() { echo "[$(date +%H:%M:%S)] $*"; }

JOBS_FILE=$(mktemp)

# Reuse EMA-validation runs (seeds 0-2) by aliasing their run-id.
for seed in 0 1 2; do
  ln -sf "$(pwd)/results/mnist/student_EMA-classical-seed${seed}.pt" "results/mnist/student_MNIST-classical-seed${seed}.pt"
  ln -sf "$(pwd)/results/mnist/student_EMA-classical-seed${seed}.json" "results/mnist/student_MNIST-classical-seed${seed}.json"
  ln -sf "$(pwd)/results/mnist/student_EMA-residual-seed${seed}.pt" "results/mnist/student_MNIST-residual-equal-seed${seed}.pt"
  ln -sf "$(pwd)/results/mnist/student_EMA-residual-seed${seed}.json" "results/mnist/student_MNIST-residual-equal-seed${seed}.json"
done

for seed in 3 4; do
  echo "classical 256 4 2 $seed MNIST-classical-seed${seed}" >> "$JOBS_FILE"
done
for seed in 3 4; do
  echo "residual 188 4 2 $seed MNIST-residual-equal-seed${seed}" >> "$JOBS_FILE"
done
for seed in 0 1 2 3 4; do
  echo "residual 142 4 2 $seed MNIST-residual-sub-seed${seed}" >> "$JOBS_FILE"
done
for seed in 0 1 2 3 4; do
  echo "residual 251 4 2 $seed MNIST-residual-over-seed${seed}" >> "$JOBS_FILE"
done
for seed in 0 1 2 3 4; do
  echo "head 188 4 2 $seed MNIST-head-seed${seed}" >> "$JOBS_FILE"
done
for seed in 0 1 2 3 4; do
  echo "time 188 4 2 $seed MNIST-time-seed${seed}" >> "$JOBS_FILE"
done
for qd in "4 1" "6 1" "6 2"; do
  read -r q d <<< "$qd"
  for seed in 0 1 2 3 4; do
    echo "residual 188 $q $d $seed MNIST-residual-q${q}d${d}-seed${seed}" >> "$JOBS_FILE"
  done
done

log "Total jobs queued: $(wc -l < "$JOBS_FILE")"
xargs -P "$MAX_PARALLEL" -L 1 bash scripts/_run_study1_one.sh < "$JOBS_FILE"
rm -f "$JOBS_FILE"
log "Study 1 campaign done."
