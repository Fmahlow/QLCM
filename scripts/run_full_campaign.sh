#!/bin/bash
# Full re-run: stronger teacher + 3-seed ablation grid + corrected multistep sampler.
# Logs progress per run; safe to inspect with `tail -f` while running.
set -uo pipefail
cd /home/mahlow/QLCM
source .venv/bin/activate

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_train () {
  local cfg=$1 placement=$2 ae=$3 tch=$4 runid=$5 seed=$6 qubits=$7 depth=$8
  local extra=""
  if [ -n "$qubits" ]; then extra="$extra --qubits $qubits"; fi
  if [ -n "$depth" ]; then extra="$extra --depth $depth"; fi
  log "TRAIN $runid (placement=$placement seed=$seed qubits=$qubits depth=$depth)"
  if python scripts/train_qlcm.py --config "$cfg" --placement "$placement" --autoencoder "$ae" --teacher "$tch" --run-id "$runid" --sampling-steps 1 --seed "$seed" $extra > "results/_log_train_${runid}.txt" 2>&1; then
    log "OK $runid"
  else
    log "FAIL $runid -- see results/_log_train_${runid}.txt"
  fi
}

run_eval () {
  local cfg=$1 placement=$2 ae=$3 student=$4 steps=$5 runid=$6 seed=$7 qubits=$8 depth=$9
  local extra=""
  if [ -n "$qubits" ]; then extra="$extra --qubits $qubits"; fi
  if [ -n "$depth" ]; then extra="$extra --depth $depth"; fi
  log "EVAL $runid steps=$steps"
  if python scripts/evaluate_qlcm.py --config "$cfg" --placement "$placement" --autoencoder "$ae" --student "$student" --steps "$steps" --run-id "$runid" --seed "$seed" --grid-samples 64 $extra > "results/_log_eval_${runid}.txt" 2>&1; then
    log "OK eval $runid"
  else
    log "FAIL eval $runid -- see results/_log_eval_${runid}.txt"
  fi
}

MAE=results/mnist/autoencoder.pt
FAE=results/fashion_mnist/autoencoder.pt
MTC=results/mnist/teacher.pt
FTC=results/fashion_mnist/teacher.pt
log "=== Using existing teachers (150 epochs, already trained): $MTC $FTC ==="

log "=== MNIST: training 8 architectures x 3 seeds = 24 runs ==="
declare -a ARCHS=(
  "classical::"
  "head:4:1"
  "head:4:2"
  "head:6:2"
  "residual:4:1"
  "residual:4:2"
  "residual:6:2"
  "time:4:2"
)
for arch in "${ARCHS[@]}"; do
  IFS=':' read -r placement qubits depth <<< "$arch"
  for seed in 0 1 2; do
    if [ "$placement" == "classical" ]; then
      runid="MNIST-classical-seed${seed}"
    else
      runid="MNIST-${placement}-${qubits}-${depth}-seed${seed}"
    fi
    run_train configs/mnist.yaml "$placement" "$MAE" "$MTC" "$runid" "$seed" "$qubits" "$depth"
  done
done

log "=== Fashion-MNIST: classical + residual(4,2) x 3 seeds = 6 runs ==="
for seed in 0 1 2; do
  run_train configs/fashion_mnist.yaml classical "$FAE" "$FTC" "FMNIST-classical-seed${seed}" "$seed" "" ""
  run_train configs/fashion_mnist.yaml residual "$FAE" "$FTC" "FMNIST-residual-4-2-seed${seed}" "$seed" 4 2
done

log "=== Evaluating all MNIST runs at steps 1/2/4 ==="
for arch in "${ARCHS[@]}"; do
  IFS=':' read -r placement qubits depth <<< "$arch"
  for seed in 0 1 2; do
    if [ "$placement" == "classical" ]; then
      runid="MNIST-classical-seed${seed}"
    else
      runid="MNIST-${placement}-${qubits}-${depth}-seed${seed}"
    fi
    student="results/mnist/student_${runid}.pt"
    run_eval configs/mnist.yaml "$placement" "$MAE" "$student" 1 "${runid}-1step" "$seed" "$qubits" "$depth"
    if [ "$seed" == "0" ]; then
      run_eval configs/mnist.yaml "$placement" "$MAE" "$student" 2 "${runid}-2step" "$seed" "$qubits" "$depth"
      run_eval configs/mnist.yaml "$placement" "$MAE" "$student" 4 "${runid}-4step" "$seed" "$qubits" "$depth"
    fi
  done
done

log "=== Evaluating all Fashion-MNIST runs at steps 1/2/4 ==="
for seed in 0 1 2; do
  run_eval configs/fashion_mnist.yaml classical "$FAE" "results/fashion_mnist/student_FMNIST-classical-seed${seed}.pt" 1 "FMNIST-classical-seed${seed}-1step" "$seed" "" ""
  run_eval configs/fashion_mnist.yaml residual "$FAE" "results/fashion_mnist/student_FMNIST-residual-4-2-seed${seed}.pt" 1 "FMNIST-residual-4-2-seed${seed}-1step" "$seed" 4 2
  if [ "$seed" == "0" ]; then
    run_eval configs/fashion_mnist.yaml classical "$FAE" "results/fashion_mnist/student_FMNIST-classical-seed0.pt" 2 "FMNIST-classical-seed0-2step" 0 "" ""
    run_eval configs/fashion_mnist.yaml classical "$FAE" "results/fashion_mnist/student_FMNIST-classical-seed0.pt" 4 "FMNIST-classical-seed0-4step" 0 "" ""
    run_eval configs/fashion_mnist.yaml residual "$FAE" "results/fashion_mnist/student_FMNIST-residual-4-2-seed0.pt" 2 "FMNIST-residual-4-2-seed0-2step" 0 4 2
    run_eval configs/fashion_mnist.yaml residual "$FAE" "results/fashion_mnist/student_FMNIST-residual-4-2-seed0.pt" 4 "FMNIST-residual-4-2-seed0-4step" 0 4 2
  fi
done

log "=== Aggregating ==="
python scripts/aggregate_results.py
log "=== CAMPAIGN COMPLETE ==="
