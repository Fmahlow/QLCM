#!/bin/bash
# Helper invoked once per training job by scripts/run_study1_campaign.sh's
# xargs loop. Kept as a standalone script (not an exported bash function)
# because function export through xargs -P proved unreliable in this
# environment.
set -uo pipefail
cd /home/mahlow/QLCM
source .venv/bin/activate

placement=$1
hidden_dim=$2
qubits=$3
depth=$4
seed=$5
runid=$6

AE=results/mnist/autoencoder.pt
TCH=results/mnist/teacher.pt

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -f "results/mnist/student_${runid}.pt" ]; then
  log "SKIP $runid (already trained)"
  exit 0
fi

extra=""
if [ "$placement" != "classical" ]; then extra="--qubits $qubits --depth $depth"; fi

log "TRAIN $runid"
if python scripts/train_qlcm.py --config configs/mnist.yaml --placement "$placement" \
    --autoencoder "$AE" --teacher "$TCH" --hidden-dim "$hidden_dim" $extra \
    --target ema --seed "$seed" --run-id "$runid" \
    > "results/_log_train_${runid}.txt" 2>&1; then
  log "OK $runid"
else
  log "FAIL $runid -- see results/_log_train_${runid}.txt"
fi
