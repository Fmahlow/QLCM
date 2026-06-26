# Execution Checklist

Nothing in this checklist should be started on a low-power machine unless the
expected runtime is acceptable. The current workspace is prepared for execution
but no training pipeline is running.

## Order of Execution

1. Smoke dataset autoencoder:
   `python3 scripts/train_autoencoder.py --config configs/smoke.yaml`

2. Smoke QLCM dry run only:
   `python3 scripts/train_qlcm.py --config configs/smoke.yaml --placement residual --autoencoder results/bars_stripes/autoencoder.pt --dry-run`

3. MNIST autoencoder:
   `python3 scripts/train_autoencoder.py --config configs/mnist.yaml`

4. MNIST latent diffusion teacher:
   `python3 scripts/train_teacher.py --config configs/mnist.yaml --autoencoder results/mnist/autoencoder.pt`

5. MNIST primary classical consistency baseline:
   `python3 scripts/train_qlcm.py --config configs/mnist.yaml --placement classical --autoencoder results/mnist/autoencoder.pt --run-id MNIST-CC-001 --sampling-steps 1`

6. MNIST primary QLCM-R:
   `python3 scripts/train_qlcm.py --config configs/mnist.yaml --placement residual --qubits 4 --depth 2 --autoencoder results/mnist/autoencoder.pt --run-id MNIST-QR-4-2-001 --sampling-steps 1`

7. Evaluate trained students:
   `python3 scripts/evaluate_qlcm.py --config configs/mnist.yaml --placement residual --autoencoder results/mnist/autoencoder.pt --student results/mnist/student_MNIST-QR-4-2-001.pt --steps 1`

8. Expand to the remaining matrix:
   `python3 scripts/generate_run_commands.py`

## Stop Conditions

- If autoencoder reconstruction is poor, do not run QLCM yet.
- If the teacher loss diverges, reduce learning rate before distillation.
- If QLCM-H collapses, prioritize QLCM-R.
- Treat 8-qubit runs as optional on CPU.

## Evidence Required Before Updating Paper Results

- Metrics JSON for every reported run.
- Sample grid for every reported run.
- Parameter counts.
- Runtime measurements.
- At least three seeds for the primary MNIST comparison.
