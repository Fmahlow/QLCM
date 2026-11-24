# QLCM

Consistency distillation of **hybrid quantum latent diffusion teachers**, with
controls that test whether the circuit contributes.

Status (2026-11-03): the v2 pilot, the 4-dataset teacher/student campaign and the
preregistered extension E0--E4 have run (3 seeds each). Distillation works (the
one-step student matches the 32-step teacher); the circuit does not contribute
measurably relative to size-matched classical controls or an exact separable
surrogate. Results: [v2/RESULTS.md](v2/RESULTS.md). Manuscript: [paper/](paper/README.md).

- [Manuscript](paper/main.tex) and [asset script](paper/analysis/make_assets.py)
- [Results log](v2/RESULTS.md), [GPU handoff](v2/HANDOFF.md)
- [Research decision, protocol, audit, literature and preregistration](v2/PROTOCOL.md)
- [CPU smoke configuration](v2/smoke.json), [GPU pilot configuration](v2/gpu_pilot.json)
- [Pilot implementation](v2/core.py), [runner](v2/run.py), [orchestrator](v2/orchestrate.py)
- Quantum diffusion / QDense track (pure-torch circuits, rescued from fff005, 2026-09-15): [v1/README.md](v1/README.md), [v1/src/qlcm/qdense.py](v1/src/qlcm/qdense.py), [v1/scripts/run_quantum_diffusion_campaign.sh](v1/scripts/run_quantum_diffusion_campaign.sh); design notes in [docs/research_design.md](docs/research_design.md)
- Historical study: [manuscript](v1/paper/main.tex), [execution log](v1/experiments/execution_checklist.md), [configs](v1/configs/), [figures](v1/figures/)

`v1/` remains unchanged. Its classical-teacher campaign is preliminary evidence,
not results for the new hypothesis. Raw campaign checkpoints and metrics remain
on the offline server. Local tables are transcriptions; figures and old smoke
artifacts do not independently validate the numerical campaign.

Historical commands must run from `v1/`; the historical campaign shell has a
Linux path hard-coded and requires adaptation before use. The historical method
has known limitations documented in the new protocol; do not silently rerun it
with v2 changes and call those the old results.
