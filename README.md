# QLCM

Active direction (approved 2026-10-06): characterize when consistency distillation
of a **hybrid quantum latent diffusion teacher** pays off under limited circuit
calls and measurement budgets. Novelty is a candidate, not established priority.

Start here: [Cold-start execution handoff](v2/HANDOFF.md). On a GPU server, the
entry point is `python -m v2.orchestrate`; first use `--dry-run`. No GPU campaign
or hardware experiment has been executed for v2.

- [Research decision, protocol, audit and literature](v2/PROTOCOL.md)
- [CPU smoke configuration](v2/smoke.json), [GPU pilot configuration](v2/gpu_pilot.json)
- [Pilot implementation](v2/core.py), [runner](v2/run.py), [orchestrator](v2/orchestrate.py)
- Historical study: [manuscript](v1/paper/main.tex), [execution log](v1/experiments/execution_checklist.md), [configs](v1/configs/), [figures](v1/figures/)

`v1/` remains unchanged. Its classical-teacher campaign is preliminary evidence,
not results for the new hypothesis. Raw campaign checkpoints and metrics remain
on the offline server. Local tables are transcriptions; figures and old smoke
artifacts do not independently validate the numerical campaign.

Historical commands must run from `v1/`; the historical campaign shell has a
Linux path hard-coded and requires adaptation before use. The historical method
has known limitations documented in the new protocol; do not silently rerun it
with v2 changes and call those the old results.
