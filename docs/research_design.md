# QLCM v2 — Research Design (pre-registered)

Status: **draft for review**. Nothing in this document should be implemented
until it is confirmed. This file is the contract for what counts as a valid
result in v2; if we later want to deviate from it, we edit this file first
and say why, rather than silently changing methodology mid-experiment (this
is exactly what caused the v1 epoch-budget inconsistency).

Target venue: an applied/methods-focused quantum machine learning journal
(e.g. Quantum Machine Intelligence, EPJ Quantum Technology tier). This means:
simulator-only experiments are acceptable, MNIST/Fashion-MNIST scale is
acceptable, and a well-controlled negative result is a publishable
contribution — but the experimental design must be tight enough that a
reviewer cannot identify a confound we didn't control for.

## Two separate studies

We ask two independent, separately pre-registered questions. Neither study's
methodology is allowed to leak into the other's (e.g. no borrowing an epoch
budget "by analogy" — this is what went wrong in v1).

### Study 1 — Parameter efficiency / inductive bias

**Question**: Given a *matched total parameter budget*, does substituting a
small parameterized quantum circuit for part of a classical latent
consistency student's network change sample quality, diversity, or
recognizability, relative to an all-classical student of the same size?

**Design**:
- Datasets: MNIST (primary), Fashion-MNIST (conditional — see EMA note
  below).
- Architectures: classical consistency student vs. QLCM-R (residual
  placement). QLCM-H (head) and QLCM-T (time-conditioning) are run once each
  at the "equal-parameter" point only, as a replication check against v1's
  findings (head fails, time is neutral) — they do not get the full curve.
- **Parameter-matching curve**: for QLCM-R, the quantum circuit
  (qubits/depth) is held fixed at a reference size (4 qubits, depth 2,
  matching v1's primary cell), and the internal classical backbone's
  `hidden_dim` is swept to produce three total-parameter points:
  - *sub-parametrized*: QLCM-R's total parameters ≈ 70% of the classical
    baseline's total.
  - *equal-parametrized*: QLCM-R's total parameters ≈ 100% of the classical
    baseline's total (this is the point v1 was missing — v1's QLCM-R was at
    ≈154%).
  - *over-parametrized*: QLCM-R's total parameters ≈ 150% of the classical
    baseline's total (this is close to what v1 accidentally reported as its
    only point).
  Exact `hidden_dim` values to hit these three targets will be computed
  during implementation and recorded here before training starts.
- Qubit/depth ablation (secondary, replication of v1): 4 and 6 qubits, depth
  1 and 2, at the equal-parametrized point only.
- Seeds: 5 per cell (up from v1's 3).
- Statistics: Welch's t-test (does not assume equal variance) between QLCM-R
  and the classical baseline at each parameter point, plus Cohen's d effect
  size. Significance threshold α = 0.05, stated up front; we report the
  effect size regardless of significance, since "no significant difference"
  and "no effect" are not the same claim.
- Metrics: classifier-feature FID-like score, classifier mean confidence,
  label entropy (as in v1) — this tooling was not the problem in v1 and is
  reused as-is.

**Pre-registered success/failure framing** (decided now, before running
anything, so we cannot rationalize the result afterward):
- If QLCM-R's FID-like is statistically indistinguishable from or better
  than the classical baseline's at the equal-parameter point, and this holds
  at sub-parametrized too, that is a positive efficiency finding.
- If QLCM-R is statistically worse at equal parameters, that is a negative
  finding, consistent with v1's direction but now on a fair comparison.
- The sub- and over-parametrized points contextualize *how* any gap changes
  with parameter budget, regardless of which of the above holds at the
  equal point.

### Study 2 — Few-step consistency vs. many-step quantum diffusion

**Question**: Does a quantum-consistency-distilled student reach comparable
sample quality in far fewer sampling steps (NFEs) than a literature-standard
quantum diffusion model reproduced at matched qubit count and matched
dataset/latent scale?

**Design**:
- Reproduce a published quantum diffusion architecture (Q-Dense,
  Koelle et al. 2024, arXiv:2401.07049, or a comparably well-specified
  alternative if Q-Dense's readout constraints turn out to be a poor fit —
  to be confirmed during implementation) adapted to run in the **same
  16-dim latent space, same shared autoencoder, same 10-class MNIST** as
  Study 1 and the rest of this paper — not the original paper's own reduced
  8x8-pixel/digits-0-1 scope, which was v1's limitation.
- Qubit count matched to QLCM-R's reference cell (4 image-encoding qubits +
  1 ancilla, or the nearest architecturally sensible equivalent for a
  16-dim latent).
- Diffusion schedule: reuse the same `timesteps=50` schedule already used
  for the classical/quantum teacher elsewhere in this paper, rather than an
  arbitrarily smaller τ, so "many-step" means the same thing across the
  whole paper.
- Training objective: direct previous-step prediction (as the architecture's
  non-negative, probability-valued readout requires), consistent with the
  source paper.
- **Stopping criterion**: identical rule to Study 1 and to every other
  trainable model in this project (see Shared Methodology below) — this is
  the direct fix for the v1 problem (150 vs. 50 epochs chosen by analogy).
- Evaluate FID-like vs. NFE across a step sweep (1 up to the full schedule,
  e.g. 1/2/4/8/16/32/50) for both QLCM-R and the quantum diffusion
  reproduction, 5 seeds each.
- Statistics: same Welch's t-test + effect size approach, comparing
  QLCM-R's quality at its native 1-4 step range against the quantum
  diffusion baseline's quality at every NFE point, so we can state precisely
  how many steps (if any) the baseline needs to close the gap, or whether it
  never does.

**Pre-registered success/failure framing**:
- If the quantum diffusion baseline needs materially more NFEs than QLCM-R
  to reach statistically indistinguishable quality, that supports the
  "few-step consistency is a genuine speed advantage over quantum diffusion"
  claim.
- If the quantum diffusion baseline cannot reach QLCM-R's quality at any
  NFE we test (as suggested, non-conclusively, by v1's small-scale version
  of this experiment), we report that directly and do not oversell it as a
  "speed" finding — quality and speed are separate axes, and a slower path
  to worse quality is not evidence of a speed advantage, just of a
  worse baseline.
- If the two are statistically indistinguishable in quality at matched NFE,
  that is a negative finding for the speed claim and gets reported as such.

## Shared methodology (applies to every trainable model in v2, no exceptions)

- **Stopping criterion**: track a moving average of the training loss over
  the last 10 epochs; stop when the relative improvement of that moving
  average over the previous 10-epoch window falls below 0.5%, subject to a
  shared hard cap of 300 epochs for every model regardless of family
  (consistency student, classical teacher, or quantum diffusion
  reproduction). The cap is generous enough that we expect most models to
  stop earlier via the convergence rule, not the cap — if a model hits the
  cap without converging, that is reported explicitly as non-convergence,
  not silently treated as a finished result.
- **Seeding**: `torch.manual_seed` before model construction and data
  loading, per run, as already fixed in v1.
- **Run identification**: every run gets a unique `--run-id`, metrics and
  sample grids are namespaced by it, exactly as in v1 (this tooling worked).
- **Aggregation**: one `results/summary.csv` via a shared aggregation
  script, one schema, both studies included with a `study` column so they
  are never accidentally compared against each other without the reader
  noticing they're different experiments.
- **Evidence before trusting a result**: every reported number must have
  (a) a metrics JSON, (b) a rendered sample grid actually looked at by a
  human before quoting quality numbers from it (this caught a real bug in
  v1's Q-Dense reproduction — FID-like alone did not flag blank images),
  and (c) the training loss curve checked for convergence under the rule
  above.

## Explicit exclusions (to prevent v1-style scope creep)

- No new experiments get added mid-stream without first writing the
  question and design into this file. If a follow-up question comes up
  during Study 1 or 2, it becomes a Study 3 candidate to be designed
  properly, not an ad hoc script bolted onto an existing study.
- Real quantum hardware validation is out of scope for v2 (consistent with
  the target venue decision).
- EMA target network for consistency distillation (Section on
  Fashion-MNIST below) is the one deliberate methodology change from v1's
  training objective; it is implemented and validated on MNIST *before* any
  decision is made about Fashion-MNIST, not applied retroactively to
  rationalize a result.

## Fashion-MNIST / EMA target network

v1's Fashion-MNIST results had an unresolved mode collapse toward one
class, persisting after every fix tried. Before deciding whether
Fashion-MNIST appears in the v2 paper at all, we implement an EMA
(exponential moving average) target network for consistency distillation —
the mechanism used in the original consistency models formulation
(Song et al. 2023), which v1 approximated with the online student under
`no_grad` instead. This is tested on MNIST first (does it change anything
there?), then on Fashion-MNIST specifically to see if it resolves the
collapse. Fashion-MNIST is only included in the final v2 paper if this
resolves the collapse to a degree that produces genuinely diverse samples;
otherwise Fashion-MNIST is dropped to an appendix note or omitted, rather
than reported with an unexplained collapse again.

**Result (resolved): EMA does not fix the collapse; Fashion-MNIST is
dropped from the v2 campaign.** Tested on both classical (hidden_dim=256)
and QLCM-R (hidden_dim=188, equal-parameter point), 3 seeds each, EMA decay
0.999, same converged-teacher/autoencoder pipeline that worked well on
MNIST. Label entropy came back *worse* than v1's already-bad numbers
(0.68-0.92 vs. v1's 1.3-1.4, out of a 2.303 max) and FID-like also worsened
(473-484 vs. v1's 419-420); 380-420 of 512 samples in every run predicted
as a single class ("T-shirt/top"). Contrast: MNIST's classical/QLCM-R EMA
runs converged cleanly and produced FID-like ~178-193 (a large improvement
over v1's 337-369), so the pipeline itself is not the problem — Fashion
-MNIST specifically remains broken, most likely for a reason unrelated to
the consistency target network (dataset scale/class-imbalance under the
10,000-image subset is the leading suspect, consistent with v1's
speculation, but not tested further here). Per the pre-registered rule
above, Fashion-MNIST is out of scope for the rest of v2; it is not tested
further, and MNIST is the sole dataset for Study 1 and Study 2. If revisited
later, the training-subset-size hypothesis would be the first thing to
test, as its own separate, pre-registered mini-study.

## Open items

1. **Resolved.** Exact `hidden_dim` values for Study 1's sub-/equal-/over
   -parametrized points, computed against the classical baseline reference
   (`ClassicalConsistencyModel`, `hidden_dim=256`, `90,640` total params;
   `QuantumLatentConsistencyModel`, `placement=residual`, 4 qubits, depth 2,
   for all three points):
   - sub (~70%): `hidden_dim=142` -> 63,260 params (69.8% of baseline)
   - equal (~100%): `hidden_dim=188` -> 90,952 params (100.3% of baseline)
   - over (~150%): `hidden_dim=251` -> 135,745 params (149.8% of baseline)
   The classical baseline itself stays fixed at `hidden_dim=256` across all
   three points; only QLCM-R's internal backbone is swept.
2. Whether Q-Dense's amplitude-embedding readout can be adapted cleanly to
   a continuous 16-dim latent (which is not naturally non-negative or
   norm-bounded like pixel intensities) — to be worked out during Study 2
   implementation and recorded here once resolved, not decided silently in
   code.
3. **Resolved.** Compute budget estimate (shared with the user before
   starting): ~58 training runs / ~5h sequential (~2-3h with light
   parallelism) for MNIST-only through Study 1 + Study 2 + EMA validation;
   roughly double if Fashion-MNIST proceeds. Approved; execution starts with
   the cheap EMA validation phase first (~6 runs) to calibrate the
   convergence criterion in practice and decide on Fashion-MNIST before
   committing to the full campaign.
