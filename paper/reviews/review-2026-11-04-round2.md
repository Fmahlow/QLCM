# Review: "Does the Circuit Matter? Controlled Ablations of Hybrid Quantum Latent Consistency Distillation"

Round 2. Reviewed:
- `paper/main.tex` at commit 3cccb96 (1,539 lines) and `paper/generated/*.tex`, `paper/generated/stats.json`.
- `paper/analysis/make_assets.py`, in particular the round-1 block (l. 704–790).
- `v2/diagnostics/{sweep_rescore,review_eval,endpoint_regression}.py`, plus `v2/core.py` for the circuit and its initialisation.
- The new raw reports in `v2/reports/review-2026-11/` (`r1-e9-rescore.json`, `r1-e10-rescore-r2-perturb.json`, `r5-endpoint-training.json`, `e6-e7-e8-review-eval.json`).
- `v2/PROTOCOL.md` with `git log -p`, and `results/logs/round1.log`.

I recomputed every number marked "recomputed" below from the raw JSON with `.venv-v2/bin/python` in a scratch directory. I did not run `make_assets.py`, because it overwrites the generated files. Line numbers refer to the current `main.tex`.

---

## Summary

The paper tests whether the 4–8-qubit variational branch of a hybrid latent diffusion denoiser, and its distilled consistency students, contributes anything that a separable surrogate (an exact simulation of the circuit without CZ gates) or a shape-matched classical block cannot provide. The revision answers round 1 with six analyses that were preregistered before they were run. The timestamps confirm the order: the R1–R6 preregistration was committed at 20:40:47 (8c21add), and `round1.log` starts at 20:42:24.
- **R1:** the E9 and E10 sweeps re-scored with 5,000 samples, P/R and latent metrics.
- **R2:** E6 branch perturbations at h ∈ {0, 4, 16}.
- **R3:** a model-level positive control from capacity pairs.
- **R4:** a single floor rule, with AE reconstructions and P/R on equal-size sets.
- **R5:** endpoint regression with 100k pairs and a held-out loss.
- **R6:** a seed bootstrap, the geometric-mean E10 rule, and the surrogate/classical-branch interval.

The manuscript is restructured into three blocks: evaluation validity, circuit, distillation. The abstract is shorter, and the E5 and E8 claims are restated correctly. The E8 grid change is now disclosed, and the protocol text has been restored with a dated amendment. `git log -p v2/PROTOCOL.md` shows no in-place edit after 8c21add. The working tree has no uncommitted protocol change.

The revision is substantive and honest. Every round-1 major issue is addressed in substance. Re-running `review_eval` for R4 reproduced all 789 previously committed FID-like values exactly (relative difference 0), which is a strong reproducibility statement.

The central claim holds under the new analyses: in this model family, entanglement does not contribute detectably, and the circuit gives no advantage beyond what the separable surrogate shares. What remains:
- **Statistics:** one new robustness analysis (R6) is anti-conservative.
- **Mechanism:** one mechanism the paper does not discuss. With even depth and near-zero trainable rotations, the entangled circuit is initialised essentially at the separable function, from the same parameters as the surrogate.
- **Reporting:** several selective-reporting and wording issues of the kind round 1 flagged, now located in the new material (R1 teacher results, R1 MMD, R2 a = 1.5, the MNIST reference-reconstruction floor, and the Conclusion's "the same held").

## Recommendation: **Minor revision**

The main scientific claim is supported. The round-1 issues that required substantive changes (positive control, E5 reading, robustness of the sweeps, floors, E8 disclosure, MMD) are resolved. None of the remaining issues needs new training. Most are [TEXT] fixes. Two are cheap [ANALYSIS] fixes that use only existing checkpoints and JSON: replacing the R6 bootstrap, and a diagnostic of how far the trained circuits are from separable.

Issues that block acceptance as I see it:
1. **New Major 1 [ANALYSIS]:** the R6 seed bootstrap is presented as a robustness check but gives *narrower* intervals than the analysis it is meant to check.
2. **New Major 2 [TEXT, plus a recommended ANALYSIS]:** the initialisation coincidence. It bears directly on how the entanglement null should be interpreted.
3. **New Major 3 [TEXT]:** R1/R2 results that go against the narrative are omitted, and the MNIST floor statement contradicts a preregistered R4 row.
4. **Carried over, human item H1:** the closest consistency-based quantum diffusion model (OQIDDM) has still not been read in full. Round 1 marked this as needed before submission.

---

## Response to round-1 issues

| Round-1 issue | Status | New location (main.tex) | Comment |
|---|---|---|---|
| **Major 1**: positive control oversold; contamination post hoc; not model-level; not where branch matters | **Addressed** (residuals below) | Abstract l. 41–42; §5.1 "Perturbing the branch" l. 520–539, Table `tab:r2`; "Differences between independently trained models" l. 556–575, Table `tab:r3`; Appendix `tab:e6full`; Discussion l. 1154–1156, 1201–1205 | The contamination control is moved to the appendix, labelled post hoc, and its caption says it is paired at the sample level. R2 (branch level, small h) and R3 (model level) were added as requested. Residuals: the preregistered a = 1.5 level was run but is omitted from `tab:r2` and from the "44 of 72" count. Several "detections" are *improvements* of FID-like (a = 1.5, a = 2). The R3 summary "10–20% with three seeds" is slightly optimistic. See New minor 1–3. |
| **Major 2**: E5 described backwards; four-step claim does not replicate | **Addressed** | §5.3.3 l. 940–948; §5.3.2 l. 889–891; Contribution 5 l. 117–121; Discussion l. 1181–1184; Limitations l. 1213–1215 | The text now states correctly that the one-step gap is about 9–13% on both seed sets, and that the four-step result was optimistic (3/3 → 3/5). Contribution 5 says the within-10% criterion is not met. |
| **Major 3**: robustness evaluation only at h = 128 | **Addressed** | §5.2.4 (R1) l. 781–798, Table `tab:r1`; abstract l. 40–41 | E9 and E10 were re-scored with 5,000 samples, P/R and latent FD, and the abstract now leads with the sweeps. Residuals: R1 MMD was preregistered and computed but is not reported, and the teacher-level circuit results are not discussed in the text (New Major 3). |
| **Major 4**: MNIST AE floor confounded; floor chosen per claim | **Addressed** (one interpretive residual) | §4 "Robustness evaluation" l. 366–375; §5.1 "Floors" l. 486–503; `tab:floors` | There is one stated rule (held-out training images), the preregistered floor is reported alongside, and reconstructions of both the held-out images and the reference are reported. Residual: the new "AE recon., reference" row for MNIST (20.3) is above the teacher (18.7), which contradicts the text's MNIST reading (New Major 3b). |
| **Major 5**: E8 grid edited in place, undisclosed | **Addressed** | §4 "Added after the preregistrations" l. 443–458; `PROTOCOL.md` amendment E8-a; Limitations l. 1218–1219 | The original sentence is restored, with a dated amendment that cites the commit. Git history confirms there are no further in-place edits. |
| **Major 6**: latent MMD not reported | **Addressed for E7**; **recurs in R1** | `tab:floors` (MMD column), `tab:e7` (ΔMMD), l. 497–499, 977–978 | R1 also preregistered MMD ("Fréchet e MMD latentes … em cada métrica"). `latent_mmd` is in both R1 JSONs but `make_assets.py` never reads it there (New minor 4). |
| Minor 1: significant effects against the narrative | **Addressed** for the items listed; **new instances** | E9 l. 763–765; E10 l. 708–712; E7/E5 l. 984–990 | The E9 teacher −2.4%, E10 MNIST h = 8 (+7.9%) and the 8-seed E5 teacher are now reported. New instances in R1 (teacher hybrid/classical branch) and in the E7 MMD hybrid-vs-no-entanglement result (+0.06 [+0.01, +0.11] ×10⁻³, only in the table): New Major 3a, New minor 5. |
| Minor 2: surrogate/classical-branch interval at h ≥ 16 | **Addressed** | l. 702–705 | −4.0% [−7.9%, −0.0%] is computed in `make_assets.py` and matches `stats.json`. |
| Minor 3: E10 rule with geometric means | **Addressed** | l. 696–697 | `e10.geometric_rule` = {1, 0, 2}, the same counts. |
| Minor 4: "size-matched" vs "shape-matched"; E9 causal claim | **Addressed** | Abstract l. 34; l. 96; l. 1249; E9 l. 766–769 | The E9 explanation is now a hypothesis and mentions 90 vs 16 parameters. |
| Minor 5: E0 range definition | **Addressed** | l. 823–832 | Says "ratio of means" and gives the +104% (−41…+364%) per-seed mean. |
| Minor 6: re-execution, not replication | **Addressed** | l. 410–417; `tab:e5` caption | |
| Minor 7: preregistration timing of the retraining | **Addressed** | l. 418–420 | |
| Minor 8: `tab:e6` omits levels | **Addressed** | Appendix `tab:e6full` | All levels are shown. The same omission now recurs in R2 (a = 1.5): New minor 1. |
| Minor 9: PneumoniaMNIST P/R set sizes | **Addressed** | l. 359–361; `review_eval.py` l. 96–99 | Mean of 10 subsamples of 624. |
| Minor 10: endpoint data budget | **Addressed** | E8 l. 1037–1041; `tab:e8ratios` | R5: 100k pairs, EMA held-out loss ≈ train loss (e.g. MNIST seed 0: 0.0085 vs 0.0088). |
| Minor 11: E4a hybrid/surrogate claim unverifiable | **Partially** | l. 1136–1137 | Now points to "rows in the repository's report". Acceptable, but one sentence with the numbers would be better. |
| Minor 12: cluster intervals ignore seed correlation | **Partially; new problem** | l. 468–470, 630–631, 700–705 | A seed bootstrap was added, but with 3 or 6 seeds the percentile bootstrap is anti-conservative and narrower than the cell-level interval (New Major 1). |
| Minor 13: ambiguous "the same two" | **Addressed** | Discussion l. 1186–1187 | Datasets named. |
| Minor 14: cross-references | **Addressed** | Contributions l. 103–121; `sec:res-e10` is its own subsubsection (l. 688) | No undefined `\ref` (checked statically). |
| Minor 15: timing rounding | **Addressed** | l. 990–996 | Consistently 329 ms; the comparison of surrogate student and hybrid teacher is explained. |
| Minor 16: E8 MNIST "not distinguishable" | **Partially** | Fixed in abstract, E8, Discussion and Conclusion; **still in E3** l. 1090–1092 | "DPM-Solver++ with four evaluations is not distinguishable from the one-step student … so there the saving is a factor of four". The comparison is *inconclusive* ([−12.0%, +24.3%], point estimate favouring the solver), so "at most a factor of four" is the supportable statement. |
| Minor 17: classifier accuracies | **Addressed** | l. 349–351 | |
| Minor 18: KMNIST split shift | **Addressed** | l. 493–496 | |
| TODO H1: read OQIDDM in full | **Not addressed** | l. 142–146; Limitations l. 1228–1229 | Still blocks submission (human item, `paper/TODO.md`). |
| TODO H2: affiliation, venue format | **Not addressed** | l. 21 | Not a scientific issue. |
| Presentation: three blocks, move tables to the appendix, update the forest plot, shorter abstract | **Addressed** | §5.1–5.3; appendix | Main text now has 13 tables and 7 figures (16 tables in the appendix). The abstract is about 245 words. The forest plot is regenerated with six-seed E10 intervals (`make_assets.py` l. 793–818). One placement issue remains (New presentation 1). |

---

## New major issues

### New Major 1. The R6 seed bootstrap is anti-conservative and is presented as a robustness check [ANALYSIS]

**Location:** §4 "Equivalence analysis" l. 468–470; E1 pooled l. 630–631; E10 l. 700–705; `make_assets.py` `boot_pool` (l. 771–777).

**Problem.** `boot_pool` resamples seeds with replacement (3 seeds for E1, 6 for E10), averages over cells, and takes the 5th and 95th percentiles. With n = 3 there are only 10 distinct resamples. With n = 6 the percentile bootstrap of a mean still understates the spread by about √((n−1)/n), and it ignores the t-quantile. The result is that every "bootstrap over seeds" interval reported is *narrower* than the cell-level t-interval it is meant to check:
- E1 hybrid/surrogate: [−2.2%, +2.1%] vs [−2.9%, +2.9%].
- E10 hybrid/classical branch at h ≥ 16: [−7.0%, −3.3%] vs [−8.9%, −1.1%].

A check that accounts for the shared autoencoder should widen the intervals, not narrow them. The natural seed-clustered analysis is a t-interval with n − 1 degrees of freedom over seeds, applied to each seed's mean log-ratio across cells. I recomputed it from the per-seed log-ratios in `stats.json → e10.cells`:

| E10 pooled comparison | cell-level t (paper) | seed bootstrap (paper) | **seed-level t (recomputed)** | per-seed means (%) |
|---|---|---|---|---|
| hybrid / surrogate (12 cells) | +1.0 [−1.4, +3.4] | [−1.8, +4.4] | **+1.0 [−3.1, +5.2]** | −3.4, +0.4, +3.0, +10.4, −1.1, −2.9 |
| hybrid / classical branch, h ≥ 16 | −5.1 [−8.9, −1.1] | [−7.0, −3.3] | **−5.1 [−7.6, −2.6]** | −0.9, −4.5, −6.5, −5.3, −3.3, −10.0 |
| hybrid / classical branch, h ≤ 8 | +16.8 [−1.1, +37.9] | — | **+16.8 [+7.2, +27.2]** | +11.2, +9.1, +22.5, +41.0, +6.8, +13.3 |

The equivalence conclusion survives: the upper bound of 5.2% is still well inside the 10% margin. But the honest interval is about twice as wide as the one the paper reports. Also, at h ≤ 8 the circuit is *significantly worse* than the classical branch under the seed-level analysis; the paper reports this comparison as non-significant.

**Fix.** [ANALYSIS] Replace the percentile bootstrap with the seed-level t-interval, or report both and base the claims on the wider one. Drop the 3-seed E1 bootstrap or replace it in the same way. Adjust l. 702–705 and l. 1160–1161 accordingly.

### New Major 2. With even depth and near-zero rotations the entangled circuit starts at the separable function; the paper does not say so [TEXT] + [ANALYSIS, existing checkpoints]

**Location:** Method l. 236–247 (circuit, initialisation); Discussion l. 1170–1179 ("With weights initialised near the identity, the circuit also starts close to a product of single-qubit rotations").

**Problem.** In `QuantumFeatures` (`v2/core.py` l. 82–86, 96–105) each layer applies rotations followed by the CZ ring. All CZs are diagonal and commute, so CZ_ring² = I. Without re-uploading, nothing data-dependent sits between layers. At w = 0 the depth-2 circuit is therefore *exactly* the CZ-free circuit, and the depth-4 circuit is close to it. Every configuration in which entanglement is reported to be equivalent to the surrogate has even depth and no re-uploading: E1/E10 (depth 2), E2 without re-uploading (depth 2 and 4), and E9 (depth 2 and 4). In addition, the surrogate is initialised with the *same* parameters: `SeparableSurrogate.weights = randn(depth, q, 2)·0.1` draws from the same RNG stream, and the minibatches are shared. So in these cells hybrid and surrogate start from almost the same function and see the same data.

I checked this on random angles in [−π, π]⁸ with weights from N(0, 0.1²). The RMS difference between the entangled and CZ-free circuit outputs, against an output SD of about 0.64, is:
- depth 2, no re-uploading: **0.037**;
- depth 4: 0.11;
- depth 3 (not used): 0.42;
- depth 2 with re-uploading: **0.53**.

Re-uploading is the only setting in which the entangled circuit does *not* start at the separable function. It is also the only setting in which entanglement made results significantly worse (+16.9%).

This does not invalidate the null result. In trained E10 students (seed 3), the rotation weights move to an RMS of 0.27–0.77 rad. The trained circuit then differs from its own CZ-free version by RMS 0.01–0.34, which is non-trivial in several cells (e.g. Fashion-MNIST h = 0: 0.30; h = 32: 0.34). In others it stays almost separable (MNIST h = 16: 0.014; h = 0: 0.030). So entanglement is functionally present in many trained models and still does not help. Reported this way, the paper becomes stronger, not weaker. As written, however, a reader cannot tell whether the entanglement null partly reflects an optimiser that never leaves the neighbourhood of a separable initialisation. The Discussion sentence understates the coincidence: it is exact cancellation at even depth, not proximity.

**Fix.**
- [TEXT] State in §3.2 that, without re-uploading, CZ_ring^L = I for even L. Hybrid and surrogate therefore start from identical parameters and (at w = 0) identical functions in E1, E2 without re-uploading, E9 and E10. Rewrite Discussion l. 1172–1173 accordingly, and add initialisation and parity of depth to the Limitations.
- [ANALYSIS, recommended] For each trained hybrid model, report a scalar "entangling contribution": the RMS difference between the trained circuit and the same weights without CZ, evaluated on the *actual* compressor angles from training latents, relative to the output SD. Plot the hybrid/surrogate log-ratio against it. If the ratio shows no trend while the contribution is large, the null is not an initialisation artefact.
- [EXPERIMENT, optional] One slice (e.g. MNIST and Fashion-MNIST, h ∈ {0, 16}) with depth 3, or with uniform initialisation of the rotations.

### New Major 3. Results that go against the narrative are again left out of the new text [TEXT]

**3a. R1 teacher-level circuit results.** `tab:r1` and `stats.json → r1_rules` show the following. I recomputed the per-cell rules from the raw JSON.
- **E9 teacher, hybrid vs classical branch:** −8.5% [−12.4%, −4.4%] in FID-like. The circuit criterion passes in **5 of 16** cells at 5,000 samples: Fashion h = 0 q4 zz, Fashion h = 0 q6 z and zz, Fashion h = 4 q4 zz, and MNIST h = 16 q4 z. At 512 samples it passed in three.
- **E10 teacher at h ≥ 16:** −6.7% [−13.0%, −0.0%].
- **E10 teacher, entanglement rule:** passes in MNIST h = 128, the same cell as the student.

The text (l. 787–798) reports only the student per-cell rules, writes "and the same for the teachers" only for hybrid/surrogate, and concludes "The conclusions of the sweeps do not depend on the sample size or on the metric." In E9 the sign of the hybrid/classical-branch effect flips between student (+38%) and teacher (−8.5%), and in latent FD between teacher FID-like (−8.5%) and teacher latent FD (+31.9%). That is metric dependence. It does not touch the entanglement conclusion, but it is exactly the kind of omission round 1 flagged (Minor 1). R1 also preregistered "for each cell … in each metric", and none of the per-cell latent-FD or MMD results is given.

*Fix:* add one or two sentences on the teacher results at 5,000 samples, including the 5 E9 teacher circuit passes and whether the surrogate shares them. The comparison surrogate/classical branch for the teacher at 5,000 samples is not computed; compute it. Soften l. 797–798 to say that the *entanglement* conclusion does not depend on sample size or metric.

**3b. The MNIST floor reading contradicts a preregistered R4 row.** l. 489–491: "On MNIST the 32-step teacher (18.7) is above the real floor (5.7) and the autoencoder's reconstructions of the same images (11.0)." The R4 row "AE recon., reference" (`tab:floors`) is **20.3** for MNIST. Per seed it is 17.0 / 22.4 / 21.6, against teacher values of 16.1 / 22.5 / 17.6 (recomputed from `e6-e7-e8-review-eval.json`). Reconstructing the reference images themselves therefore scores *worse* than the teacher. The text never mentions this row. Three readings of the AE floor on MNIST (11.0, 9.9, 20.3) span a factor of two, so the statement "on Fashion-MNIST and KMNIST the autoencoder is the bottleneck", read as excluding MNIST, is not supported. On the reference-reconstruction reading, the MNIST teacher is also at the AE limit.

*Fix:* report the 20.3 and say that on MNIST the AE floor depends on which images are reconstructed. Either say that the teacher is at the reconstruction level of the reference itself, or argue why the held-out-train reconstruction is the right decoder floor. That argument is not obvious, since FD(recon(held-out train), ref) < FD(recon(ref), ref).

---

## New minor issues

1. **[TEXT/ANALYSIS] R2 omits a preregistered level and counts improvements as detections.**
   - The round-1 preregistration lists a ∈ {0, 0.5, 1.5, 2, 3}. a = 1.5 was run (36 rows in `r1-e10-rescore-r2-perturb.json`) but `R2P` in `make_assets.py` (l. 739) drops it. The "44 of the 72" count therefore excludes 12 preregistered perturbations; with them it is 47 of 84.
   - I recomputed a = 1.5. It is "detected" in 3 of 12 cases (MNIST h = 0 student −16.6%, Fashion h = 0 student −26.8%, Fashion h = 0 teacher −28.3%), and all of these are *improvements*. With a = 2, 8 of 12 cases are also improvements (down to −34%).
   - The R2 text presents all detections as evidence that "the evaluation detects changes of the branch's function". It does, but a reader should be told that amplifying the branch's input-dependent part by 1.5–2× *improves* the one-step FID-like by up to a third at small h. That is the same under-dispersion effect the paper documents for latent noise, and it shows that FID-like differences between branch types may partly reflect dispersion.
   - *Fix:* add the a = 1.5 column, give the count over all preregistered perturbations, and add one sentence on the sign.
2. **[TEXT] R2 permutation sentence is incomplete.** l. 535–537 says permutation is detected "for the teacher at h = 0". `tab:r2` also marks the MNIST h = 4 teacher (+34*). *Fix:* "for the teacher at h = 0 (both datasets) and MNIST h = 4".
3. **[TEXT] R3 summary is optimistic, and two R3 pairs were not preregistered.**
   - l. 568–570: "resolve differences of roughly 10–20% with three seeds". With three seeds at 5,000 samples, an 18.6% difference (Fashion MLP h = 16 vs 32, [−5.6%, +49.0%]) and an 8.4% difference (Fashion hybrid 32 vs 128) are *not* resolved. A fairer summary is "differences of about 15–20% or more, not reliably below".
   - The preregistration lists MLP 16/32, MLP 8/16 and hybrid 32/128. The rows "MLP only, h = 32 vs 128" and "Hybrid, h = 16 vs 32" are additions, and the second one is used in the text ("detected everywhere"). Mark them as added after the preregistration in l. 443–458.
4. **[ANALYSIS] R1 latent MMD is not reported** (see Major 6 above). For the record, my recomputed hybrid − surrogate MMD differences (×10³, cells as units) are all consistent with zero and small against values of 7–14 (×10³):
   - E9 student +0.65 [−0.39, +1.69];
   - E9 teacher +0.03 [−0.10, +0.15];
   - E10 student +0.17 [−1.16, +1.50];
   - E10 teacher +0.20 [−0.18, +0.57].

   One column in `tab:r1` or one sentence suffices.
5. **[TEXT] The E7 MMD interval that excludes 0 is not mentioned.** l. 984–988 lists the E7 intervals that exclude 1 "while staying inside the margin". It omits the hybrid vs no-entanglement teacher in latent MMD, +0.06 [+0.01, +0.11] ×10⁻³ (`tab:e7`). R6 preregistered "todos os intervalos que excluem 0 nos dois sentidos".
6. **[TEXT] Conclusion and Discussion overstate in places.**
   - l. 1249–1252: "entanglement never beat an exact separable surrogate; the same held with six seeds per cell, with a two-qubit correlator readout and with 5,000 samples". Not literally true: with correlators entanglement passed once (E9 MNIST h = 0, 1/8), and at 5,000 samples once in each sweep (E9 MNIST h = 0 zz; E10 MNIST h = 128). *Fix:* "met the criterion at most once per sweep, never reproducibly".
   - l. 1157–1158: "A classical layer of the same shape matches the circuit or beats it". At h ≥ 16 the circuit is significantly better than the classical branch by about 5%, and in MNIST h = 32 it meets the criterion. The next sentence qualifies this, but the claim itself should say "matches it within the margin or beats it".
   - l. 1159–1160: "never improved … in a way that survived replication". The E9 pass was never tested on new seeds, so it did not "fail" replication. Write "that was replicated".
7. **[TEXT] Abstract wording.** l. 39–41: "differ by +1.0% (90% CI [−1.4%, +3.4%]), also with 5,000 samples and latent-space metrics" reads as if the same numbers held at 5,000 samples. R1 gives +1.9% [−1.9%, +5.8%] (E10) and −1.0% (E9). *Fix:* "and remain equivalent with 5,000 samples and latent-space metrics". Also, "Perturbing the branch is detected wherever the branch carries the model" (l. 41–42) depends on how "carries" is defined. At Fashion h = 16 no teacher perturbation except readout zeroing is detected. "where the branch carries most of the model (h ≤ 4)" is accurate.
8. **[TEXT] E3 wording.** l. 1090–1092 still says "not distinguishable … a factor of four" (Minor 16 above).
9. **[TEXT] Loose "indistinguishable"/"equivalent".** l. 972 "indistinguishable DDIM curves". Round 1 asked to keep "equivalent" to the TOST meaning and to avoid "indistinguishable".
10. **[TEXT] R5 control is only partial.** The held-out loss was added only for the 100k model, so the 10k model's overfitting was not measured directly. The inference "the deficit is not overfitting" rests on the 100k model giving the same FID-like (l. 1040–1041), which is a fair argument. Say so in those terms. Note also that 100k pairs × 200 epochs is about 10× the consistency student's gradient steps, which favours the baseline. That is worth one clause, because it strengthens the conclusion.
11. **[ANALYSIS] R1 pools single-qubit and correlator cells in E9.** `R1CELLS['E9']` pools 8 zz and 8 z cells into one "(16)" family. With zz the classical branch has 90 parameters, with z it has 36. The hybrid/classical-branch pooled numbers in `tab:r1` (+38.4%; teacher −8.5%) therefore mix two different controls. Report them separately, or say so in the caption.
12. **[TEXT] `generated/tab_e6.tex` is stale and unused** (the paper uses `tab_e6_full.tex`). Delete it to avoid confusion.

---

## Numerical spot-check table

Sources: `paper/generated/stats.json` (S), generated tables, and recomputation (R) from `v2/reports/review-2026-11/*.json`. All rows are claims new or changed in this revision.

| # | Claim in text | Location | Source value | Match |
|---|---|---|---|---|
| 1 | Circuit criterion 1/23, entanglement 0/23 | Abstract l. 37–38 | S `criteria_primary`: e1 {1, 0}, e2 {0, 0}; `unique_primary_cells` = 23 | yes |
| 2 | Six seeds: entanglement 0/12; correlators 1/8 | Abstract l. 38–39 | S `e10.entanglement` = 0; `e9.passes.student/ent` = 1 | yes |
| 3 | Pooled +1.0% [−1.4%, +3.4%] | Abstract l. 39–40; l. 700 | S `e10.pooled.hybrid/surrogate` +0.96 [−1.44, +3.43] | yes |
| 4 | One-step student matches/beats teacher on three datasets; 9–13% worse on MNIST | Abstract l. 44–45 | E7: KMNIST +2.07 [−0.58, +4.79] (equiv.), Fashion −2.44, Pneumonia −18.52; MNIST +13.05 (sel.) and +9.05 (E5) | yes |
| 5 | Floors: teacher 18.7, real floor 5.7, AE recon held-out 11.0; Pneumonia 5.8 vs 5.2 | l. 489–492 | `tab_floors`: 18.7, 5.7, 11.0; 5.8, 5.2 | yes (but see New Major 3b: AE recon of reference is 20.3) |
| 6 | Fashion/KMNIST AE recon 30.2 / 159.6, recall 0.30 / 0.08; teacher 42.9 / 212.5 | l. 492–493 | `tab_floors` identical | yes |
| 7 | KMNIST unseen train 28.3 vs test 2.1 | l. 494–495 | `tab_floors` 28.3 / 2.1 | yes |
| 8 | MNIST MMD 11.9 and 11.8 vs 11.3 (×10⁻³) | l. 498 | `tab_floors` 11.9 / 11.8 / 11.3 | yes |
| 9 | Preregistered MNIST floor 17.3, recon 9.9 | l. 501–503 | `tab_floors` 17.3 / 9.9 | yes |
| 10 | E6 branch perturbations move FID-like by −9% to +7% | l. 525–527 | S `e6_branch_range` [−8.83, +7.01]; `e6_branch_any_detect` = false | yes |
| 11 | R2: 44 of 72 meet rule | l. 532–533 | S `r2_detected` = 44, `r2_total` = 72 | yes as computed; **incomplete**: a = 1.5 omitted (47/84 with it; New minor 1) |
| 12 | R2: a = 0 worsens student by 14–274% in all six cells | l. 534–535 | S `r2_a0_range_student` [14.18, 274.35]; `r2_a0_detected_student` = 6 | yes |
| 13 | Permutation detected for teacher at h = 0 and student MNIST h = 16, Fashion h = 4 | l. 535–537 | `tab_r2`: also MNIST h = 4 teacher +34* | **partial** (one cell omitted) |
| 14 | R3: MLP halving/quartering +9% to +78% | l. 559–560 | `tab_r3` min +8.5 (Fashion 32/128, 6 seeds), max +78.0 | yes |
| 15 | R3: all MLP differences significant with six seeds; with 3 seeds/5,000 all but Fashion h = 16 vs 32 | l. 560–562 | `tab_r3`: all 6-seed lower bounds > 0; Fashion 16/32 [−5.6, +49.0] | yes |
| 16 | Hybrid h = 16 vs 32 +16 to +31% detected everywhere; h = 32 vs 128 +2 to +11%, only Fashion with six seeds | l. 563–566 | `tab_r3`: 15.9–31.1, all excl. 0; 1.6–10.7, only Fashion 6-seed [+3.4, +8.9] | yes |
| 17 | Contamination 5%: +7 to +19% | l. 572 | S `e6_contamination5_range` [6.90, 19.23] | yes |
| 18 | Latent noise decreases FID-like up to 32% at s = 0.4; precision falls, recall rises | l. 572–574 | `tab_e6_full` −32 (Pneum. teacher); R: MNIST teacher P 0.802→0.726, R 0.776→0.861 (same direction on all datasets) | yes |
| 19 | Power 0.43/0.76/0.95; false pass 3–10% | l. 622–624 | S `power`: 0.426 / 0.764 / 0.946; 0.027 / 0.095 | yes |
| 20 | E1 bootstrap over seeds [−2.2%, +2.1%] | l. 630–631 | S `r6_bootstrap.E1` [−2.15, +2.10] | yes (method anti-conservative; New Major 1) |
| 21 | E10: MNIST h = 32 −11.5%, 5/6; h = 16 −6.3%, 4/6; geometric rule same counts | l. 694–699 | S `e10.cells`: −11.48 (5), −6.35 (4); `geometric_rule` {1, 0, 2} | yes |
| 22 | E10 bootstrap [−1.8%, +4.4%]; h ≥ 16 −5.1% [−8.9%, −1.1%], bootstrap [−7.0%, −3.3%]; surrogate/cl. br. −4.0% [−7.9%, −0.0%], bootstrap [−6.6%, −1.4%] | l. 700–705 | S `r6_bootstrap` and `e10.pooled`: identical to 0.1 | yes (method: New Major 1; seed-level t gives [−3.1, +5.2], [−7.6, −2.6]) |
| 23 | E10 per-cell: six intervals exclude 1; circuit better by 6–11%, worse by 34–64%; MNIST h = 8 +7.9% [+3.6%, +12.4%] | l. 707–712 | S `e10.significant_cells`: −11.48, −6.20, −7.27; +64.37, +33.63; +7.89 [+3.57, +12.39] | yes |
| 24 | R1: E9 −1.0% / −0.3%, E10 +1.9% / +2.7%, all within ±6.5%; P/R < 2 pp | l. 788–791 | `tab_r1`: −1.0 [−4.8, +3.0], −0.3 [−3.8, +3.3], +1.9 [−1.9, +5.8], +2.7 [−0.8, +6.4]; ΔP/ΔR −0.6/+1.6, +0.3/−0.4 | yes |
| 25 | R1 per-cell: one entanglement pass per sweep (E9 MNIST h = 0 zz; E10 MNIST h = 128); one circuit pass per sweep (E9 MNIST h = 16 z; E10 MNIST h = 32); ten classical-branch passes, all h ≤ 8 | l. 791–796 | S `r1_rules`; R: per-cell list confirms cells; against-passes 7 (E9, all h ∈ {0, 4}) + 3 (E10, h ∈ {0, 4, 8}) | yes (teacher passes not reported; New Major 3a) |
| 26 | R1 h ≥ 16 hybrid/cl. br. −4.7% [−11.5%, +2.6%], latent −1.3% | l. 796–797 | `tab_r1` identical | yes |
| 27 | E0: PneumoniaMNIST student 5.8 → 22.2 with readout zeroed | l. 823 | `tab_e0` 5.8 / 22.2 | yes |
| 28 | Student vs teacher at 5,000: precision −4, recall +5 pp; four-step +42% latent FD | l. 885–886, 892–893 | `tab_e7`: −4.0 / +4.6; +42.4 [+15.1, +76.3] | yes |
| 29 | Campaign E7: max bounds 2.1% runs, 3.2% datasets, 4.9% latent FD; ΔMMD ≤ 0.13 ×10⁻³; P/R < 0.5 pp | l. 976–979 | `tab_e7`: 2.1, 3.2, 4.9; 0.13; max |ΔP/ΔR| 0.3 | yes |
| 30 | E8: DPM++ with 6 NFE within 1% of / better than DDIM-32 on every dataset | l. 1028–1029 | `tab_e8`: 17.7/18.7, 43.3/42.9 (+0.9%), 192.1/212.5, 5.6/5.8 | yes |
| 31 | E8: −32.8%, −66.1%, KMNIST −1.1% [−4.0%, +2.0%], MNIST +4.6% [−12.0%, +24.3%]; DPM++8 +17.4%, +10.5% | l. 1031–1036 | `tab_e8_ratios` identical | yes |
| 32 | R5: 100k pairs +7.0%, +23.4%; −5.8%, +1.3%; held-out loss equals train loss | l. 1038–1041 | `tab_e8_ratios`; R: `r5-endpoint-training.json` e.g. MNIST s0 val 0.0085 / train 0.0088, Fashion s0 0.0081 / 0.0097 | yes |
| 33 | Timing: 10.8 vs 1.3 ms (×8); 329 vs 26 ms (×13); surrogate 5–6× faster (2.2 vs 10.8; 54 vs 329) | l. 990–996 | ratios 8.3, 12.7 (25.5 ms), 4.9 and 6.1 | yes ("5–6" rounds 4.9 up) |
| 34 | Retraining "three minutes before" the preregistration commit | l. 418–419 | `phase1.log` 15:44:41 vs 538048d 15:47:50 | yes |
| 35 | R1–R6 preregistered before running | l. 421–427 | 8c21add 20:40:47; `round1.log` START 20:42:24 | yes |

Summary: 35 checks. 32 match exactly. 2 are correct as computed but incomplete or based on a flawed method (rows 11, 20/22). 1 is a partial mismatch (row 13). I found no transcription errors between the raw JSON, `stats.json` and the generated tables. Re-running `review_eval` for R4 reproduced all 789 previously committed FID-like values exactly.

---

## Presentation

1. **Block structure works**, with one misplacement. "Hybrid and classical models are equivalent at h = 128" (§5.3.4, l. 967) is a circuit result. It sits inside the distillation block, after E5, and the timing paragraph is attached to it. Move the equivalence part to the circuit block, perhaps next to E0, which explains it, and keep the timing paragraph in the distillation block.
2. **Abstract** (about 245 words) is much improved and accurate except for the two phrasings in New minor 7. Target length and the ordering (sweeps first, distillation in one sentence) are as requested.
3. **Main text** is still long: 13 tables and 7 figures. `tab:e7` and `tab:e5` could move to the appendix with one sentence each, since the circuit question no longer depends on h = 128. `tab:r1` belongs in the main text.
4. **Forest plot** caption l. 734–737 says "E1 cells with six seeds (E10)". Given New Major 1, consider showing seed-level intervals or saying they are per-cell paired t-intervals.
5. **Terminology.** Use "equivalent" only in the TOST sense (l. 972). "Detected" in R2 should mean "detected change" in either direction, and the paper should say that explicitly (New minor 1).
6. **Method section** should state the parity of depth and the shared initialisation (New Major 2). It is one sentence and pre-empts the most obvious question a QML reviewer will ask about the entanglement null.
