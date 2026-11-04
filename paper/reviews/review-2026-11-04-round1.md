# Review: "Does the Circuit Matter? Controlled Ablations of Hybrid Quantum Latent Consistency Distillation"

Round 1 of the revised version (after the simulated review of 2026-11-03). Reviewed: `paper/main.tex` (commit b9e7460), `paper/generated/*.tex`, `paper/generated/stats.json`, `paper/analysis/make_assets.py`, `v2/PROTOCOL.md` (including its git history), `v2/core.py`, `v2/diagnostics/{review_eval,endpoint_regression,capacity_sweep}.py`, and the raw reports in `v2/reports/review-2026-11/`. Every number in the spot-check table below was recomputed or read from those sources. I also ran the unit tests (`python -m unittest v2.test_core`: 7/7 pass), and I checked the DPM-Solver++(2M) and Heun implementations against an analytic Gaussian-data denoiser (details in Major 5).

---

## Summary

The paper asks whether the variational circuit in a hybrid latent diffusion denoiser contributes anything that a classical block of the same shape could not. It does this in a consistency-distillation setting, where cutting circuit calls would matter on hardware. The model is a 16-D autoencoder latent with an MLP backbone plus an additive 4–8-qubit branch (compressor → circuit → linear readout). The controls are an exact separable surrogate (the circuit without CZ, simulated as Bloch vectors), a shape-matched classical branch (tanh(Linear(2q→q))), a backbone-only MLP, and branch ablations at evaluation. There are three preregistered sweeps: E1 over backbone width (h = 0…128), E2 over qubits, depth and re-uploading at h = 0, and E9 with a two-qubit correlator readout. Across these sweeps the entanglement criterion is met in one cell out of 43 (23 + 12 + 8). The circuit criterion is met in about one cell per sweep. Pooled with cells as the unit, hybrid and surrogate students are equivalent within about ±3%. The revision adds six preregistered review experiments:
- E5: MNIST replication on seeds 3–7.
- E6: positive control.
- E7: 5,000-sample re-scoring, floors, precision/recall and latent metrics.
- E8: DPM-Solver++(2M), Heun and direct endpoint regression baselines.
- E9: correlator readout.
- E10: three more seeds for the E1 grid.

The paper also reports a full retraining of the lost campaign checkpoints.

Overall this is an unusually careful negative-results paper. It has a written preregistration with dated addenda, every number is regenerated from committed JSON, and it states its limits candidly. The central conclusion holds up well under the new experiments: in this architecture, entanglement does not contribute detectably, and the small advantage of the circuit over the classical branch at wide backbones (about 5%) is shared by the separable surrogate. My concerns are about the new material:
1. The post hoc positive control is oversold, and it is not the control the previous review asked for.
2. The E5 replication is described backwards in two places.
3. The robustness evaluation (E7) was applied only to the h = 128 regime, which the paper itself calls uninformative about the circuit. The sweeps, where the circuit question is actually decided, are still scored with 512-sample FID-like only.
4. The autoencoder "floor" on MNIST is confounded, and the choice of floor varies across claims.
5. The E8 sampler grid was edited in the preregistration text itself, without an addendum, and the paper does not disclose it.
6. A preregistered metric (latent MMD) was computed but is never reported.

None of these needs new training. Most need only text or analysis changes. Two need inference-only runs on checkpoints that exist on the new server.

## Recommendation: **Major revision** (close to minor)

The main scientific claim, that the circuit and its entanglement make no detectable contribution in this model family, is well supported, and the revision strengthens it (E9 and E10 especially). But several claims are stated more strongly than the evidence supports, or are misreported:
- The abstract says a positive control shows that the evaluation "detects small, known degradations". That control is post hoc (not labelled so in the abstract), fails the preregistered detection rule at the advertised 2% level, and is paired at the sample level. So it says little about the sensitivity of comparisons between independently trained models.
- The E5 text says the fresh-seed MNIST gap is larger than the selection seeds suggested, when for the one-step student it is smaller (+9.1% vs +13.1%). What fails to replicate is the four-step claim, and that claim is still asserted in the distillation section.
- The 5,000-sample and latent-metric robustness check covers only the configuration the paper says is least informative.

Fixing these is mostly rewriting plus two cheap inference-only experiments: re-scoring the E9/E10 checkpoints with the E7 protocol, and branch perturbations at small h. I would expect to recommend acceptance after that. I keep "major" rather than "minor" because the abstract's evaluation-validity claim and the distillation claims have to change in substance, not just in wording.

---

## Status of each previous TODO item (`paper/TODO.md`)

| # | Item | Status | Comment |
|---|---|---|---|
| 1 | Replicate MNIST on fresh seeds with the fixed budget | **Addressed** (experiment); **interpretation wrong** | E5 ran on seeds 3–7 as preregistered, and all numbers match `stats.json`. The equivalences replicate. The text misstates what the replication shows about the distillation gap (Major 2). The 8-seed pooled column includes the three selection seeds, and the retrained versions of those seeds are effectively identical to the originals (Minor 6). |
| 2 | Positive control: perturb the branch in a known way and show that a change of about 10% is detected | **Partially addressed** | The preregistered branch perturbations (scaling, permutation) move FID-like by at most −9% to +7% and never meet the preregistered rule, so the control asked for was not achieved. The post hoc contamination control shows that the *metric* responds to bad samples. It does not show that the *pipeline* detects a ~10% change in a model's function, and it was not applied where the branch carries the model (Major 1). |
| 3a | Real-vs-real and AE floors for all datasets | **Addressed, with a confound** | Floors exist for all four datasets. The MNIST AE floor is computed on the non-exchangeable test half and scores *better* than the raw images it reconstructs (9.9 vs 17.3), so it is not a floor. The choice of floor changes from claim to claim (Major 4). |
| 3b | More samples (2,000–5,000) in the main comparisons | **Partially addressed** | 5,000 samples were used only for the h = 128 campaign. The comparisons that decide the circuit question (E1/E2/E9/E10) still use 512 samples (Major 3). |
| 3c | Precision/recall | **Addressed** | The implementation is correct (k-th neighbour excluding self, and the standard Kynkäänniemi definitions). The minor issue for PneumoniaMNIST is that the sets are unequal (624 real vs 5,000 generated) (Minor 9). |
| 3d | Latent-space metric (Fréchet or MMD) | **Partially addressed** | Latent Fréchet is reported. The unbiased RBF-MMD was preregistered, computed (`review_eval.py`, `latent_mmd`) and described in §4 (l. 373–376), but it is never reported (Major 6). |
| 4 | Few-step baselines (DPM-Solver++/Heun/UniPC, endpoint regression) | **Addressed**, with a disclosure problem | DPM-Solver++(2M), Heun and endpoint regression are all implemented. I verified both solvers numerically: they converge at second order. The preregistered evaluation grid was changed by editing the preregistration in place, and the paper does not disclose that change (Major 5). Progressive distillation was optional and was not done; that is acceptable. The endpoint baseline has a smaller data budget than the student (Minor 10). |
| 5 | Correlator readout or fixed readout rank | **Addressed** (correlators) | E9 is well designed: the surrogate's correlators are exact products, and unit tests check factorisation without CZ and its breakdown with CZ. The fixed-readout-rank variant was not done; the paper says so in the Discussion. The E9 classical branch then has 90 parameters against the circuit's 16, which explains the +62% (Minor 4). |
| 6 | Qualitative sample grid | **Addressed** | `fig_samples.pdf` (seed 0, 4 datasets, shared noise). Samples are visually near-identical across teacher, hybrid and classical students. |
| 7 | More seeds in the noisiest E1/E2 cells | **Partially addressed** | E1 now has six seeds (E10). E2 still has three seeds, and its re-uploading cells were the noisiest (per-cell SDs up to 62; pooled RU hybrid/surrogate interval [+0.2%, +36.4%]). |
| H1 | Read OQIDDM in full and fix the bib/novelty claim | **Not addressed** | Still described as "abstract and partial preview" (l. 156–158; Limitations l. 1363–1364). This is the closest consistency-based quantum diffusion model, so it should be resolved before submission. |
| H2 | Affiliation, acknowledgements, venue format | **Not addressed** | Placeholder footnote (l. 21). |
| H3 | Overleaf branch points to `v1/` | n/a | Not a manuscript issue. |

---

## Major issues

### Major 1. The positive control does not show what the abstract and Discussion claim [TEXT] + [EXPERIMENT, inference only]

**Location:** Abstract l. 49–51; §5 "E6" l. 1036–1045; Discussion l. 1297–1298 and l. 1336–1340; Conclusion l. 1387–1388; Table `tab:e6`.

**Problem.**
1. *The abstract misreports the result.* "Replacing 2% of the samples is detected on every dataset" uses "detected" in the sense of "90% interval excludes 0". The preregistered E6 detection rule (PROTOCOL.md, E6; restated in the paper at l. 1025–1026) also requires a change of at least 10%. Contamination at 2% gives +3% to +7% (Table `tab:e6`, all marked °, none marked *), so it **fails the preregistered rule**. The contamination control is also post hoc (PROTOCOL addendum dated after E6 was seen; commit 8f4f3e7), and the abstract does not say so.
2. *The control is paired at the sample level, so it does not test the relevant sensitivity.* In `review_eval.py` the contaminated set is `cat([bad[:k], clean[k:]])`: 95–98% of the samples are *identical* to the unperturbed reference, and the only randomness left in the ratio comes from the replaced samples. Its narrow interval therefore reflects resolution of the *metric* for a fixed sample set. The equivalence tests compare *independently trained* models, whose intervals are dominated by training variance between seeds. That is exactly why the 5,000-sample intervals are barely narrower than the 512-sample ones (l. 1082–1084). The narrowness of the equivalence intervals already shows their precision. The claim the previous review asked to support is different: that a known change in the *model's function* of about 10% would be detected. Contamination with 2-step DDIM samples (FID-like 446 on MNIST) is a change to the *sample set*, not to the denoiser.
3. *The one control that does target the branch says the opposite at h = 128.* The preregistered branch scaling (×0 to ×3 of the input-dependent part) and the permutation move FID-like by only −9% to +7% (`stats.json: e6_branch_range`). I recomputed the unreported a ∈ {0.5, 1.5}: −2.7% to +2.9%. So at h = 128, the metric cannot see large changes in the branch's input-dependent function. The paper says so ("consistent with E0"), but it then concludes "the equivalence intervals … are not an artefact of an insensitive metric" (l. 1043–1045). For the circuit question at h = 128, the metric is insensitive to the branch, by construction of the model.
4. The Discussion claim (l. 1297–1298) that "the positive control (E6) shows that pooled intervals of a few percent are meaningful" is a non sequitur. E6 is on the retrained h = 128 campaign with 5,000 samples. The pooled intervals it is invoked for (E1/E10) use 512 samples, original classifiers for seeds 0–2, and cells as units.

**Fix.**
- [TEXT] In the abstract, either remove the positive-control sentence or rewrite it along these lines: "a post hoc control (contaminating 5% of samples) is detected under the preregistered rule on 3 of 4 datasets; the preregistered branch perturbations were not detected." Check the wording against the table: at 5%, the rule is met on MNIST, Fashion and PneumoniaMNIST but not KMNIST (+7°). Label it post hoc. Delete or qualify l. 1043–1045 and l. 1297–1298.
- [ANALYSIS] Add a *model-level* positive control from data that already exists: pairs of independently trained pipelines with a known ordering, run through the same paired TOST machinery. Candidates: the student budget 200/0.999 vs 200/0.95 (32.3 vs 28.0, about +15%, Table `tab:budget`); E10 hybrid at h = 32 vs h = 128 (31.5 vs 28.3); MLP-only h = 16 vs h = 32. These show whether the pipeline detects a ~10–15% difference between models, which is the question the equivalence results depend on.
- [EXPERIMENT, inference only] Run the preregistered E6 branch perturbations (scale, permutation, readout zeroed) on the **E10 seeds 3–5 checkpoints at h ∈ {0, 4, 16}**, where the branch carries most of the model. That is the regime where the circuit question is informative, and the place where a branch-level positive control can succeed. The checkpoints exist on the new server.

### Major 2. The E5 replication is described backwards, and the four-step MNIST claim does not replicate [TEXT]

**Location:** §5 E5 l. 527–530; Limitations l. 1347–1350; §5 "Distillation transfers" l. 808–809 and l. 815–816; Contributions l. 120–121; Discussion l. 1316–1318.

**Problem.** The one-step student on the selection seeds (0–2) is +13.1% worse than the 32-step teacher (`tab:studteach`; `stats.json: equivalence_campaign/student1_vs_teacher32/mnist` = +13.12%). On the five fresh seeds it is +9.1% [+6.8%, +11.4%]. The fresh-seed gap is **smaller**, not "slightly larger than one would read from the three selection seeds" (l. 529), and the Limitations repeat the error ("somewhat larger than the selection seeds suggested", l. 1349–1350). The quantity that does get worse on fresh seeds is the **four-step** student: −2.2% and 3/3 seeds within 10% on the selection seeds, against +8.9% [−4.6%, +24.3%] and 3/5 on fresh seeds. Yet §5.5 still states that on MNIST "four steps are needed" and that "the advance criterion is met in all seeds … at 4 steps on MNIST" (l. 809, l. 815–816). Those statements come from the selection seeds only and are not supported out of sample. E7 adds that the four-step student is +42% worse than the teacher in latent Fréchet distance. Finally, Contribution 5 says distillation "replicates on fresh seeds", but the preregistered E5 distillation criterion failed (3/5 seeds; the interval upper bound of 11.4% exceeds 10%).

**Fix.** [TEXT]
- Rewrite l. 527–530 and l. 1347–1350 to state the actual pattern: the one-step gap is about 9–13% on both seed sets, and the four-step result was optimistic on the selection seeds, falling from 3/3 to 3/5 within 10%.
- Qualify the four-step MNIST statements in §5.5 with the E5 numbers.
- In Contribution 5, replace "replicates on fresh seeds" with "the MNIST gap replicates on fresh seeds; the preregistered within-10% criterion is not met".

### Major 3. The robustness evaluation (E7) was applied to the regime that does not bear on the circuit [EXPERIMENT, inference only] + [TEXT]

**Location:** §4 "Robustness evaluation" l. 367–383; §5.7 l. 972–978; §5.9 l. 1081–1094; Abstract l. 50–54; Limitations l. 1346–1347.

**Problem.** The paper says clearly (l. 972–976) that the h = 128 campaign "is the one in which equivalence is least informative about the circuit", and that the evidence about the circuit comes from the sweeps. Yet all of the new robustness work (5,000 samples, precision/recall, latent Fréchet/MMD, floors) was done on h = 128 only. E1, E2, E9 and E10 are still scored with 512-sample FID-like from a small classifier. This is exactly the metric whose validity TODO item 3 questioned, and E6 shows that it rewards added variance. The abstract then advertises the 5,000-sample equivalences as a main result. Previous TODO 3b asked for more samples "in the main comparisons", and for the circuit question those are the sweeps.

**Fix.**
- [EXPERIMENT, inference only] Re-score the existing E9 checkpoints (seeds 0–2, 8 cells) and the E10 new-seed checkpoints (seeds 3–5, 12 cells) with `review_eval`'s `Scorer`: 5,000 samples, P/R, latent FD and MMD. Then recompute the per-cell and pooled hybrid/surrogate and hybrid/classical-branch intervals in each metric. This is cheap (the models are tiny) and gives the circuit question the same robustness check the h = 128 regime received.
- [TEXT] In the abstract, give the h = 128 equivalences less weight than the sweep results, or replace them with the re-scored sweep results.

### Major 4. Floors: the MNIST AE floor is confounded, and the choice of floor changes from claim to claim [EXPERIMENT, inference only] + [TEXT]

**Location:** §4 l. 377–382; §5.9 l. 1069–1079; Table `tab:floors`; PROTOCOL addendum to E7.

**Problem.**
1. `review_eval.py` reconstructs the **preregistered floor images**, i.e. test images 5,000–9,999 (`recon = auto.decode(auto.encode(floor_images))`). On MNIST these raw images score 17.3 against the reference, but their AE reconstructions score **9.9** (seeds: 8.4/11.2/10.2). Reconstruction makes them look *more* like the reference: the AE pulls the second test half (a different writer population) toward the training manifold. A quantity that improves on the real data it reconstructs is not a floor. The sentence "not far from the autoencoder's own reconstructions (9.9 …)" (l. 1072–1073) therefore compares the teacher with a confounded number.
2. Floors are chosen per claim. For MNIST the text uses the post hoc unseen-train floor ("the relevant one", l. 1079). For the latent FD claim "at the level of the real floor on MNIST, KMNIST and PneumoniaMNIST" (l. 1075–1077), KMNIST silently uses the post hoc floor too (0.46 vs teacher 0.45). The preregistered KMNIST latent floor is 0.03, which puts the teacher 15× above it. On KMNIST the unseen-train FID-like floor is 28.3, against 2.1 test-vs-test, so train and test also differ there. The MMD numbers (not reported) show the same pattern: KMNIST teacher MMD 0.0048, unseen-train floor 0.0047, test floor about 0. The post hoc floor is arguably the right reference for models trained on the training split, but that argument has to be made once and applied to all datasets, with both floors reported in the text.

**Fix.**
- [EXPERIMENT, inference only] Compute AE reconstructions of (a) the unseen-train images and (b) the reference images themselves, and report both.
- [TEXT] State one rule for which floor is relevant (for example, "held-out images from the training distribution, because generators are trained on the training split") and apply it to every dataset and metric. Report the preregistered floor alongside, and drop the claim "AE … is close to teacher" for MNIST unless (a) supports it.

### Major 5. The E8 preregistration was edited in place, and the change is not disclosed [TEXT]

**Location:** §4 "Added after the preregistration" l. 448–459; §5.6 l. 894–897; `v2/PROTOCOL.md` E8 paragraph.

**Problem.** The preregistration commit (538048d, 15:47) specifies "DPM-Solver++(2M) com 1/2/4/8 avaliações e Heun … com 3/5/7 avaliações". The implementation commit nine minutes later (43bdeaa, 15:56, before any E8 result) rewrote that sentence *inside the preregistration paragraph* to "3/4/6/8" and "4/6/8", and inserted the Heun note. This was done without the "Adendo post hoc" marking used for the E6/E7 changes. The paper discloses only the Heun first-step change. It does not disclose that the NFE grid was changed. Substantively the change is benign: DPM++ at 1–2 NFE equals DDIM, and the change preceded the results. But the protocol now reads as if the new grid had been registered, which undermines the paper's otherwise exemplary transparency.

Implementation check: I verified both solvers with an exact denoiser for Gaussian data (scratch script). DPM-Solver++(2M) error vs the exact ODE solution: 0.10 at 8 NFE, 0.037 at 16, 0.017 at 24, 0.0064 at 32, against DDIM's 0.051 at 32. Heun: 0.029 at 16, 0.0047 at 32. Both are second order and correct. Low-NFE accuracy is dominated by the first-order first step across the large terminal log-SNR gap (ᾱ_T = 10⁻⁶). On that toy problem the time-uniform grid used here beat a log-SNR-uniform grid and a "denoise to 0 from the last level" grid at 3–4 NFE, so I do **not** consider the solver baseline handicapped.

**Fix.** [TEXT]
- In PROTOCOL.md, restore the original sentence and add a dated amendment.
- In §4 (l. 448–459), list the NFE-grid change next to the Heun change.
- Optionally [EXPERIMENT, inference only], add a one-line grid sensitivity check (e.g. log-SNR-uniform levels) for the preregistered question "student vs best solver at ≤4 NFE". The MNIST answer is inconclusive ([−12.0%, +24.3%]).

### Major 6. A preregistered metric (latent MMD) is computed but never reported [ANALYSIS]

**Location:** §4 l. 373–376 (MMD described); Tables `tab:floors` and `tab:e7` (MMD absent); PROTOCOL E7 ("métricas no latente: Fréchet e MMD").

**Problem.** `review_eval.py` computes `latent_mmd` for every row, and the method section says it is reported, but no table or sentence gives an MMD value. Omitting a preregistered metric is selective reporting, even when the result is benign. From the raw report the MMD results are in fact informative and supportive. MNIST teacher/student about 0.012, matching the unseen-train floor (0.011), against 0.024 for the preregistered test floor. KMNIST teacher equals the unseen-train floor. Fashion teacher 0.0006–0.0012 against a floor of about 0.

**Fix.** [ANALYSIS] Add MMD (with interval) to `tab:floors` and `tab:e7`, or move it to the appendix with one sentence in the text.

---

## Minor issues

1. **[TEXT] Significant effects that go against the narrative are described as "only" or left out.** l. 756–758: "hybrid and surrogate teachers differ by only −2.4%". The interval [−4.0%, −0.8%] excludes 0, so with correlators entanglement makes the teacher slightly but significantly better. E10 MNIST h = 8 (Table `tab:e10`): the surrogate beats the hybrid in **6/6** seeds, +7.9% [+3.6%, +12.4%], so entanglement hurts significantly. The text does not mention it. E5 8-seed hybrid vs classical teacher: −2.5% [−4.4%, −0.6%] excludes 0. *Fix:* report every interval that excludes 0, in both directions, in one sentence or table column ("significant but inside the margin").
2. **[ANALYSIS] An unverified claim in E10.** l. 641–643 says the 5.1% advantage over the classical branch at h ≥ 16 "the separable surrogate shares". That interval is not computed. From `stats.json` I get surrogate/classical-branch at h ≥ 16 = −4.0% [−7.9%, −0.0%], which supports the claim only marginally. *Fix:* compute it in `make_assets.py` and report it.
3. **[ANALYSIS] E10 mixes classifiers within cells.** Seeds 0–2 are scored with the original classifiers and seeds 3–5 with the retrained ones (caption l. 653–655). The ratios are paired within a seed, which is fine. But the preregistered rule's "mean at least 10% lower" is implemented as a ratio of *arithmetic* means pooled over seeds (`verdict6`), and the mean ± SD columns mix classifiers. On Fashion-MNIST the classifier shifts FID-like by about 5% for the h = 128 models. Whether the shift is multiplicative for FID ~150 models is unknown. *Fix:* also report the rule with the geometric mean of paired ratios (as shown in the table), and say that the two agree. Note that seeds 0–2 cannot be re-scored because the checkpoints were lost.
4. **[TEXT] "Size-matched" classical branch.** Abstract l. 41, l. 108, Conclusion l. 1382. The classical branch has 36 parameters against the circuit's 16 (and 90 against 16 with correlators). It is *shape-matched*, not size-matched. The E9 statement that the classical branch wins "because its wider output now feeds a 10- or 21-dimensional linear readout" (l. 759–761) is a causal explanation that was not tested. *Fix:* use "shape-matched", and phrase the E9 explanation as a hypothesis, mentioning the parameter difference.
5. **[TEXT] E0 range inconsistent with its own table.** l. 1137: "zeroing its readout raises the teacher's FID-like by 19–100%". This matches ratios of means in `tab:e0` (19% … 100%). Table `tab:e0rel` gives the mean of per-seed changes, and for the PneumoniaMNIST teacher that is +104% [−41, +364]. *Fix:* say which definition is used, or write "19–104%" and mention the PneumoniaMNIST interval, which includes negative values.
6. **[TEXT] The retrained campaign is a re-execution, not a replication.** The retrained models give pixel variances identical to 5–6 significant digits and identical label entropies (e.g. MNIST seed 0 teacher-32: 0.2188713 vs 0.2188683), and the per-seed retrained/original log-ratios are constant within each dataset (`stats.json: reproduction`). CPU training was deterministic. The differences come only from the retrained classifiers, which l. 430–433 already suspects. *Fix:* say so explicitly; it is a stronger reproducibility statement than "within 1%". Also note that the E5 "seeds 0–7" column therefore includes the selection seeds unchanged, so only the seeds 3–7 column is out of sample.
7. **[TEXT] Preregistration timing.** Phase 1 (campaign retraining) started at 15:44:41 (`phase1.log`), three minutes before the preregistration commit (15:47:50) and before the implementation commit (15:56). E5 started at 16:25. Nothing with a criterion could have been seen, but "written into the protocol before any of these runs" (l. 420–421) cannot be verified for the retraining from the commit timestamps. *Fix:* one sentence of disclosure.
8. **[TEXT] Table `tab:e6` omits preregistered levels.** a = 0.5 and 1.5, s = 0.05 and 0.2, and 20% contamination are not shown. I recomputed them: a ∈ {0.5, 1.5} gives −2.7% to +2.9% (consistent with the text); s = 0.2 gives −10.5% on PneumoniaMNIST. *Fix:* show all preregistered levels (or put them in the appendix) and say where the full grid is.
9. **[ANALYSIS] Precision/recall for PneumoniaMNIST uses 624 real images against 5,000 generated ones.** The k-NN radii then come from sets of very different density, which biases P/R (Kynkäänniemi et al. use equal sizes). *Fix:* subsample the generated set to 624 (averaged over subsamples) for PneumoniaMNIST P/R.
10. **[EXPERIMENT, optional] The endpoint-regression data budget is not matched.** `endpoint_regression.py` trains on 10,000 *fixed* noise→endpoint pairs for 200 epochs. The consistency student sees 10,000 images × 200 epochs with fresh noise and levels in every batch. With a 16-D latent and a cheap teacher, 100k pairs cost almost nothing. The baseline's +9.4% (Fashion) and +23.8% (PneumoniaMNIST) deficits may be an overfitting artefact. *Fix:* report the train loss (in `student_endpoint_training.json`), add a held-out loss, or rerun with more pairs, or soften "competitive alternative" / "worse".
11. **[TEXT] Unverifiable statement in E4a.** l. 1241 says "Hybrid and surrogate students behave the same", but Table `tab:e4a` shows only the classical student. *Fix:* add those rows or point to the report.
12. **[ANALYSIS] Cluster intervals ignore seed-level correlation.** In E1/E2/E10 all cells of a seed share the same autoencoder. The cell-as-unit interval (`cluster_ci`) treats cells as independent. *Fix:* a two-way (cell × seed) mixed model or a bootstrap over seeds, as a robustness check of the pooled intervals.
13. **[TEXT] Ambiguous sentence.** l. 1320–1322: "direct endpoint regression is a competitive alternative on the same two". In context, "the same two" reads as the datasets where the student was "much better" (Fashion, PneumoniaMNIST), but endpoint regression ties the student on MNIST and KMNIST. *Fix:* name the datasets.
14. **[TEXT] Cross-references.** l. 422: "Sections \ref{sec:res-e5}, \ref{sec:res-e10}–\ref{sec:res-e8}" resolves to 5.2, 5.3–5.6, which leaves out E6 (5.8) and E7 (5.9). `sec:res-e10` is placed after a `\paragraph` and resolves to the E1/E2 subsection. Contribution 3 (l. 115) points to `sec:res-equivalence` (the h = 128 campaign) for "pooled equivalence intervals and a power analysis", which are in §5.3. Contribution 5 (l. 123) leaves out E5 (§5.2). *Fix:* give E10 its own subsection or `\subsubsection`, and fix the ranges.
15. **[TEXT] Timing rounding.** l. 813 gives "328 ms" and l. 982–983 give "329 ms" for the same quantity (source: 328.9 ms). Also, l. 985 compares the 32-step *surrogate student* (54 ms) with the 32-step *hybrid teacher* (329 ms). Same network size, but say so.
16. **[TEXT] E8 MNIST phrasing.** The abstract (l. 57–58) and Conclusion (l. 1391–1392) say the student is "not distinguishable" from DPM-Solver++ at 4 NFE on MNIST and KMNIST. KMNIST is equivalent ([−4.0%, +2.0%]). MNIST is merely inconclusive ([−12.0%, +24.3%], point estimate favouring the solver). *Fix:* "equivalent on KMNIST, inconclusive on MNIST". Also mention in the abstract that DPM-Solver++ with 6–8 NFE is better than the one-step student on MNIST (+17.4%) and KMNIST (+10.5%).
17. **[TEXT] Classifier accuracies.** l. 362–363 give the accuracies of the original classifiers. E5–E8 use the retrained ones (0.9592 / 0.8893 / 0.9317 / 0.8253, `review-2026-11/*validation.json`). Report both, or say they agree within 0.002.
18. **[TEXT] KMNIST train/test shift.** The unseen-train floor (FID-like 28.3, latent FD 0.46) against the test-vs-test floor (2.1, 0.03) shows that the KMNIST test split differs from the training split. That deserves a sentence in E4b/E7, because it means part of the gap to the floor is a split shift that no generator trained on the training split could close.

---

## Numerical spot-check

Sources: `paper/generated/*.tex`, `paper/generated/stats.json`, and for some rows a recomputation from `v2/reports/review-2026-11/*.json` and `v2/reports/campaign-datasets-teachers/*.json`.

| # | Claim in text | Location (main.tex) | Source value | Match |
|---|---|---|---|---|
| 1 | E10 hybrid vs surrogate, pooled +1.0%, [−1.4%, +3.4%] | Abstract l. 44–45; l. 639–640 | `e10.pooled.hybrid/surrogate`: +0.965, [−1.441, +3.430] | yes |
| 2 | E9 entanglement 1/8 cells; pooled −0.8%, [−8.3%, +7.3%] | Abstract l. 46–47; l. 749–755 | `e9.passes.student/ent` = 1; pooled −0.831, [−8.326, +7.276] | yes |
| 3 | Equivalence intervals within ±3.5% (FID-like) and ±5% (latent FD) at 5,000 samples | Abstract l. 52–54; l. 1084–1086 ("2.1%, 3.2%, 4.9%") | max \|bound\| over hybrid-vs-control rows: runs 2.08, datasets 3.17, latent FD 4.95 | yes |
| 4 | "Replacing 2% of the samples is detected on every dataset" | Abstract l. 50–51; l. 1041–1042 | `e6_contamination2_all_sig` = true, but changes are +2.7% to +7.5% (< 10%), so the preregistered rule is not met | **partial** (true only under the non-preregistered "CI excludes 0" criterion; see Major 1) |
| 5 | E5: student(1)/teacher(32) +9.1% [+6.8%, +11.4%], 3/5 within 10%; FID-like 29.5±2.5 vs 27.1±2.5; 4-step +8.9%, 3/5 | l. 524–527 | `e5`: +9.05 [+6.77, +11.37]; within10_1 = 3; means 29.53±2.46 vs 27.10±2.50; 4-step +8.90, within10_4 = 3 | yes |
| 6 | E5: hybrid vs classical teacher −2.5%; hybrid vs surrogate student +0.3%; all intervals inside the margin, also with 8 seeds | l. 521–523 | −2.54 [−5.81, +0.85]; +0.27 [−0.62, +1.17]; all `equivalent` = true (5 and 8 seeds) | yes |
| 7 | "The MNIST distillation gap is … slightly larger than one would read from the three selection seeds" | l. 528–529; l. 1349–1350 | Selection seeds +13.1% (`equivalence_campaign/student1_vs_teacher32/mnist`) vs fresh seeds +9.1% | **no** (fresh-seed gap is smaller; Major 2) |
| 8 | E7 MNIST student vs teacher +13.1% [+5.1%, +21.7%]; KMNIST +2.1%; Fashion −2.4%; PneumoniaMNIST −18.5% | l. 1088–1090 | `e7_equivalence/student1_vs_teacher32/*`: +13.05 [+5.05, +21.66]; +2.07; −2.44; −18.52 | yes |
| 9 | Floors: teacher 18.7 vs floor 5.7 (MNIST), 5.8 vs 1.7 (PneumoniaMNIST); AE 9.9 / 5.2; Fashion/KMNIST AE 30.8 / 203.2, recall 0.29 / 0.07, teacher 42.9 / 212.5; preregistered MNIST floor 17.3 | l. 1070–1079 | `floors`: 18.71 / 5.655; 5.754 / 1.707; 9.906 / 5.175; 30.85 / 203.24; 0.287 / 0.071; 42.92 / 212.52; 17.27 | yes (but see Major 4 on interpretation) |
| 10 | Latent FD of teacher/student "at the level of the real floor on MNIST, KMNIST and PneumoniaMNIST" | l. 1075–1077 | KMNIST: teacher 0.45, student 0.46; preregistered floor 0.03, post hoc 0.46 | **partial** (holds only with the post hoc floor for KMNIST) |
| 11 | E8: student vs best ≤4 NFE −32.8% (Fashion), −66.1% (PneumoniaMNIST), −1.1% [−4.0%, +2.0%] (KMNIST), +4.6% [−12.0%, +24.3%] (MNIST); vs DPM++8 +17.4% / +10.5%; endpoint +9.4% / +23.8% | l. 900–908 | `e8`: −32.76; −66.09; −1.06 [−4.02, +1.99]; +4.59 [−11.95, +24.25]; +17.36; +10.50; +9.37; +23.77 | yes |
| 12 | DPM-Solver++ at 6 NFE within 1% of or better than 32-step DDIM on every dataset | l. 896–898 | 17.7 vs 18.7; 43.3 vs 42.9 (+0.9%); 192.1 vs 212.5; 5.6 vs 5.8 | yes |
| 13 | E10: MNIST h = 32 −11.5%, 5/6; h = 16 −6.3%, 4/6; hybrid/classical branch h ≥ 16 −5.1% [−8.9%, −1.1%]; h ≤ 8 +16.8% [−1.1%, +37.9%]; 7/24 per-cell intervals inside the margin; new seeds alone 1 circuit / 0 entanglement pass | l. 633–645 | `e10`: −11.48 (5 wins); −6.35 (4 wins); −5.12 [−8.94, −1.13]; +16.78 [−1.08, +37.86]; `equivalent_cells` = 7; `new_seeds_only` = {1, 0} | yes |
| 14 | E9: teacher hybrid zz / z −6.2% [−10.7%, −1.6%]; students +1.5% / +1.6%; hybrid/classical branch +62% | l. 754–759 | `e9.pooled*`: −6.23 [−10.65, −1.59]; +1.53; +1.65; +62.04 | yes |
| 15 | Retrained campaign within 1% on MNIST / KMNIST / PneumoniaMNIST and 4–5% higher on Fashion | l. 429–432 | `reproduction`: +0.35…+0.41; +0.78…+0.97; −0.34; Fashion +4.38…+5.22 | yes ("4–5%" rounds +5.2) |
| 16 | E7 intervals "0.8–1.8 times narrower" than with 512 samples | l. 1082–1083 | ratios of interval widths (512/5,000): 1.24, 1.03, 1.84, 0.84, 1.32 | yes |
| 17 | Zeroing the readout raises teacher FID-like by 19–100% | l. 1137 | `tab_e0` (ratio of means): +100 / +19 / +19 / +72; `tab_e0_relative` (mean per-seed): +100 / +19 / +19 / **+104** | **partial** (definition-dependent; Minor 5) |
| 18 | features_mean: −0.1% to +6.9% on average; one seed +13.7%; varying-part RMS 5–11% | l. 1139–1143 | `tab_e0_relative` mean range −0.1…+6.9; `e0_features_mean_range_pct` max 13.74; `e0_varying_range` 0.048–0.112 | yes |
| 19 | Pooled E1/E2: −0.0% [−2.9%, +2.9%]; +2.0% [−2.0%, +6.1%]; +16.9% [+0.2%, +36.4%]; −4.0% [−9.3%, +1.5%]; +14.1% [−1.7%, +32.4%]; RU pooled +111% | l. 585–598 | `sweep_families`: −0.05 [−2.89, +2.88]; +1.97 [−1.99, +6.08]; +16.90 [+0.19, +36.39]; −4.01 [−9.26, +1.54]; +14.11 [−1.67, +32.41]; +110.8 | yes |
| 20 | Power: 0.43 / 0.76 / 0.95 at SD 0.10; 0.31 / 0.47 / 0.64 at SD 0.20; false-pass 3–9% | l. 577–580 | `power`: 0.426 / 0.764 / 0.946; 0.306 / 0.468 / 0.635; 0.027 / 0.095 | yes (9.5% shown as "9") |
| 21 | Campaign timing: 1-step 10.8 vs 1.3 ms; 32-step teacher 328 ms (l. 813) and 329 ms (l. 982); surrogate 2.2 / 54 ms | l. 812–813, 980–985 | hybrid-teacher-32 mean 328.9; classical 25.5; no-ent 329.4; surrogate-1 2.2; surrogate-32 53.9 | **partial** (328 vs 329 for the same value; Minor 15) |
| 22 | E3: 27.9 (1 step, 64 shots) vs 52.2 (4-step teacher); 144.4→158.4; 110.7→99.7; 144.4→138.7 | l. 1184–1198 | `tab_e3` rows identical | yes |
| 23 | E4b: reconstruction 204, real floor 13.7, latent-64 reconstruction 25, student 124.6 vs teacher 113.8 | l. 1258–1262 | `tab_e4b` 204.1 / 24.9 / 124.6 / 113.8; `e4b_floor` 13.67 | yes |
| 24 | Classifier accuracies 0.959 / 0.889 / 0.931 / 0.827 | l. 362–363 | `classifier_accuracy` (original classifiers): 0.9589 / 0.8889 / 0.9314 / 0.8269. Retrained (used in E5–E10): 0.9592 / 0.8893 / 0.9317 / 0.8253 | yes for the original classifiers (Minor 17) |
| 25 | Campaign dataset-level interval student vs teacher [−16.5%, +15.8%] | l. 811 | `student1_vs_teacher32/by_dataset` [−16.54, +15.75] | yes |

Summary: 19 of 25 checks match exactly. 4 match only partially (rows 4, 10, 17, 21: definition or rounding issues, or a non-preregistered criterion). 1 is a substantive mismatch in interpretation (row 7). I found no transcription errors between `stats.json` and the generated tables; the pipeline from reports to tables looks sound.

---

## Presentation

**Length and structure.** The `.tex` is about 1,480 lines, with **21 tables and 6 figures in the main text** plus 5 appendix tables. That is probably 30 or more pages compiled. The Results section has 12 subsections in an order that makes the argument hard to follow:
- E5 (a distillation replication) appears *before* the sweeps and refers to campaign equivalences that are only presented four subsections later.
- E6/E7 (validity of the evaluation) come after the campaign equivalence they are meant to support.
- E0 (the mechanism that explains everything) comes near the end.

I suggest three blocks:
1. **Is the evaluation valid?** E7 floors, E6 controls (moved first).
2. **Does the circuit contribute?** E1/E2/E10/E9, pooled intervals, power, E0 as mechanism.
3. **Does distillation work?** Campaign, E5, E8, E3.

E4 and the per-cell E2/E9 tables can go to the appendix.

**Tables and figures.** Too many tables carry overlapping information:
- `tab:students`, `tab:campaign`, `tab:studteach`, `tab:equivalence` and `tab:e7` all report the h = 128 campaign.
- `tab:e1` and `tab:e10` report the same grid with 3 and 6 seeds.

Keep `tab:e10` and move `tab:e1` to the appendix. Merge `tab:students` into `tab:campaign`. Merge `tab:equivalence` and `tab:e7` into one table with 512- and 5,000-sample columns. `tab:e6` should show either all preregistered levels or only the controls discussed. The forest plot (Fig. `fig:forest`) should be updated to the six-seed E10 intervals, since the text now relies on them.

**Abstract.** About 330 words, with more than 15 numbers and three nested result clauses. Its most prominent new claim (the positive control) is the weakest one (Major 1), and the h = 128 robustness equivalences get space they do not deserve (Major 3). Target about 200 words:
- the question and setting;
- the controls;
- the sweep result (1 of 23, 0 of 12 with six seeds, correlators 1 of 8, pooled ±3%);
- the mechanism (learned offset);
- distillation in one sentence, with the MNIST gap and the solver comparison stated as "better than or equivalent to DPM-Solver++ at 4 NFE on three datasets, inconclusive on MNIST";
- the scope disclaimer.

Drop the positive-control sentence or label it post hoc and state it under the preregistered rule.

**Clarity.** The writing is precise and unusually honest about what was preregistered and what was added later; the "Added after the preregistration" paragraph is a model to follow and should be complete (Major 5). Two small points:
- The term "FID-like" is introduced well.
- "Equivalent" is used both for TOST at ±10% and loosely ("indistinguishable"). Keep it to the TOST meaning throughout.
