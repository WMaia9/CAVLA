# ICML 2027 Review, Round 4: "When Fewer Steps Mean Lower Temperature: Euler Sampling of Flow-Matching Driving Policies" (v4.1)

**Reviewer:** balanced / area-chair style. Earlier rounds: 4, 4, 5.
**Material reviewed:** `versions/main_v4.1.pdf` and `.tex` (8 pages: about 5 of main text plus references and appendices; 15 TODOs), `tables/*`, `figures/*`, the "Rodada 3" section of `CONSOLIDADO.md`, `artifacts/paper_numbers_v2_20261026/summary.json` and `src/cavla/paper_numbers.py`.
**What I ran (CPU only):** `fingerprint.py`, in my round-4 scratch directory, on the existing 60-clip × 6-seed plans for all three Alpamayo models. Numbers marked **[recomputed]** come from it (§2). It uses a joint clip bootstrap with 2000 resamples.

---

## 1. Summary

The paper argues that few-step Euler sampling of flow-matching driving policies mostly contracts samples toward the one-step conditional mean, and that an evaluation should account for this. The argument runs in three steps:
- **Mechanism.** One Euler step returns the conditional mean. A closed-form Gaussian factor, ∏cos θᵢ, predicts the median contraction (0.67 for E4, 0.86 for E10).
- **Effective temperature.** Each sampler gets an effective temperature τ_eff, defined by matching its seed spread to the spread of a Heun-32 reference started from τz. The released E10 samplers of three Alpamayo generations sit at τ_eff ≈ 0.83. At τ_eff, the reference reproduces the ADE₁ and flip-rate changes of E4 and E10.
- **Controls.** Heun-3 and Heun-5 fail the same test (flip residuals of +16.5 and +9.1 pt).

New in v4/v4.1:
- **Shrinkage control (Shrink_k).** Each reference sample is moved linearly toward E₁ until its spread matches E_k's, with no integrator involved. It passes the test as well, and the paper reports this.
- **New framing.** The paper now presents itself as an evaluation protocol: "report accuracy gains against the reference solver at matched effective temperature". The mechanistic claim is weakened to "a contraction toward the mean that rate-level evaluation cannot distinguish from other contractions of the same size". Temperature is presented as the explicit, tunable knob.
- **Qwen-Drive** is kept as a boundary case: a nearly deterministic flow where the solver adds spread.

---

## 2. What I checked this round **[recomputed]**

### 2.1 Per-sample, temperature and linear shrinkage are distinguishable, and Euler resembles temperature

The new paragraph in the Discussion says shrinkage "contracts every sample by the same factor … whereas Euler contracts 90–93% of samples and less on the median". The paper never says what the *reference at τ_eff* does on this statistic, and that is the comparison that matters. The statistic is the median ratio of each sample's distance to E₁ over the reference sample's, with the same noise; the last column is the share of samples with a ratio below 1.

| Model | k | Euler E_k | A_τeff (interpolated) | Shrink_k | Share contracted: E_k / A_τ (grid) / Shrink |
|---|---|---|---|---|---|
| A1.5 | 4 | 0.68 | 0.69 | 0.57 | 92% / 88% / 100% |
| A1.5 | 10 | 0.86 | 0.84 | 0.80 | 90% / 89% / 100% |
| R1 | 4 | 0.66 | 0.72 | 0.58 | 93% / 82% / 100% |
| R1 | 10 | 0.84 | 0.87 | 0.80 | 93% / 84% / 100% |
| A2 | 4 | 0.66 | 0.68 | 0.56 | 92% / 88% / 100% |
| A2 | 10 | 0.85 | 0.84 | 0.79 | 90% / 90% / 100% |

What this shows:
- **Temperature has the same per-sample signature as Euler.** At τ_eff the reference's median contraction lies within 0.01–0.06 of Euler's, and it also leaves roughly 10–15% of samples uncontracted.
- **Linear shrinkage has a different signature.** It needs a much smaller factor (c ≈ 0.57) to reach the same spread, because pairwise spread falls faster than radial distance under both Euler and temperature.
- **This restores part of the temperature claim the paper gave up.** Among contractions that pass the rate test, temperature matches Euler's per-sample signature and linear shrinkage does not.
- **What to do in the paper.** Put this in Finding 1 or the Discussion, and make it a pre-registered mechanism endpoint (§6, Q3).
- **The case against linear shrinkage specifically is stronger than the per-sample distance test.** Pooled over models, the distance E10→A_τeff minus E10→Shrink₁₀ is −0.024 m [−0.065, +0.005], which is borderline. For E4 it is +0.020 [−0.029, +0.067], which is not significant.

### 2.2 Rate-level residuals: the shrink arm and Euler cannot be separated, but the shrink residuals all have the same sign

Flip residuals relative to the temperature curve, pooled over the three models:

| Arm | k = 10 | k = 4 |
|---|---|---|
| Euler | +0.84 pt [−0.39, +2.17] | +0.30 [−1.82, +2.49] |
| Shrink | +1.15 [−0.11, +2.47] | +1.44 [−0.99, +3.98] |
| Euler − Shrink | −0.31 [−1.58, +1.02] | −1.15 [−3.45, +1.30] |

- **All six Shrink flip residuals in Table 1 are positive** (0.6–2.6 pt), while Euler's change sign.
- **minADE₆ points the other way.** Euler's residual is positive in 5 of 6 cells, so Euler loses more best-of-six error than temperature does. Shrink's residuals are ≤ 0.02 m in every cell, and negative in four of six.
- **Conclusion.** At n = 60 the rate-level test indeed does not separate the arms. This supports the new wording.

### 2.3 E₁ is not deterministic in the released networks, which affects the theory and the shrink control

- **E₁ is not deterministic.** Its seed spread is 0.23, 0.32 and 0.21 m (A1.5, R1, A2), or 9–14% of the reference spread. The sentence "the one-step sampler has no diversity at all" holds for the exact minimizer, not for the trained field. Say so. Then either report E₁'s residual spread or define the contraction with respect to the seed-averaged E₁, as `clip_metrics` already does for `del_mean`.
- **Consequence for the shrink control.** `add_shrinkage` sets c = Spr(E_k)/Spr(A₁). That formula assumes a single E₁ for all seeds, but the code uses each seed's own E₁. As a result, Shrink_k is not exactly spread-matched:
  - its spread is 2–7% above E_k's (for A1.5 E4, 1.21 vs 1.16 m);
  - its τ_eff is 0.64–0.66, against Euler's 0.61–0.64.
- **Fix.** Solve for c numerically, or shrink toward the seed mean of E₁. This is unlikely to change any conclusion, but a control named "matched spread" should be matched.

### 2.4 Text versus code
1. **Equivalence interval.** Section 4 says S is equivalent when "the 95% interval of the residual lies inside a margin". `paper_numbers.py` uses the 5–95% quantiles, which is a 90% interval and the correct one for TOST. Fix the text.
2. **"Declared in advance".** Section 4 calls the margins (2 pt, 0.05 m) "declared in advance". They were set after the 60-clip data were seen (as I noted in round 3). They will be pre-declared only for the confirmation. Say that.
3. **Equivalence in v4.1.** The only cell that is equivalent on both metrics is R1 E10. A1.5 E6 passes on ADE₁ only. The text's "too wide … in most cells" is accurate. It would help to report this count explicitly ("1 of 6 E4/E10 cells passes on both metrics").
4. **Shrink intervals.** "All with intervals that contain zero, just as Euler does" is not quite right. Euler's A2 E10 interval excludes zero, which the paper itself says in Finding 2.

---

## 3. Status of round-3 requests

| Round-3 request | Status |
|---|---|
| Shrinkage arm in Table 2 | **Done**, and it is generated by code. The per-sample comparison is only qualitative, and the decisive part (A_τeff's signature) is missing (§2.1). |
| Re-scope to "mean-directed contraction" | **Done**, in the Discussion and abstract. The title and introduction still say "temperature" without qualification (W1). |
| E1/E2 reported, scope k ≥ 4 | **Partly done.** The E₁ limit paragraph is good. The τ grid has not been extended, and the claim's range of k is not stated in the abstract. |
| minADE₆ and entropy | **Done** (Finding 2, plus Appendix figure). |
| "Fraction explained" above 100% | **Explained.** The abstract still leads with "79–121%". I would lead with residuals and their intervals instead. |
| Disclose prior use of the 436 clips | **Done.** Fresh clips and seeds are promised. |
| Margins described as post hoc | **Not done.** The text now says the opposite (§2.4). |
| Heun on R1/A2; a displacement-matched Heun | **Not done** (TODO). |
| Proposition 3.1: proof or demotion | **Not done.** It is still stated as a proposition with a TODO, and the Lipschitz caveat is unresolved. |
| Qwen-Drive generated from code | **Not done.** |
| Figures | **Done, and good.** The teaser and the two-row curve figure make the Euler/Heun contrast immediate. This is the biggest presentational gain of the round. |
| Pre-registered confirmation, FlashDrive, NAVSIM, closed loop | **All pending.** |

---

## 4. Strengths

1. **The paper is now honest and internally consistent.**
   - Its claims match what 60 clips can support.
   - The limits are stated: the match is distributional only, E₁ is out of range, stops are untestable, and shrinkage passes.
   - Every number is macro-generated from one script with a joint clip bootstrap.
2. **The negative control discriminates.** Heun-5 has the released budget yet flips 9.7% of decisions at τ_eff ≈ 0.98, so the test is not vacuous.
3. **Simple, transferable mechanism.** E₁ is the mean, and the ∏cos θ factor predicts the median contraction across three generations with no fitting. Practitioners can use it directly.
4. **A practical, cheap recommendation.** "Report τ_eff, and compare against the reference solver at τ_eff" costs about one extra reference sweep. It applies to any first-order or cached flow sampler, including FlashDrive-style caching, ProbeFlow, AdaVLA and Coda.
5. **The figures do real work.** Figure 1(b) and Figure 3 convey the whole result in seconds.

---

## 5. Weaknesses

**W1. The new framing is safer but thinner, and the paper has not decided which paper it is.**
- The abstract now says, in effect, "Euler behaves like lower temperature, and any equal contraction would pass the test".
- The title and the first contribution bullet still sell temperature as *the* explanation.
- An ICML reader will then ask what is new beyond "first-order integration under-disperses, so single-sample error falls". That is the familiar bias–variance effect (contraction toward the mean always lowers ADE₁), and it is close to the Gaussian Euler analysis of Claustre et al.
- A methodology paper is judged by **whether the method changes a conclusion someone has published**, and by whether it transfers. At present it has changed none: the FlashDrive slot is a TODO, and the one changed conclusion is the authors' own schedule study, which lives in an empty appendix.
- **On whether the repositioning helps:** it makes the paper more correct and more robust, but less significant on its own. It moves all of the significance onto the pending experiments.
- **There is a way to recover part of the stronger claim.** The per-sample signature in §2.1 shows that temperature, unlike linear shrinkage, matches Euler sample by sample. With that evidence the paper can again say "Euler acts as a hidden temperature", and the protocol becomes a principled choice rather than one of many indistinguishable references.

**W2. No significance-bearing result has landed since round 3.**
- Equivalence is established in 1 of 6 E4/E10 cells.
- No out-of-sample confirmation has been run.
- There is a single model family (Qwen-Drive is a boundary case, and its numbers are hand-typed).
- There is no standard driving metric (PDMS, closed-loop incidents), and no published acceleration has been re-evaluated.

**W3. The assay-sensitivity control is only on one model.** Heun-3/5 exist only for A1.5. The argument that the test can fail needs the control on every model in which the test is claimed to pass.

**W4. Theory hygiene.**
- **Proposition 3.1.** Its hypothesis, h·Lip < 1, is likely violated in the last step for low-variance targets. Either bound Lip or demote the proposition to a remark.
- **"No diversity at all" for E₁.** This is contradicted by the 9–14% residual spread (§2.3).
- **Range of k.** The Gaussian factor is quoted for s ≥ 0.3. The text should state the range of k over which the claim holds (k ≥ 4 on current data).

**W5. Unfinished draft.** The PDF has 15 TODOs, and Appendices B–E are empty. The bibliography still has placeholders (Alpamayo, Qwen-Drive, AlpaSim, PhysicalAI). The text inconsistencies in §2.4 need fixing.

---

## 6. Questions for the authors

1. **FlashDrive.** What does matched-τ *predict* for FlashDrive's reported metric, before you unblind?
   - Your own data show that Euler's minADE₆ residual is positive, meaning worse than temperature.
   - If FlashDrive reports minADE₆ (or minADE with k > 1), the protocol might attribute *less* than the whole effect to temperature.
   - Write the prediction down now.
2. **NAVSIM.** DiffusionDrive is a truncated-diffusion model with anchors, not a rectified flow started from N(0, I). Its "E₁ = mean" step does not apply.
   - Will you state, before running, which of ReCogDrive and DiffusionDrive should land on the curve and which should behave like Qwen-Drive?
   - A rule stated in advance that predicts the boundary would turn Finding 4 from an anecdote into a result.
3. **Mechanism endpoint.** Will the confirmation pre-register the per-sample contraction signature?
   - Proposed endpoint: the median of E_k's ratio is within δ of A_τeff's and outside δ of Shrink_k's, pooled over models.
   - With δ = 0.05, the current data pass both conditions in two of three models for E10 (R1's Shrink₁₀ is only 0.04 from Euler) and two of three for E4 (R1's A_τeff is 0.06 from Euler). The test needs a pre-specified δ and the fresh sample.
4. **Closed loop.** With 58 scenes and one rollout each, what effect size can the AlpaSim comparison detect?
   - If the minimum detectable effect is several times the open-loop flip residual, present it as a sanity check, not as evidence of equivalence.
   - Was the `failed.0` restart in two arms resolved, and is it reported?
5. **Confirmation design.** Will it include Heun-3 on all three models as an assay-sensitivity gate? Will its margins be frozen before any analysis seed is run, and will the clips come from chunks disjoint from the 60-clip, schedule, and AlpaSim sets?

---

## 7. Which pending experiments matter most

I rank them by how much each moves the ICML decision.

1. **Official FlashDrive at matched τ (highest value).**
   - This is the experiment that makes a methodology paper significant: it either changes the reading of a published driving acceleration or certifies it.
   - Either outcome is publishable, provided it is reported with the same residual and TOST machinery and with weights, cache and temperature decomposed.
   - Pre-register the prediction (Q1).
2. **Pre-registered confirmation on fresh clips and seeds.**
   - It turns "consistent with" into "equivalent within a margin".
   - Primary endpoint: pooled E10 TOST on flips and ADE₁ (n ≈ 400 gives more than 0.99 power at half the margin, per my round-3 calculation).
   - Assay-sensitivity gate: Heun-3 on all models.
   - Mechanism endpoint: the §2.1 signature.
   - This is cheap relative to its value. Without it, the core quantitative claim stays exploratory.
3. **NAVSIM second family (ReCogDrive, and/or DiffusionDrive) with PDMS at matched τ.**
   - It removes the single-family objection and ties the protocol to a standard benchmark metric.
   - It is the main 6→7 lever, especially combined with a boundary rule stated before the run.
4. **AlpaSim closed loop (lowest priority for the decision).**
   - Driving reviewers will want it, but at 58 scenes × 1 rollout it will almost certainly be inconclusive.
   - It helps only if presented as a sanity check, or if it shows a *difference* that open loop predicts: for example, A₁ versus E4 at τ = 0.65 diverging in the direction the flip analysis predicts.
   - A powered version (enriched scenes, ≥ 3 rollouts per scene) would count more, but should not displace items 1–3.

Cheap CPU fixes that should happen regardless:
- the §2.1 signature table;
- the exact spread matching for Shrink (§2.3);
- the text and code fixes in §2.4;
- Heun-3/5 on R1 and A2 (GPU, but small);
- Proposition 3.1 demoted or proved;
- Qwen-Drive numbers generated from code;
- the empty appendices filled or removed.

---

## 8. Score

**Score: 5/10 (borderline reject). Confidence: 4/5.**

Reasons the score holds:
- The draft is cleaner, honest, and well illustrated.
- The shrinkage control was added promptly and reported against the authors' own interest. I credit that.
- But no result that bears on significance or generality has arrived since round 3.
- The repositioning, while correct, lowers the novelty of the stand-alone science. A protocol paper with no re-evaluated published method and one model family is below the ICML main-track bar.

### Minimum for a 6/10 (weak accept)
1. **Official FlashDrive at matched τ, reported whichever way it goes.** Use the same residual and TOST machinery, with a prediction stated before unblinding.
2. **A pre-registered confirmation on fresh clips and seeds that passes its primary endpoint** (pooled E10 equivalence on flips and ADE₁), with Heun-3 rejected on all three models.
3. **Draft fixes:**
   - the §2.4 inconsistencies (90% vs 95% interval, "declared in advance");
   - E₁'s nonzero spread acknowledged, and Shrink exactly spread-matched;
   - Heun-3/5 on R1 and A2;
   - Proposition 3.1 proved or demoted;
   - Qwen-Drive from code;
   - no TODOs, and the appendices either filled or cut;
   - the title and contribution bullets aligned with the contraction framing, or the temperature claim backed with §2.1.

### Minimum for a 7/10 (accept)
All of the above, plus:

4. **A second driving family on NAVSIM, with PDMS at matched τ.** It should either land on the curve or land where a rule stated before the run predicts.
5. **At least one of the following:**
   - (a) **The protocol changes a published reading.** For example, FlashDrive's gain shrinks substantially at matched τ.
   - (b) **The per-sample temperature signature (§2.1) is confirmed out of sample** as a pre-registered mechanism endpoint, which justifies the title.
   - (c) **Closed loop agrees with open loop** with a scene-clustered CI that is tight enough to be informative.

If all three of (a)–(c) hold, I would consider 7–8.
