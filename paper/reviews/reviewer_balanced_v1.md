# ICML 2027 Review, Round 2: "When Fewer Steps Mean Lower Temperature: Mean Collapse in Accelerated Flow-Matching Driving Policies"

**Reviewer:** balanced / area-chair style. **Round 1:** `reviewer_balanced.md` (4/10; projected ~7/10 if actions 1–5 were done and came out positive).

**Material reviewed:**
- The draft: `paper/main.tex`, the generated tables in `paper/tables/`, and the diff against `versions/main_v0.tex`.
- The authors' response: `reviews/CONSOLIDADO.md` and `docs/CRONOGRAMA_ICML2027.md`.
- The analysis code: `src/cavla/matched_temperature_stats.py` and `src/cavla/gaussian_contraction_prediction.py`. Both are identical to their `source_snapshot.py`.
- The artifacts: `artifacts/matched_stats_v1_20261026/summary.json` and `artifacts/gaussian_contraction_v1_20261026/summary.json`.
- The raw data behind them: `artifacts/temperature_probe{,_r1,_a2}_v1_20261022/plans.npz` and `artifacts/native_solver_physicalai_v1_20261022/predictions.npz`.

Every number marked **[recomputed]** comes from my own CPU passes over those files, using the authors' `per_clip` / `summarize` / `contraction` functions unchanged. No GPU jobs were run.

---

## 0. Bottom line

The revision fixes the *inferential infrastructure* I asked for:
- clips are the resampling unit;
- τ_eff is re-estimated inside each bootstrap resample;
- effect, residual, fraction-explained and TOST are computed together;
- the analysis uses a single 60-clip sample;
- the safety headline is withdrawn.

That is real progress, and the code is clean and auditable.

However, the two new headline claims do not survive an audit of that same code and data.

**1. The "tail" result (Euler removes stopping) is pseudo-replicated, the same error as the round-1 collision claim, in a new place.**
- All stopping decisions in the 60-clip sample, in all three models and every configuration, come from **two clips**:
  - `context_113`, where the ego speed is 0.0 m/s;
  - `context_141`, where the ego speed is 0.77 m/s.
- The pooled CI in the abstract was obtained by resampling clips **independently per model**, but the three models share the same clips.
- With a joint clip bootstrap, the pooled stopping residual is:
  - Euler-4: −0.80 pp, 95% CI [−2.2, 0.0], P(≥0) = 0.14;
  - Euler-10: −0.40 pp, 95% CI [−0.98, 0.0], P(≥0) = 0.054.

**2. The Gaussian "parameter-free prediction" matches only at the level of medians, and anti-predicts per clip.**
- Across clips, predicted and measured contraction are **negatively** correlated: Spearman −0.46 / −0.39 / −0.37 for Euler-4/8/10.
- The measured value is reproducible across seeds (seed-to-seed ρ = 0.52–0.62), so this is not noise in the measurement.
- A constant (the c_k at s=1) has lower error than the data-driven prediction.

What the data *do* support is close to the **round-1 thesis** that the authors stepped back from: Euler-k ≈ explicit temperature.
- Fraction of the effect explained by temperature: 0.79–1.21 for ADE₁ and flips in all six headline cells.
- Heun-3/5 fail the same test clearly. This is a negative control that I recomputed but that is still not in the paper.
- No beyond-temperature effect is established at the clip level. Equivalence at the declared margins is also not yet established, because 60 clips are too few.

The reframing therefore moved the paper's distinctive claim onto its weakest evidence.

**Score: 4/10 (unchanged), confidence 4/5.** The trajectory is positive. The fixes are mostly writing plus one cheap confirmation run (§5).

---

## 1. Status of my round-1 actions

| # | Round-1 action | Status | Comment |
|---|---|---|---|
| 1 | Fix statistics: cluster tests, paired residual CIs with τ_eff uncertainty, TOST, fraction explained, distinct-clip counts; remove "halves collisions" | **Mostly done; one critical regression** | Clip bootstrap with τ_eff and the curve re-estimated per resample: done, correctly. TOST and fraction-explained are computed but **not reported**. The collision claim was withdrawn and described per clip: done well. But the new pooled stopping CI repeats the pseudo-replication across models, and there are no distinct-clip counts for stopping (it is 2). See §3. |
| 2 | Official FlashDrive at matched temperature | **Not done (running)** | `flashdrive_probe_nvidia_v1` is at clip 113. Finding 4 is still a TODO. |
| 3 | Distributional matching (energy distance / MMD) and the Heun-k negative control in the main table | **Not done** | The data exist: A1.5 has `heun3_t*` and `heun5_t*` configs, but `matched_temperature_stats.py` filters samplers to `euler*`. I recomputed the control (§3.6); it works and should be in Table 2. No distributional test yet. |
| 4 | Language-seed ablation | **Not done** | Still a TODO in the Limitations section. |
| 5 | Closed-loop v2, pre-registered and powered | **Not done** | v1 is running. No v2 protocol or power analysis yet. |
| 6 | Gaussian closed-form prediction of τ_eff(k) | **Done, but over-interpreted** | It reproduces the *median* contraction to 0.02–0.04, but it is A1.5-only, measured in 128-d action space rather than in trajectory spread, and anti-correlated per clip (§4). It does not yet "explain why three generations land at the same τ_eff", since R1 and A2 were not tested. |
| 7 | Confirmation on the 436 unseen clips | **Not done** | Now the most important cheap run (§5). |
| 8 | Two-mode proof or conjecture | **Partially done** | The proposition is honestly scoped ("TODO: proposition for two modes, or state as empirical regularity"). The 408,769-flip number still has no saved artifact (`mean_attraction_v1_20261026` now exists; please cite it). |
| 9 | Narrative and clarity | **Largely done** | Shorter, cleaner, findings-centred. "FlashDrive-style" ambiguity removed by deferring. The W5 good-vs-bad framing is resolved ("set temperature explicitly"). The schedule negative result is in Related Work plus an appendix. Remaining inconsistencies are listed in §3.7. |

**Summary:** of the five actions I rated decisive (1–5), one is substantially done, and it contains a new instance of the old error. The other four are pending. My round-1 projection of ~7/10 assumed items 1–5 were done and positive; that condition is not yet met.

---

## 2. New strengths and weaknesses

### Strengths (new or strengthened)

- **S1. Auditable statistics pipeline.**
  - Tables are generated from JSON (`make_tables.py`) and source snapshots are saved with each artifact.
  - Clips are the resampling unit, and τ_eff and the temperature curve are recomputed inside every bootstrap resample. This is exactly the right design, and it let me find the problems below within an hour. Few submissions are this transparent.
- **S2. A single 60-clip sample across the three Alpamayos with a common Heun-32 reference.** This removes most of the round-1 inconsistency. The exception is the contraction table (§3.7).
- **S3. The temperature-equivalence core is now quantitatively strong.**
  - Effects are large and their CIs exclude zero: ADE₁ −0.13 to −0.38 m; flips +4.7 to +14.7 pp.
  - Residuals are small. Fraction explained ranges over 0.86–1.21 (ADE₁) and 0.79–1.19 (flips) in the six headline cells. Pooled flip residuals are +0.8 pp (Euler-10) and +0.3 pp (Euler-4). **[from summary.json; pooled recomputed]**
  - Explicit temperature applied to Alpamayo's own Euler-10 sampler (`euler10_t0.8/0.9`) also lands on the Heun curve: ADE₁ residual 0.000 and 0.013 m; flip residual +1.4 and −0.4 pp. **[recomputed]**
- **S4. The negative control works, and it is the most persuasive evidence in the paper (once added).** Heun-3 and Heun-5 sit close to τ_eff ≈ 0.95–0.98 by spread, yet both have large residuals with CIs that exclude zero. Few-step error that is *not* temperature-like is detected. **[recomputed, A1.5]**

  | Sampler | Flip residual | ADE₁ residual | minADE₆ residual |
  |---|---|---|---|
  | Heun-3 | +16.5 pp [10.0, 22.6] | +0.15 m [0.02, 0.31] | +0.46 m |
  | Heun-5 | +9.1 pp [4.4, 13.9] | +0.08 m | +0.25 m |

- **S5. Honest withdrawal of the safety claim.** The new Safety paragraph reports 7 clips with any collision and 5 clips driving the differences, and it explicitly declines a safety claim.
- **S6. A real-data analog of the toy mean-attraction result.** Among flips where the reference decision differs from the Euler-1 (conditional-mean) decision, 83% of Euler-4 flips and 75% of matched-temperature flips land on the mean's decision class (Euler-10: 75% vs 72%). **[recomputed, 3 models pooled]** This is a nice result, it is not in the paper, and it supports "both knobs are mean-directed".

### Weaknesses (new or remaining)

- **N1 (major). The headline tail claim rests on two clips and a mis-specified pooled bootstrap.** See §3.1. The abstract's "−0.8 pp, 95% CI [−1.6, −0.2]" is not supported.
- **N2 (major). The "mean collapse beyond temperature" thesis has no supporting clip-level evidence yet.** Every candidate beyond-temperature signal (stopping, entropy, minADE₆, minority decisions, direction of flips) fails or is ambiguous under joint clip resampling (§3.3–3.5). The reframing was formed after seeing the data (CONSOLIDADO, problem 2), so even a confirmed version must be labelled exploratory until replicated.
- **N3 (major). The Gaussian prediction is overstated in the abstract and contributions.** See §4.
- **N4 (major, unchanged). No result yet on the two experiments that carry significance for ICML:** official FlashDrive at matched temperature, and closed loop. The abstract still ends in a TODO.
- **N5 (moderate). The equivalence claim is stated more strongly than the authors' own TOST supports.** Finding 2 says residuals are "within ±0.03 m … within 2 pp, with intervals that contain zero". That is the absence-of-evidence framing from round 1. By the authors' own TOST flags:
  - ADE₁ equivalence passes in **2 of 6** headline cells (R1 and A2 Euler-10);
  - flip equivalence passes in **1 of 6** (R1 Euler-10);
  - A2 Euler-10 has a flip residual of +2.04 pp, 95% CI [−0.04, +4.19]. This is at the margin, not within it.
- **N6 (moderate). Driving relevance of the "stopping" class.** Both stopping clips are near-stationary egos (0.0 and 0.77 m/s). There, "stop" vs "keep" separates *remaining stopped* from *creeping*, and the boundary is a threshold artefact. This is not the "brake for a hazard" decision the Impact Statement implies.
- **N7 (moderate, unchanged).** No language-seed ablation and no cross-family driving model. ReCogDrive and DiffusionDrive on NAVSIM are planned in `CRONOGRAMA`, but not started.
- **N8 (minor).** Euler-1 and Euler-2 τ_eff are clamped at the bottom of the grid (0.40) by `np.interp`, so their residuals are not meaningful. Either extend the τ grid to {0.3, 0.2, 0.1, 0} or omit them; currently they sit in the JSON with large "residuals" that a reader could misuse.

---

## 3. Statistics audit

### 3.1 Pooled stopping residual: pseudo-replication across models (critical)

**Distinct clips with any "stop" decision** (60 clips × 6 seeds = 360 samples per model). **[recomputed]**

| Model | Heun-32, τ=1 | Heun-32, τ=0.6 | Euler-10 | Euler-4 |
|---|---|---|---|---|
| A1.5 | 113: 5, 141: 1 | 113: 6, 141: 1 | 113: 5, 141: 1 | 113: 5 |
| R1 | 113: 4, 127: 1, 141: 1 | 113: 5, 141: 2 | 113: 4 | 113: 4 |
| A2 | 113: 4, 124: 1, 141: 1 | 113: 5, 141: 3 | 113: 4, 141: 1 | 113: 4 |

- The base stopping rate on this sample is **1.67% in all three models**. The paper says 1.7–3.7%; the 3.7% belongs to the old 100-clip set.
- The entire pooled residual is a difference of 1–3 samples per model in clips 113 and 141.
- Leave-one-clip-out results:
  - dropping clip 113 gives pooled residuals of −0.56 / −0.25 pp (Euler-4 / Euler-10);
  - dropping clip 141 gives −0.25 / −0.19 pp.

**Bootstrap reconstruction.** The pooled number is not in any artifact; it is a script-free claim. I reproduced the paper's interval only when I resampled clips *independently* for each model:

| Sampler | Point | Independent per model (≈ paper) | **Joint over shared clips (correct)** | P(residual ≥ 0), joint |
|---|---|---|---|---|
| Euler-4 | −0.80 pp | [−1.67, −0.19] | **[−2.2, 0.0]** | 0.14 |
| Euler-10 | −0.40 pp | [−0.86, −0.06] | **[−0.98, 0.0]** | 0.054 |

Pooling three models that see the same 60 clips is legitimate only if clips are resampled jointly. Three Alpamayo generations stopping in the same two scenes is one piece of evidence, not three.

Also, by the authors' **own declared margin** for stopping (1 pp), even the point estimates (−0.4 and −0.8 pp) lie *inside* the equivalence region. Under the paper's own framework, the tail effect is not practically meaningful in absolute terms. If it matters, it matters in relative terms: about 30–50% of a rare class. A relative endpoint needs to be pre-declared.

**Per-cell intervals.** Every per-cell stopping CI has an upper bound of exactly 0.0. This is a symptom of zero-inflation: the percentile bootstrap produces a degenerate interval whenever a resample excludes the 1–2 stop clips. The percentile bootstrap is unreliable here.

**Fix:**
- Report distinct-clip counts for every rare-event metric.
- Use a joint (clip-level) bootstrap or an exact clip-level sign or permutation test.
- Remove the stopping claim from the abstract until the enriched confirmation in §5 exists.

### 3.2 τ_eff recomputed per bootstrap: correct

- `summarize()` recomputes spreads, re-sorts the curve and re-interpolates τ_eff and every metric curve for each resample. The uncertainty of τ_eff is therefore propagated into the residuals. Good.
- Linear interpolation between grid points spaced 0.1 apart is adequate: the curves are smooth and monotone over [0.6, 1].
- Two caveats:
  - (i) The `np.interp` clamping noted in N8.
  - (ii) Matching is on the *mean spread over clips*, a single global τ_eff per sampler. A per-clip τ_eff would test whether the temperature equivalence holds clip by clip, a stronger and more interesting claim. It is cheap.

### 3.3 Equivalence margins

- **Timing.** The code comment says the margins were "declared before computing residuals (26/10/2026)". However, residuals on these exact probes were computed on 22/10 (`temperature_report_*`) and discussed in round 1, including A2 flips 9.7% vs 7.7%. The margins were therefore set after the point estimates were known. Say so, and do not call them pre-registered. For the 436-clip confirmation, pre-register them with a timestamp.
- **Values.** ADE₁ 0.05 m and flips 2 pp are reasonable: about 15–40% of the effects. Stopping at 1 pp is too loose relative to a 1.7% base rate. minADE₆ at 0.05 m is tight relative to its clip variance; no cell passes, and that is a power problem, not evidence of a residual.
- **Power.** With 60 clips, the 90% residual interval for flips has a half-width of about 1.5–3.5 pp. Equivalence at 2 pp is unreachable even if the true residual is 0. With 436 clips the width shrinks by about √(436/60) ≈ 2.7×, and TOST would pass in most cells if the residuals are near zero. This is the cheapest decisive experiment in the paper.
- **Reporting.** `fraction_explained` and its CI are computed but not shown. They are the clearest summary of the main claim, so add a column. Note that fraction-explained CIs are unstable when the effect is near 0 (e.g., R1 Euler-10 minADE₆: [−5.6, 4.4]); show them only for metrics with a significant effect.

### 3.4 Multiple comparisons

- The artifact contains 3 models × 5 samplers × 6 metrics = **90 residual CIs**. Excluding the clamped Euler-1/2 leaves 54.
- Of those 54, only about one has a 95% CI that strictly excludes 0: the R1 Euler-4 deceleration residual, +1.3 pp [0.2, 2.6]. A few others touch 0. That is what one expects under a global null.
- The stopping headline was chosen *after* inspecting these cells, so it is a post-hoc selection from 54+ tests, with no correction.
- The table shows 5 of the 6 metrics. **Entropy is computed but omitted**, even though the introduction cites "lowers the entropy of its driving decisions".

**Recommendation:**
- One **primary** endpoint per claim:
  - (a) equivalence: pooled flip and ADE₁ residuals for Euler-10 and Euler-4, TOST at the stated margins, joint clip bootstrap;
  - (b) beyond-temperature: one pre-declared tail metric (§5).
- Holm correction across the secondary metric × cell family, shown as a column or a footnote.
- Report all six metrics, including entropy.

### 3.5 Other candidate beyond-temperature signals (my checks; joint clip bootstrap, 3 models pooled) **[recomputed]**

| Signal | Euler-10 | Euler-4 | Verdict |
|---|---|---|---|
| Decision entropy residual | −0.024 [−0.054, +0.005], P(≥0) = 0.05; negative in all 3 models | +0.011 [−0.039, +0.064] | Suggestive at Euler-10 only. It does not grow with fewer steps, which mean collapse would predict. |
| minADE₆ residual | +0.049 m [−0.012, +0.142] | +0.045 m [−0.032, +0.131] | Consistent sign, not significant. "Positive in all six cells" is false: R1 Euler-4 is −0.0015. |
| Minority-decision rate residual (mode taken from Heun curve samples) | −0.05 pp [−1.4, +1.4] | **+2.9 pp [+0.8, +5.3]** | Opposite to mean collapse. Biased against Euler, though, because the mode is estimated on the curve samples. Needs held-out seeds. |
| P(flip lands on the Euler-1 decision class), Euler − temperature, per clip | −2.2 pp [−7.5, +1.7] | −1.8 pp [−9.8, +5.5] | No difference. Both knobs are equally mean-directed (72–83%). |

Taken together, the data do not show that Euler is *more* mean-collapsing than temperature at the decision level. They show that both are mean-directed and largely interchangeable. That is a fine result, but it is the round-1 thesis, not the round-2 one.

### 3.6 Negative control (should be in Table 2)

The Heun-3/5 results in S4 show that the residual test has power against non-temperature numerical error. Add them as rows of Table 2.

### 3.7 Remaining inconsistencies

- **Table 1 (contraction) vs Table 2.** Table 1 uses 200 clips × 2 seeds, a **Heun-160** reference and 128-d *action* space. Table 2 uses 60 clips × 6 seeds, Heun-32 and xy-*trajectory* spread. The contraction claim and the τ_eff claim are on different samples, references and spaces.
- **Claim about the action scale s.** The text says c₁₀ ∈ [0.82, 0.88] "for s_d ∈ [0.3, 2], which is why policies with normalized actions land near the same value". But the artifact's conditional-std quartiles are **0.018 / 0.11 / 0.39**, so most dimensions have s < 0.3, where c₁₀ drops (0.62 at s = 0.1). The median prediction lands near 0.86 only because the s²-weighting is dominated by a few high-variance dimensions. The stated explanation is incorrect as written.
- **"Heun-5 and RK4 show no contraction" (Finding 1).** Not in the artifact. I confirmed it from `native_solver_physicalai_v1` (Heun-5: median ratio 0.998, 50% of samples contracted; RK4-2: 1.048, 22%). Save it in the artifact.
- **FlashDrive-style cache.** Shrinkage median 0.497 vs spread τ_eff 0.555. The two estimators still disagree for the cache, as in round 1. Whichever is reported, report both.
- **Base stopping rates** are given as 1.7–3.7%, but are 1.67% on the stated sample (§3.1).
- **minADE₆** is said to be "positive in all six cells"; it is 5 of 6.

---

## 4. Is the Gaussian prediction strong enough to headline?

**No, not in its current form.** **[recomputed from `native_solver_physicalai_v1`, 200 clips × 2 seeds, A1.5]**

| | Euler-4 | Euler-8 | Euler-10 |
|---|---|---|---|
| Predicted median (authors) | 0.672 | 0.823 | 0.856 |
| Measured median | 0.637 | 0.804 | 0.840 |
| Constant c_k(s=1), no data at all | 0.720 | 0.851 | 0.879 |
| Per-clip Spearman(predicted, measured) | **−0.46** | **−0.39** | **−0.37** |
| Seed-to-seed Spearman of measured | 0.62 | 0.54 | 0.52 |
| MAE: data-driven prediction | 0.124 | 0.084 | 0.073 |
| MAE: constant s=1 | **0.109** | **0.068** | **0.058** |
| Fraction of clips where predicted > measured | 0.60 | 0.60 | 0.61 |

Interpretation:
- **What the result shows.** c_k(s) is flat over the high-variance range, so the uniform grid alone fixes the *typical* contraction near 0.64 / 0.80 / 0.84. That is a real and useful observation: it explains why three Alpamayo generations with similar normalization share τ_eff ≈ 0.83. However, it is a property of the **grid**, and the data-dependent part of the prediction adds nothing.
- **Why the per-clip test fails.** The measured contraction varies across clips reproducibly (ρ ≈ 0.5–0.6 between seeds), yet the Gaussian model ranks clips in the **wrong order**. The likely causes:
  - (i) s_d is estimated from a single pair of samples per clip, |a−b|/√2, which is a one-degree-of-freedom estimate per dimension;
  - (ii) diagonal rather than full covariance;
  - (iii) genuine non-Gaussianity: multimodal clips contract differently.

  Cause (iii) is scientifically interesting; causes (i) and (ii) are fixable.

**What would make it headline-worthy:**
1. Estimate the conditional covariance from ≥16 reference samples per clip. Use its eigenbasis and eigenvalues, not diagonal one-pair estimates.
2. Show a per-clip calibration plot (predicted vs measured) with a constant-c_k null, and report ρ and MAE against that null.
3. Predict **τ_eff in trajectory spread space** for **all three Alpamayos plus SmolVLA**, for k = 2, 4, 6, 10, and for the cache grid. That is the claim in the contributions list.
4. If per-clip prediction still fails, make the failure the finding. The Gaussian model gets the grid-level contraction right, and the per-clip deviation from it measures non-Gaussianity (multimodality). Then test whether that deviation predicts where decisions flip. This would connect the theory to the decision-level story and make "mean collapse" more than a name.

Until then: "a uniform 10-step Euler grid contracts by about 0.84 for any target scale within the high-variance range, which matches the measured 0.84". State this as an explanation of the common τ_eff, not as a "parameter-free prediction to within 0.04".

---

## 5. Significance, and how to sharpen the thesis

**Is "mean collapse" significant enough for ICML?**
- As currently supported, the beyond-temperature part rests on 2 clips, and the positive part (Euler ≈ temperature) is the round-1 thesis.
- The paper's genuine contribution is:
  - (i) an evaluation confound with a cheap, validated correction: τ_eff plus matched comparison, with Heun-k as the failing control;
  - (ii) the observation that released driving samplers already run at τ ≈ 0.83 and that published "accuracy gains" may be this temperature;
  - (iii) a pre-registered negative result (the schedules).
- That is an ICML-worthy evaluation paper **if** (ii) is demonstrated on a published method's own checkpoint (FlashDrive) and the equivalence is confirmed out of sample.
- "Mean collapse erases rare decisions" would make the paper much more significant for driving. Right now it is a hypothesis.

**How to sharpen it.** Pick one of two theses and commit:
- **(A) "Fewer steps = lower temperature", as a measurement and evaluation paper.** The headline is equivalence (TOST passes on 436 clips), the negative control, and the FlashDrive reinterpretation. The tail becomes a pre-registered secondary endpoint, and a null there is fine. This is the safer path and it is currently supported.
- **(B) "Same temperature, different tails".** Here the headline is a beyond-temperature decision effect. It needs a design with power for rare decisions, which random clips cannot give (2 of 60 have any stop). Mean collapse makes a sharp, testable prediction that does not single out stopping:

  > In clips whose reference decision distribution is multimodal, Euler-k moves probability mass toward the decision of E₁(c) (the conditional-mean plan) more than explicit temperature at matched spread does. The excess grows as k decreases and is largest at k=1 vs A₀.

  My check (§3.5) finds no such excess on the current data (−2 pp, CI ±5–8 pp, 49 clips). So the hypothesis is open, not refuted.

I recommend writing the paper as (A) with (B) as a pre-registered secondary test. Promote (B) only if the enriched confirmation in §6 supports it.

---

## 6. Updated prioritized plan (about 3 months; deadline ~end of January 2027)

**The single most valuable next experiment:** a **pre-registered confirmation on new clips, enriched for decision ambiguity**, for the three Alpamayos.
- **Clip set:** the 436 unseen clips, plus screening for ambiguity among additional clips.
- **Seeds and configs:**
  - 6 screening seeds, plus 12 fresh analysis seeds (so the mode or ambiguity selection is not reused, avoiding regression to the mean);
  - Euler-1/2/4/6/10, Heun-3/5, Heun-32 at τ ∈ {1, 0.9, …, 0.3, 0.2, 0};
  - Alpamayo's own Euler-10·τ.
- **Pre-register, with timestamps:**
  - **Primary 1:** pooled flip and ADE₁ residuals of Euler-4 and Euler-10, with TOST at 2 pp and 0.05 m, using a joint clip bootstrap.
  - **Primary 2:** the "mass toward the E₁ decision, Euler minus matched temperature" excess on the ambiguity-enriched subset. Use a clip-level test and declare a relative effect size.
  - **Secondary, Holm-corrected:** stopping, entropy, minADE₆, and the clip-level Gaussian calibration.
- **Cost:** about 2–4 GPU-days per model, reusing existing probe code.
- **Why this one:**
  - It turns Table 2 from exploratory into confirmatory.
  - It is the only way TOST can pass.
  - It is the only design with power to settle the new thesis either way.
  - It is decoupled from the FlashDrive and AlpaSim risks.

| # | Action | Effort | Why | When |
|---|---|---|---|---|
| 1 | **Correct the draft now (CPU).** Remove the stopping claim from the abstract, or restate it as "2 clips, exploratory". Recompute the pooled residual with a joint clip bootstrap and save it in the artifact. Add distinct-clip counts for rare metrics. Restate Finding 2 using fraction-explained plus TOST outcomes, not "intervals contain zero". Report entropy. Fix the 1.7–3.7%, the "all six cells" and the s-range errors. Add Heun-3/5 rows. Admit that the margins were set post hoc. | 1–2 days | Prevents a reviewer from finding round 1's pseudo-replication again. The negative control is the strongest single addition. | This week |
| 2 | **Enriched pre-registered confirmation** (above). | 2–4 GPU-days/model, plus ~3 days of analysis | Confirmatory equivalence, plus a powered test of mean collapse. | Weeks 1–3 (interleave with FlashDrive) |
| 3 | **Official FlashDrive at matched temperature (Gate 2).** Decompose the result into fine-tuned weights, cache and temperature, separately for minADE₁ and minADE₆, on the same 60 clips, with the residual/TOST machinery. Report both cache τ_eff estimators. | Running; ~1 week | The novelty lever for ICML. A published gain reinterpreted on its own checkpoint. | Weeks 0–2 |
| 4 | **Gaussian model done properly (CPU, plus 16 reference seeds on 60 clips).** Full-covariance s from ≥16 samples. Per-clip calibration vs a constant null. Predict τ_eff in trajectory space for all three Alpamayos and SmolVLA, plus the cache. If per-clip prediction fails, test whether non-Gaussianity predicts flips. | 2–3 days CPU + <1 GPU-day | Turns an over-claim into either a real prediction or a mechanistic link to decisions. | Weeks 1–2 |
| 5 | **Closed-loop v2, pre-registered and powered**, on the **ambiguity-enriched** scenes from #2 rather than random scenes. Include the explicit-temperature arm, an equivalence margin and a scene-clustered test. | 5–7 machine-days | Random scenes give about 2/60 informative ones (cf. stopping). Enrichment is what makes v2 powered. | Weeks 3–6 |
| 6 | **Cross-family driving model** (ReCogDrive or DiffusionDrive on NAVSIM, PDMS as metric), running the same protocol. | ~1 week | Removes "one family" (N7). PDMS gives a standard driving metric. | Weeks 2–5 |
| 7 | **Language-seed ablation.** | 2–3 GPU-days | Deployment relevance (unchanged from round 1). | Weeks 1–2 |
| 8 | **Distributional matching:** per-clip energy distance, Euler-k vs Heun·τ_eff, vs Heun·τ=1, vs Heun-k. **Per-clip τ_eff** as a stronger form of the equivalence. Add the mean-directed-flip statistic (S6) to the paper. | 2 days CPU | Answers "you matched one scalar". | Weeks 1–2 |
| 9 | Proof or conjecture for two modes. Save the toy artifact. Extend the τ grid below 0.4 (or drop Euler-1/2). | 1–2 weeks CPU | Hygiene. | Weeks 3–5 |

**Decision rule.**
- If #2 confirms equivalence and #3 shows the FlashDrive gain is temperature: write as thesis (A). I would expect 6–7/10.
- If #2 additionally confirms the E₁-directed excess with a pre-registered test: thesis (B) becomes the headline, which is plausibly 7–8/10 for a driving-focused ICML paper.
- If #3 fails and #2 shows equivalence only: a solid TMLR or strong-workshop paper. ICML is borderline.

---

## 7. Score and confidence

- **Current draft: 4 / 10 (weak reject). Unchanged in number, improved in substance.**
  - The infrastructure is now right.
  - The core temperature equivalence is quantitatively strong, though not yet confirmed as equivalence.
  - However, both new headline claims (the stopping tail and the parameter-free Gaussian prediction) are over-stated against the authors' own data. The one I can check most directly repeats the round-1 pseudo-replication error.
  - The two significance-carrying experiments are still pending.
- **Projected:**
  - 6/10 with actions 1–3 done and positive;
  - 7/10 adding #4–5;
  - higher if the enriched test confirms mean collapse beyond temperature.
- **Confidence: 4 / 5.** I re-ran the authors' code paths on the raw plans and verified every number cited here.

### Recomputation log (all [recomputed])

1. Stop decisions come from 2 clips (`context_113`, speed 0.0 m/s; `context_141`, speed 0.77 m/s), with single extra samples in 127 and 124 at τ=1 only. The base rate is 1.67% in all three models.
2. Pooled stopping residual, joint clip bootstrap: Euler-4 −0.80 pp [−2.2, 0.0]; Euler-10 −0.40 pp [−0.98, 0.0]. With independent per-model resampling (the paper's interval): [−1.67, −0.19] and [−0.86, −0.06].
3. TOST passes (summary.json): ADE₁ 2/6, flips 1/6, in the headline cells.
4. Heun-3: τ_eff 0.952, flip residual +16.5 pp [10.0, 22.6]. Heun-5: τ_eff 0.977, flip residual +9.1 pp [4.4, 13.9]. Both A1.5.
5. Gaussian model, per-clip Spearman −0.46 / −0.39 / −0.37 (Euler-4/8/10). A constant s=1 beats it on MAE. The conditional-std quartiles are 0.018 / 0.11 / 0.39.
6. Contraction of Heun-5 is 0.998 (50% of samples contracted). RK4-2 is 1.048 (22%). Cached Euler-8 is 0.497 (100%), against a spread τ_eff of 0.555.
7. Mean-directed flips: 83% (Euler-4) vs 75% (temperature). The per-clip difference is −1.8 pp [−9.8, +5.5].
8. Entropy residual for Euler-10, pooled: −0.024 [−0.054, +0.005]. The minority-rate residual for Euler-4 is +2.9 pp [+0.8, +5.3], biased against Euler.
