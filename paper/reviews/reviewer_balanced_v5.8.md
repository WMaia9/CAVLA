# ICML 2027 Review, Round 5.8: "When Fewer Steps Mean Lower Temperature" (v5.8)

**Reviewer:** balanced / area-chair style. Earlier rounds: 4, 4, 5, 5, 5, 5, 5 (v5.4).

**Material reviewed:**
- `versions/main_v5.8.tex`, compared with v5.4;
- the macros in `tables/numbers.tex` and `tables/numbers_tables.tex`;
- these artifacts:
  - `artifacts/paper_numbers_v4_20261026/summary.json`
  - `artifacts/flashdrive_analysis_nvidia_v2_20261026/summary.json`
  - `artifacts/qwendrive_analysis_rl_v2_20261027/summary.json`
- `docs/PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md`, addenda 11, 11b, 12 and 13;
- the run logs of the held-out probe and the Qwen-Drive SFT v2 probe, read only to check their status.

**Context the authors disclosed to the AC, not in the manuscript:**
- 57 of the 60 main clips are in Alpamayo's *train* partition, and so are 391 of the 436 confirmation clips.
- A pre-registered held-out replication (addendum 13) is running. A1.5 was at context_031 of 120 when I checked.
- Qwen-Drive was re-run with a corrected reference (v2). The RL planner is finished; the SFT planner is at about context_114.

## 1. Summary

The paper argues that few-step Euler sampling of flow-matching driving policies contracts samples toward the conditional mean. As a result, the effect of fewer steps on rate metrics (ADE₁ and the decision-flip rate) is reproduced by an accurate reference solver run at a matched, lower "effective temperature".

The evidence is:
- three Alpamayo generations, 60 clips × 6 seeds, with τ_eff(E₁₀) ≈ 0.83;
- a Heun control that fails the same test;
- a linear-shrinkage control that passes it;
- a FlashDrive velocity cache that behaves like E₄;
- a v2 Qwen-Drive section in which step count maps to τ but temperature does not move the error.

The proposed deliverable is a protocol: report τ_eff, and compare against the reference at τ_eff before crediting an accuracy gain to a sampler.

## 2. What changed since v5.4

| v5.4 item | Status in v5.8 | Verdict |
|---|---|---|
| N1: temper the FlashDrive headline | Abstract: "behaves like four Euler steps … close to but not within our equivalence margin"; the finding title is now "behaves like" | **Done.** |
| N2: state the scope where the claim is made | Abstract: "On the original Alpamayo-1.5 checkpoint and the same clips"; finding: "We report ADE₁ only" | **Mostly done.** Contribution 2 still says the cache "acts as a lower temperature" without qualification. |
| N3: explained fraction against E₈ | Not stated. From the artifact it is (2.5065 − 2.3972)/0.1177 ≈ 93% at the point estimate | **Not done.** |
| N4: flips and minADE₆ for the cache | "decision flips for this run are pending" | **Not done.** It needs no GPU time. |
| N5: the introduction overclaims | "can inherit this temperature shift; we test one of them" | **Done.** |
| N6: z-lab decomposition | Still a TODO ("running") | **Not done.** |
| Qwen-Drive numbers unified | Rewritten on v2 (RL); SFT is a TODO | **Half done.** The v2 design is correct. |
| 436-clip confirmation, Heun on R1/A2, Proposition 3.1 proof, appendices B–E | Still TODOs; 12 TODO markers, unchanged | **Not done.** |
| Training-partition overlap (new) | Not disclosed. The paper says "PhysicalAI-AV test split" (abstract, line 159) | **New and serious** (W1). |

**The numbers check out.** I checked every macro I could trace against the artifacts and found no discrepancy:
- τ_eff(E₁₀) = 0.83 [0.80, 0.86], 0.84 [0.82, 0.85] and 0.83 [0.81, 0.85];
- the Heun-3/5 flips, 17.8/9.7% against 1.3/0.6% matched, with residuals of 16.5/9.1 points;
- the maximum absolute Euler flip residual of 2.0 (R1 E₄ −2.04, A2 E₁₀ +2.04 [0.03, 4.45]);
- the explained fractions of 79–121%;
- the FlashDrive values: 0.12 [0.09, 0.14], 0.64 [0.63, 0.65], −0.001 [−0.033, 0.032], 0.008 [−0.034, 0.051], E₄ upper bound 0.070, E₈ 0.016 [−0.014, 0.048], E₁₀ 0.012 [−0.016, 0.041];
- the Qwen-Drive values: spread 0.14, ADE₁ 1.55, τ_eff(E₁₀) = 0.98, E₄ 0.42 [0.41, 0.44], E₄ residual 0.041 [0.022, 0.062], matched gain 0.001, flips 4.4/1.8%, Heun-64 convergence 1.5 mm.

## 3. Strengths

1. **Clean mechanism with a usable closed form.**
   - E₁ returns the conditional mean.
   - The contraction factor ∏cos θᵢ has the cap cos(π/2k)^k.
   - The cap matches the median contraction.
2. **The test discriminates.** Heun-5, at the released budget, fails by about 9 points of flips at τ_eff ≈ 1, so landing on the curve is not automatic.
3. **Unusual honesty about what the test identifies.**
   - The shrinkage control passes the rate test.
   - Matching is at the level of rates over clips, not of individual samples.
   - Post-hoc margins and failed predictions are reported.
4. **The FlashDrive finding is now stated correctly.** "The cache ≡ E₄" (Δ ADE₁ −0.001 ± 0.03, τ_eff 0.64 against 0.63) is concrete and actionable for practitioners.
5. **The Qwen-Drive v2 redesign is principled.**
   - The reference stays inside the trained time range, t ≤ 0.9.
   - The convergence check (Heun-64) and the fp32 expert are sound.
   - The section reports a case where the accuracy story does *not* transfer, rather than hiding it.
6. **Full traceability from macro to artifact.** The pre-registration addenda, including the self-correction in 11b, are good scientific practice.

## 4. Weaknesses

**W1 (new, most important). The "test split" description is wrong for this model family, and it confounds the headline effect.** 57 of the 60 clips are in Alpamayo's training partition (addendum 13), yet the paper calls them the "PhysicalAI-AV test split" twice. As written this is a factual misstatement, and it must be corrected in the manuscript whatever the held-out run shows.

It is more than a labeling issue:
- The paper's motivating phenomenon is that fewer steps *lower* ADE₁, and that temperature reproduces the drop.
- That drop depends on the conditional mean being close to the logged trajectory.
- On memorized clips the mean should be unusually close, so the "accuracy gain from contraction" may be inflated or created by training-set membership.
- The Qwen-Drive contrast makes this sharper. There, temperature does not move ADE₁ (0.001 m), and the paper attributes this to error "set by the conditional mean". An equally simple reading is that Qwen-Drive did not train on these clips and Alpamayo did.
- The mechanism (contraction ≈ temperature in spread) probably survives, because it is a property of the integrator. The *practical claim* in the abstract and recommendations, that accuracy gains from fewer steps are a temperature artifact, is exactly what the contamination threatens.
- With only 3 held-out clips in the current sample, the paper cannot separate the two readings.

Addendum 13's prediction 2 (E₁₀'s raw ADE₁ effect is negative on all three models, and its residual is under 0.05 m) is the right test. Prediction 1 (τ_eff) tests the mechanism.

**W2. Equivalence is established in almost none of the main cells.** Of the six Alpamayo Euler cells (3 models × E₄/E₁₀):
- flips: only R1 E₁₀ passes TOST at 2 points;
- ADE₁: only R1 E₁₀ passes at 0.05 m;
- A2 E₁₀'s flip residual excludes zero.

The abstract's "reproduces … (residuals within 2.0 points)" reads as equivalence, but it is a statement about point estimates. The margins were also set after seeing these 60 clips. The held-out run fixes the margin problem and the contamination problem together, which is why it is the decisive experiment.

**W3. Qwen-Drive is a partial failure of the thesis, and the paper should say so plainly.**
- E₄'s ADE₁ residual is 0.041 m [0.022, 0.062]. The interval excludes zero and crosses the margin.
- Addendum 12 predicted residuals within ±0.05 m, and that prediction failed.
- E₁/E₂ hit the τ-grid floor (0.3).
- On this model, E₄ is *not* temperature in error: the whole effect is a non-temperature discretization cost.

The discussion frames this well ("fewer steps only cost accuracy"). Contribution 4, however, presents it as supporting evidence ("step count still maps to temperature") without saying that the pre-registered residual prediction failed. Say so, as addendum 12 itself requires. The SFT planner is still pending.

**W4. The "pre-registered" wording for FlashDrive is too strong.** Addendum 11b's own correction records that the τ_eff guess was saved 3 s before `summary.json`, in the same command as the analysis. "Our pre-registered guess" should read "our guess, recorded in the same command as the analysis, before any metric was printed", or the sentence should be dropped. Leaving it in hands a hostile reviewer an easy target.

**W5. Items carried over from v5.4, all cheap:**
- the explained fraction against E₈, with a clip-bootstrap interval (≈93% at the point estimate; it falls to ≈57% if the residual sits at its upper bound);
- flips and minADE₆ for the cache, from the existing `plans.npz`;
- the bf16 time grid, noted in the appendix;
- contribution 2's unscoped "acts as a lower temperature".

**W6. Completeness.**
- 12 TODOs remain.
- Appendices B–E are placeholders, including the 436-clip negative result that the related work cites as motivation (those clips are also 391/436 in-train).
- Proposition 3.1 is still a sketch.
- AlpaSim, NAVSIM and the z-lab decomposition are absent.

## 5. Questions to the authors

1. On the 3 test-partition clips in the current sample, and on the held-out clips when they land: is E₁₀'s raw ADE₁ gain against Heun-32 at τ = 1 of the same size? Please report it separately from the residual, because the gain is what contamination would inflate.
2. Can you confirm whether Qwen-Drive's training data include PhysicalAI-AV clips 100–159? This decides whether the Qwen/Alpamayo contrast is about the mean-dominated error or about memorization.
3. If the held-out τ_eff matches but the ADE₁ gain shrinks, will the abstract be reframed around spread and decisions, with the accuracy claim restricted?
4. On Qwen-Drive E₄, is the 0.041 m residual explained by the endpoint jump at t = 0.9, which is coarser with 4 steps? A τ-curve with E₄ on [0, 0.9] would tell.

## 6. Assessment

Between v5.4 and v5.8 the paper again improved its **wording**, not its **evidence**:
- the FlashDrive overclaim and the introduction overclaim are fixed;
- the Qwen-Drive section is rebuilt on a correct reference;
- the numbers remain fully traceable.

None of the evidential items I set for a 6 has landed: the decomposition, the cache flips/minADE₆, the confirmation and a complete manuscript.

The partition finding also changes the picture:
- the main sample is not held-out data for the model family that carries the positive result;
- the manuscript currently says it is;
- the Qwen-Drive contrast, now the paper's only second family, has a contamination-based alternative explanation.

To the authors' credit, they found this themselves and pre-registered a clean replication before running it. That is the right response, and it is why I do not lower the score.

## 7. Score

**Score: 5/10 (borderline reject). Confidence: 4/5.**

If v5.8 were submitted as-is, with "test split" and without the held-out result, I would move to **4**, because a reviewer who checks the split will treat the misstatement as disqualifying.

## 8. What would move the score

**To 6 (weak accept):**
1. **The addendum-13 held-out replication (120 clips, all three models), reported as pre-registered, with every prediction marked as passed or failed.**
   - If predictions 1–3 hold, make it the primary result and demote v1.
   - If ADE₁ gains shrink, reframe the claim around spread and decisions.
   This is now the single most important item.
2. Correct the "test split" text. Disclose the 57/60 and 391/436 overlap, and its implication for the 436-clip schedule result.
3. Report Qwen-Drive E₄'s failed residual prediction explicitly, and add the SFT planner.
4. Make the cheap FlashDrive additions: flips, minADE₆, the explained fraction with its interval, and the corrected "pre-registered" wording.
5. Bring the TODOs to zero, either by filling appendices B–E or by cutting them, and prove or demote Proposition 3.1.

**To 7 (accept):** all of the above, plus two of the following:
- the z-lab decomposition of FlashDrive's *published* gain at matched τ;
- the per-sample signature endpoint confirmed on held-out clips;
- a second family where temperature *does* move the error and Euler lands on the curve, such as NAVSIM PDMS;
- AlpaSim closed-loop results agreeing with open loop, with scene-clustered intervals.
