# ICML 2027 Review, Round 5.8: Reviewer 2

**Paper:** `versions/main_v5.8.tex` / `main_v5.8.pdf` (27 Oct 04:48, 9 pp.), diffed against `main_v5.4.tex`.
**Macros:** `tables/numbers.tex`, which is byte-identical to `artifacts/paper_numbers_v4_20261026_numbers.tex`, and `tables/numbers_tables.tex`.
**Artifacts checked:**
- `paper_numbers_v4_20261026/summary.json`
- `flashdrive_analysis_nvidia_v2_20261026/summary.json`
- `qwendrive_analysis_rl_v2_20261027/summary.json`
- `physicalai_test_contexts_v1_20261022/records.json`
- `physicalai_splits_20261027/splits/{alpamayo_aligned,default}.parquet`

**Protocol:** `docs/PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md`, Addenda 11–13.

## Summary

The paper argues that few-step Euler sampling of flow-matching driving policies amounts to a hidden reduction in sampling temperature.
- **Mechanism.** One Euler step returns the conditional mean. More steps contract samples toward it by a Gaussian factor ∏cos θᵢ, which v5.8 caps at cos(π/2k)^k on any grid.
- **Main evidence.** On 60 PhysicalAI-AV clips × 6 seeds and three Alpamayo generations, the released E₁₀ sampler sits at τ_eff ≈ 0.83. A Heun-32 reference started from τz reproduces most of the ADE₁ and flip-rate changes of E₄ and E₁₀. Heun-3 and Heun-5 do not.
- **Controls and extensions.** A linear shrinkage passes the same rate test. FlashDrive's velocity cache behaves like E₄. A rewritten Qwen-Drive section (the v2 probe) finds that step count maps to temperature there as well, but temperature does not change error, so fewer steps only cost accuracy.
- **Proposal.** Accuracy gains from accelerated samplers should be reported against a reference solver at matched effective temperature.

## Numbers check

Spot-checked against the artifacts. All of these match:
- every `tab_matched` cell;
- `ExplMin/Max` 79/121;
- the Heun residuals 16.5 [9.8, 23.1] and 9.1 [4.3, 14.2];
- the shrink flip residuals 0.7–3.4;
- the signature differences;
- the stop counts;
- every FlashDrive macro, including the new `FDEEightResid` 0.016 [−0.014, 0.048] and `FDGainEEightCI` [0.09, 0.14];
- every Qwen-Drive macro: spread 0.136, ADE₁ 1.550, E₁₀ τ_eff 0.981 with residual 0.0005 [−0.0001, 0.0013], E₄ τ_eff 0.422 [0.406, 0.438] with residual 0.041 [0.022, 0.062], flips 4.4% vs 1.8%, and Heun-64 vs Heun-32 at 1.46 mm.

The new closed-form cap is correct: cos(π/20)^10 = 0.883, cos(π/8)^4 = 0.729 and cos(π/4)^2 = 0.5. The angles in the (1−t, ts) quadrant do sum to π/2, so the Jensen argument holds.

Small discrepancies:
- **`FlipResidMaxAbs` = "2.0" is a rounding.** The two extreme cells are +2.04 (Alpamayo-2, E₁₀) and −2.04 (R1, E₄). Both point estimates lie *outside* the 2-point equivalence margin. "Residuals within 2.0 points" in the abstract is therefore the same kind of margin-adjacent rounding I flagged for "0.051 m" in v5.4.
- **The Qwen matched-temperature change is −0.0007 m**, printed as "changes ADE₁ by 0.001 m". The sign is lost, though this is harmless.
- **The Qwen τ grid extends to 0.3**, while Definition 4.1 fixes {1, …, 0.4}.

## Status of my v5.4 concerns

| v5.4 item | v5.8 |
|---|---|
| The abstract spun a failed FlashDrive TOST ("within 0.051 m") | **Fixed.** It now reads "0.008 m, close to but not within our equivalence margin". |
| Title of `find:flash`: "is a lower temperature" | **Fixed:** "behaves like a lower temperature". |
| E₈ verdict, gain CI, interval levels, linear-interpolation note | **All fixed.** |
| Linear vs PCHIP interpolation sensitivity | Not done. |
| FlashDrive flip residual | Not done. It is now *disclosed* as pending, which is the honest minimum. |
| "Indistinguishable from E₄" | **Fixed:** "close to", with the CI. |
| 11b heading contradicted the timestamp correction | **Fixed** in the protocol. |
| Qwen-Drive hand-typed from another sample | **Fixed.** It is now 60 clips, macros come from an artifact, and the reference stays inside the trained time range. This is a substantive and honest correction: the v1 "solver adds spread" boundary claim was an artifact of querying t > 0.9, and the authors withdrew it rather than defend it. |
| Incomplete manuscript | **Not fixed.** There are 12 `\TODO`s (13 in v5.4); Appendices B–E are empty; Prop. 3.1 has no proof; "pre-registration documents, data splits" are a TODO. |
| Independent replication | **Not in the paper.** Addendum 13 is running; Alpamayo-1.5 has processed 32 of 120 contexts. |
| Rate-level negative controls; a TOST verdict column in Table 1 | Not done. |

## Strengths

1. **A clean, falsifiable idea with a correct closed form.** The new grid-independent cap cos(π/2k)^k ("choosing where steps go can reduce the contraction toward this cap but cannot remove it") is a genuinely useful addition. It turns an empirical observation into a statement about every first-order schedule.
2. **The test discriminates.** Heun-3 and Heun-5 fail by 9–17 points with intervals far from zero, so landing on the temperature curve is not automatic.
3. **Unusual candor in the body.**
   - The margins are stated to have been chosen post hoc.
   - The fact that shrinkage also passes the rate test is foregrounded.
   - The FlashDrive equivalence failure is stated plainly.
   - The authors admit the pre-registered τ_eff guess for FlashDrive was wrong.
4. **The Qwen-Drive v2 rewrite is the right scientific move.** The original result was built on an out-of-distribution reference. It was rerun inside the trained range in reasoning mode with fp32, and it produced a different and more interesting conclusion: fewer steps pay discretization cost with no temperature benefit when error is set by the conditional mean.
5. **Numbers are machine-generated from snapshot-stamped artifacts.** I found no transcription errors.

## Weaknesses (ranked)

### W1. The paper calls its evaluation data "test" clips and does not disclose that 57 of the 60 are in Alpamayo's training partition (critical; honesty)

I verified this independently by mapping `records.json` → `clip_id` → `alpamayo_aligned.parquet`:
- context_100–159: **57 train / 3 test**;
- the "A1.5 holdout" 160–199 that I recommended in v5.4: **35 train / 5 test**;
- per Addendum 13, the 436-clip schedule confirmation: **391 / 436 train**.

All 60 clips are in PhysicalAI-AV's *default* test split, so the label is technically defensible. It is nonetheless misleading in context. The paper repeats "PhysicalAI-AV test split" or "test clips" in:
- the abstract;
- Sec. 4 (Data);
- the Fig. 3 and Table 1 captions.

Contribution 2 says "on **real test data**". Appendix C and Related Work describe the 436-clip confirmation as "on 436 clips not used to design it", which is true, but those clips are almost all inside the policy's training set. The data manifest itself lists "Alpamayo training exposure to test clips is not independently verified". The authors have since verified it, and the answer is unfavorable.

**Why it matters scientifically.**
- *ADE₁ claims.* On memorized clips the conditional mean is plausibly closer to the logged human trajectory. That is exactly the regime in which contracting toward the mean (lower τ, fewer steps) looks like an "accuracy gain". The ADE₁ half of the central claim, and the FlashDrive 0.12 m gain, could be inflated by training exposure.
- *Flip claims.* These depend on the shape of the learned field near the data. Training clips may have sharper conditionals and different spread structure.
- *Scope of the mechanism.* The mechanism (Eq. 1) does not depend on the split. The *empirical* "temperature reproduces 79–121%" claim does.

**What the authors knew.** They know this, and they wrote Addendum 13 at 02:48, two hours before this PDF was built at 04:48. The paper still says nothing. Even the `find:matched` TODO frames the confirmation as needing fresh clips only because "the 436 clips … were already used", not because they are training clips.

A pre-registered replication that is running is good practice. It does not license a draft that describes the current sample as test data. An ICML reviewer who found this through the public split file would treat it as a credibility problem for the whole paper, not as a limitation.

**Minimum fix:** one sentence in Data and in Limitations:
> "57 of the 60 clips (and 391/436 of the schedule confirmation) lie in Alpamayo's training partition; results on 120 held-out clips are in Sec. X."

Also:
- drop "real test data";
- make the same disclosure for R1 and Alpamayo-2. Does `alpamayo_aligned` even describe their training sets? See Q1.

### W2. The headline equivalence is stated more strongly than the tests support, and nothing is held out

The abstract says that at τ_eff "the reference solver *reproduces*" both ADE₁ and flip changes, with "residuals within 2.0 points". From `summary.json`, using the 90% TOST at the paper's own margins, here is the verdict for E₄/E₁₀:

| Cell | ADE₁ | Flips |
|---|---|---|
| Alpamayo-R1 E₁₀ | pass | pass |
| All other E₄/E₁₀ cells (5 of 6) | fail | fail |

In the failing cells:
- A1.5 E₁₀ has an ADE₁ residual of +0.027 m, with a 95% upper bound of 0.095 m;
- A2 E₁₀ has a flip residual of +2.04, with its interval excluding zero;
- R1 E₄ has a flip residual of −2.04.

Two of the six point estimates are already outside the 2-point margin, and "within 2.0" hides this by rounding. On top of that:
- the margins were chosen after seeing these 60 clips;
- all of the evidence comes from a sample that is mostly in-training (W1);
- "fraction explained" values above 100% (up to 121%) mean the explicit temperature overshoots.

What the paper has shown is "consistent with temperature, not shown equivalent". The body (`find:matched`) says roughly this. The abstract and Contribution 2 do not. Table 1 still lacks an ADE₁ residual interval and a TOST verdict column.

### W3. Still an incomplete manuscript, and two of the four contributions are only partly evidenced

The incomplete parts:
- 12 TODOs;
- four empty appendices, including the protocol and reproducibility appendix that would hold the decision thresholds and split lists;
- no proof of Prop. 3.1, whose own TODO concedes the Lipschitz hypothesis "can fail near t = 1";
- Heun only on one model;
- the FlashDrive *shipped* system (W4A8 + DFlash + streaming) still "running";
- FlashDrive flips "pending";
- the Qwen SFT planner "running".

The FlashDrive contribution tests a re-implementation of one component, on ADE₁ only, on the training-partition clips.

The Qwen section is now honest, but:
- it is one planner;
- it does not report that Addendum 12's pre-registered prediction for E₄ (ADE₁ residual within ±0.05 m) failed on the interval (upper bound 0.062);
- its flips are untestable.

### W4. Addendum 13 is weaker than the paper's TODO promises

These problems should be fixed *before* its results are unblinded:
- **Not fresh seeds.** It reuses seeds 9201–9206, while the paper's TODO promises "fresh clips and seeds".
- **A weaker standard for ADE₁.** Prediction 2 tests a *point estimate* |·| < 0.05 m, which is weaker than the paper's own TOST.
- **No primary endpoint.** It drops the paper's stated primary endpoint ("E₁₀ equivalence averaged over models") and states no decision rule for it.
- **Undefined trigger.** "If the numbers change materially" is not defined.
- **Missing pieces.** The shrinkage-signature secondary endpoint that the Discussion says is "fixed before it runs" is not in Addendum 13, and FlashDrive is not included.
- **No integrity record.** There are still no hashes or version control for the registration.
- **Power.** At n = 120, flip-residual 95% intervals of about ±2.5–3.5 points shrink by only about √2. Equivalence at 2 points will likely still be inconclusive for E₄.

### W5. Minor issues

- **Inconsistent interval levels.** Table 1 uses 95% intervals, while equivalence uses 90%. Say so in the text, as you now do for FlashDrive.
- **Grid mismatch.** The Qwen τ grid (down to 0.3) contradicts Definition 4.1.
- **Unexplained Heun error.** Heun's ADE₁ residual is also large and positive (+0.15 m for Heun-3), and the paper never comments on it. It supports "Heun has its own integration error", but readers will ask.
- **Stop counts.** "Stops come from 4 clips" pools clips across models; each model has 2–3.
- **Sign.** The Qwen "changes ADE₁ by 0.001 m" is actually −0.0007 m.
- **Interpolation sensitivity** for FlashDrive is still not reported.

## Questions

1. Is `alpamayo_aligned` the training split of Alpamayo-1.5 only, or of R1 and Alpamayo-2-Super as well? If Alpamayo-2 was trained on more data, how many of its 60 clips are truly held out? Was Qwen-Drive trained on PhysicalAI-AV?
2. On the 3 + 5 + 10 Alpamayo-test clips already inside contexts 000–199, do the ADE₁ effects of E₄/E₁₀, and their residuals, differ in sign or size from the training clips? This is underpowered but is a free sanity check, and it should be declared as exploratory.
3. If the held-out replication shows the same τ_eff and flip residuals but a *smaller* ADE₁ gain, will the abstract's ADE₁ claim be revised? Addendum 13 should state this decision rule now.
4. Why does Addendum 13 reuse seeds 9201–9206 when the paper promises fresh seeds?
5. Was Addendum 12's prediction for Qwen E₄ (equivalence at 0.05 m) evaluated as pass or fail? The interval says fail.

## Score

**4 / 10 (reject; would return to 5 with disclosure). Confidence: 4/5.**

The paper moved forward on every axis I raised in v5.4:
- the FlashDrive framing is now accurate;
- the Qwen rewrite is a real correction;
- the grid-independent cap strengthens the theory.

On those merits alone I would have held 5.

I am lowering the score because the draft, as written, describes its evaluation as "test data" without mentioning that the authors already know 95% of it is in the evaluated model's training partition. A running pre-registered replication is the right response. Leaving the current text unqualified is not.

This is easily fixed, and I want to be clear that I see no fabrication: every number checks out. The issue is omission of a fact that is material to the ADE₁ claims.

## The three changes that would most raise my score

1. **Disclose the training overlap now and reframe the existing sample** (no compute needed; brings the score back to 5).
   - Replace "test split" / "real test data" with an accurate description, plus the 57/60 and 391/436 counts in Data, Limitations and Appendix C.
   - Answer Q1 for R1 and Alpamayo-2.
   - Fix the abstract's "reproduces … within 2.0 points" to report the equivalence verdicts: 1 of 6 E₄/E₁₀ cells passes both metrics.
   - Add a TOST column to Table 1.
2. **Make the held-out replication (Addendum 13) the main result** (to 6–7).
   - Tighten the addendum before unblinding: fresh seeds or an explicit justification, TOST rather than point estimates, the primary endpoint and decision rule, the shrinkage-signature endpoint, and hash the registration.
   - Report every prediction's pass or fail, and move the 60-clip sample to a secondary analysis.
   - If the ADE₁ gain shrinks on held-out clips, say so in the abstract. That would itself be an interesting finding about memorization and temperature.
3. **Finish the manuscript.**
   - Remove all TODOs, fill Appendices B–E (including the split lists and thresholds), and write the proof of Prop. 3.1 with the near-t = 1 caveat.
   - Add the FlashDrive flip residual and the shipped FlashDrive checkpoint, and Heun on R1 and Alpamayo-2.
   - Without this the paper is not reviewable as a complete ICML submission, whatever the replication shows.
