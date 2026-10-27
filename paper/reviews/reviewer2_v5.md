# ICML 2027 Review, Round 5: Reviewer 2 (focused round)

**Paper:** `versions/main_v5.tex` / `main_v5.pdf` (both 26 Oct 18:47), compared with `main_v4.1.tex` by `diff`.
**Artifact:** `artifacts/paper_numbers_v3_20261026/summary.json`. `src/cavla/paper_numbers.py` is byte-identical to its `source_snapshot.py`.
**My scripts:** `/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/r2_round5/sig.py`, output in `sig.out`. I used the paper's primitives and a joint clip bootstrap (1000 resamples, seed 20261026).

## What was fixed (verified)

- **Shrink is now spread-matched exactly.** `add_shrinkage` bisects c, and τ_eff(Shrink_k) equals τ_eff(E_k) in all 6 cells (Table 1). Closes my round-4 W1 (second half) and Q4. One leftover: c is still fit once on the full sample and not re-fit inside the bootstrap.
- **The abstract no longer says "any contraction passes".** The new text is limited to linear shrinkage toward E₁, which matches my round-4 controls. Closes Q2.
- **E₁ spread** is now given as 9–14%, computed from the artifact (`\EOneSpreadMin/Max`). Closes W5(a).
- **The equivalence criterion is now 90% TOST,** matching the code, and the paper admits the margins were chosen after seeing the data. Closes W6.2. The paper still reports no verdict column. Under that rule only R1 E₁₀ is equivalent on ADE₁ and flips, and A1.5 E₆ on ADE₁ only.
- **The intro's minADE sentence is fixed.** The minADE₆ effect is positive in all 6 Euler cells. Closes W6.1.
- **The stop-clip speed is generated:** `\StopClipMaxSpeed` = 2.6 (context_124, 2.64 m/s). Closes W6.4.
- **The collision sentence** is softened to "a handful of clips". It is still not backed by an artifact, but it is no longer a false count.
- **Every new macro** (`Sig*`, `Shrink*`, `EOneSpread*`, `StopClipMaxSpeed`) and every Table 1 cell agrees with `summary.json`. I also recomputed the signature medians independently, and they agree (e.g., Euler E₄ 0.678 / 0.655 / 0.667; temperature 0.693 / 0.727 / 0.692).

## The signature claim (`find:shrink`, abstract, Discussion)

**What the paper claims.** "Euler's per-sample signature resembles temperature's rather than a uniform contraction's." The abstract says the per-sample contraction of shrinkage "differs from Euler's, which resembles temperature's." The finding is labeled exploratory and same-sample, which is correct. **But it has no interval, and with intervals it is only half true.**

Paired per-model differences in median contraction ratio and in the fraction of samples contracted (95% clip bootstrap):

| Cell | Euler − Temp (median) | Euler − Temp (fraction contracted) | Euler − Shrink (median) |
|---|---|---|---|
| A1.5 E₄ | −0.014 [−0.066, +0.026] | **+0.045 [+0.002, +0.088]** | **+0.131 [+0.101, +0.168]** |
| A1.5 E₁₀ | +0.020 [−0.012, +0.052] | +0.015 [−0.029, +0.064] | **+0.075 [+0.062, +0.087]** |
| R1 E₄ | **−0.072 [−0.120, −0.033]** | **+0.122 [+0.070, +0.177]** | **+0.119 [+0.099, +0.136]** |
| R1 E₁₀ | **−0.032 [−0.051, −0.011]** | **+0.103 [+0.062, +0.146]** | **+0.057 [+0.046, +0.068]** |
| A2 E₄ | −0.025 [−0.077, +0.025] | **+0.046 [+0.009, +0.087]** | **+0.133 [+0.106, +0.152]** |
| A2 E₁₀ | +0.006 [−0.016, +0.024] | +0.026 [−0.010, +0.063] | **+0.071 [+0.056, +0.082]** |

**1. Euler is separable from temperature on the paper's own signature.**
- R1 differs on the median at both budgets.
- The fraction contracted differs in 4 of 6 cells. Euler contracts *more* samples than temperature does.
- The paper's own ranges already show this: 90–93% against 81–89%, and 0.66–0.68 against 0.69–0.73 at E₄. Neither pair overlaps. Yet the text presents them as alike ("both leave some samples uncontracted").
- The supported statement is narrower: "Euler is closer to temperature than to uniform shrinkage, but distinguishable from temperature in some models."

**2. The separation from Shrink is close to true by construction.**
- For Shrink, every per-sample ratio equals c exactly. So its median is c, and "contracts all of them" (100%) is a definition, not a finding.
- Spread is a *mean* over pairwise distances. Any contraction whose ratios vary from sample to sample, and that is matched on spread, will have a median different from c.
- Shrink's bootstrap interval is degenerate, because c is fit on the full sample.
- The large Euler − Shrink gap is robust, but it shows that Euler's contraction varies across samples. It does not show that the contraction is temperature-like.

**3. The median signature is not specific to temperature.** I built two more spread-matched controls:
- **Uniform shrinkage toward the empirical mean of the 6 reference samples (`toMean`).**
  - At E₁₀ its median matches Euler's (Euler − control: −0.003 / −0.018 / +0.000).
  - At E₁₀ it is within the temperature interval in 2 of 3 models.
  - It also passed the rate test at E₁₀ in my round 4: −0.83 [−1.75, +0.06].
- **A radial contraction toward E₁ that shrinks large deviations more, `g = min(1, κ·d^(−1/2))`.**
  - At E₄ it matches Euler's median in all three models, with intervals containing zero.
  - Its upper quantiles are different (q75 0.85–0.95 vs 0.76–0.81).

**4. "Heun fails both tests"** (Discussion) is not backed by any reported number.
- In the artifact, Heun-3/5 contract 70%/69% of samples, while the matched temperature contracts 42%/21%.
- That matched temperature value is interpolated near the τ=1 end of the grid, where τ_eff is clamped.

**Verdict on strength.**
- The abstract sentence is stronger than the evidence. It is same-sample, exploratory, has no interval, and is contradicted on one model and on the fraction-contracted statistic.
- It needs a clip-bootstrap interval before it can appear anywhere, and it should come out of the abstract.
- In the body, report paired Euler − Temp and Euler − Shrink intervals per model (a table like the one above). Compare the full quantile profile, not only the median. Include at least one non-uniform control.
- Then reword: "Euler's per-sample contraction is heterogeneous, unlike uniform shrinkage, and closer to temperature's than to shrinkage's; it is not identical to temperature's (R1; fraction contracted)."
- As a pre-registered secondary endpoint, declare the statistic (e.g., the paired difference in median ratio, or a two-sample quantile distance), the margin and the direction now.

## Remaining issues, ranked

1. **Signature claim overstated, with no interval** (above). New in v5; it is now in the abstract.
2. **The manuscript is still incomplete.**
   - There are 13 `\TODO` markers in the source and 15 in the PDF. Appendices B–E are empty, Prop. 3.1 has no proof, the bib notes still say "TODO verify", Qwen-Drive numbers are hand-typed, and the FlashDrive contribution is missing.
   - This remains my prerequisite for a score above 5.
3. **The Heun control is still on one model, and partly structural.** τ_eff is clamped at 1. No displacement-matched Heun has been run, and the reference's convergence has not been checked (Heun-32 vs Heun-64 in fp32). The abstract still generalizes from Alpamayo-1.5 alone.
4. **The rate-level controls are still absent.** The paper now says that only a uniform shrinkage toward E₁ passes, which is accurate. It still does not show what the test *rejects*: mixture collapse at +6.5 pp, the empirical-mean centre at E₄, the constant-velocity centre. Showing that is what makes a protocol paper. The per-clip energy distance and TV analysis (my round-4 W2) is also still absent, and "a sizable share of individual decisions" still has no number.
5. **Leftover wording and statistics problems.**
   - "Sampled before any analysis" is still false for A1.5.
   - The A1.5 holdout clips (160–199) are unused.
   - The abstract still leads with fractions explained (79–121%).
   - No TOST verdict column; reporting one would show that 1 of 6 Euler cells is equivalent.
   - Shrink's c is not re-fit inside the bootstrap.
   - The Shrink minADE₆ effect is smaller than Euler's (A1.5 E₁₀: 0.05 vs 0.20 m). That is a second rate-level discriminator the paper does not mention.

## Score

**5 / 10 (weak reject), unchanged. Confidence 4/5.**

The corrections are real and verified: exact spread matching, an honest abstract on rates, TOST, minADE, E₁ spread, and the stop speed. Every number reproduces.

But the one genuinely new scientific claim is stated more strongly than a bootstrap supports, and it sits in the abstract. The manuscript is also still incomplete. Fixing the signature claim (intervals, out of the abstract, rewording as above) is cheap. Moving above 5 still requires a complete manuscript.
