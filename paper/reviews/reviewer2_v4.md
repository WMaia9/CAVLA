# ICML 2027 Review, Round 4: Reviewer 2

**Paper:** "When Fewer Steps Mean Lower Temperature: Euler Sampling of Flow-Matching Driving Policies" (`versions/main_v4.1.tex` / `main_v4.1.pdf`, 8 pages, compiled 26 Oct 17:06).
**Previous reviews:** `reviewer2.md` (4/10), `reviewer2_v1.md` (4/10), `reviewer2_v3.md` (5/10).
**Authors' response:** `CONSOLIDADO.md`, section "Rodada 3".

## What I checked (CPU only, no GPU)

**Provenance.**
- `src/cavla/paper_numbers.py` is byte-identical to `artifacts/paper_numbers_v2_20261026/source_snapshot.py`.
- `summary.json` now records `boot: 1000`, which closes my round-3 provenance item.
- I re-derived every macro in `tables/numbers.tex` and `tables/numbers_tables.tex`, and every cell of `tab_matched.tex` and `tab_contraction.tex`, from `summary.json`. All of them agree.
- The PDF agrees with the `.tex`.

**New analyses.** Scripts are in `/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/r2_round4/`:
- `controls.py`: spread-matched contraction controls with different mechanisms; pooled and paired residuals; the Alpamayo-1.5 holdout.
- `misc.py`: spread curves, convergence order, per-clip energy distance and decision TV for the shrinkage arm, per-noise disagreement, and τ_eff against the Gaussian factor.
- Raw output is in `main.out`, `hold.out`, `all100.out` and `misc.out`.

All analyses use the paper's own primitives: `pn.clip_metrics`, `pn.spread`, `decisions`, the paper's matching rule, and a joint clip bootstrap (1000 resamples, seed 20261026).

---

## Summary

The paper argues that k-step Euler sampling of flow-matching driving policies acts as an implicit lower sampling temperature.

**Mechanism.** One Euler step returns the learned conditional mean E₁. More steps contract samples toward it by roughly the Gaussian factor ∏cos θᵢ.

**Empirical core.** On 60 PhysicalAI-AV clips × 6 seeds, for Alpamayo-1.5, R1 and 2:
- E₁₀ has τ_eff ≈ 0.83.
- The reference solver (Heun-32) at τ_eff reproduces 79–121% of the change in ADE₁ and decision flips.
- Heun-3/5 (Alpamayo-1.5 only) do not.

**New in v4/v4.1.**
- A shrinkage arm Shrink_k = E₁ + c(A₁ − E₁) is added to Table 1.
- The authors concede that rate-level evaluation cannot distinguish Euler from "any contraction toward the mean of the same size".
- The contribution is repositioned as a *matched-temperature evaluation protocol*.

## Strengths

1. **Reproducibility is now exemplary.** One artifact, one sample and one reference produce every macro and table, and I regenerated all of them. Ranges such as `\ExplMin`–`\ExplMax`, `\FlipResidMaxAbs` and the Shrink ranges are computed, not typed.
2. **Most of the number errors I flagged were fixed:**
   - 79–121% instead of "90–100%";
   - "at most 2.0 points" instead of "±2";
   - the Alpamayo-2 E₁₀ interval that excludes zero is now stated;
   - minADE₆ and entropy residuals are reported;
   - the Heun budgets are stated correctly;
   - the "436 unseen clips" contradiction is resolved.
3. **The authors ran the control I asked for and reported an unflattering outcome**: shrinkage passes too. The claim was weakened accordingly. That is the right reflex.
4. **The core E₁₀ result replicates out of sample.** Alpamayo-1.5 was probed on clips 100–199, but the paper uses only 100–159. On the unused 40 clips (160–199):
   - E₁₀ flip residual: −0.01 pp [−1.72, +1.67];
   - E₄: +1.95 pp [−1.62, +6.31];
   - Heun-3/5: +18.8 and +13.8 pp.

   On all 100 clips, the A1.5 E₁₀ 90% interval [−0.78, +1.41] pp lies inside the ±2-point margin. That establishes **equivalence**, which the paper does not currently claim for this model. The authors should use these clips (see Q1).
5. **Contractions with the wrong mechanism can be caught by the test** (my W1), which the authors have not shown. A protocol paper can make more of this than it currently does.

## Weaknesses (most severe first)

### W1. The new central concession is wrong in both directions, and the shrinkage control cannot support it.

The abstract says: "any contraction toward the mean of the same size would pass it". The Discussion says: "rate-level evaluation cannot tell apart [Euler] from other contractions of the same size".

**Shrink is close to temperature by construction.** For a Gaussian target, the reference at temperature τ is A_τ(z) = m + τ(A₁(z) − m). That is *exactly* linear shrinkage toward the mean. So Shrink_k is a linearized temperature with the learned E₁ as its centre. It passing tells us almost nothing about "other contractions". It is the null model, not an alternative.

**I built five other spread-matched contractions.** In each, c (or p) is bisected so that the clip-mean spread equals E_k's exactly. Pooled flip residual over the three models, joint bootstrap (`controls.py`):

| Control (spread = E_k) | E₄: pooled flip residual (pp) [95%] | E₁₀: pooled flip residual (pp) [95%] |
|---|---|---|
| Euler (paper) | +0.30 [−1.81, +2.64] | +0.84 [−0.38, +2.24] |
| Shrink toward E₁ (paper's c) | +1.44 [−1.01, +3.99] | +1.15 [−0.17, +2.51] |
| Shrink toward E₁, exact spread match | +1.87 [−0.79, +4.67] | +1.02 [−0.33, +2.47] |
| Shrink toward the **clip mean of the 6 reference samples** | **−2.76 [−4.83, −0.95]** | −0.83 [−1.75, +0.06] |
| Shrink x only (≈ longitudinal) toward E₁ | **−3.22 [−5.62, −1.01]** | **−1.66 [−3.15, −0.29]** |
| Mixture: replace a sample by E₁ with probability p | **+6.49 [+2.91, +9.79]** | **+4.21 [+1.31, +7.24]** |
| Shrink toward constant-velocity straight line | **+16.1 [+10.3, +22.6]** | **+4.45 [+1.78, +7.40]** |

(Shrinking y only cannot reach E_k's spread: the spread is dominated by x.)

Three conclusions follow:
- **"Any contraction toward the mean would pass" is false on the paper's own data.** A plain linear contraction toward an empirical estimate of the conditional mean fails at E₄, and its interval excludes zero. A mode-collapse mixture toward E₁ fails at both budgets.
- **"Cannot tell apart from other contractions" is also false.** The rate test has real power against wrong centres and against mixture collapse. Residuals across the contraction centres I tried span about 5 pp at E₄. Euler falls within that span; the paper's Shrink happens to fall near Euler.
- **The defensible claim is narrower and more interesting:** clip-averaged rates identify both the *size* and the *centre* of the contraction. Euler matches a contraction of the Gaussian-predicted size toward the *learned one-step mean*, to within about ±2 pp. That claim is testable, and the authors could make it with the controls above. The current text instead weakens the paper with an assertion that their data contradict.

**The paper's Shrink arm is not spread-matched as the caption claims.** In `add_shrinkage`, c = mean spread(E_k) / mean spread(A₁), pooled over clips. But E₁ is not constant across seeds (see W5), so Shrink's spread overshoots:
- τ_eff(Shrink) exceeds τ_eff(E_k) in **all 6 cells**, by 0.01–0.04 (e.g., R1: 0.65 vs 0.61).
- c is also fixed on the full sample, not re-estimated within the bootstrap.
- With an exact match, A2 Shrink₄'s residual is +3.4 pp [−0.5, +7.9]. Its 90% interval [−0.07, +7.2] nearly excludes zero.

The caption ("to E_k's spread") and the code docstring should be corrected, or the matching made exact.

### W2. The paper discards the one analysis that does separate Euler, shrinkage and temperature: per-clip distributions.

The Discussion concedes that "per-clip rates differ" but quantifies nothing. I computed per-clip trajectory energy distance and decision-histogram TV to the bracketing temperatures (`misc.py`, 60 clips):

| Model | Adjacent τ (0.6 vs 0.7) | **E₄** vs A₀.₆ / A₀.₇ | **Shrink₄** vs A₀.₆ / A₀.₇ | E₁₀ vs A₀.₈ / A₀.₉ | Shrink₁₀ vs A₀.₈ / A₀.₉ |
|---|---|---|---|---|---|
| A1.5 energy | 0.066 | 0.190 / 0.155 | **0.325 / 0.275** | 0.092 / 0.081 | 0.118 / 0.097 |
| R1 energy | 0.080 | 0.247 / 0.191 | **0.362 / 0.311** | 0.110 / 0.077 | 0.144 / 0.112 |
| A2 energy | 0.072 | 0.193 / 0.165 | **0.341 / 0.293** | 0.080 / 0.072 | 0.135 / 0.109 |
| Decision TV (A1.5 / R1 / A2), vs A₀.₆ | – | 7.2 / 8.3 / 9.4% | 9.7 / 12.2 / 14.7% | 2.5 / 4.4 / 5.0% | 3.9 / 5.8 / 7.5% |

Per-noise decision disagreement with the nearest temperature:
- E₄: **7.8–9.4%**, against an adjacent-τ baseline of 2.5–4.2%.
- Shrink₄: 9.7–15.0%.
- E₁₀: 2.5–5.0%, against 2.2–3.9%.

Per clip, the ordering is temperature < Euler < shrinkage at both budgets.
- So the authors' own data *can* tell Euler from shrinkage, and Euler is *more* temperature-like than linear shrinkage.
- E₄ is still 2–3× further from its matched temperature than adjacent temperatures are from each other.

This should be in the paper. It strengthens the E₁₀ story and bounds the E₄ story. Leaving it out while asserting "cannot tell apart" is the wrong choice for a paper that now sells itself as an evaluation protocol.

**The coverage metric discriminates as well.** The paired pooled minADE₆ residual, Euler minus Shrink, at E₁₀ is **+0.080 m [+0.010, +0.174]**. Euler loses best-of-6 coverage that a same-spread shrinkage does not.

### W3. The manuscript is still incomplete, and this is now the binding constraint.

My round-3 condition for any score above 5 was a complete manuscript. It is not met:
- 13 `\TODO` boxes in the source (15 "TODO" strings in the PDF), including:
  - the FlashDrive / closed-loop contribution;
  - the Prop. 3.1 proof;
  - the Heun control on R1 and A2 and the displacement-matched Heun;
  - the reference at τ ∈ {0.3, 0.2, 0.1};
  - the Qwen unification;
  - the language ablation;
  - **all of Appendices B–E**.
- App. E ("decision thresholds, seeds, data splits") is a TODO, yet §4 says the thresholds were "fixed in advance (App. E)". The thresholds exist in `trajectory_decision_gate.py` (1.5 m extent, ±20°, ±1.75 m, 0.15·v band); just print them.
- `references.bib` still has 4 entries with `note = {TODO verify ...}`.
- The "Related Work" negative result (a schedule that "passed a pre-registered confirmation on 436 clips ... yet its gain disappeared at matched temperature") points to App. C, which is a TODO.
- Contribution 4 (Qwen-Drive) still uses hand-typed numbers from a different sample (clips 100–200, different report), as its own TODO admits.

### W4. The Heun negative control is still on one model, and it is partly structural.

Nothing new was run. The TODOs remain: Heun on R1 and A2, a displacement-matched Heun, and a reference convergence check. New evidence on why this matters:
- **Heun inflates spread out of sample.** On A1.5 clips 160–199:
  - spread(Heun-3)/spread(A₁) = **1.22** and spread(Heun-5)/spread(A₁) = **1.12**;
  - τ_eff is therefore *above* the grid. `np.interp` clamps it to 1.000, where the matched flip rate is 0 by construction.

  On the paper's sample, the τ_eff intervals already reach 1.00. The control "fails" mainly because the temperature curve has zero flips at τ=1 by definition. Any inaccurate sampler that does not contract fails it. My round-3 displacement-matched residuals (+6.8 and +4.5 pp; Heun-5's lower bound is +0.6 pp) remain the honest size of this control.
- **Convergence orders are anomalous.** Mean distance to A₁ at equal clips and noise:
  - Euler: 0.678 → 0.500 → 0.348 m for k = 4, 6, 10, an empirical order of about 0.73;
  - Heun: 0.777 → 0.403 m for 3 → 5 steps, an order of about 1.3, not 2.

  Extrapolated, Heun-32's own error is about 0.04 m, roughly 2% of the spread. That suggests the reference is adequate for ADE. But the estimate is self-referential, and the low orders point to a non-smooth field near t=1 or to bf16 effects: exactly the Lipschitz caveat the Prop. 3.1 TODO leaves open. A Heun-32 vs Heun-64 fp32 check on 10 clips is still the cheapest way to close this.
- **The abstract generalizes from one model.** "Second-order Heun samplers with a similar budget fail the same test" rests on Alpamayo-1.5 only. Heun-3 (6 NFE) is not "a similar budget" to E₁₀.

### W5. Mechanism claims that the data contradict or do not support.

**(a) "One Euler step ... for every z: the one-step sampler has no diversity at all"** (§3), unchanged since round 2.
- The learned E₁ has a seed spread of **0.23 / 0.32 / 0.21 m**, 9–14% of A₁'s (2.0–2.3 m).
- This is precisely why the paper's Shrink arm is not spread-matched (W1).
- State it as a population identity and report the learned E₁ spread.

**(b) "A contraction ... of the size the Gaussian factor predicts"** (Discussion; abstract: "matches at the median").
- The factor matches the *median per-sample ratio to E₁*: 0.66–0.68 vs 0.67, and 0.84–0.86 vs 0.86.
- It does not match the quantity that defines τ_eff, which is spread.
  - E₄'s spread ratio is 0.56–0.58 and E₁₀'s is 0.79–0.80.
  - The Gaussian prediction for τ_eff (0.672 for E₄, 0.856 for E₁₀) lies outside the bootstrap interval of τ_eff in **4 of 6 cells**:
    - R1 E₄ [0.59, 0.63]; A2 E₄ [0.55, 0.67]; R1 E₁₀ [0.82, 0.85]; A2 E₁₀ [0.81, 0.85].
- The paper's own observation that Euler "contracts ... less on the median" than Shrink, yet reaches the same or lower spread, says the same thing. About 15% of Euler's spread loss is not explained by contraction toward E₁. Likely causes are heterogeneous contraction, with the largest deviations shrinking most, or samples collapsing toward each other.
- The closed form explains the median ratio; it does not predict τ_eff.
- In addition, the Gaussian column still comes from a different sample and space (200 clips, action space, Heun-160). The caption now discloses this, but Finding 1 still places the two side by side as an agreement.

**(c) Prop. 3.1** remains a sketch with a TODO. No experiment tests it: the gate's thresholds are not hyperplanes through a symmetry axis. Contribution 3 ("a symmetry condition under which no integrator changes a decision") overstates a sketch that assumes h·Lip < 1.

### W6. Internal contradictions and residual number or wording errors.

1. **Intro vs Finding 2 on minADE.**
   - The intro still says "averaged metrics such as minADE register [the temperature shift] as an accuracy gain".
   - Finding 2 now says minADE₆ is *worse* than the reference.
   - The artifact agrees with Finding 2. The minADE₆ effect is positive in all 6 cells: A1.5 E₄ +0.38 m [+0.09, +0.81], E₁₀ +0.20 m; R1 E₄ +0.18 m [+0.05, +0.33]; A2 E₄ +0.36 m, E₁₀ +0.20 m [+0.02, +0.49].
   - Only ADE₁ improves. This is the fourth round this sentence has survived.
2. **The equivalence criterion differs between text and code.**
   - §4 says equivalence holds when "the **95%** interval of the residual lies inside a margin".
   - The code uses `np.quantile(res, [.05, .95])`, a 90% interval (standard TOST).
   - Fix one or the other, and report the verdicts as a column. Under the code's rule only R1 E₁₀ is equivalent, on both ADE₁ and flips, out of 6 Euler cells. A1.5 E₆ is equivalent on ADE₁ but is not in the table.
3. **The abstract leads with fractions explained (79–121%)** although §4 says "we therefore read residuals and their intervals, not fractions, as evidence". Their bootstrap intervals (my round 3) span roughly 0.6–1.4 and up to 2.67. Lead with residuals.
4. **"From 4 clips in which the vehicle is nearly stationary"**: `context_124` is at 2.64 m/s. Unfixed since round 3.
5. **"All differences between samplers come from 5 distinct clips"** (collisions): still no artifact. In round 3 I counted 6 clips on the Euler + reference curve and 8 clips with Heun. Unfixed for the fourth round; either produce the artifact or delete the sentence.
6. **"Sampled before any analysis"**: unchanged. `context_100..159` was chosen after the A1.5 100-clip results existed.
7. **Some ranges are still hard-coded in min/max order:**
   - the abstract's `\AOneFiveETenTau--\ROneETenTau`;
   - Finding 1's `\ROneEFourContraction--\AOneFiveEFourContraction` and its siblings;
   - the "90–93%" typed by hand in the Discussion.

   All are correct today. Generate them anyway.
8. **Table 1 / Table 5:** Heun-3's contraction (0.90, 70% < 1) is still omitted from Table 5, although Fig. 2 plots it.
9. **"Sizable share of individual decisions"** (Discussion): give the number, 7.8–9.4% vs a 2.5–4.2% adjacent-τ baseline (W2).

### W7. The contribution has shrunk to a protocol whose novelty and consequence are not demonstrated.

After the repositioning, the main contribution is "compare an accelerated sampler against the accurate solver at the temperature that matches its spread". That is a sensible reporting norm. But:
- It is applied only to samplers the authors constructed (E_k, Heun-k).
- No published acceleration (FlashDrive, ProbeFlow, AdaVLA, Coda) is evaluated with it: the FlashDrive finding is a TODO.
- No closed-loop or safety consequence is shown.
- The only cross-family evidence is a boundary case (Qwen-Drive) where the view "does not apply", with numbers from a different sample.

The title ("When fewer steps mean lower temperature") still asserts the identity that the Discussion now says rate-level evaluation cannot establish.

## Questions for the authors

1. **The holdout clips.** Why does Alpamayo-1.5 use 60 clips when 100 were probed? On all 100 clips, E₁₀ meets your equivalence criterion (90% interval [−0.78, +1.41] pp). On the unused 40 it replicates (−0.01 pp). Will you report clips 160–199 as an internal replication, or use all 100?
2. **The abstract's claim.** Will you replace "any contraction toward the mean of the same size would pass it" with a statement that matches the controls in W1? Mixture collapse and wrong-centre or empirical-mean shrinkage are rejected.
3. **Per-clip results.** Will you report per-clip energy distance and decision TV (W2) as part of the protocol? They separate temperature, Euler and shrinkage, which rates do not.
4. **The Shrink arm.** Will you make it exactly spread-matched (bisect c, or match per clip) and re-estimate c inside the bootstrap?
5. **Heun spread.** Why does Heun-k increase spread above the reference on clips 160–199? Is the Heun-32 reference converged in fp32?
6. **Equivalence intervals.** Which interval defines equivalence, 90% or 95%?

## Score

**Score: 5 / 10 (weak reject).** Unchanged.

**Why not lower.** The pipeline is fully reproducible. Nearly all of last round's number errors are fixed. The authors ran the control I asked for and reported its unflattering result. The E₁₀ result replicates on 40 held-out Alpamayo-1.5 clips.

**Why not higher.**
- The manuscript is still incomplete (13 TODOs, empty appendices, no proof). This was my stated prerequisite.
- The new concession in the abstract is factually contradicted by the paper's own data (W1).
- The analyses that would make the protocol informative (per-clip distributions; contraction controls with a different mechanism) are absent.
- The Heun control is still single-model and partly structural.

**Confidence: 4 / 5.** Every number was recomputed from `summary.json` and the raw `plans.npz`. All new analyses use the paper's own primitives and bootstrap.

## What would change my score

- **+0.5:**
  - Fix W6: the intro minADE sentence, 90% vs 95%, a TOST column, the per-noise number, `context_124`, the collision sentence or an artifact for it, "sampled before any analysis", and the hard-coded ranges.
  - Fix W5(a) and W5(b) wording.
  - Make Shrink exactly spread-matched.
- **+1:**
  - Replace the "any contraction" concession with a contraction-control table like W1's, showing what the rate test rejects (mixture collapse, wrong centre, empirical-mean shrinkage) and what it does not (learned-mean shrinkage).
  - Add the per-clip energy and TV analysis (W2).
  - Together these make the protocol contribution real.
- **+0.5:** Use all 100 Alpamayo-1.5 clips, or report 160–199 as a replication. Add the displacement-matched Heun and the reference convergence check (Heun-32 vs Heun-64, fp32, 10 clips).
- **+1 to +2:** A second, non-Alpamayo stochastic flow policy; Heun on R1 and A2; and **one published acceleration** (FlashDrive) evaluated with the protocol, ideally with a closed-loop or safety consequence.
- **Prerequisite for > 5:** a complete manuscript, with no TODO boxes, filled appendices (thresholds, seeds, schedules), a proof of Prop. 3.1 or its removal, and a clean bibliography.
