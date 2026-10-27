# ICML 2027 Review, Round 3: "When Fewer Steps Mean Lower Temperature: Euler Sampling of Flow-Matching Driving Policies"

**Reviewer:** balanced / area-chair style.
**Previous rounds:** `reviewer_balanced.md` (4/10) and `reviewer_balanced_v1.md` (4/10). In round 2 I projected ~6/10 if actions 1–3 came out positive and ~7/10 if actions 4–5 did too.

**Material reviewed:**
- `paper/main.tex` (identical to `versions/main_v3.tex`) and `main.pdf` (6 pages, 13 TODOs).
- `paper/tables/*` and `reviews/CONSOLIDADO.md` ("Rodada 2").
- `docs/CRONOGRAMA_ICML2027.md`, `docs/AUDITORIA_ICML_20261026.md` and `docs/PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md`.
- `artifacts/paper_numbers_v1_20261026/summary.json` and `src/cavla/paper_numbers.py`.
- The raw plans in `artifacts/temperature_probe{,_r1,_a2}_v1_20261022/plans.npz`.

**What I ran:** CPU only. Every number marked **[recomputed]** comes from my own passes over those plans, using the authors' `decisions`, `load` and `matched` functions unchanged. I used two scratch scripts, both described in §7 so they can be re-implemented:
- `shrink_control.py`, a new shrinkage-toward-the-mean control;
- `power.py`, which estimates clip-level residual SDs and the power of the equivalence test (TOST).

---

## 0. Bottom line

**This revision does what I asked in round 2, and the paper is now honest.**
- The stopping claim is gone from the abstract and is now a counted, exploratory paragraph (6/360 plans from 4 clips).
- The Gaussian factor is explained as a median value, not as a per-clip predictor.
- The Heun-3/5 negative control is in Table 2.
- The equivalence result is reported as underpowered rather than "reproduced".
- Matching holds for distributions, not for individual samples, and the text now says so.

**The narrowed thesis is coherent.** Its three parts fit together:
- Euler-k sits on the temperature curve at the level of distributions;
- Heun-k at the same compute budget does not, which shows the test can fail;
- a nearly deterministic flow (Qwen-Drive) sits outside the regime.

If the pending runs come out as expected, this is a defensible ICML paper. I still score it below the bar today, for three reasons:

1. **Every result that carries significance is pending.** These are the out-of-sample equivalence test, FlashDrive, a second model family and closed loop.
2. **New issue: the test does not single out temperature.** A trivial post-hoc shrinkage of the reference sample toward the one-step mean, with no integrator at all, reproduces Euler-k's flips and ADE₁ as well as explicit temperature does (§2.1, **[recomputed]**). The test separates *mean-directed contraction* from Heun-type error. It does not separate *temperature* from other contractions. The claim and the title need to absorb this.
3. **The "436 unseen clips" are not unseen.** They were the confirmation set for the schedule study. Alpamayo-1.5's uniform Euler-4/6/8 flip rates against a Heun-80 reference have already been observed on them (seeds 9201–9204). The paper's Related Work says so itself, and `docs/AUDITORIA_ICML_20261026.md` flags it too. The confirmation is still valuable, but it must be described and designed accordingly (§5).

**Score: 5/10 (borderline reject), confidence 4/5.** That is up one point from round 2, for honesty and coherence. The minimum results for a 6 and for a 7 are listed in §4.

---

## 1. Status of my round-2 actions

| # | Round-2 action | Status | Comment |
|---|---|---|---|
| 1 | Correct the draft (stopping, joint bootstrap, TOST, Heun rows, s-range, post-hoc margins) | **Done** | The joint clip bootstrap is in `paper_numbers.py`, and TOST uses a 90% interval, which is correct. Heun is in Table 2. The s-range statement has been fixed. Two gaps remain: the margins are still not described as set after the 60-clip data were seen, and "fraction explained" is shown without intervals (§3). |
| 2 | Enriched pre-registered confirmation | **Planned** | The design needs changes (§5). |
| 3 | Official FlashDrive at matched temperature | **Running** | `flashdrive_probe_nvidia_v1` is at about clip 125 of 60–100 windows. |
| 4 | Gaussian model done properly | **Descoped, acceptably** | It is now explicitly a median-level explanation. The per-clip test is gone rather than fixed. That is fine for a methodology paper. |
| 5 | Closed loop, powered and on enriched scenes | **v1 running** | The v1 arms are E10, E4, Heun-32 at τ ∈ {1, 0.84, 0.65}, and the matched arms were chosen in advance, which is good. With 58 scenes and one rollout per scene it has low power (§4). Two arms show a `failed.0` restart. Report that. |
| 6 | Second driving family (NAVSIM) | **Data downloaded** | Not run. |
| 7 | Language-seed ablation | Not done | Still a TODO. |
| 8 | Distributional matching (energy distance), per-clip τ_eff, mean-directed flips | Not done | My shrinkage check (§2.1) shows why a per-sample or distributional comparison now matters more. |
| 9 | Proof; τ grid below 0.4 | Not done | The Lipschitz caveat is acknowledged. The τ grid still stops at 0.4, so E1 and E2 cannot be placed on the curve (§2.2). |

---

## 2. New findings from the data

### 2.1 Shrinkage toward the mean passes the same test **[recomputed]**

**The control.** Define S_c(z) = E₁ + c·(A₁(z) − E₁). It takes the reference sample and pulls it linearly toward the one-step mean, with no integration involved. The spread of S_c is exactly c times the reference spread, so I set c = Spr(E_k)/Spr(A₁), which matches Euler-k's spread. On the same 60 clips × 6 seeds, with the joint clip bootstrap:

| Model, sampler | τ_eff | c | Flips: E_k / A_τeff / S_c | Flips, E_k − S_c | ADE₁ (m): E_k / A_τeff / S_c | Per-sample distance (m): E_k→A_τeff vs E_k→S_c |
|---|---|---|---|---|---|---|
| A1.5 E4 | 0.64 | 0.57 | 10.6 / 10.3 / 10.0 | +0.6 [−3.1, 4.2] | 2.58 / 2.58 / 2.54 | 0.33 vs 0.34, n.s. |
| A1.5 E10 | 0.83 | 0.80 | 4.7 / 5.0 / 5.3 | −0.6 [−2.5, 1.4] | 2.73 / 2.70 / 2.68 | **0.20 vs 0.23**, diff −0.038 [−0.079, −0.005] |
| R1 E4 | 0.61 | 0.58 | 10.8 / 13.3 / 12.5 | −1.7 [−5.0, 1.4] | 2.10 / 2.15 / 2.11 | 0.44 vs 0.34, diff +0.098 [−0.003, 0.197] |
| R1 E10 | 0.84 | 0.80 | 6.1 / 7.2 / 6.9 | −0.8 [−3.1, 1.1] | 2.28 / 2.29 / 2.28 | 0.22 vs 0.19, n.s. |
| A2 E4 | 0.62 | 0.56 | 14.7 / 13.6 / 15.8 | −1.1 [−4.7, 2.5] | 2.61 / 2.56 / 2.61 | 0.36 vs 0.35, n.s. |
| A2 E10 | 0.83 | 0.79 | 9.7 / 8.6 / 8.3 | +1.4 [−0.6, 3.3] | 2.78 / 2.76 / 2.76 | **0.18 vs 0.24**, diff −0.059 [−0.108, −0.021] |

Notes on the table:
- The A_τeff column uses the nearest grid τ for flips and the linearly interpolated plan for ADE₁. It therefore differs slightly from Table 2, which interpolates the metric curves.

What this means:
- **At the level of distributions, the test cannot tell temperature from shrinkage.** For flips and ADE₁, spread-matched shrinkage is as good a match for Euler-k as A_τeff.
  - The protocol's real content is therefore "**matched spread with a mean-directed contraction**".
  - Heun-k fails because its error is not mean-directed. That makes the negative control more interpretable, not less.
- **Sample by sample, temperature is the better description for E10 in two of three models.** E10 is significantly closer to A_τeff than to S_c in A1.5 and A2. For R1 E4 the point estimate goes the other way.
  - This is the result that distinguishes "Euler acts like temperature" from "Euler acts like any shrinkage". It is cheap to add, because the plans already exist.
- **What the paper should do:**
  - Add S_c as a second reference arm in Table 2.
  - State the claim as "few-step Euler is a mean-directed contraction. At matched spread, its rate-level effects are those of any such contraction. The explicit temperature is the natural, tunable member of that class."
  - Use the per-sample distance comparison as the evidence for preferring temperature, and pre-register it (§5).
- **This does not hurt the evaluation recommendation.** "Compare against an accurate solver at matched spread" stays valid whichever contraction is the better mechanistic description. It does hurt any sentence implying that the solver *is* a temperature knob in a sense stronger than spread matching.

### 2.2 Euler-1/2 fall outside the grid, and Euler-1 fails the test **[from summary.json]**

- **Euler-1 cannot be placed on the curve.** τ_eff is clamped at 0.4 (`in_range: False`) for all three models. Its flip residual is +11.1, +10.8 and +15.3 points, and every interval excludes 0.
- **The paper hides this.** The abstract opens with "One Euler step returns the conditional mean", but Euler-1 is in neither table.
- **Euler-1 is a real limit, not only a grid artefact.** E₁ is the conditional mean. A₀ is the flow started from z = 0. For a multimodal target these differ, and neither equals the other's decision. So the temperature view must break down as k → 1.
- **What to do:** report E1/E2. Extend Heun-32 to τ ∈ {0.3, 0.2, 0.1, 0} (CONSOLIDADO item 4). State the range of k where the equivalence is claimed, which is k ≥ 4 on present data.

### 2.3 Metrics the paper omits **[from summary.json]**

- **minADE₆.** This is the metric accelerations usually report. For A1.5 E10, temperature explains only 46% of the change (residual +0.11 m [−0.03, 0.33]). For R1 it explains 63%. The residual is not significant, but a reader comparing with FlashDrive will look for this row. Include it.
- **Decision entropy.** The E10 residual is negative in all three models (−0.020, −0.028, −0.022). Euler-10 lowers entropy slightly more than matched temperature does. This is the only consistent hint of an effect beyond temperature. **Pre-register its direction** (§5) rather than discovering it later.

### 2.4 Clip-level variance and power **[recomputed]**

I estimated each clip's residual as the sampler's metric minus the clip's own temperature curve, evaluated at the global τ_eff. The bootstrap intervals in the paper also include uncertainty in τ_eff. My per-model SDs reproduce them to within about 10%.

| Endpoint | Per-model SD | Cross-model correlation | Pooled-over-models SD | Clips for 90% TOST power: true residual 0 / ¼ margin / ½ margin (pooled) | Same, per model (worst model) |
|---|---|---|---|---|---|
| E10 flips (margin 2 pt) | 0.063–0.087 | −0.07 to 0.45 | 0.051 | 70 / 98 / 220 | 207 / 291 / 654 |
| E10 ADE₁ (margin 0.05 m) | 0.12–0.17 | −0.26 to 0.72 | 0.098 | 42 / 59 / 133 | 123 / 172 / 387 |
| E4 flips (margin 2 pt) | 0.13–0.15 | 0.0 to 0.59 | 0.098 | 260 / 365 / 820 | 626 / 881 / 1982 |
| E4 ADE₁ (margin 0.05 m) | 0.26–0.40 | −0.21 to 0.61 | 0.207 | 187 / 262 / 590 | 688 / 968 / 2178 |

How to read this:
- The figures assume 6 seeds, α = 0.05 on each side, a normal approximation, and no enrichment.
- "Pooled" means the residual is averaged over the three Alpamayo generations clip by clip. Those residuals are only weakly correlated, so pooling cuts the variance by a factor of about 2–2.5.
- The observed pooled E10 flip residual is +0.84 pt, about 0.4 of the margin.

---

## 3. Remaining issues with the draft

1. **Wording of the confirmation set.** Two sentences describe the same 436 clips as "unseen": the Related Work sentence and the TODO in Finding 2. Describe them as "held out from the temperature analysis; previously used for the schedule confirmation (A1.5, seeds 9201–9204)".
2. **"Fraction explained" from 79% to 121%.** A ratio above 100% with no interval confuses readers. Report the residual with its CI and the margin as a share of the effect. On the 60 clips, the 2-pt margin is 43% of A1.5's E10 flip effect (4.7 pt). A pass at that margin only shows that temperature explains at least 57% of the effect. Say so, or add a relative criterion (§5).
3. **The ADE₁ "gain" is partly mechanical.** Any contraction toward the conditional mean lowers single-sample error. That is the bias–variance trade-off. Readers will discount it. Lead with decisions and minADE₆, and treat ADE₁ as the expected consequence.
4. **Qwen-Drive.** The numbers are typed by hand, the sample is not the 60-clip sample, and best-of-6 has not been checked against their protocol.
   - As written, the boundary is an observation.
   - It becomes a contribution if a rule stated in advance predicts it. One candidate: when the conditional spread s → 0, the contraction factor acts on almost nothing, so integration error dominates, and the theory predicts τ_eff > 1.
   - State that rule *before* NAVSIM, and test it on ReCogDrive and DiffusionDrive.
5. **Proposition 3.1.** The condition h·Lip < 1 is the crux, because the Lipschitz constant of v grows like 1/(1−t) for low-variance targets. Either bound Lip for the last step or present the result as a heuristic remark. A flawed proposition costs more than having none.
6. **Production.** The draft is 6 pages with no figures and 13 TODOs. The teaser and the temperature-curve figure are where the Heun-vs-Euler contrast gets understood. Budget a week for them.
7. **Data provenance.** The protocol doc notes that nobody verified PhysicalAI-AV test clips are absent from Alpamayo's training data. This is irrelevant to the solver comparison, since every arm uses the same frozen model, but say so in one sentence.

---

## 4. Is this an ICML contribution? Positioning, and minimum results

### Positioning: an evaluation-methodology paper, with the science as evidence that the method is valid

Arguments for methodology framing:
- What survives every check so far is an **evaluation confound plus a cheap control**:
  - report the spread-matched τ_eff;
  - compare against an accurate solver at that τ;
  - use a non-mean-directed sampler (Heun) to show the test can fail.
- This deliverable holds whether the best mechanistic description is "temperature" or "mean-directed shrinkage" (§2.1).
- It also turns most of the open risks into results, not threats:
  - a FlashDrive gain that vanishes at matched τ is a headline;
  - a gain that survives is still a correctly measured result;
  - a NAVSIM model that behaves like Qwen-Drive becomes a tested boundary.

Arguments against science framing:
- An empirical-science framing ("Euler *is* temperature") would need per-sample or distributional evidence that singles out temperature among contractions.
- It would also need the k → 1 breakdown explained, and confirmation in more than one model family.
- Today the data support that framing only partially: two of three models favour temperature per sample, and Euler-1 fails.

**Recommended structure:**
- Title along the lines of "*Matched-temperature evaluation of accelerated flow-matching driving policies*" or "*Fewer Euler steps are a hidden temperature: …*".
- Contribution 1: the protocol and its validity checks (Heun control, shrinkage arm).
- Contribution 2: the confirmatory equivalence test.
- Contribution 3: the reinterpretation of published accelerations (FlashDrive, and Coda if feasible).
- Contribution 4: the closed-form contraction and the boundary regime.

ICML accepts evaluation-methodology papers when they **change the reading of a published result** and **generalize beyond one model family**. Those two items carry the 6→7 step.

### Minimum results for a **6/10** (weak accept)
1. **The 436-clip confirmation passes its primary endpoint.** That endpoint is TOST equivalence of the pooled E10 flip and ADE₁ residuals at the margins of §5. The assay-sensitivity check must also hold: Heun-3 is rejected, run on all three models.
2. **Official FlashDrive at matched temperature,** reported whichever way it goes. The decomposition separates weights, cache and temperature for minADE₁ and minADE₆, using the same residual and TOST machinery.
3. **Draft fixes** (CPU, about a week):
   - the shrinkage arm and the per-sample comparison (§2.1);
   - E1/E2 rows and an extended τ grid, or an explicit k ≥ 4 scope (§2.2);
   - minADE₆ and entropy rows;
   - Qwen-Drive numbers generated from code;
   - disclosure of the prior use of the 436 clips;
   - figures;
   - the proposition fixed or demoted.

### Minimum results for a **7/10** (accept)
All of the above, plus:

4. **A second driving family on NAVSIM** (DiffusionDrive and/or ReCogDrive).
   - Run the same protocol, and **report PDMS at matched τ**.
   - It counts if the model either lands on the curve, or lands where a rule stated before the run predicts (the boundary rule of §3, item 4).
   - This removes the "one family" objection and attaches the protocol to a standard driving metric.
5. **One of the following two:**
   - (a) **Closed loop agrees with open loop.** AlpaSim incident or progress metrics with E4/E10 within the CI of their matched Heun-32·τ arms, and the A₁ arm distinguishable where open loop predicts a difference. With 58 scenes × 1 rollout this will likely be inconclusive. Report a scene-clustered CI and treat it as supporting evidence. A powered v2 on enriched scenes, with ≥3 rollouts per scene, would make it count.
   - (b) **The pre-registered tail endpoint is positive** on the enriched stratum. For example, E4 removes stops relative to A_τeff, or the E10 entropy residual is confirmed. A positive tail result, in a driving-specific direction, would move the paper toward 7–8. A null is fine for a 6.

If FlashDrive's gain *survives* matched temperature, the paper still stands at about 6, provided the protocol is the headline and the result is reported honestly.

---

## 5. Design guidance for the confirmation study

### 5.1 What the clip set can and cannot claim
- **Prior exposure of the 436 clips:**
  - A1.5's uniform Euler-4/6/8 and Heun-80 plans (seeds 9201–9204) have already been examined, for flip rates against Heun-80.
  - The *matched-temperature residual* has not been computed on these clips. R1 and A2 have never run on them.
- **Requirements that follow:**
  - **Fresh analysis seeds** (e.g., 9401–9412).
  - **A new Heun-32 reference.**
  - **No selection or stratification that uses the earlier Euler outputs.**
  - Describe the study as a *frozen-design confirmation on clips held out from the temperature analysis*.
- **Overlap check.** Confirm in the protocol, by clip_id hash, that none of the 436 belong to chunks used by the 60-clip sample, the schedule calibration set (100–199) or the AlpaSim scenes.

### 5.2 Two strata: random for the primary, enriched for the tail
- **Stratum R (representative): all 436 clips, no subsampling.** The primary endpoints are estimated here, so the estimand is the population of PhysicalAI-AV test clips. Enriching this stratum would change the estimand and inflate per-clip variance (flips concentrate in ambiguous clips).
- **Stratum T (tail enrichment): about 200–300 *additional* clips** from test-split chunks not yet used.
  - If the clips must come from already-used chunks, take at most one clip per chunk, and cluster the bootstrap by chunk.
  - Select them **only on pre-outcome information**: data available before any policy runs, plus at most a reference-only screen on discarded seeds. Fix the criteria in a hashed document and a committed selection script before inference.
  - Inclusion rules (a clip qualifies through any one of them):
    1. **Stop initiation.** Logged ego speed at t₀ ≥ 1.0 m/s, and the logged human future reaches < 0.5 m/s within 6.4 s.
    2. **Stop release.** Speed at t₀ < 0.5 m/s, and the human future exceeds 2 m/s within 6.4 s.
       Rules 1 and 2 exclude stationary-and-staying clips, the trivial stops behind the current 6/360.
    3. **Strong deceleration.** The human future has a minimum acceleration ≤ −2 m/s² while speed at t₀ ≥ 3 m/s.
    4. **Lateral manoeuvre.** The human future's lateral offset is ≥ 2 m, or the heading change is ≥ 30°, classified by the paper's existing thresholds.
    5. **Optional: reference ambiguity.** The decision entropy of Heun-32 at τ=1 over **4 screening seeds** (disjoint from the analysis seeds, then discarded) is ≥ 0.5 nats, or at least two decision classes appear. This uses only the reference model, never a comparison outcome. Fresh analysis seeds remove the regression-to-the-mean bias of selecting on it.
  - Map or traffic-control metadata (intersection, stop sign, signal within 50 m) may be added **if it exists in `clip_index.parquet` or `records.json` before any run**.
  - Record which rule each clip qualified through, and report results per rule.
- **Combining strata.** For population statements use stratum R only. For the tail, use stratum T, reported as a conditional estimand ("among stop-transition clips"). Do not pool R and T without design weights.

### 5.3 Endpoints and multiplicity

**Family 1: equivalence (stratum R; α = 0.05, fixed-sequence gatekeeping, so no α split).**
- **P1 (primary).** For E10, the clip-level residual averaged over the three models, for both flips and ADE₁.
  - Margins: **2.0 pt and 0.05 m.** Test: TOST, 90% interval from a joint clip bootstrap that re-estimates τ_eff in each resample (≥ 5000 resamples).
  - Both metrics must pass (an intersection–union test, which needs no multiplicity correction).
  - Power at n = 436: above 0.99 for true residuals up to half the margin (§2.4).
- **P2 (tested only if P1 passes).** The same test for E4. Power at n = 436 is about 0.94 if the true residual is ¼ of the margin, and about 0.69 for flips if it is ½. State this in the protocol now.
- **Key secondary for the paper's claim, declared now.**
  - (i) The lower bound of the 90% interval of fraction explained is ≥ 0.7 for E10 flips. This guards against a pass at a margin that is 30–45% of the effect.
  - (ii) A per-model E10 TOST, Holm-corrected over the three models. The worst model has about 0.97 power at ¼ margin.
  - (iii) A per-model E4 TOST. It is **declared underpowered** (about 0.6), so a failure is not evidence of a difference.
- **Margins.** The 2-pt and 0.05-m margins were chosen after seeing the 60-clip data. Say so. Freezing them before the 436-clip run is what makes them legitimate.
- **Seeds.** Use 12 analysis seeds if compute allows. The SDs in §2.4 assume 6, so 12 is conservative. Most of the power comes from the number of clips, so with 8 seeds the primary endpoint is still above 0.95 powered.

**Assay sensitivity (stratum R; this condition must hold for P1/P2 to be interpretable, and it spends no α).**
- Heun-3 on all three models. Its pooled flip residual must have a 95% lower bound above +2 pt. On the 60 clips it is +16.5 pt [9.8, 23.1] for A1.5.
- This mirrors the role of an active-control arm in a non-inferiority trial. If Heun-3 also "passes", the test has no power to discriminate and P1 means nothing.

**Family 2: mechanism (stratum R; Holm over 3 tests).**
- **M1.** Per-sample distance: E10→A_τeff is smaller than E10→S_c (shrinkage arm, §2.1), pooled over models, one-sided. This is the endpoint that distinguishes "temperature" from "any contraction". It costs nothing extra, since S_c is computed from existing outputs.
- **M2.** The E10 entropy residual is below 0 (the direction seen on the 60 clips), pooled, one-sided.
- **M3.** Among flips where the A₁ decision differs from the E₁ decision, more Euler-k flips than matched-temperature flips land on the E₁ decision. In round 2 I measured 83% vs 75% for E4.

**Family 3: tail (stratum T; α = 0.05 for its own primary, declared as a separate family).**
- **T1.** Stop rate of E4 minus stop rate of A_τeff(E4) is below 0. Use a clip-level paired difference and a cluster bootstrap. The τ_eff is the one estimated on stratum R, frozen and not re-estimated on T, so the tail test is not contaminated by its own spread match.
  - Report the **number of distinct contributing clips** next to every rate.
  - Sizing: with a per-clip SD of the stop residual of 0.2–0.3 at 12 seeds, detecting a 5-pt effect with 80% power, one-sided, needs **≈ 100–220 stop-eligible clips**. Re-estimate this SD from the 4 screening seeds before unblinding, and write the final n into the protocol before any analysis seed is run.
- **Secondary (Holm):** the same test for E10, lateral flips on rule-4 clips, and minADE₆ residuals.

**Exploratory (no claims):** collisions in log replay, the clip-level Gaussian calibration, and Alpamayo's own Euler-10·τ arm.

### 5.4 Minimum configurations per clip and seed
- E1, E2, E4, E10.
- Heun-3 and Heun-5.
- Heun-32 at τ ∈ {1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0}.
- S_c, computed from existing outputs at no cost.
- E6 and Euler-10·τ only if budget allows. Heun-32 × 10 temperatures dominates the cost. If needed, drop τ = 0.9 and 0.1–0.2 before dropping clips.
- Keep the language seed fixed, as on the 60 clips. Language-seed variation is a separate ablation.

### 5.5 Pre-registration mechanics
Before running any analysis seed, commit and record a hash (and an OSF or GitHub timestamp) for:
- the clip lists for R and T;
- the selection script for T;
- the screening outputs;
- the analysis seeds;
- the margins;
- the endpoint definitions and the gatekeeping order;
- the analysis code (`paper_numbers.py` extended).

Report every endpoint, including failures, in an appendix table.

---

## 6. Score

- **Current draft: 5/10 (borderline reject); confidence 4/5.**
  - Positives: the scope is honest, the negative control is real and well integrated, and the statistics pipeline is auditable. The thesis is coherent.
  - Negatives: every significance-bearing result is pending. The new shrinkage control shows that the "temperature" label is stronger than the distribution-level test supports. Euler-1/2 are hidden.
- **Projected:**
  - **6/10** if the 436-clip P1 passes with assay sensitivity, FlashDrive is reported at matched τ, and the §4 draft fixes are made.
  - **7/10** if, in addition, a NAVSIM family is confirmed with PDMS at matched τ, and either closed loop agrees or the tail endpoint T1 is positive.
  - **7–8/10** if, in addition, M1 confirms that temperature is a better description than shrinkage sample by sample.

---

## 7. Recomputation log (all [recomputed], CPU)

1. **Shrinkage control** (§2.1).
   - Definition: S_c = E₁ + c·(A₁(z) − E₁), with E₁ averaged over seeds and c = Spr(E_k)/Spr(A₁).
   - Data: 60 clips × 6 seeds; decisions from `trajectory_decision_gate.decisions`; 2000 clip-bootstrap resamples.
   - Flip difference E_k − S_c across the six cells: from −1.7 to +1.4 pt, with every interval containing 0.
   - Per-sample E10 distance to A_τeff minus distance to S_c: A1.5 −0.038 [−0.079, −0.005]; A2 −0.059 [−0.108, −0.021]; R1 +0.034 [−0.017, 0.083].
2. **Euler-1 residuals** (`summary.json`): +11.1 / +10.8 / +15.3 pt, with τ_eff out of range in all three models. Euler-2 is out of range for R1 and A2.
3. **minADE₆, A1.5 E10:** fraction explained 0.46. **Entropy residual, E10:** −0.020 / −0.028 / −0.022.
4. **Per-clip residual SDs and power** (§2.4).
   - Method: `paper_numbers.load` and `matched`, with each clip's curve interpolated at the global τ_eff, 6 seeds, normal approximation to TOST.
   - The pooled E10 flip residual on the 60 clips is +0.84 pt, with SD 0.051.
5. **Stopping:** the E4 stop residual is nonzero in only 2 of 60 clips, so no power calculation from current data is possible. This is why stratum T needs a screening step.
