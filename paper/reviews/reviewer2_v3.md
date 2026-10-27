# ICML 2027 Review, Round 3: Reviewer 2

**Paper:** "When Fewer Steps Mean Lower Temperature: Euler Sampling of Flow-Matching Driving Policies" (`paper/main.tex`, compiled `main.pdf`, identical to `versions/main_v3.tex`).
**Previous reviews:** `reviewer2.md` (round 1, 4/10) and `reviewer2_v1.md` (round 2, 4/10).
**Authors' response:** `CONSOLIDADO.md`, section "Rodada 2".

## What I checked (CPU only, no GPU jobs)

**Reproducibility of the numbers.**
- `src/cavla/paper_numbers.py` is byte-identical to the artifact's `source_snapshot.py`.
- I re-ran it on `physicalai_test_contexts_v1_20261022` and the three `temperature_probe_*` artifacts.
  - With `--boot 1000` (the count stated in the paper), `numbers.tex` is reproduced exactly. `summary.json` matches up to float32 jitter in the contraction medians (about 1e-7).
  - With the script's *default* `--boot 2000`, one macro changes (`\ATwoEFourTauLo`: 0.55 → 0.56). The invocation is not recorded in the artifact.
- `make_tables.py` regenerates `tab_matched.tex`, `tab_contraction.tex` and `numbers_tables.tex` byte-identically.
- I checked every macro and every typed number in the text against `summary.json` (§2).

**New analyses.** The scripts are in my scratchpad (`r3/heun_control.py`, `r3/frac_ci.py`, `r3/energy.py`):
- a displacement-matched version of the Heun negative control;
- a "trivial shrinkage toward the one-step mean" control;
- bootstrap CIs on the "fraction explained";
- a pooled, jointly resampled flip residual;
- per-clip energy distance and decision-histogram total variation between samplers;
- a recount of log-replay collisions on clips 100–159.

---

## 0. Summary of the revision

The revision does what I asked for in round 2. The thesis is now narrower, and the numbers are honest:

- The stopping "tail" headline is withdrawn and demoted to a counted observation.
- The per-noise claim is withdrawn, and the Discussion says Euler-4 and A_τeff disagree sample by sample.
- The Gaussian factor is downgraded to "matches the median".
- "Parameter-free" is gone, and the 99.95% mixture statistic is out of the main text.
- Every quoted number in the Results now comes from one artifact, one 60-clip sample, one Heun-32 reference, and a clip bootstrap drawn jointly across models.

This is the right direction. I credit it.

What is left is narrow:

> On 60 clips and one model family, the released 10-step Euler samplers have a spread-matched τ_eff ≈ 0.83. At that τ, an accurate solver reproduces the clip-averaged ADE₁ and the clip-averaged flip rate to within about 2 points (point estimates 79–121% "explained"). Heun-3/5 on one model do not.

The abstract and title present this as equivalence "at the level of the distributions". My new analyses show that this is true only for clip-averaged rates:

- For Euler-4, the per-clip conditional distributions differ from A_τeff about as much as Heun-5 differs from its own matched temperature (W3-A).
- A trivial post-hoc shrinkage toward the one-step mean passes the paper's test about as well as Euler does (W3-B).
- About half of the Heun negative control's margin is explained by Heun simply being a *less accurate* integrator than Euler at equal NFE here, which is itself an unexplained anomaly (W3-C).

The manuscript is also still incomplete: 13 TODO markers, a teaser placeholder, all four appendices empty, and no proof.

---

## 1. Status of my round-2 issues

| # | Round-2 issue | Status | Evidence / remaining problem |
|---|---|---|---|
| W-A | Stopping headline: CI pseudo-replicated across models; rests on two standstill clips | **Fixed** | Withdrawn from the abstract and contributions. Now a "Rare decisions" paragraph with counts (6/6/6 of 360; E4 5/4/4; τ=0.4 7/8/9), which match `summary.json["stops"]`. Small error: "4 clips in which the vehicle is nearly stationary" includes `context_124` at **2.64 m/s**, which is not nearly stationary (113: 0.0, 127: 0.23, 141: 0.77 m/s). |
| W-B | TOST verdicts never reported; fail in most cells | **Partly fixed** | The text now says flip intervals "are not narrow enough to establish equivalence at a 2-point margin in most cells", which is accurate: flip equivalence holds in 1/6 E4/E10 cells (R1 E10). The ADE₁ verdicts (margin 0.05 m) are not mentioned at all: 1/6 cells (R1 E10). The margins and the TOST rule have disappeared from §4, so readers cannot see the criterion. Report the verdicts as a column. |
| W-C | Per noise, E4 is not its matched temperature | **Acknowledged, not quantified** | The Discussion says "a sizable share of individual decisions" without a number. Give the 8–9% vs 2.5–4% adjacent-τ baseline. My new per-clip *distribution* analysis (W3-A) shows that the gap is not only per noise. |
| W-D | Gaussian prediction agrees only in the median; per clip anti-correlated | **Fixed in wording; new inconsistency** | Now "matches the median but not the ranking". But the Gaussian column of Table 1 comes from `gaussian_contraction_v1_20261026`: **action space, 200 clips × 2 seeds, Heun-160 reference**. It sits next to **trajectory-space** ratios on 60 clips × 6 seeds with Heun-32. In its own space the measurement was 0.637/0.840 vs prediction 0.672/0.856. The agreement now looks tighter (0.67 vs 0.66–0.68) only because two different quantities are compared. The "why the same τ_eff" sentence was correctly softened, but s_d is still measured only for Alpamayo-1.5. |
| W-E | 99.95% mixture statistic does not discriminate | **Fixed** | Removed from the main text; the appendix TODO says it does not single out first-order samplers. |
| W-F | Intro numbers against the wrong baseline | **Partly fixed** | Now "compared with an accurate solver", with no number for the error. But the sentence still reads "four Euler steps instead of the released ten ... changes about one decision in ten". The 10.6% is E4 vs Heun-32 at τ=1, not vs E10. Say "relative to an accurate solver" in the first clause, or quote E4-vs-E10. |
| W-G | Incomplete manuscript | **Open** | 13 TODO boxes: teaser figure, contribution list, Prop. 3.1 proof, Finding 2 confirmation, Qwen unification, FlashDrive finding, Limitations, and all four appendices. The thresholds are "fixed in advance (App. D)", but App. D is a TODO. Two bibliography entries still carry `TODO` notes (Alpamayo report, Qwen report). |
| W-H(a) | E₁ = E[x₁\|c] "for every z" is exact only at the population minimizer | **Open** | Still stated as an identity ("the one-step sampler has no diversity at all"). The learned E₁ has a nonzero seed spread, and R1's E₁ stops in 1/6 seeds of `context_113`. |
| W-H(b) | Euler-1/2 outside the τ grid | **Moot** | Not reported in the paper; still clamped at 0.4 in the artifact. |
| W-H(c) | 60-clip subset not "sampled before any analysis" | **Open** | Text unchanged. `context_100..159` was chosen after the A1.5 100-clip results were known. |
| W-H(d) | minADE₆ claim unsupported | **Over-corrected** | The claim was removed, and so was *every* minADE₆ and entropy number, although the artifact computes both. See W3-D. |
| W-H(e) | Heun-5 contraction not in an artifact | **Fixed** | Now in `summary.json`. |

**Round-1 items still open:**
- W5: thresholds and frozen language branch; no threshold sensitivity analysis; language ablation is a TODO.
- W6: Prop. 3.1 is still a sketch with a TODO on the Lipschitz condition near t=1. That condition is now directly relevant: see W3-C.
- W7: one model family. Heun was run on **one** model.
- W8: Qwen numbers are still typed by hand from a different sample; the TODO admits it.
- W9: the contamination statement is still absent, although `data manifest` says training exposure "is not independently verified".
- W10: the noise-scaling centre is not the conditional mean. Moot for the main claim now that stopping is withdrawn.

---

## 2. Numbers in the draft vs `summary.json`

| # | Location | Draft | Artifact / recomputation | Status |
|---|---|---|---|---|
| 1 | Abstract | τ_eff(E10) 0.83–0.84 | 0.833 / 0.837 / 0.833 | OK |
| 2 | Contribution 2 and Finding 2 | temperature accounts for **"90–100% of the decision flips"** of E10 | A1.5 **90%**, R1 **100%**, A2 **79%** | **Mismatch.** The range omits Alpamayo-2. It should be 79–100%. The macro range `\AOneFiveETenExplFlip--\ROneETenExplFlip` hard-codes which model is min and max. |
| 3 | Contribution 2 and Finding 2 | ADE₁ explained 92–121% (E10); 86–114% (E4); flips 86–119% (E4) | 121/92/115; 97/86/114; 86/119/90 | OK |
| 4 | Finding 2 | "the flip residuals are within ±2 points" | R1 E4 **−2.04**, A2 E10 **+2.04** pp | Only after rounding. |
| 5 | Table 2, A2 E10 | flip residual 2.0 [0.0, 4.4] | [**+0.03**, 4.45] pp | **The 95% CI excludes zero**, but the table's rounding shows "0.0". This is the newest model's *released* sampler. It is the one cell where temperature significantly under-explains the flips. It should be stated, with the caveat that 6 cells were tested. |
| 6 | Finding 1 | 92–93% (E4) and 90–93% (E10) contracted; medians 0.66–0.68 and 0.84–0.86 | 92.5/93.3/91.7%; 90.3/93.3/89.7%; 0.678/0.655/0.667; 0.858/0.841/0.851 | OK |
| 7 | Finding 1 and Table 1 | Gaussian 0.67 / 0.86 | `gaussian_contraction_v1`: 0.672 / 0.856, in **action space**, 200 clips, Heun-160 | Number OK; comparison invalid (W-D above) |
| 8 | Finding 1 and Table 1 | "Heun samples do not [contract]"; Table 1 lists Heun-5 only | Heun-5 0.958 (69%); **Heun-3 0.898 (70%)** | Heun-3 is omitted from Table 1. Its median ratio of 0.90 is close to E10's 0.86, which does not fit "barely contract". |
| 9 | Finding 3 | Heun-3/5: τ_eff 0.95/0.98; flips 17.8/9.7%; matched 1.3/0.6%; residual +16.5/+9.1 | 0.952 [0.85, **1.00**] / 0.977 [0.91, **1.00**]; 17.8/9.7; 1.33/0.65; 16.45/9.07 | OK. Both τ_eff CIs hit the grid edge τ=1, so the matched value there is extrapolation-adjacent. |
| 10 | Abstract | Heun "with the same budget" | Heun-5 = 10 NFE (= E10); **Heun-3 = 6 NFE** (= E6, which is not in Table 2) | Half true |
| 11 | Rare decisions | 6/6/6 of 360 reference stops; from 4 clips "nearly stationary" | Counts OK; union of clips {113, 124, 127, 141}; 124 at 2.64 m/s | Wording |
| 12 | Rare decisions | collision "differences between samplers come from 5 clips" | Clips 100–159, Euler + Heun-32 curve: collisions in **6** clips (107, 118, 127, 131, 138, 146), all with between-sampler differences. Adding Heun-3/5, which are in Table 2: **8** clips (adds 109 and 116, where *only* Heun collides). No artifact. | **Mismatch / untraceable** (third round in a row) |
| 13 | §4 | "1000 bootstrap resamples" | Artifact reproduces with `--boot 1000`; script default is 2000 | Record the invocation in the artifact |
| 14 | Finding 4 (Qwen) | 0.04 m, 0.16 m, 1.42 vs 1.56 m | Unchanged from round 2 (clip range 100–200, `temperature_report_qwendrive_*`) | Typed by hand; different sample |
| 15 | Finding 2 TODO; Related Work | "pre-registered confirmation on **436 unseen clips**" (future) **and** "passed a pre-registered confirmation on 436 unseen clips" (past, schedules) | Same 436 clips. The authors' own audit (`docs/AUDITORIA_ICML_20261026.md`, l. 64) says they "must not be described as never-examined data". | **Contradiction.** The planned confirmation cannot use "unseen" clips that the schedule study already used. |
| 16 | Intro | "averaged metrics such as minADE register it as an accuracy gain" | minADE₆ gets **worse** under Euler in every cell: A1.5 E4 1.25 → 1.63 m, E10 → 1.45 m; R1 E4 1.05 → 1.23 m; A2 E10 1.33 → 1.53 m | **Wrong for minADE_k with k > 1.** Only ADE₁ improves. |

Bootstrap CIs on the "fraction explained" (my `frac_ci.py`, same seed and resamples as the artifact):

| Cell | ADE₁ explained | Flips explained |
|---|---|---|
| A1.5 E4 | 0.97 [0.65, 1.23] | 0.86 [0.58, 1.15] |
| A1.5 E10 | 1.21 [0.85, 2.67] | 0.90 [0.58, 1.38] |
| R1 E4 | 0.86 [0.59, 1.15] | 1.19 [0.91, 1.67] |
| R1 E10 | 0.92 [0.71, 1.17] | 1.00 [0.74, 1.42] |
| A2 E4 | 1.14 [0.95, 1.32] | 0.90 [0.72, 1.13] |
| A2 E10 | 1.15 [0.94, 1.40] | 0.79 [0.62, **1.00**] |

All lower bounds are ≥ 0.58, so "most" is defensible. The intervals are wide (often 0.6–1.4), and that width should be in the table rather than point percentages quoted to the unit.

Pooled flip residual across the three models (joint resampling):
- E10: **+0.84 pp [−0.38, +2.24]**.
- E4: +0.30 pp [−1.81, +2.64].

---

## 3. Is the Heun negative control sound?

**Setup: same reference, partly the same budget.**
- Heun-3/5 are evaluated against the same Heun-32 reference, on the same 60 clips × 6 noises, with the same decision gate. Good.
- The budget matches only for Heun-5 (10 NFE vs E10). Heun-3 has 6 NFE, which matches E6, not E4 or E10.
- Heun was run only on Alpamayo-1.5.
- Everything runs under `torch.autocast(bfloat16)` (`temperature_probe` source, l. 39), including the reference.

**(a) The control is partly built in.** Flips are defined against A_ref under the same noise, so the temperature curve has exactly 0 flips at τ=1 by construction. Heun's τ_eff is 0.95–0.98 (CIs reach 1.0), so its "matched" flip rate is forced to near 0 (1.3%, 0.6%). Any sampler that leaves the spread alone but has *any* integration error therefore fails this test. The control shows that the test rejects error that is not a contraction. Table 1 already shows that, so it adds little beyond the contraction table.

**(b) Heun's flips come largely from its own discretization error, and Heun is a worse integrator than Euler here.** Per noise, mean xy distance to A_ref(z) (`heun_control.py`):

| Sampler | NFE | Distance to A_ref (m) | Spread (m) |
|---|---|---|---|
| E10 | 10 | **0.348** | 1.61 |
| **Heun-5** | 10 | **0.403** | 1.96 |
| E6 | 6 | 0.500 | 1.40 |
| **Heun-3** | 6 | **0.777** | 1.90 |

At equal NFE, the "second-order" sampler is *further* from the accurate solution than first-order Euler, even though Euler deliberately contracts. That is not what second-order convergence looks like.

Plausible causes:
- the field is not smooth near t=1. This is exactly the Lipschitz caveat the authors leave as a TODO in Prop. 3.1, and Heun's corrector evaluates v at t=1;
- bf16 arithmetic.

Either cause also bears on whether **Heun-32 in bf16** is a converged reference. The Heun-32 vs Heun-160 convergence check I requested in round 2 is still absent.

**(c) A fairer control matches on displacement, not spread.** Interpolate the temperature curve's flip rate at the *same mean displacement from A_ref* instead of the same spread:

| Sampler | Flips | Curve flips at same spread | Curve flips at same displacement | Displacement-matched residual [95% CI] |
|---|---|---|---|---|
| Heun-3 | 17.8% | 1.3% | 11.0% | **+6.8 pp [+2.1, +11.7]** |
| Heun-5 | 9.7% | 0.6% | 5.2% | **+4.5 pp [+0.6, +9.4]** |
| E4 (A1.5) | 10.6% | 9.1% | 9.6% | +1.0 [−1.3, +4.0] |
| E10 (A1.5) | 4.7% | 4.3% | 4.6% | +0.2 [−1.1, +1.6] |

The control survives in this sharper form: Heun's error changes more decisions per metre of displacement than temperature does. But the margin is about half of what the paper reports (16.5 → 6.8; 9.1 → 4.5 pp), and Heun-5's lower bound is +0.6 pp. The paper's framing attributes all of the margin to "not being temperature". About half of it is "being inaccurate".

**(d) The test cannot distinguish temperature from trivial shrinkage.** I built a post-hoc sampler S_c(z) = E₁(z) + c·(A_ref(z) − E₁(z)) in trajectory space: plain linear shrinkage toward the one-step mean, no ODE. At c chosen to match the E4 / E10 spread, its flip residual against the temperature curve is:

| Model | Shrink at E4 spread | Euler E4 | Shrink at E10 spread | Euler E10 |
|---|---|---|---|---|
| A1.5 | about +1.0 pp | +1.5 | about +0.7 | +0.5 |
| R1 | about +1.5 | −2.0 | about +1.6 | 0.0 |
| A2 | about +3.3 | +1.4 | about +0.6 | +2.0 |

With 60 clips, a deterministic shrinkage toward E₁ lands on the "temperature curve" about as well as Euler does. So the evidence supports "Euler-k ≈ contraction toward the conditional mean, and any such contraction has rate effects similar to temperature". It does not support the stronger "Euler-k is equivalent to temperature". That is consistent with the authors' mechanism, but the thesis should be stated at that strength.

**Verdict on (3).** Same reference: yes. Same budget: only for Heun-5. Informative: yes, but less than claimed. It shows the test rejects samplers whose error is not contraction. It does not show that the test can reject a sampler that contracts in a non-temperature way: shrinkage passes, and per-clip distributions fail for Euler as well (W3-A). The sentence "It is specific to first-order integration" generalizes from two Heun configurations on one model, with an anomalously inaccurate Heun. Required:
- report the displacement-matched residual;
- add RK4 / midpoint, and Heun on R1 and A2;
- show reference convergence: Heun-32 vs Heun-64/160, and fp32 vs bf16, on a subset.

---

## 4. Other new weaknesses

### W3-A. "Distribution-level" equivalence holds only for clip-averaged rates. Per clip, E4's conditional distribution is as far from A_τeff as Heun-5's is from its own matched temperature.

The object of a stochastic policy is p(x₁ | c), a per-clip distribution. I computed, per clip (6 samples each), the trajectory energy distance and the total variation between decision histograms. Mean over 60 clips (`energy.py`):

| Model | E10 vs A0.8 / A0.9 | A0.8 vs A0.9 (Δτ = 0.1) | **E4 vs A0.6 / A0.7** | A0.6 vs A0.7 | A0.5 vs A0.7 (Δτ = 0.2) |
|---|---|---|---|---|---|
| A1.5 energy | 0.092 / 0.081 | 0.073 | **0.190 / 0.155** | 0.066 | 0.166 |
| A1.5 decision TV | 2.5 / 2.5% | 2.2% | **7.2 / 6.9%** | 3.9% | 6.1% |
| R1 energy | 0.110 / 0.077 | 0.080 | **0.247 / 0.191** | 0.080 | 0.210 |
| R1 decision TV | 4.4 / 4.4% | 2.8% | **8.3 / 7.2%** | 3.9% | 8.9% |
| A2 energy | 0.080 / 0.072 | 0.081 | **0.193 / 0.165** | 0.072 | 0.164 |
| A2 decision TV | 5.0 / 5.6% | 3.3% | **9.4 / 8.1%** | 2.5% | 5.3% |

For comparison, the A1.5 negative control: **Heun-5 vs A1.0 energy 0.173, TV 9.2%**.

- **E10** is about as close to its bracketing temperatures as adjacent temperatures are to each other. Its decision TV is somewhat larger (A2: 5.0–5.6% vs 3.3%).
- **E4's per-clip distribution** is 2–3× further from both bracketing temperatures than they are from each other. That is comparable to a 0.2 mismatch in τ, and indistinguishable from Heun-5's distance to the reference.

The flip *rate* matches because E4 adds flips in some clips and removes them in others. The per-clip distribution does not match.

The abstract ("at the level of the distributions they produce, few-step Euler samplers behave like an accurate solver run at a lower sampling temperature") should say "in clip-averaged rates". The Euler-4 claim should be limited accordingly.

### W3-B. The metrics the thesis is most sensitive to were dropped.

The artifact computes minADE₆ and decision entropy for every cell. Neither appears in the paper, yet the intro asserts that "averaged metrics such as minADE register [the temperature shift] as an accuracy gain".

- minADE₆ gets *worse* under Euler in every cell (#16 above).
- It is also where temperature explains least: A1.5 E10 minADE₆ residual +0.106 m [−0.03, +0.33], fraction explained **46%**; A2 E4 +0.075 [−0.004, +0.167].

A paper claiming distribution-level equivalence should report its coverage metric. Report minADE₆ and entropy residuals in Table 2, even if their CIs include zero.

### W3-C. Integrator anomaly and reference validity.

See §3(b). If Heun-5 (10 NFE) is further from Heun-32 than Euler-10 is, two questions follow:
- Is Heun-32 itself converged?
- Is the field near t=1 regular enough for the Gaussian/linear-field reasoning of §3?

Neither is addressed. A Heun-32 vs Heun-64 (fp32) check on 10 clips would settle it on a small GPU budget.

### W3-D. The internal confirmation plan reuses data called "unseen".

See #15. The schedule study already consumed the 436 clips. The TODO in Finding 2 and the Related Work sentence must be reconciled. Either draw a new confirmation sample, or describe the 436 clips as previously used, with a frozen analysis plan.

### W3-E. Smaller issues

- **Rounding in Table 2 hides the A2 E10 CI.** The lower bound +0.03 pp prints as "0.0". Use two decimals, or mark CIs that exclude zero.
- **Hard-coded macro ranges.** `X--Y` ranges assume which model is min and max. This produced the wrong "90–100%". Generate min/max programmatically.
- **The bootstrap holds the six seeds fixed.** Still not stated.
- **Notation conflict.** "Heun-k uses 2k field evaluations" while "Heun with 32 steps (64 evaluations)" is the reference: fine. But Table 1's "Heun does not contract" uses the Heun-5 row only (see #8).
- **Prop. 3.1 is disconnected from the experiments.** It concerns reflection-symmetric decisions about a hyperplane, but the gate's decisions are thresholds (1.5 m extent, ±20°, ±1.75 m) that are not hyperplanes through a symmetry axis. No experiment tests it. Either connect it (e.g., left/right flips in clips with near-symmetric lateral distributions) or move it to the appendix.

---

## 5. Strongest remaining reason to reject

The paper claims distribution-level equivalence between few-step Euler and explicit temperature. The evidence is two clip-averaged scalars on 60 clips from one model family. Spread, the third scalar, is matched by construction.

- By the paper's own margins, equivalence is established in 1 of 6 cells.
- The test cannot distinguish temperature from a trivial post-hoc shrinkage toward the mean.
- Per clip, Euler-4's conditional distribution is as far from its matched temperature as the negative control's (W3-A).
- The negative control is weaker than presented: half of its margin is integration inaccuracy, on one model.

In addition, the submission is not complete: 13 TODOs, empty appendices including the one that defines the decision thresholds, no proof, and the FlashDrive/closed-loop contributions still placeholders.

What remains is a well-executed, reproducible, but small measurement note: "E10 in Alpamayo ≈ τ 0.83 in rates". That does not meet the bar for a main-track ICML paper.

---

## 6. Score and confidence

**Score: 5 / 10 (weak reject).** Up from 4.

**What earned the +1:**
- The authors did what I asked for: the tail claim is withdrawn, the TOST outcome is acknowledged, the per-noise failure is admitted, and the Gaussian claim is downgraded.
- They built a reproducible pipeline: I regenerated every macro and table from the artifact.

**What prevents a higher score:**
- The manuscript is incomplete. I said in round 2 that completeness is a prerequisite for anything above 5.
- The "distribution-level" claim does not survive a per-clip distribution comparison for E4.
- The negative control is weaker than presented.
- One number range is wrong (flip "90–100%" should be 79–100%), and the A2 E10 cell significantly fails, which the paper does not mention.

**Confidence: 4 / 5.** All numbers were recomputed from the artifacts and code. The new analyses use only saved plans.

**What would move the score.**
- **+0.5:** Fix the number errors (#2, #4, #5, #12, #15, #16) and the Gaussian cross-space comparison. Add minADE₆, entropy and the TOST columns. Rescope the abstract to "clip-averaged rates", and add per-clip energy/TV as an honest limitation, or as a finding about E10 vs E4.
- **+1:** Strengthen the control: the displacement-matched residual; Heun and RK4 on all three models; a shrinkage-toward-mean comparator; and a reference convergence check (Heun-32 vs Heun-64, fp32), which would also explain the Heun anomaly.
- **+1:** A truly fresh, pre-registered confirmation sample (not the 436 schedule clips) with ≥ 200 clips, powered for the declared ADE and flip margins.
- **+1 to +2:** A second, non-Alpamayo stochastic flow policy (ReCogDrive or DiffusionDrive, as planned), plus the official FlashDrive at matched τ_eff, or a powered closed-loop result. Either would turn the measurement note into a paper with a consequence.
- **Prerequisite for > 5:** a complete manuscript: no TODOs, appendices with thresholds and seeds, a proof or a removal of Prop. 3.1, and a fixed bibliography.
