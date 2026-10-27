# ICML 2027 Review, Round 5.2 (focused): "When Fewer Steps Mean Lower Temperature" (v5.2)

**Reviewer:** balanced / area-chair style. Earlier rounds: 4, 4, 5, 5, 5.

**Material reviewed:**
- `versions/main_v5.2.tex`, compared with v5 by diff;
- `tables/numbers_tables.tex`, macros `\FD*`;
- `artifacts/flashdrive_analysis_nvidia_v1_20261026/summary.json` and its `source_snapshot.py`;
- `artifacts/flashdrive_probe_nvidia_v1/plans.npz`, for the clip ids;
- `docs/PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md`, addenda 11 and 11b;
- `src/cavla/paper_numbers.py`, for the main sample's definition.

## 1. Change versus v5

**Text fixes F1–F3 from my v5 review are done:**
- **F1:** Finding 5 now says that shrinkage has a constant contraction ratio by construction.
- **F2:** The paper now states "closer to temperature than to a uniform shrinkage, but not identical", and reports pooled differences.
- **F3:** The abstract no longer claims that the sample-level result resembles temperature, and "Heun fails both tests" is now "fails the rate test".
- **Pre-registered signature endpoint:** It now has a fixed statistic (pooled median difference at E10's spread) and a fixed margin (0.05). This is what I asked for.

**One item from v5 is still not done:** Reviewer 2's arms showing contractions that *fail* the test have not been added. Those are a mean of six samples, random collapse to E₁, and a constant-velocity line.

**New: the "A published acceleration" paragraph.** It reports FlashDrive's velocity cache, run through FlashDrive's own code on the original nvidia Alpamayo-1.5 checkpoint, at matched temperature. This is the first evidence in the paper on a third-party method.

## 2. FlashDrive paragraph: numbers against the artifact

| Macro | Paper | Artifact | Verdict |
|---|---|---|---|
| `\FDClips` | 60 | `clips: 60` | matches |
| `\FDGain` | 0.28 m | Heun32·τ=1 ADE₁ 2.688 − cache 2.405 = 0.283 | matches, but see W2 |
| `\FDTau` | 0.64 | 0.640, 90% CI [0.627, 0.651] | matches (the CI is not reported) |
| `\FDResid` | 0.008 [−0.034, 0.051] | 0.0077 [−0.0344, 0.0513] | matches |
| `\FDETenTau` / `\FDETenResid` | 0.84 / 0.012 [−0.016, 0.041] | 0.837 / 0.0117 [−0.0158, 0.0412] | matches |

The analysis script follows addendum 11b exactly:
- the clip is the unit;
- τ_eff is found by linear interpolation on the Heun32·τ spread curve;
- 1000 clip-bootstrap resamples, with τ_eff re-solved in each one.

Addendum 11b is timestamped before any metric was computed. The paper also reports honestly that the prediction τ_eff ≈ 0.5–0.6 failed (the CI excludes it) and that equivalence was not established. That is good practice.

**Factual and wording errors:**
- **E1. "On 60 *further* clips" is wrong.** The FlashDrive probe uses `context_100`–`context_159`. `paper_numbers.py` defines the main sample as the same `context_100-159`. These are the same 60 clips, with different windows (4 consecutive windows ending at t0, on a different code stack). This is not an independent sample. The paper must say "the same 60 clips". The E10 re-match (0.84 against 0.83) is therefore a reproducibility check across stacks, not a replication.
- **E2. "The upper bound sits at our 0.05 m margin" understates the miss.** The bound is 0.0513, so it slightly *exceeds* the margin. Say "just exceeds".
- **E3. The paragraph omits the τ_eff interval [0.63, 0.65].** That interval is what shows the pre-registered prediction failed rather than being merely imprecise. Report it.

## 3. Does this add the generality I asked for?

**Partly. It adds relevance, not generality.**

**What it does show:**
- The protocol can be applied to a published acceleration, run through that method's own code.
- On that acceleration, a temperature change accounts for the cache's ADE₁ gain within about 0.05 m.
- So the paper's claim that published accelerations inherit a temperature shift now has one supporting instance, not zero.

**What it does not show:**
- **The same model family, the same clips and the same reference solver.** The new data point is another sampler on Alpamayo-1.5. It is not a new policy, dataset or metric. My v5 generality concern (one family, 60 clips) is untouched.
- **The decomposition I asked for is not there.**
  - Addendum 11 promises to split FlashDrive's reported gain into (a) a checkpoint difference (W4A8 + DFlash + streaming) and (b) the cache's effect. The paper gives only (b).
  - The z-lab run is still a TODO, and its log is still loading the model.
  - Without (a), the paper cannot say whether FlashDrive's *published* gain is mostly temperature.
- **ADE₁ is the only metric.** The paper's main claim is about decision flips as well as ADE₁, but the FlashDrive analysis computes no flip residual and no minADE₆. The trade-off the introduction advertises (better ADE₁, worse minADE₆) is not shown for FlashDrive.

## 4. Strengths

1. **Pre-registered prediction, honestly reported.** A clean protocol with a falsifiable prediction, timestamped before unblinding, and an honest report of a partial miss. This is now consistent across the main results, the Shrink control and FlashDrive.
2. **A real test on a third-party method.** FlashDrive's own code and released weights are run, not a re-implementation.
3. **The rate-level claim is now stated precisely and backed by controls.** The sample-level signature is correctly framed as secondary, with a margin fixed in advance.
4. **Numbers are reproducible.** All FlashDrive numbers trace to the artifact without discrepancy. The analysis script is short and matches the protocol.

## 5. Weaknesses

**W1. A same-run result the paper does not report: the cache behaves like 4-step Euler.**
- In the same run, Euler-4 has τ_eff 0.626 [0.601, 0.653] and ADE₁ 2.406.
- The cache has τ_eff 0.640 [0.627, 0.651] and ADE₁ 2.405.
- The cache does 8 steps with steps 3–6 reused, which leaves 4 fresh velocity evaluations. It is statistically indistinguishable from plain Euler-4.
- This is the cleanest mechanistic sentence available, and it strengthens the thesis: the cache is a coarser first-order integrator, hence lower temperature. State it, with intervals.

**W2. The baseline for the gain is the wrong one.** "0.28 m relative to the reference at τ = 1" compares against Heun32, which no one ships. Against the samplers a practitioner would compare with:

| Baseline | Gain from the cache | Share explained by temperature, at the residual's upper bound |
|---|---|---|
| Heun32 at τ = 1 (as in the paper) | 0.28 m | ≈ 82% |
| Euler-8, same step count, cache off | 0.118 m | ≈ 57% |
| Euler-10 | 0.143 m | — |

- Against Euler-8, "almost all of the gain is a temperature effect" holds at the point estimate (≈ 93%) but not at the interval bound.
- Report the gain against Euler-8 and state the explained fraction with its interval.

**W3. Generality is unchanged.** It is still one model family and 60 clips (now reused).
- The confirmation is not run (TODO).
- There is no NAVSIM or second family, and AlpaSim is absent.
- Heun is run on A1.5 only, and Qwen-Drive is not unified.
- **The introduction overclaims.** It says ProbeFlow, AdaVLA and Coda "inherit this temperature shift", but none of them was tested. Coda in particular *corrects* the endpoint, so the prediction may fail for it. Restrict the claim to FlashDrive, or label the others as predictions.

**W4. Euler-4 fails the 0.05 m margin in this run** (upper bound 0.070). The paper's E₄ equivalence statements come from the main run. Add a footnote noting that, on this stack, E₄'s ADE₁ residual is not equivalent at the margin.

**W5. The manuscript is incomplete.**
- There are 12 TODO markers.
- The contribution list still carries a TODO asking to add FlashDrive. Now that a result exists, a contribution bullet should state it, with its scope: one checkpoint, ADE₁ only.
- Appendices B–E are still placeholders, and Proposition 3.1 is still not proved.

**W6. Reviewer 2's failing-contraction arms are still missing.** They are the best evidence that the rate test discriminates between contractions. I asked for them in v5.

## 6. Reassessment of the contribution

**The paper is now a tidy, honest diagnostic study:**
- a protocol;
- a rate-level equivalence on one production family, with a shrinkage control;
- one application to a published acceleration.

**FlashDrive moves the practical significance from hypothetical to demonstrated once.** That was my first requirement for a 6, but it is only half-met:
- the checkpoint decomposition is missing;
- flips and minADE are missing for FlashDrive;
- the result is on reused clips.

The other requirements for a 6 are still open: the confirmation, and a manuscript with no TODOs.

## 7. Score

**Score: 5/10 (borderline reject, upper end). Confidence: 4/5.**

This is a real step toward 6, not yet a 6. If the z-lab decomposition and the confirmation land as described, I expect to move to 6 without needing further evidence of generality.

## 8. Concrete path to acceptance

### For 6 (weak accept)

1. **Fix the FlashDrive paragraph:**
   - say "same 60 clips" (E1);
   - say "just exceeds" (E2);
   - report the τ_eff CI (E3);
   - add the Euler-4 equivalence (W1);
   - report the gain against Euler-8, with the explained fraction and its interval (W2).
2. **Add flip residuals and minADE₆ for the cache from the same `plans.npz`.** No new GPU time is needed. The rate claim should be tested on the same two metrics as in Table 2.
3. **Finish the z-lab run and report the decomposition:**
   - (a) optimized checkpoint against nvidia at the same τ;
   - (b) cache against the temperature curve.
   - Report it whichever way it comes out, and state how much of FlashDrive's *published* ADE₁ gain each part accounts for.
4. **Run the pre-registered 436-clip confirmation:**
   - primary endpoint: pooled E10 TOST on flips and ADE₁;
   - Heun-3 rejected on all three models;
   - signature endpoint at E10 with margin 0.05.
5. **Complete the manuscript:**
   - zero TODOs;
   - a FlashDrive contribution bullet with honest scope;
   - Heun-3/5 on R1 and A2;
   - Proposition 3.1 proved or demoted;
   - Qwen-Drive numbers unified;
   - appendices filled or cut;
   - Reviewer 2's failing-contraction arms added;
   - the introduction's claim about ProbeFlow, AdaVLA and Coda limited to tested methods.

### For 7 (accept)

All of the 6 requirements, plus:

6. **A second family (NAVSIM, PDMS at matched τ)** that lands on the curve, or lands where a rule stated before the run predicts.
7. **One of the following:**
   - (a) the decomposition shows that FlashDrive's published gain changes materially at matched τ;
   - (b) the signature endpoint is confirmed out of sample;
   - (c) AlpaSim agrees with open loop, presented as a sanity check with a scene-clustered interval.
