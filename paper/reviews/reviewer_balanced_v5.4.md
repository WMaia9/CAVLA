# ICML 2027 Review, Round 5.4 (focused): "When Fewer Steps Mean Lower Temperature" (v5.4)

**Reviewer:** balanced / area-chair style. Earlier rounds: 4, 4, 5, 5, 5, 5.

**Material reviewed:**
- `versions/main_v5.4.tex`, diffed against v5.2;
- `tables/numbers_tables.tex`, macros `\FD*`;
- `artifacts/flashdrive_analysis_nvidia_v2_20261026/summary.json` and its `source_snapshot.py`;
- `artifacts/flashdrive_probe_nvidia_v1/source_snapshot.py` and `manifest.json`;
- FlashDrive's own `euler_with_cache` (`.deps/flashdrive/flashdrive/diffusion.py`) and `scripts/infer.py`;
- `artifacts/flashdrive_probe_zlab_v1.log`, for the status of the decomposition run;
- `src/cavla/paper_numbers.py`, for the definitions of the sample, the seeds and the reference.

## 1. Change versus v5.2

The FlashDrive paragraph has been rewritten and promoted to Finding `find:flash`. It is now cited in the abstract and in contribution 2. I checked it against my v5.2 requests.

| v5.2 item | Status in v5.4 | Verdict |
|---|---|---|
| E1: "further clips" should be "same clips" | "using the same \FDClips{} clips and seeds" | **Done.** The seeds (9201–9206) and the clips (context_100–159) match `paper_numbers.py`. |
| E2: "sits at" the margin should be "just exceeds" | "just exceeds the 0.05 m margin, so equivalence is not established" | **Done.** |
| E3: report the τ_eff CI | `\FDTauCI` = [0.63, 0.65] | **Done.** It matches the artifact (0.627, 0.651). |
| W1: the cache is equivalent to Euler-4 | ADE₁ difference −0.001 [−0.033, 0.032]; E₄ τ_eff 0.63 | **Done.** This is now the mechanistic core of the finding. |
| W2: use Euler-8 as the baseline, and state the explained fraction | The gain is 0.12 m against E₈ | **Half done.** The baseline is corrected. The explained fraction and its interval are not stated (see N3). |
| W4: E₄ fails the margin on this stack | "neither is it for E₄ (upper bound 0.070)" | **Done.** |
| W5: a contribution bullet with honest scope | One clause was added to contribution 2 | **Done, but the scope is missing** (see N2). |
| W3: the introduction overclaims for ProbeFlow, AdaVLA and Coda | Line 79 is unchanged: "inherit this temperature shift" | **Not done.** |
| Flip residual and minADE₆ for the cache | Not computed. `source_snapshot.py` computes ADE₁ only. | **Not done.** No GPU time is needed. |
| z-lab decomposition | Still TODO. The log shows an earlier crash (shape mismatch in the position embedding), then a restart that was at context_106 of 100–159 at 22:47. | **In progress.** |
| 436-clip confirmation | Still TODO | **Not done.** |
| Reviewer 2's arms showing contractions that fail the test | Absent | **Not done.** |
| Zero TODOs | 12 TODOs remain, previously 12 | **Not done.** |

**The numbers check out.** Every `\FD*` macro traces to `summary.json`:
- 0.64 and [0.63, 0.65];
- 0.12, which is the paired −0.1177;
- −0.001 [−0.033, 0.032];
- 0.63 and 0.070;
- 0.008 [−0.034, 0.051] and 0.051;
- 0.84 and 0.012 [−0.016, 0.041].

**The implementation is now described correctly.** In v5.2 the paper said the cache was "run through FlashDrive's own code". It now says "re-implemented inside FlashDrive's own inference code". This is the accurate description:
- The probe's `integrate()` skips the field evaluation on steps {3, 4, 5, 6}, zero-indexed. It keeps the last fresh velocity, which comes from step 2.
- This matches `euler_with_cache` together with `CACHE_STEPS = [3, 4, 5, 6]` and `DIFFUSION_STEPS = 8`. That gives 4 fresh evaluations, at steps 0, 1, 2 and 7.
- The one difference is that FlashDrive builds its time grid in bf16, while the probe uses float `i*h`. The effect is probably negligible, but it should be noted in the appendix.

## 2. Strengths

1. **The finding is now mechanistic and correct.** "Cache ≡ E₄" (ADE₁ difference −0.001 ± 0.03, τ_eff 0.64 against 0.63) is a clean explanation:
   - a step cache with 4 fresh evaluations is a coarse first-order integrator;
   - so it inherits the contraction of 4 Euler steps.
   This is the most persuasive single sentence the paper has about a third-party method.
2. **The failures are reported honestly.** The finding states three of them:
   - the pre-registered τ_eff prediction (0.5–0.6) missed, and the CI excludes it;
   - equivalence is not established for the cache;
   - equivalence is not established for E₄ on this stack.
   This is consistent with the pre-registration discipline I have praised in earlier rounds.
3. **The baseline is now the fair one.** A practitioner would compare against plain E₈, not against Heun32 at τ = 1.
4. **The run reproduces across stacks.** On FlashDrive's code path, E₁₀ lands at τ_eff 0.84, against 0.83 in the main pipeline, and it passes the margin.
5. **The numbers are fully traceable.** There is not a single discrepancy between the macros and the artifact.

## 3. Weaknesses

**N1. The finding's title and abstract sentence are stronger than the evidence.**
- "A published acceleration *is* a lower temperature" is the headline. Yet the finding's own test does not establish equivalence: the upper bound is 0.051 against a margin of 0.05.
- The abstract says "the reference at that temperature is within 0.051 m of it". This presents the upper bound of a failed TOST as a closeness guarantee. It also does not tell the abstract reader that 0.05 m is the margin.
- Suggested title: "A published acceleration behaves like four Euler steps and is close to a lower temperature".
- Suggested abstract wording: "…its residual against the reference at that temperature is 0.008 m (90% interval up to 0.051 m, just above our 0.05 m margin)".
- The causal clause "It does so by contracting further" should read "consistent with contracting further". The data show co-occurrence on one model.

**N2. The scope is not stated where the claim is made.** Both the abstract and contribution 2 carry the claim. Neither says that it rests on:
- one checkpoint (the unoptimized nvidia A1.5, not the W4A8/DFlash stack FlashDrive ships);
- one metric (ADE₁);
- the same 60 clips as the main study.

Promoting a single, partially failed equivalence on reused clips to the abstract is premature unless these qualifiers travel with it.

**N3. The explained fraction is still not stated.** The paper gives the 0.12 m gain against E₈, but not how much of it temperature reproduces. From the artifact:
- the reference's ADE₁ difference between E₈'s τ_eff (0.80) and the cache's (0.64) is 2.507 − 2.397 = 0.110 m;
- that is about 93% of 0.118 at the point estimate;
- if the cache's residual sits at its upper bound (0.051), it is about 57%.

This is the number a practitioner wants. Report it with a bootstrap interval, paired on clips, re-solving both τ_eff in each resample.

**N4. Only ADE₁ is tested.** The paper's thesis is that matched temperature reproduces *both* ADE₁ and decision flips, and that minADE₆ gets worse. For FlashDrive, only ADE₁ is shown. The plans are already in `plans.npz`.
- Flip residuals are needed to call this a test of the paper's claim rather than a partial one.
- The minADE₆ comparison against E₈ is needed to show that the "accuracy gain" has the trade-off the introduction advertises.

This costs no GPU time. Of everything listed here, it gives the most evidence per hour of effort.

**N5. The introduction still overclaims (W3 from v5.2).** Line 79 says the accelerations of FlashDrive, ProbeFlow, AdaVLA and Coda "inherit this temperature shift". Only FlashDrive has been tested. Coda corrects the few-step endpoint, so the prediction may fail for it. Write "FlashDrive inherits it (Finding 6); we predict the same for step-adaptive methods".

**N6. The decomposition FlashDrive's published gain needs is still missing.** The z-lab run is in progress and was at 7 of 60 clips at last log. Until it lands, the paper cannot say what share of FlashDrive's *reported* improvement is temperature. The finding concerns the cache on the original checkpoint, which is not the published system. The TODO at line 200 says so, but the abstract does not.

**N7. The remaining items carry over unchanged:**
- generality: one family, 60 clips, no NAVSIM, AlpaSim not yet run;
- the 436-clip confirmation is a TODO;
- Heun on R1 and A2 is a TODO;
- the Qwen-Drive numbers are not unified;
- Proposition 3.1 is not proved;
- Appendices B–E are placeholders;
- 12 TODO markers remain;
- Reviewer 2's arms showing contractions that fail the test are missing.

## 4. Reassessment

**The FlashDrive result is now correctly stated and more interesting than in v5.2.** "The cache is E₄" is a real insight. It turns the paper's thesis into a concrete warning about a published method.

**The paper has improved its *wording*, not its *evidence*.** None of the evidence I asked for to reach 6 has arrived:
- the decomposition;
- flips and minADE₆ for the cache;
- the confirmation;
- a complete manuscript.

**The promotion to the abstract also brings a mild new overclaim (N1, N2).** A partially failed equivalence on reused clips, measured on ADE₁ only, is headlined as an identity.

## 5. Score

**Score: 5/10 (borderline reject, upper end, unchanged). Confidence: 4/5.**

The text corrections are complete and accurate, and they make the paper stronger. However, the bar I set for 6 was evidential, and none of it has arrived. The paper is one focused week from 6, and the cheapest items are trivial.

## 6. Path to acceptance

### For 6 (weak accept)

1. **No new compute is needed for these:**
   - (a) Temper the title and abstract sentence of `find:flash` and state the scope there: one checkpoint, ADE₁, the same 60 clips (N1, N2).
   - (b) Report the explained fraction against E₈ with a clip-bootstrap interval (N3).
   - (c) Add flip residuals and minADE₆ for the cache, E₄ and E₈ from the existing `plans.npz` (N4).
   - (d) Restrict the claim in line 79 to tested methods (N5).
   - (e) Note the bf16 time grid in the appendix.
2. **Finish the z-lab run and report the decomposition**, whichever way it comes out:
   - checkpoint against nvidia at matched τ;
   - cache against the temperature curve.
3. **Run the 436-clip confirmation.** It has a pre-registered primary endpoint and the signature endpoint.
4. **Complete the manuscript:**
   - zero TODOs;
   - Heun-3/5 on R1 and A2;
   - Proposition 3.1 proved or demoted;
   - Qwen-Drive numbers unified;
   - appendices filled or cut;
   - Reviewer 2's failing-contraction arms added.

### For 7 (accept)

All of the 6 requirements, plus:

5. **A second family (NAVSIM PDMS at matched τ).** It should land on the curve, or land where a rule stated before the run predicts.
6. **One of the following:**
   - (a) the decomposition shows that FlashDrive's published gain changes materially at matched τ, with flips included;
   - (b) the signature endpoint is confirmed out of sample;
   - (c) AlpaSim agrees with open loop, with a scene-clustered interval.
