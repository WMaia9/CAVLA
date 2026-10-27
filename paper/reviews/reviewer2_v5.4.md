# ICML 2027 Review, Round 5.4: Reviewer 2 (focused round)

**Paper:** `versions/main_v5.4.tex` / `main_v5.4.pdf` (26 Oct 22:47), diffed against `main_v5.2.tex`. Macros: `tables/numbers_tables.tex`.
**Artifact:** `artifacts/flashdrive_analysis_nvidia_v2_20261026/summary.json`. Its `source_snapshot.py` is byte-identical to `src/cavla/flashdrive_analysis.py`.
**Protocol:** `docs/PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md`, Addenda 11 and 11b, including the 20:53 timestamp correction.
**Scope of this round:** the new `find:flash`, the new abstract sentence, and the clause added to Contribution 2. Nothing else in the paper changed relative to v5.2.

## Numbers check

Every FlashDrive macro matches `summary.json`. The rendered PDF shows the same values.

| Macro | Paper | Artifact |
|---|---|---|
| `FDGainEEight` | 0.12 | cache − E₈ = −0.1177, 90% CI [−0.143, −0.091] |
| `FDTau`, `FDTauCI` | 0.64 [0.63, 0.65] | 0.6402 [0.6270, 0.6508] |
| `FDVsEFour` | −0.001 [−0.033, 0.032] | −0.0007 [−0.0328, 0.0324] (90%) |
| `FDEFourTau` | 0.63 | 0.6257 |
| `FDResid` / `FDResidHi` | 0.008 [−0.034, 0.051] / 0.051 | 0.0077 [−0.0344, 0.0513] |
| `FDEFourResidHi` | 0.070 | 0.0697 |
| `FDETenTau`, `FDETenResid` | 0.84, 0.012 [−0.016, 0.041] | 0.8369, 0.0117 [−0.0158, 0.0412] |

No transcription errors. `\FDGain` (0.28) is still defined in the macros but is no longer used, which is correct.

## Change since v5.2

The authors addressed most of my FlashDrive points directly and honestly.

| v5.2 issue | Status in v5.4 |
|---|---|
| W1: "60 further clips" | **Fixed.** The text now says "the same 60 clips and seeds". "Again" is gone. The protocol correction also states that E₁₀ at 0.84 is a re-run, not a replication. |
| W2: 0.28 m gain conflated E₈ with the cache | **Fixed.** The gain is now 0.12 m relative to plain E₈. |
| W3: the cache ≈ E₄ comparison was missing | **Adopted** and made central: "indistinguishable from E₄", with τ_eff 0.63 vs 0.64. |
| W4: "run through FlashDrive's own code" | **Fixed.** The text now says "re-implemented this cache inside FlashDrive's own inference code". |
| Pre-registered verdicts | **Mostly fixed.** The cache fails, E₄ fails and E₁₀ passes. **E₈ is still omitted**, although it passes (upper bound 0.048). |
| W5: per-clip heterogeneity | Partly covered: the abstract and the Discussion say the match holds "for rates over clips". The finding itself still gives no per-clip spread (SD of the residual 0.21 m). |
| W6: pre-registration timing | **Acknowledged** in the protocol. The correction says 11b was saved in the same minute as the analysis, not before it. This is honest. However, the 11b heading still reads "declaradas antes de abrir os dados" ("declared before opening the data"), which the correction contradicts. The paper does not call the FlashDrive analysis pre-registered, which is correct. |
| Linear-interpolation sensitivity, interval on the fraction explained | **Not done.** |

## Is the new text supported?

**Supported:**
- The cache sits at a lower effective temperature than E₈ and E₁₀ (0.64 vs 0.80 and 0.84, with tight intervals).
- Its ADE₁ gain over E₈ (0.118 m, 90% CI [0.091, 0.143]) is accompanied by a matching drop in spread.
- In ADE₁ and τ_eff it is not distinguishable from E₄. Mechanistically this makes sense: the cache makes 4 fresh field evaluations.
- The finding's body reports the failed equivalence plainly ("equivalence is not established; neither is it for E₄") and admits that the prediction of 0.5–0.6 missed.

**Overstated:**

1. **The abstract spins a failed equivalence test as a success.** "The reference at that temperature is within 0.051 m of it" quotes the upper bound of a 90% interval, which is exactly 0.05 m (the pre-specified margin) plus 0.001 m. A reader who does not know the margin will read this as a pass. The finding itself says equivalence is not established, so the abstract and the body disagree in tone. Suggested replacement: "...its single-sample error is within 0.01 m of the reference at that temperature (90% CI upper bound 0.051 m, just outside our 0.05 m equivalence margin)". Alternatively, drop the residual from the abstract and keep "behaves like four Euler steps", which is the robust part.

2. **The finding's title, "A published acceleration *is* a lower temperature", claims more than the test shows.**
   - The protocol's own verdict is a failure, and the same is true for E₄.
   - With PCHIP or spline interpolation, which I computed in v5.2, the upper bound moves further out, to 0.054–0.056.
   - "Behaves like a lower temperature in single-sample error" is what the data support.
   - Contribution 2 ("acts as a lower temperature") is acceptable as worded.

3. **Only ADE₁ is tested; decision flips are not.** The paper's central claim, repeated in the abstract, is that temperature reproduces *both* ADE₁ and the decision-flip rate. The FlashDrive analysis reports neither a flip rate nor a flip residual, yet the abstract places the FlashDrive sentence right next to that two-metric claim. Either compute the flip residual on the existing plans (`plans.npz` has everything needed) or say "single-sample error" explicitly, as the abstract already half does.

4. **"Indistinguishable from E₄"** rests on a 90% interval of ±0.033 m, with the level not stated in the text. That is an absence of evidence for a difference, not a demonstrated equivalence. Write "not distinguishable (90% CI ...)", and add the paired spread difference, which I computed in v5.2 as +0.031 [−0.04, 0.11].

5. **Minor issues:**
   - The 0.12 m gain is given without its interval.
   - Neither the interval level of `FDVsEFour` nor the bootstrap is stated in the finding.
   - Sec. 4 defines the τ grid as {1, …, 0.4}, but the FlashDrive curve stops at 0.5.
   - The interpolation is not said to be linear.
   - E₈'s pass is omitted.
   - The "fraction explained" interval ([82%, 112%]) is still absent.

**Overall:** the finding's body is now accurate and fair. The abstract sentence is the one place where the FlashDrive result is oversold. It is also the sentence most readers will see, and the paper's thesis is precisely that headline accuracy claims hide what is going on.

## Remaining weaknesses (unchanged from v5.2)

- **W7. Incomplete manuscript.**
  - There are 13 `\TODO`s.
  - Appendices B–E are empty.
  - Prop. 3.1 has no proof.
  - The Qwen-Drive numbers are hand-typed from a different sample.
  - The FlashDrive optimized checkpoint (W4A8, DFlash, streaming) is still "running". That checkpoint is what FlashDrive actually ships, so the paper tests the cache component, not the published system.
- **No independent replication.** The FlashDrive run reuses the 60 clips on which the 0.05 m margin was chosen. The A1.5 holdout (clips 160–199) is still unused.
- **The Heun control** is on one model, with τ_eff near 1.
- **No rate-level negative controls**, and no TOST verdict column in Table 1.
- **Pre-registration cannot be audited.** The directory is not under version control and there are no hashes.

## Score

**5 / 10 (weak reject), unchanged. Confidence 4/5.**

The FlashDrive section improved substantially. Each factual error I raised in v5.2 was corrected, and cache ≈ E₄ is a genuinely nice, mechanistically clean result. The protocol correction about the timestamp is to the authors' credit.

The score does not move for two reasons:
- The manuscript is still a draft with 13 TODOs and empty appendices.
- The new abstract sentence presents a failed pre-specified equivalence ("within 0.051 m") in language that reads as a pass.

## What would raise the score

1. **Immediately (no new compute):**
   - Rewrite the abstract sentence so it names the margin and the verdict, or drop the residual.
   - Retitle `find:flash` to "behaves like a lower temperature".
   - Add E₈'s verdict, the gain's interval, interval levels, the linear-interpolation note and its sensitivity.
   - Fix the 11b heading.
2. **To 6:**
   - A complete manuscript: no TODOs, the appendices, the proof, a unified Qwen-Drive sample.
   - The flip-rate residual for the cache, E₄ and E₈ on the FlashDrive run.
3. **To 7:**
   - A confirmation run on fresh clips (the A1.5 holdout 160–199 plus new seeds), including FlashDrive, with hashed pre-registration.
   - The optimized FlashDrive checkpoint decomposed into a checkpoint effect and a cache/temperature effect.
   - Rate-level negative controls.
4. **Beyond:** a closed-loop (AlpaSim) consequence, or a second model family where the matched-temperature protocol changes a published conclusion.
