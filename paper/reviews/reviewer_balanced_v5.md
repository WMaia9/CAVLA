# ICML 2027 Review, Round 5 (focused): "When Fewer Steps Mean Lower Temperature" (v5)

**Reviewer:** balanced / area-chair style. Earlier rounds: 4, 4, 5, 5.
**Material reviewed:**
- `versions/main_v5.tex` and `.pdf`, compared with v4.1 by diff;
- `reviewer_balanced_v4.md` and the "Rodada 4" sections of `CONSOLIDADO.md`;
- `artifacts/paper_numbers_v3_20261026/summary.json` and `src/cavla/paper_numbers.py`.

**What I ran (CPU only):** two scripts in `scratchpad/balanced_round5/`.
- `sig_ci.py` puts joint clip-bootstrap intervals (1000 resamples) on the per-sample signature in Finding 5.
- `sig_center.py` repeats the signature using the seed-mean of E₁ as a common center.

## 1. What changed

- **Two claims of different strength.** The paper now separates a rate-level claim from a sample-level one:
  - *Rate level:* Euler is a contraction toward the mean, and explicit temperature reproduces it.
  - *Sample level:* the contraction's signature is temperature-like. This is labelled exploratory and is a pre-registered secondary endpoint.
- **New Finding 5.** It reports the spread-matched shrinkage control. The shrinkage is now matched exactly by bisection.
- **Text and code agree again:**
  - the equivalence test uses a 90% interval (TOST);
  - the margins are disclosed as chosen post hoc;
  - E₁ is described as keeping 9–14% of the reference spread;
  - the introduction now says that minADE₆ gets worse.

All the round-4 text/code items (§2.3–2.4 of my v4 review) are fixed. I checked the new numbers against `summary.json`, and they match.

## 2. Is the framing coherent? Mostly yes, with three flaws

**What the signature check found** [recomputed, pooled over models, 95% CI over clips]:

| Comparison (median contraction ratio, same spread) | E4 | E10 |
|---|---|---|
| Euler − Shrink | +0.128 [+0.107, +0.146] | +0.067 [+0.057, +0.076] |
| Euler − Temperature | −0.037 [−0.070, −0.011] | −0.002 [−0.019, +0.016] |
| Fraction contracted, Euler − Temperature | +7.1 pt [+4.5, +10.0] | +4.8 pt [+1.9, +7.8] |

- **Euler is clearly closer to temperature than to shrinkage.** The Euler–shrinkage gap is separated in all six model×k cells.
- **At E10 Euler and temperature are indistinguishable on the median.**
- **At E4 they differ,** mostly because of R1: −0.072 [−0.122, −0.035].
- **The result does not depend on which E₁ is the center.** It is unchanged when ratios are measured to the seed-mean of E₁ instead of each seed's own E₁, so it is not an artifact of E₁'s residual spread.

So the sample-level claim holds up as an exploratory observation. These are the three flaws in how it is presented:

**F1. Shrinkage's signature is fixed by construction.**
- Every Shrink_k sample has a ratio of exactly c (the standard deviation of the ratios is 0). "Shrinkage contracts all of them" is therefore a tautology, not an observation.
- The only real information is a single number: Euler and temperature reduce pairwise spread more than radial distance to the mean (0.67 vs c ≈ 0.55 at E4), and a linear map cannot do that.
- Say this plainly, and report intervals.
- Also note why it matters: for a Gaussian target, temperature *is* uniform shrinkage toward m. The separation therefore comes from the non-Gaussian structure of the real policies, which is itself interesting.

**F2. "Euler and temperature both leave some samples uncontracted" hides a significant difference.**
- Euler contracts 5–7 pt more samples than temperature.
- Reviewer 2's energy distances show the same ordering: temperature < Euler < shrinkage.
- The honest sentence is: "Euler is much closer to temperature than to uniform shrinkage, but not identical to it."

**F3. The abstract contradicts itself.**
- It first says "the match holds for rates over clips, not for individual samples", and then that the per-sample contraction "resembles temperature's".
- Fix: same-noise pairs disagree, while the *distribution* of per-sample contraction is temperature-like.

**Two smaller points:**
- **Reviewer 2's failing contractions are left out.** In round 4 he showed that shrinking toward the six-sample mean, collapsing random samples to E₁, and shrinking toward a constant-velocity line all *fail* the rate test. v5 still reads as though any shrinkage passes.
  - Add those arms. They are the strongest support for the rate-level claim: the test rejects most contractions, and passes only those toward E₁.
  - Also note that all six Shrink flip residuals are positive (0.7–3.4 pt), while Euler's change sign.
- **"Heun fails both tests" is not supported.** Heun's per-sample signature is never reported, and with τ_eff ≈ 1 it is ill-defined. Say "fails the rate test".

## 3. Significance and title

- **Significance is unchanged.** The two-tier framing is now internally consistent and partly recovers the temperature claim. That is a real improvement in correctness. But it adds no new evidence about generality or impact.
- **The paper is still a single-family, 60-clip study:**
  - 12 TODOs remain;
  - Heun is run on A1.5 only;
  - Proposition 3.1 is unproved;
  - Appendices B–E are empty;
  - no published acceleration has been re-evaluated (the FlashDrive, AlpaSim, NAVSIM and confirmation artifacts do not exist yet).
- **The title fits.** "When Fewer Steps Mean Lower Temperature" is defensible at the rate level and is supported by the E10 signature. Keep it.
  - Revisit it only if the confirmation's signature endpoint fails, in which case use "...Act Like Lower Temperature".
  - The contribution bullets should name the two-tier claim and the shrinkage controls, which they do not yet.

## 4. Minimum remaining work

### For a 6 (weak accept)
1. **Official FlashDrive at matched τ, reported whichever way it goes.** State the prediction before unblinding, and decompose weights, cache and temperature. FlashDrive reports minADE, and Euler's minADE₆ residual is positive, so the prediction is not obvious.
2. **A pre-registered confirmation on fresh clips and seeds:**
   - primary endpoint: pooled E10 TOST on flips and ADE₁;
   - Heun-3 rejected on all three models;
   - signature endpoint with δ fixed in advance (for example |E−T| < 0.05 and |E−S| > 0.05). On current data this passes at E10 in all three models but fails at E4 for R1, so pre-register E10 as the primary signature cell.
3. **Text fixes F1–F3**, plus Reviewer 2's failing-contraction arms in Table 2 or the appendix.
4. **A complete manuscript:**
   - no TODOs;
   - Heun-3/5 on R1 and A2;
   - Proposition 3.1 proved or demoted;
   - Qwen-Drive numbers generated from code;
   - appendices filled or cut.

### For a 7 (accept)
All of the 6 requirements, plus:

5. **A NAVSIM second family with PDMS at matched τ.** It should either land on the curve or land where a rule stated before the run predicts. DiffusionDrive is not a rectified flow from N(0, I).
6. **At least one of the following:**
   - (a) FlashDrive's reported gain changes materially at matched τ;
   - (b) the signature endpoint is confirmed out of sample;
   - (c) AlpaSim closed loop agrees with open loop, with a scene-clustered interval that is informative.

**On AlpaSim:** at 58 scenes × 1 rollout, present it as a sanity check, not as equivalence. It is the lowest-priority pending item.

## 5. Score

**Score: 5/10 (borderline reject). Confidence: 4/5.**

The framing is now coherent. It is close to what I asked for in round 4, and the sample-level evidence survives my checks at E10. But significance still depends entirely on pending experiments.
- **FlashDrive plus a passing confirmation plus a clean manuscript would move me to 6.**
- **Adding NAVSIM and one of 6(a)–(c) would move me to 7.**
