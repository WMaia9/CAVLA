# ICML 2027 Review, Round 5.2: Reviewer 2 (focused round)

**Paper:** `versions/main_v5.2.tex` / `main_v5.2.pdf` (26 Oct 20:49), compared by `diff` with `main_v5.tex` and `main_v5.1.tex`.
**New artifact:** `artifacts/flashdrive_analysis_nvidia_v1_20261026/summary.json`. `src/cavla/flashdrive_analysis.py` is byte-identical to its `source_snapshot.py`. Raw data: `artifacts/flashdrive_probe_nvidia_v1/plans.npz` and `manifest.json`.
**My script:** `/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/r2_v52/fd.py`, output in `fd.out`. It imports the paper's `per_clip` and uses the same 1000-draw clip bootstrap (seed 20261026). It adds monotone (PCHIP) and cubic-spline interpolation, gain intervals, paired sampler contrasts, per-clip residuals and leave-one-out.

## Summary of the paper

The paper argues that few-step Euler sampling of flow-matching driving policies (Alpamayo-1.5, R1, 2) contracts samples toward the conditional mean. It argues that a Heun-32 reference run at a lower initial-noise temperature τ_eff, matched on seed spread, reproduces the sampler's effect on rate metrics (ADE₁, decision-flip rate). It proposes a matched-temperature evaluation protocol with clip-bootstrap intervals and TOST equivalence.

v5.2 adds one paragraph, "A published acceleration", which applies the protocol to FlashDrive's velocity cache on the original nvidia Alpamayo-1.5 checkpoint.

## What changed since v5 (verified)

**v5.1 addressed my round-5 main point, the overstated per-sample signature claim.**
- The signature sentence is out of the abstract.
- `find:shrink` now reports pooled paired differences with intervals (`SigDiff*`). It says plainly that Euler is "closer to temperature than to a uniform shrinkage, but not identical to it (largest for R1)". It also says that a matching median does not identify temperature.
- The Discussion now limits the claim to the rate level and declares the per-sample statistic and margin (0.05) as a pre-registered secondary endpoint.
- The abstract now gives the Heun flip residuals as numbers instead of "fail the same test".
- "Sampled before any analysis" is removed.

All of these are what I asked for. **Round-5 issue 1 is closed.**

**v5.2 adds the FlashDrive paragraph.** Every macro matches `summary.json`:

| Macro | Paper | Artifact |
|---|---|---|
| `FDClips` | 60 | 60 |
| `FDTau` | 0.64 | 0.6402 |
| `FDGain` | 0.28 | 2.6883 − 2.4048 = 0.2834 |
| `FDResid` | 0.008 [−0.034, 0.051] | 0.0077 [−0.0344, 0.0513] |
| `FDETenTau` | 0.84 | 0.8369 |
| `FDETenResid` | 0.012 [−0.016, 0.041] | 0.0117 [−0.0158, 0.0412] |

I reproduced all of them independently.

**Unchanged:** 13 `\TODO`s in the source (one was replaced by the new paragraph and its TODO), empty appendices, no proof of Prop. 3.1, hand-typed Qwen-Drive numbers.

## Is the new analysis sound?

**Clip bootstrap.** Sound.
- Clips are resampled with replacement.
- τ_eff and the matched ADE₁ are re-solved in every draw.
- The same draws are shared across samplers.
- All configurations share the six noise seeds and the VLM reasoning within a window, so the contrasts are paired.
- The spread curve is strictly monotone in all 1000 draws, so `np.interp` never extrapolates silently, and `boot_in_range` = 1.0.
- A 90% percentile interval is the correct TOST interval for α = 0.05.
- Leave-one-clip-out moves the cache residual only within [−0.003, +0.016].

**τ_eff interpolation.** Adequate, with a small one-sided bias.
- The Heun-32 ADE₁ curve is convex in τ (2.344, 2.376, 2.429, 2.506, 2.589, 2.688). Piecewise-linear interpolation therefore slightly overstates the matched ADE₁, which biases the residual toward zero.
- With PCHIP or cubic-spline interpolation, the cache residual becomes 0.010 [−0.033, 0.054–0.056], and P(residual > 0.05) goes from 0.052 to 0.074–0.078.
- τ_eff is unchanged (0.641).
- So the failure of equivalence is not a rounding artifact; a better interpolator moves the upper bound further past the margin. The paper should state that the interpolation is linear, give the grid (0.1), and report this sensitivity in one line.

**Equivalence honesty.** Mostly honest, with one omission.
- "Equivalence is not established" is correct: 0.0513 > 0.05.
- "Sits at our margin" is fair, but "just outside" would be more accurate.
- "Almost all of the gain is a temperature effect" holds for the point estimate (97% explained). The 90% interval on the fraction explained is [82%, 112%], and the paper should give it.
- **The omission:** Addendum 11b pre-registered four configurations (cache, E₄, E₈, E₁₀) with the prediction that every 90% interval lies inside ±0.05 m. Two of the four failed:
  - cache: upper bound 0.051;
  - E₄: 0.016 [−0.036, **0.070**].
- The paper reports only the cache and E₁₀. It also does not say that E₁₀ *passes* equivalence on this run (upper bound 0.041). Both facts belong in the text, since the protocol's value depends on reporting its pre-registered verdicts, not only those for selected configurations.

## Main weaknesses

### W1. "60 further clips" is false. The FlashDrive run reuses the paper's main sample.

- The probe manifest has `range [100,160]` and seeds 9201–9206. These are the same clip IDs and seeds as the main temperature probes (R1/A2 `range [100,160]`; A1.5 `[100,200]`, with 160–199 as holdout).
- Window w3 reproduces the main sample's Heun-32 τ=1 plans almost exactly: the mean absolute difference is 0.028 m on context_100 and 0.003 m on context_130.
- So the FlashDrive inputs are the same 60 scenes. The extra windows w0–w2 are 0.1–0.3 s earlier frames of the same scenes, with nearly identical futures, and they add little independent information.
- Consequently, "On the same run, E₁₀ *again* has τ_eff 0.84" is a pipeline re-run on the same clips and noises, not a replication. The agreement with `\AOneFiveETenTau` = 0.83 is a reproducibility check of the FlashDrive input path, which is useful but much weaker than a replication.
- The paper must say "the same 60 clips, through FlashDrive's input pipeline (4 consecutive windows each)". The unused A1.5 holdout clips (160–199) were the natural place for this run.

### W2. The headline "0.28 m" conflates Euler-8 with the cache.

- The gain is measured against Heun-32 at τ=1, which no one deploys.
- Decomposed:
  - E₈ alone lowers ADE₁ by 0.166 [0.117, 0.209].
  - The cache's own increment over its uncached E₈ is 0.118 m [0.087, 0.148] (paired, 95%).
  - Against Alpamayo's released E₁₀, the cache gains 0.143 m.
- "Lowers the checkpoint's ADE₁ by 0.28 m" will be read as FlashDrive's gain, and it is about twice that. Report the gain against the sampler FlashDrive replaces, with its interval.

### W3. The paper misses the most informative comparison: the cache is Euler-4.

- Steps 3–6 reuse the step-2 velocity, so the cache makes four fresh evaluations (steps 0, 1, 2, 7).
- On this run, cache − E₄ = −0.001 m in ADE₁ [−0.038, 0.038] and +0.031 in spread [−0.04, 0.11]. τ_eff is 0.640 vs 0.626.
- This is the mechanistic point of the paper: a four-evaluation first-order sampler sits at the temperature of four Euler steps. It is a sharper and more convincing statement than the miss against the smoke-test prediction (0.5–0.6).
- It also means that E₄ failing equivalence (W-equivalence above) is the same finding as the cache failing it.

### W4. "Run through FlashDrive's own code" overstates it.

- `flashdrive_probe_nvidia_v1/source_snapshot.py` hooks `euler_with_cache` and integrates every configuration, including the cache, with its own `integrate()`. The cache trajectories are a reimplementation. FlashDrive's code supplies the model loader, the inputs and `step_fn`.
- I compared it with `.deps/flashdrive/flashdrive/diffusion.py`. The logic is identical (reuse the last fresh velocity on steps 3–6). One difference: FlashDrive builds the time grid in bfloat16 and the probe in the state dtype.
- The official output is computed and discarded. A single check of the reimplementation against FlashDrive's own sampler, on matched noise, would close this. Until then, write "FlashDrive's inputs and model, with its cache schedule".

### W5. Clip-level heterogeneity is large and goes unmentioned in the new paragraph.

- At the global τ_eff, per-clip ADE₁ residuals of the cache have SD 0.21 m and range from −0.50 to +0.74 m.
- The aggregate match is a mean over clips. The existing "Distribution, not trajectory" paragraph covers this in principle, but the FlashDrive paragraph, which practitioners will read in isolation, should say that the match is on the clip average.

### W6. The pre-registration timing cannot be audited.

- Addendum 11b is stamped "26/10, 20:55". The protocol file's mtime is 20:48:53, the analysis code's is 20:48:53, the artifact's is 20:48:56 and the paper's is 20:49:32.
- The declared stamp therefore *postdates* the result. This may be a clock or labeling slip, but the directory is not under version control, so "written before computing" cannot be verified at all.
- For a paper whose central contribution is a pre-registered protocol, this matters. Commit or hash the addendum (for example, a SHA-256 posted to a timestamped location) before each run.

### W7. Still carried over.

- The manuscript is incomplete: 13 TODOs, empty appendices B–E, no proof of Prop. 3.1, hand-typed Qwen-Drive numbers, and the FlashDrive optimized checkpoint pending.
- The Heun control is on one model with τ_eff clamped near 1.
- Rate-level negative controls are missing (what the test rejects).
- No TOST verdict column in Table 1.
- The equivalence margins were chosen after seeing the main 60 clips. This run reuses those clips, so W1 compounds this: the 0.05 m margin is being tested on the sample it was tuned on.

## Score

**5 / 10 (weak reject), unchanged. Confidence 4/5.**

v5.1 fixed the signature overclaim exactly as requested. The new FlashDrive analysis is statistically sound in its mechanics: the joint clip bootstrap re-solves τ_eff, the curve stays monotone, the interpolation bias is small and conservative in direction, and every number reproduces. The text is honest that equivalence is not established.

But the paragraph:
- calls the main sample "further clips" and presents a re-run as a replication ("again");
- attributes Euler-8's effect to the cache;
- leaves out the pre-registered failure of E₄ and the pre-registered pass of E₁₀;
- misses the obvious cache ≈ E₄ result.

The manuscript is still incomplete.

## What would raise the score

1. **To 6:**
   - A complete manuscript (no TODOs, appendices, the proof, a unified Qwen-Drive sample).
   - The FlashDrive paragraph corrected as follows:
     - "same 60 clips";
     - the gain vs E₈/E₁₀ with intervals;
     - the fraction explained with its interval;
     - all four pre-registered verdicts (cache ✗, E₄ ✗, E₈ ✓, E₁₀ ✓);
     - cache vs E₄ reported;
     - linear-interpolation sensitivity in a footnote;
     - "reimplemented cache schedule" unless validated against the official sampler.
2. **To 7:**
   - The pre-registered confirmation run on genuinely fresh clips and seeds, including the A1.5 holdout 160–199 and FlashDrive on them, with verifiable pre-registration timestamps.
   - The optimized FlashDrive checkpoint (W4A8, DFlash, streaming) decomposed into checkpoint effect vs cache/temperature effect, as Addendum 11 promises.
   - Rate-level negative controls showing what the test rejects.
3. **Beyond that:** a closed-loop (AlpaSim) consequence of the temperature shift, or a second model family where the matched-temperature protocol changes a published conclusion.
