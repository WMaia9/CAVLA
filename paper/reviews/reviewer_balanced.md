# ICML 2027 Review: "Your ODE Solver Is a Temperature Knob: How Accelerating Flow-Matching Driving Policies Changes What They Decide"

Reviewer role: balanced / area-chair style. Draft reviewed: `paper/main.tex` (early draft with TODOs), checked against `docs/RESULTADOS_DECISAO_E_INFORMACAO_20261022.md`, `docs/TEORIA_INVERSOES_DE_DECISAO.md`, `docs/ESBOCO_PAPER_TEMPERATURA_NUMERICA.md`, `docs/PROTOCOLO_*.md`, `docs/CRONOGRAMA_ICML2027.md`, and the JSON summaries in `artifacts/`. All numbers marked "recomputed" below come from my own light Python passes over those JSON files.

---

## 1. Summary

The paper claims that, in flow-matching VLA driving policies, accelerating the action decoder with first-order samplers (few Euler steps, velocity caching) acts almost exactly like lowering the sampling temperature (scaling the initial noise, x0 = τz). Samples contract toward the conditional mean. The mechanism is simple: under rectified flow, one Euler step returns E[x1 | c], and for Gaussian targets every Euler grid contracts deviations by a product of cos θ_i. The paper defines an effective temperature τ_eff as the noise scale at which an accurate solver (Heun-32) matches the sampler's seed spread. It reports:

- For Alpamayo-1.5, 400/400 Euler-type samples lie closer to the conditional mean than the accurate solver's sample from the same noise.
- The released 10-step samplers of Alpamayo-1.5, R1 and 2 share τ_eff ≈ 0.83–0.84. At matched τ_eff, single-sample ADE, minADE6 and decision-flip rates of Euler-k are close to those of the accurate solver.
- Lower temperature shifts decision frequencies away from stopping and decelerating, and lowers decision entropy.
- In non-reactive log replay, Euler-4 has fewer collisions than the accurate solver at τ=1, but not fewer than the accurate solver at its τ_eff.
- The published single-sample gains of FlashDrive and Coda's "fewer steps, better actions" are proposed to be temperature effects.
- Qwen-Drive is a counterexample: its flow is nearly deterministic, and the released solver adds diversity instead of removing it.

A symmetry proposition (decisions protected by a reflection symmetry never flip) and toy-mixture evidence (flips move toward the mean) provide the decision-level theory. The recommendation is to report τ_eff and compare accelerations at matched effective temperature. Closed-loop AlpaSim results, official FlashDrive checkpoints, a language-seed ablation and the formal mixture proof are pending.

## 2. Strengths

1. **A clear, useful and easily adopted idea.** "Compare accelerated samplers at matched effective temperature" is a concrete evaluation rule that the VLA-acceleration community (FlashDrive, ProbeFlow, AdaVLA, Coda) can apply directly. It is cheap to compute, since it needs only seed spread plus an accurate solver with τ-scaled noise. If it holds up, it changes how such papers are refereed.
2. **The core empirical regularity is striking and replicates.** τ_eff(Euler-10) = 0.839 / 0.837 / 0.833 for Alpamayo-1.5 / R1 / 2 (recomputed from `artifacts/temperature_report_{v1,r1_v1,a2_v1}_20261022/summary.json`). Every Table 1 cell matches the JSON to the reported precision. The R1 and Alpamayo-2 replication was pre-registered before execution (Addendum 9 in `PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md`). This is out-of-sample evidence, not post-hoc fitting.
3. **A built-in negative control exists, though it is not yet shown.** Few-step Heun samplers do *not* lie on the temperature curve. For example, Heun-5 has spread 1.886, close to τ=1, yet a flip rate of 11.3% against 0% for the reference (`temperature_report_v1_20261022`). This shows the matching test can fail, which is what makes the Euler result informative.
4. **Unusually good research hygiene.** The protocols were registered before the runs, with dated addenda. The lab reports a negative result honestly: the decision-information schedule passed a pre-registered 436-clip confirmation, then lost its gain at matched temperature (Addenda 6 and 8). It also discloses an analysis it discarded because of leakage (AUROC 0.88 → 0.48–0.51). This negative result is the best argument for the protocol and should be promoted.
5. **The decision-level view is the right lens for driving.** Showing that minADE_k flips its verdict as k grows (Euler-10 vs converged: −0.18 m at k=1, +0.05 m at k=6) is a crisp demonstration that trajectory-averaged metrics can hide distributional changes.
6. **The Qwen-Drive counterexample** keeps the thesis honest. It shows the effect depends on the parameterization and on how stochastic the learned flow is, rather than being a universal law.

## 3. Weaknesses

### W1. Statistical inflation in the headline safety result (major)
The abstract and contributions report "Euler-4 halves collisions (0:12 discordant, p<10⁻³)". I recomputed this from `artifacts/safety_metrics_v1_20261025/rows.json`:
- The 1,302 "paired units" are clip × seed × model. The same `context_100…159` clips appear in all three Alpamayo models, and seeds are nested within clips.
- **The 12 discordant pairs come from only 5 distinct clips, and only 7 distinct clips produce any collision at all** under Heun-32 τ=1 or Euler-4. Collision events total 21 vs 9.
- A clip-level sign test on 5 concordant-direction clips gives p = 2/2⁵ = 0.0625, not p < 10⁻³. The "0:2 at matched τ" comes from a single clip.
- The human logged trajectory itself collides in 1.0% of clips (results log §9). That is the same order as the effect (1.61% vs 0.69%), so label and box noise is a live confound.

"Halves collisions" should therefore not appear in the abstract until closed-loop or larger-sample evidence supports it. The safety analysis also has no pre-registered protocol (there is no `PROTOCOLO_*` for safety metrics), so it should be labelled exploratory.

### W2. The central claim is an equivalence claim, but it is tested as "no significant difference" (major)
"Explained by temperature" means the residual (sampler − temperature curve at τ_eff) is negligible. The draft instead shows point values ("6.1 / 6.1%") with a "TODO confidence intervals". What is needed:
- Paired residual CIs per cell. These already exist for Alpamayo-1.5 uniform Euler-4/6/8 in `artifacts/schedule_matched_report_v1_20261022/summary.json` (e.g., flip residual for uniform_4 = +0.018 [−0.006, +0.045]).
- An equivalence margin set before looking at the data (TOST).
- A "fraction explained" statistic with CI: 1 − |residual| / |raw sampler − τ=1 difference|.

Some residuals are *not* null, and the paper should report them rather than hide them:
- Decision-entropy residual for uniform Euler-8 is −0.032 [−0.068, −0.001], and Euler-10 entropy is 0.392 vs 0.425 on the curve.
- The minADE6 residual is +0.04 to +0.06 m in every Alpamayo-1.5 row (results log §6).
- On Alpamayo-2, Euler-10 flips are 9.7% vs 7.7%.

So the honest statement is "temperature explains most of the effect; Euler additionally reduces decision entropy and coverage beyond its spread". That is still a strong result, and more credible than the current framing.

### W3. τ_eff is matched on a single scalar, and the "independent validation" is not independent
Mean pairwise spread and the median shrinkage ratio are two estimators of the same isotropic scale. Their agreement (0.84 vs 0.84) does not validate the claim that the *distributions* match. Agreement on ADE1 and entropy is also partly mechanical, because both are monotone in spread.

The draft itself shows the contraction is anisotropic. The Gaussian factor depends on the per-direction target scale s, and per-coordinate or PCA de-shrinking made everything worse (results log §5). Distributional evidence is needed: per-clip energy distance or MMD between Euler-k samples and Heun·τ_eff samples, compared against Heun·τ=1 and against Heun-k. This needs only saved plans, no GPU.

For the cache the two estimators also disagree: spread-matched τ_eff = 0.555 [0.531, 0.578] (`schedule_matched_report_v1_20261022`, `flashdrive_cached8_4`), while the intro says "about 0.5", which is the shrinkage ratio.

### W4. Abstract and contributions overreach the evidence in several places (fixable)
- **"Every Euler-type sample … across three generations … contracts":** the 400/400 shrinkage test exists only for Alpamayo-1.5 (results log §5; Finding 1 has a TODO for R1/A2). For R1 and Alpamayo-2 only spread-based evidence exists.
- **"Velocity cache of FlashDrive":** this is the lab's FlashDrive-*style* re-implementation inside an Alpamayo-1.5 Euler-8 schedule, not the official FlashDrive system, which is not started per `CRONOGRAMA_ICML2027.md`. The reinterpretation of FlashDrive's published numbers is so far an analogy.
- **"6.1% vs 6.1%" in the abstract** is the best-matching cell (R1). The worst is Alpamayo-2 at 9.7% vs 7.7%. Report the range.
- **Finding 5 mixes two data subsets and two references.** "Converged: 2.20 m" ADE1 comes from the 200-clip × 2-seed analysis with a Heun-160 reference. Table 1 uses holdout clips 100–199 with Heun-32, where τ=1 ADE1 is 2.465 m. Readers will see two different "converged" ADE1 values. The results log also states that the minADE_k comparison was against **Heun16**, not the converged flow ("vantagem do Euler10 sobre o Heun16", results log §5); please check. The k=6 difference (+0.046 [−0.023, +0.153]) is not significant, so "minADE6 degrades" is a direction, not a result.
- **Setup says "200 clips for open-loop analyses"**, but Table 1 uses 100 clips (Alpamayo-1.5) and 60 clips (R1, A2), and Qwen-Drive uses 60 clips with an **Euler-64** reference (with floor min(0.1, 1/k)), not Heun-32.
- **Qwen-Drive:** "this spurious diversity is what improves its best-of-six error" is causal language. Two observations complicate it. Euler-10 also improves *single-sample* ADE1 (1.540 vs 1.571, RL). And spread is non-monotone in k (Euler-1 0.055, Euler-2 0.045, Euler-4 0.083, Euler-10 0.161, Euler-64 0.038; `temperature_report_qwendrive_rl_v1_20261022`). That points to a parameterization artefact (x-prediction endpoint near t=1) rather than "the solver adds diversity" in general.

### W5. The paper is ambiguous about whether lower temperature is good or bad
Finding 3 and the Impact Statement frame "fewer cautious manoeuvres" as a risk. Finding 4 says lower temperature *halves* collisions. Against the human, the decision log-score shows no detectable harm (+0.018 [−0.027, +0.057], results log §5), and modal agreement improves slightly. The paper needs one coherent position. My suggestion: temperature is a hidden policy hyperparameter, neither good nor bad, and its hiddenness is the problem. Optimal τ is task- and metric-dependent, which is why comparisons must hold it fixed.

### W6. Validity threats that a driving reviewer will raise
- **Reasoning trace fixed per clip.** In deployment Alpamayo samples a chain-of-thought. If language-level variance dominates action-noise variance, then τ_eff on the action expert may matter little for the policy's real output distribution. This ablation is planned for week 0 and is essential.
- **Reference convergence.** The setup has "Heun-80 and Heun-160 agree to within TODO". The earlier protocol (`PROTOCOLO_ERRO_NUMERICO_DECISOES_V1.md`) found that Euler-40 had not converged relative to Euler-80 in BF16 on a hard seed. Report Heun-32 vs Heun-160 distances, and flip rates *relative to the τ=0.9 effect size*.
- **Test-split contamination** is unverified ("Não verificamos de forma independente se esses clipes ficaram fora do treino", results log, Dados). Say this in the limitations, or check it against the released training manifests.
- **Fixed-threshold decision classes.** Flips concentrate near thresholds, so a threshold-sensitivity analysis (±20%) is needed.
- **Closed-loop has not been run yet.** AlpaSim v1 is 60 scenes × 1 rollout with a 34% baseline incident rate. With paired binary outcomes, a 5-point difference typically yields fewer than 10 discordant scenes. Even a perfect v1 can only be a pilot.

### W7. The theory is thin relative to its billing
- The Gaussian contraction (∏cos θ_i ≤ 1) is acknowledged as implicit in Claustre et al. (2026).
- The symmetry proposition is correct in spirit. However, (a) the condition h·Lip(v) < 1 is restrictive near t → 1 for multimodal targets, where Lip(v) blows up, and (b) driving decisions are not reflection-symmetric, so it explains why flips are rare, not where they go.
- The two-mode monotonicity result, which is the decision-level result that matters, is a TODO and currently empirical (408,769 toy flips).
- The "k^{−p} flip-rate" proposition in `TEORIA_INVERSOES_DE_DECISAO.md` is not in the paper.

Either prove the two-mode case or present it explicitly as a conjecture with evidence. The paper does not need deep theory, but it must not claim "decision-level theory" as a contribution without a theorem.

### W8. Terminology
Scaling the initial noise is a "temperature" only in a loose sense: it equals tempering p^{1/T} only for Gaussians. Call it "noise-scale temperature" once, cite the Glow/BigGAN-truncation lineage, and use the Alpamayo `temperature` argument as the explicit control. Otherwise ML reviewers will object.

## 4. Questions for the authors

1. How many *distinct clips* drive each collision comparison, and what is the clip-level (cluster) test result? My recomputation gives 5 clips for 0:12 and 1 clip for 0:2.
2. Does the Gaussian closed form predict τ_eff? A quick computation with the paper's own formula (ratio = Euler/exact deviation on a uniform grid) gives, for target scale s ≈ 0.3 or s ≈ 3, Euler-2/4/6/10 ratios of about 0.28–0.30 / 0.59–0.62 / 0.72–0.73 / 0.82–0.83. The ratio for Euler-10 is at most 0.88 for any s. The measured values are 0.42 / 0.65 / 0.75 / 0.84. If a one-parameter Gaussian fit reproduces the whole k-curve for all three Alpamayos, then "common τ_eff ≈ 0.84" is a *predicted* consequence of the grid plus the data scale, not a coincidence. That would greatly strengthen the theory section.
3. With the language trace re-sampled per seed, does τ_eff of the action expert still predict changes in the *policy-level* decision distribution?
4. What is the Heun-32 vs Heun-160 discrepancy, in metres and in flip rate?
5. In Finding 5, is the k-sweep comparison against Heun-16 or against the converged flow?
6. For Qwen-Drive, why is spread non-monotone in k? Is the Euler-64 reference with the min(0.1, 1/k) floor converged?
7. Does the Alpamayo multi-sample selector, if enabled in deployment, change which τ is optimal or undo the contraction?
8. Will the official FlashDrive checkpoint be evaluated with *its* 8-step schedule and fine-tuned weights, so that the fine-tuning vs temperature attribution is actually separated (FT weights at τ_eff vs original weights at τ_eff)?

## 5. Significance and novelty

**Who would care.** Three groups:
- Driving-VLA and robot-VLA practitioners who ship accelerated flow decoders (NVIDIA Alpamayo, FlashDrive, π0-style policies).
- Authors and reviewers of acceleration papers, who currently accept "near-lossless minADE" claims.
- Evaluation and safety researchers, who would gain a principled reason why acceleration can silently change the decision distribution.

The practical message ("your speed-up may just be a temperature change; report τ_eff") is actionable and likely to be cited.

**What is genuinely new vs prior work:**
- *Not new:* one Euler step equals the conditional mean (folklore in rectified flow and consistency models). Euler contraction for Gaussians (Claustre et al., 2026). Few-step loss of rare modes and mean collapse (SubFlow, GMFlow). Noise-scale temperature (Glow, truncation tricks). Readers will also recall that coarse deterministic diffusion samplers produce blurrier, mean-like samples (DDIM/EDM discussions).
- *New:* (i) an operational τ_eff definition and a matched-temperature comparison protocol for flow *policies*; (ii) the empirical finding that Euler-k lies on the explicit-temperature curve across three generations of a 10B driving VLA, while Heun-k does not; (iii) decision-level consequences (entropy, stop/decelerate frequencies, minADE_k verdict reversal); (iv) a concrete, testable reinterpretation of published acceleration gains; (v) a pre-registered negative result showing that even a confirmed out-of-sample gain can be a temperature artefact.

**Assessment.** The novelty is in measurement and evaluation methodology, not mechanism. For ICML main track this is acceptable, as an "evaluation confound" paper, if the matched-temperature claim is demonstrated with equivalence statistics and at least one published acceleration is reinterpreted on its own checkpoint. Without the official FlashDrive result and without a closed-loop effect, the paper reads as a well-executed workshop-level observation about Alpamayo.

## 6. Soundness of the statistical analyses

| Aspect | Current state | Verdict / fix |
|---|---|---|
| Pairing | Common random numbers: the same noise z across samplers, the same clips and seeds; integrators run inside each model's own sampling call. | Excellent design. State it prominently. |
| CIs in Table 1 | TODO. The JSON has marginal clip-bootstrap CIs, which are very wide (e.g., minADE6 1.220 [0.920, 1.605]) because they include clip heterogeneity. | Report **paired residual** CIs (sampler − interpolated curve), bootstrapping clips and refitting the curve, as already done in `schedule_matched_report`. Propagate the uncertainty in τ_eff itself. |
| Equivalence | "Difference vanishes" is inferred from non-significance (0:2, p=0.5). | Use TOST with pre-registered margins, or report "fraction explained" with CI. Absence of evidence is not equivalence, especially with only 2 discordant pairs. |
| Clustering | McNemar on 1,302 clip×seed×model units treats them as independent. | Use a clip-level cluster bootstrap or permutation test, or a GLMM with a clip random effect. Report distinct-clip counts. On recomputation the headline p-value moves from <10⁻³ to ≈0.06. |
| Multiple comparisons | Many schedules × NFE × metrics × models. The authors already correctly declined to promote the isolated field-turn NFE-4 residual. | Declare one primary endpoint per claim (e.g., flip residual of Euler-4 and Euler-10, pooled over models). Apply Holm correction to the secondary set. Mark the safety and decision-frequency analyses as exploratory. |
| Pre-registration | Strong for schedules (Addenda 6/8), the multi-model replication (Addendum 9), robotics and closed-loop v1 (`PROTOCOLO_CLOSEDLOOP_ALPASIM_V1.md`, C1–C3 with a falsification criterion). None for log-replay safety; Table 1 itself began as exploratory (Addendum 7). | Put the protocol timestamps in an appendix. Separate confirmatory and exploratory results explicitly in the paper. Pre-register closed-loop v2 with a power analysis and an equivalence margin. |
| Sample sizes | 100 / 60 / 60 clips × 6 seeds for the core result. The 436-clip confirmation set has been used only for schedules. | Rerun the core temperature probe (Euler-4/10, Heun-32·τ grid) on the 436 unseen clips for at least Alpamayo-1.5. This is the cheapest way to turn Table 1 into a confirmatory result. |
| Leakage / forking paths | Detected and disclosed (the AUROC case). | Keep it. It builds trust. |

Overall: the experimental design is sound and better than typical for the area. The *inferential* layer (clustering, equivalence, primary endpoints) still has to be built, and the one headline p-value I checked is materially overstated.

## 7. Concrete prioritized plan to reach a clear accept

Ordered by expected score gain per unit of effort, mapped onto `CRONOGRAMA_ICML2027.md` (deadline assumed end of January 2027; today 26/10). Items marked "CPU" need no GPU and can run in parallel with the closed-loop and FlashDrive jobs.

| # | Action | Effort | Why it matters | When |
|---|---|---|---|---|
| 1 | **Fix the statistics now (CPU).** Clip-clustered tests for all collision claims. Paired residual CIs with τ_eff uncertainty for every Table 1 cell. TOST equivalence with a stated margin. A "fraction explained" column. Distinct-clip counts. Remove "halves collisions" from the abstract unless closed-loop supports it. | 2–3 days | Removes the most likely reject reason: an overstated p-value on the safety headline. Turns "explained by temperature" into a testable claim. | Week 0 |
| 2 | **Official FlashDrive at matched temperature (Gate 2).** Use the original weights and the FlashDrive fine-tuned weights, each with and without the cache, each against Heun-32·τ_eff. Decompose minADE1 and minADE6 into fine-tuning, cache, and temperature. | ~1 GPU-week | The single strongest novelty lever: a published "accuracy gain" reinterpreted on its own checkpoint. If it fails, cut the claim, as the timeline already plans. | Weeks 0–1 (as planned) |
| 3 | **Distributional matching plus Heun-k negative control in the main table (CPU).** Per-clip energy distance or MMD of Euler-k vs Heun·τ_eff, vs Heun·τ=1, vs Heun-k at equal NFE. Add the Heun-5 / Heun-3 rows to Table 1. Report the entropy and minADE6 residuals as "beyond-temperature" effects. | 2–3 days | Answers the "you matched one scalar" objection. Shows the test can fail, and turns the residuals into a finding rather than a hidden weakness. | Week 0–1 |
| 4 | **Language-seed ablation** (Alpamayo-1.5 and R1, reasoning re-sampled per seed). Report τ_eff and the decision-distribution shift at the policy level. | 2–3 GPU-days | Pre-empts the most obvious driving-reviewer objection (W6). If action-temperature effects survive language variance, the deployment relevance is established. | Week 0 (as planned) |
| 5 | **Closed-loop v2, pre-registered and powered.** Use 150–200 scenes × 3 seeds with the planned configs plus the explicit-temperature arm (Euler-10·τ via Alpamayo's own parameter). Pre-specify the primary endpoint (offroad-or-collision-at-fault), a scene-clustered test, an equivalence margin for "Euler-4 ≈ Heun·τ_eff", and a power calculation using v1's 34% base rate and its observed discordance. Treat v1 as a pilot. | 5–7 machine-days | Converts a non-reactive, 5-clip safety signal into a closed-loop result: the difference between an analysis paper and a driving paper. A null at matched τ with tight equivalence bounds is still publishable. | Weeks 1–4 |
| 6 | **Gaussian closed-form prediction of τ_eff(k) (CPU).** Fit per-direction target scales from Heun-32 samples (PCA), predict τ_eff for k = 1, 2, 4, 6, 10 with ∏cos θ_i, and compare across the three Alpamayos and SmolVLA. Explain the Qwen-Drive exception through its x-prediction parameterization. | 2–3 days | Turns "common τ_eff ≈ 0.84" from a coincidence into a prediction, and strengthens theory more cheaply than the mixture proof. My quick check suggests the fit is plausible. | Weeks 1–2 |
| 7 | **Confirmation on the 436 unseen clips** for the core Euler-4/10 vs Heun-32·τ result (Alpamayo-1.5; R1 if time allows), pre-registered with item 1's equivalence margins. | ~2–3 GPU-days | Makes the central table confirmatory rather than exploratory, at low cost, reusing existing infrastructure. | Weeks 2–3, while v2 runs if GPU sharing allows; otherwise in the December buffer |
| 8 | **Two-mode monotonicity:** prove it, or state it as a conjecture with the toy evidence. Move the symmetry proposition's full proof to the appendix. Reframe the contribution as "decision-level analysis", not "theory". | 1–2 weeks (CPU) | Prevents an overclaim objection. Low marginal score gain unless the proof is clean. | Weeks 2–3 (as planned) |
| 9 | **Narrative and clarity fixes.** One consistent clip subset and reference per table. "FlashDrive-style" everywhere it applies. Report the τ_eff range in the abstract (0.83–0.84, flips 5.0–9.7% vs 4.7–7.7%). Resolve the good-vs-bad framing of low temperature (W5). Promote the schedule negative result to a main-text paragraph. Address contamination, threshold sensitivity, and reference convergence. Move SmolVLA to a one-paragraph generality check. | 3–4 days | Clarity is cheap score. Reviewers punish internal inconsistencies (e.g., the 2.20 vs 2.465 m "converged" ADE1). | Draft v1 (13/12) |

**Decision rule.** If Gate 2 (FlashDrive) fails *and* closed-loop v2 shows nothing, the paper still has a defensible contribution: a measurement protocol plus a pre-registered negative result plus a decision-level analysis across 3–4 driving VLAs. It would fit better as a strong workshop or TMLR paper than as an ICML main-track paper. If either succeeds, with items 1–4 done, I would expect a 6–7.

## 8. Score and confidence

- **Current draft: 4 / 10 (weak reject).** The idea is good and the design is careful, but the headline closed-loop and FlashDrive results are missing, the safety headline is statistically overstated, and "explained by temperature" lacks equivalence statistics.
- **Projected with items 1–5 completed and positive: 7 / 10 (accept).**
- **Confidence: 4 / 5.** I am familiar with flow-matching samplers and with driving-policy evaluation, and I verified the key numbers against the artifacts.

### Summary of verified number mismatches (with sources)

1. Collision 0:12, p<10⁻³: driven by **5 distinct clips** (7 with any collision); 0:2 comes from 1 clip. Clip-level sign test p ≈ 0.06 (`artifacts/safety_metrics_v1_20261025/rows.json`).
2. "FlashDrive cache τ_eff ≈ 0.5": spread-matched τ_eff = **0.555 [0.531, 0.578]**; 0.50 is the shrinkage ratio (`artifacts/schedule_matched_report_v1_20261022/summary.json`; results log §5).
3. "Every Euler sample contracts across three generations": the 400/400 result exists only for **Alpamayo-1.5** (results log §5).
4. Finding 5 "converged 2.20 m" (200 clips, Heun-160) vs Table 1 Heun-32 τ=1 ADE1 = **2.465 m** (`temperature_report_v1_20261022`). The minADE_k sweep is labelled "Heun16" in the results log §5.
5. "200 clips for open-loop analyses": Table 1 uses **100 / 60 / 60 clips**. Qwen-Drive uses 60 clips with an **Euler-64** reference.
6. Abstract "6.1 vs 6.1%" is the best cell. Range across models: Euler-10 flips 5.0–9.7% (sampler) vs 4.7–7.7% (curve).
7. Non-null residuals not reported: entropy residual uniform Euler-8 **−0.032 [−0.068, −0.001]**; minADE6 residual **+0.04 to +0.06 m** (Alpamayo-1.5).
8. Table 1 cells themselves: all reproduced to the stated precision (τ_eff 0.648 / 0.839 / 0.613 / 0.837 / 0.619 / 0.833).
