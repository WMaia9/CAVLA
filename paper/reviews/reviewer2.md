# ICML 2027 Review: Reviewer 2

**Paper:** "Your ODE Solver Is a Temperature Knob: How Accelerating Flow-Matching Driving Policies Changes What They Decide" (draft `/mnt/datasets/CAVLA/paper/main.tex`)

**Basis for this review.** The draft, plus the evidence it cites: the results log (`docs/RESULTADOS_DECISAO_E_INFORMACAO_20261022.md`, sections 1-9), the theory notes (`docs/TEORIA_INVERSOES_DE_DECISAO.md`), the paper plan (`docs/ESBOCO_PAPER_TEMPERATURA_NUMERICA.md`), the protocols (`docs/PROTOCOLO_CLOSEDLOOP_ALPASIM_V1.md`, `docs/PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md` addenda), the timeline (`docs/CRONOGRAMA_ICML2027.md`), and these JSON artifacts, which I recomputed: `artifacts/temperature_report_{v1,r1_v1,a2_v1,qwendrive_rl_v1,qwendrive_sft_v1}_20261022/summary.json`, `artifacts/safety_metrics_v1_20261025/rows.json`, `artifacts/schedule_matched_report_v1_20261022/summary.json`, `artifacts/mode_collapse_theory_v1_20261022/summary.json`, `artifacts/flashdrive_smoke_zlab/manifest.json` and `artifacts/alpasim/closedloop_v1/`. I count every TODO as not done.

---

## 1. Summary

The paper argues that in flow-matching action decoders (rectified-flow interpolation, velocity prediction), first-order few-step or cached ODE samplers act as a hidden reduction of the sampling "temperature". Here, temperature means scaling the initial noise, `x0 = tau z`. Four arguments support this:

- **(i) Elementary facts.** One Euler step returns E[x1|c], and for Gaussian targets Euler contracts every sample toward the mean on any grid, by a factor of a product of cos(theta_i). The authors attribute the latter to the proof of Prop. 2 of Claustre et al. 2026.
- **(ii) A symmetry proposition.** A reflection-invariant target with a reflection-hyperplane decision never flips under explicit RK with h·Lip(v)<1.
- **(iii) Measurements on Alpamayo-1.5, R1 and 2 (PhysicalAI-AV test split).** Euler-type samples contract toward the conditional mean. Defining tau_eff by matching the seed spread to Heun32·tau gives about 0.84 for the released Euler-10 in all three models. At matched tau_eff, ADE1, minADE6 and the decision-flip rate are close to the explicit-temperature curve.
- **(iv) Consequences.** Lower temperature shifts the decision distribution, log-replay collisions fall with fewer steps but not at matched temperature, and published single-sample gains (FlashDrive, Coda) are claimed to be temperature effects.

Qwen-Drive is presented as a counterexample: its flow is nearly deterministic and the solver adds diversity. The paper recommends reporting tau_eff and comparing accelerations at matched tau_eff.

The core phenomenon is real and is measured carefully in open loop. However, three of the five headline contributions are TODOs: closed-loop, FlashDrive at matched temperature, and the mixture theory. The safety result is statistically much weaker than presented. Several decision-level claims are contradicted by the authors' own artifacts.

---

## 2. Strengths

1. **Clear, testable mechanism with a protocol.** The tau_eff definition is simple and model-agnostic. The idea of decomposing a sampler's effect into "temperature + numerical residual" is useful, and reviewers of acceleration papers could adopt it.
2. **The open-loop matched-temperature evidence is internally consistent across three checkpoints.**
   - I recomputed every cell of Table 1 by linear interpolation on the Heun32·tau grid in the `temperature_report_*` summaries, and all of them reproduce to the stated precision. For example, Alpamayo-1.5 Euler-10 has spread 1.472, which gives tau_eff = 0.838, ADE1 2.316, minADE6 1.157 and flips 4.7%. Alpamayo-2 Euler-4 gives tau_eff = 0.619.
   - The contrast with Heun-5, which has large non-temperature error (ADE1 2.563 vs 2.465 and minADE6 1.308 vs 1.074 at tau=1), is a good internal control. It shows the decomposition is not vacuous.
3. **Unusual methodological honesty for the area.** The authors pre-registered protocols. They report a schedule "method" that passed its own pre-registered out-of-sample confirmation (436 clips) and then withdraw it because the gain vanishes at matched tau_eff. They report Qwen-Drive as a counterexample instead of dropping it, and they discard a leaky AUROC analysis (log §2). This is the kind of self-falsification the field needs.
4. **Relevant problem.** Acceleration papers for driving VLAs do report trajectory-averaged metrics only. The observation that minADE_k gives opposite verdicts as k grows is a useful warning even if it is not new in spirit.
5. **The commitment analysis is a nice empirical counterpart to the theory.** No numerical flips occur in clips where all six seeds agree, and Spearman rho = 0.76 between policy entropy and flip rate.

---

## 3. Weaknesses (most severe first)

### W1. Three of five contributions are unsubstantiated, and the abstract and intro already state them as results.

- **Closed-loop.** AlpaSim is a TODO. `artifacts/alpasim/closedloop_v1/` holds one result directory per configuration, and the log ends mid-run at Euler-10. The protocol is 60 scenes × 1 rollout with a 34% baseline incident rate in the first 29 scenes (`CRONOGRAMA_ICML2027.md`). Even when it finishes, it will have very little power.
- **FlashDrive reinterpretation.** Contribution 4 says the FlashDrive single-sample gains are "reproduced by explicit temperature", and the abstract says the pattern was "previously attributed to fine-tuning". The only FlashDrive artifact is a smoke test with `range: [100, 101]` (one clip, 3 windows; `artifacts/flashdrive_smoke_zlab/manifest.json`). All quantitative FlashDrive statements in the draft come from a re-implemented "FlashDrive-style" cache on the original NVIDIA checkpoint. FlashDrive's reported numbers use a different checkpoint: W4A8, DFlash speculative decoding of the reasoning, and streaming fine-tuning. That stack also changes the reasoning text, which this paper holds fixed. Asserting that their attribution is wrong is not acceptable until Adendo 11 is run at scale.
- **Coda.** The intro says that "fewer steps can improve closed-loop success" disappears at matched temperature. This was never tested. Worse, the authors' only closed-loop setting (SmolVLA/LIBERO, log §8) did *not* reproduce the Coda effect: "P3 (fewer steps improves success) did not appear", and success is flat in tau.
- **Mixture theory.** "Decisions flip toward the mean" is a conjecture backed by a toy, and the toy number is not reproducible (see W6).

Remove these items, and what remains is an open-loop measurement study on one model family plus a weak log-replay result.

### W2. The safety result (Finding 4, contribution 5, abstract) rests on 5 driving scenes, and its p-values ignore clustering.

I recomputed this from `artifacts/safety_metrics_v1_20261025/rows.json`.

- **The "1,302 paired units" are not independent.** They are 6 seeds × ~99/59/59 clips × 3 models, and the three models are evaluated on the *same* clips.
- **The discordant pairs are concentrated in a handful of clips:**
  - Euler-4 vs Heun32 tau=1, the 0:12 pairs: 9 (model, clip) units and **5 unique clips**. `context_127` alone contributes 6 of the 12.
  - Heun32 tau=1 vs tau=0.6, the 10:0 pairs: the same 5 clips.
  - Heun32 tau=1 vs tau=0.4, the 12:0 pairs: the same 5 clips. These are literally the same events.
  - The "matched" Euler-4 vs Heun32 tau=0.6 comparison, 0:2 pairs: 1 clip (`context_127`).
- **McNemar over clip-seed-model units treats within-clip replicates as independent.** "p<10^-3" and "p=0.002" are therefore badly anti-conservative. A clip-level or clustered test on 5 informative clips cannot reach anything like these p-values.
- **The total number of events is tiny.** There are 21 collisions vs 9. The human log collides in 1/99 clips (1.0%), so the label/box noise floor is of the same order as the effect (1.61% vs 0.84%). Heun32 tau=1 collides 6 times in `context_193`, which is the clip where the human log itself collides, so it is plausibly a labeling artifact.
- **Lower temperature does not give monotonically fewer collisions.** Euler-1, the conditional mean and the limit of the paper's own temperature axis, collides 15/1302 times (1.15%). That is *more* than Euler-2 (7), Euler-4 (9) and Heun32 tau=0.4 (9). For Alpamayo-1.5 alone, Euler-1 collides 14 times vs 11 for Heun32 tau=1. This is the classic mode-averaging hazard: averaging a left and a right swerve drives straight at the obstacle. The draft omits it, and it directly undercuts the reading "lower temperature = safer".
- **Non-reactive log replay is biased in favour of plans that stay close to the logged ego.** Logged agents reacted to the logged ego. Contracting toward E[x1|c] lowers ADE to the human, so fewer collisions with logged agents follow mechanically. This is an evaluation artifact, not a safety property. The paper's own Limitations line ("log-replay agents do not react and no map is used") understates how decisive it is. With no drivable-area check, a contracted plan that cuts a corner is never penalised.

**Bottom line:** "Euler-4 halves collisions (0:12, p<10^-3)" must not appear in the abstract or contributions in its current form.

### W3. At the decision level, "Euler ≈ explicit temperature" fails for the safety-critical class the paper emphasises (stopping).

- **Finding 3** is titled "Lower temperature shifts the decision distribution away from cautious manoeuvres" and leads with "caching reduces stopping by 0.75 points".
- **In the authors' own temperature reports, explicit temperature does the opposite for stop.** Under Heun32 from tau=1 to tau=0.4, stop frequency *increases*:

  | Model | Stop at tau=1 | Stop at tau=0.4 |
  |---|---:|---:|
  | Alpamayo-1.5 | 3.7% | 4.2% |
  | R1 | 1.7% | 2.2% |
  | Alpamayo-2 | 1.7% | 2.5% |

- **Euler-type samplers *decrease* stop.**

  | Model | Heun32 tau=1 | Euler-10 | Euler-1 |
  |---|---:|---:|---:|
  | R1 | 1.7% | 1.1% | 0.3% |
  | Alpamayo-2 | 1.7% | 1.4% | 0.0% |

- **Consequence.** The stop shift caused by Euler/cache is a non-temperature residual, with the opposite sign, in exactly the decision that matters most for safety. Only "decelerate" moves consistently with tau (A1.5: 21.0% to 17.3%).
- The events are rare and the sign evidence is noisy. But the draft cannot both attribute the stop shift to lower temperature and claim that matched temperature explains decision changes. This needs a per-class matched analysis with CIs.

### W4. Matching a single scalar (seed spread) and then showing that other scalars match is a weak test of "almost exactly a temperature".

- **ADE1 and flip-rate-vs-reference are close to monotone functions of spread** for *any* sampler that contracts toward the mean. Matching on spread makes their agreement largely expected, not an independent confirmation.
- **The "independent" shrinkage ratio is not independent.** Distance to the conditional mean and pairwise spread are both second-moment summaries. For a Gaussian-like field they must agree. Their agreement (0.84/0.84, 0.65/0.64) shows the field is near-linear, not that Euler equals temperature.
- **The paper's own Gaussian theory predicts *anisotropic* contraction, which isotropic noise scaling cannot reproduce.** The product of cos(theta_i) depends on the per-coordinate s. The log even reports that per-coordinate/PCA de-shrinking "makes everything worse" (§5), which suggests non-isotropic structure.
- **The decisive test is missing: a paired, per-noise comparison.**
  - For the same z, compare Euler-k(z) with Heun32(tau_eff · z): their paired distance, and their decision disagreement rate.
  - Compare these numbers with the run-to-run distance between Heun32(tau · z) and Heun32(tau' · z) for nearby tau.
  - If the equivalence is "almost exact", as the abstract says, these should be small. Report them.
- **minADE6 shows a systematic residual.** Euler is worse than its matched temperature in all six Table 1 cells: +0.04, +0.06, +0.00, +0.01, +0.08 and +0.02 m. This one-sided residual is exactly the "genuine numerical effect" the paper says the protocol isolates. The draft passes over it in silence (the log says this explicitly in §6).
- **Flips: Alpamayo-2 Euler-10 is 9.7% vs 7.7%.** That is 26% relative excess. The abstract instead quotes the best cell (R1: 6.1 vs 6.1). The flip residuals also change sign across models (R1 Euler-4: 10.8 vs 12.9).
- **No CIs on any Table 1 cell** (TODO). The marginal clip-bootstrap CIs in the summaries are wide; for example, A1.5 Euler-4 ADE1 is [1.74, 2.68]. Paired residual CIs are needed. "Vanishes at matched temperature" needs an equivalence test (TOST) with a pre-stated margin, not a failure to reject.

### W5. The "decisions" are not the VLA's decisions, and the language branch is frozen.

- Alpamayo is a *reasoning* VLA, and its explicit decision lives largely in the reasoning trace / meta-action. The paper fixes the language branch per clip (Limitations; commitment manifest: "Language branch fixed by seed 123").
- It reads "decisions" from the trajectory with fixed, unreported thresholds (stop/decelerate/keep/accelerate, left/right/lane shift/straight). The title, "...Changes What They Decide", is therefore an overstatement. The paper measures threshold crossings of a trajectory conditioned on an unchanged decision text.
- Flips near a fixed threshold are sensitive to the threshold choice, and no sensitivity or margin analysis is given.
- In deployment, reasoning-sampling variance may dominate action-noise variance. Without the multi-seed language ablation (TODO, "not started" in the timeline), we do not know whether solver temperature matters at the policy level at all.

### W6. Theory: one restated known result, one proposition of doubtful applicability, and an unreproducible toy number.

- **One Euler step returns the conditional mean.** This is folklore for rectified flow / flow matching. It is the starting point of consistency, shortcut and mean-flow lines of work, and of the well-known blurriness of one-step deterministic samplers. The Gaussian contraction is, by the authors' own admission (log §5; plan), in the proof of Prop. 2 of Claustre et al. 2026. Restating an inequality "as a contraction" is not a contribution.
- **Prop. 3.1 (symmetry) is correct in spirit but has two problems.**
  - (a) It requires h·Lip(v)<1. For multimodal or near-degenerate targets, Lip(v(·,t)) grows like 1/(1-t) as t→1, so the condition fails in exactly the coarse-step regime the paper studies. With step h=0.1-0.25 the hypothesis is unlikely to hold near t=1.
  - (b) No driving decision is a reflection-hyperplane decision of a reflection-invariant conditional. Speed thresholds are not symmetric.

  It is a curiosity that explains why some toy decisions never flip. It is not a tool for the paper's setting. The proof itself is only a sketch (TODO).
- **The mixture claim ("99.95% of 408,769 flipped samples end closer to the global mean") cannot be traced to any artifact.**
  - `artifacts/mode_collapse_theory_v1_20261022/summary.json` contains 733,868 flips in total. The breakdown is Euler 289,245, Heun 172,947, RK 155,137 and cached 116,539. Euler plus cached is 405,784.
  - The saved `source_snapshot.py` does not compute any distance-to-mean quantity.
  - The toy also shows that Heun and RK4 flips go to the more probable mode about 70% of the time, so "flip toward the mean/majority" is not specific to first-order samplers.
  - The two-mode monotonicity "proposition" is explicitly TODO.

### W7. Generality is overstated: "four driving VLAs" is one model family plus one counterexample.

- **Alpamayo-1.5, R1 and 2 are one model family.** They share a vendor, the same action-expert design and the same 10-step uniform Euler sampler. Given the Gaussian theory, a common tau_eff ≈ 0.84 is *predicted* by an identical grid applied to fields with similar noise-to-signal ratios. Presenting "three generations share one effective temperature" as a finding is circular, not corroborating.
- **Qwen-Drive does not support the thesis.** Its accurate-solver spread is 0.037 m and it is insensitive to tau.
- **The Qwen "solver adds diversity" result depends on an author modification.** The accurate reference is Euler64 with a floor of min(0.1, 1/k), which differs from the released integration. The released floor prevents k>10 from reaching the endpoint (Adendo 10). So the "accurate solver" is a different sampler from the one the model was trained with, and the "spurious diversity" claim is entangled with that modification.
- **Qwen's navigation command is derived from the logged future heading** (Adendo 10). This leaks the future into the input.
- **The title and abstract generalize to "flow-matching driving policies"**, but there is no evidence for:
  - samplers other than RK-family (DPM-Solver-style multistep, stochastic samplers);
  - distilled, consistency or shortcut one-step policies, which are a major acceleration class;
  - any non-NVIDIA stochastic flow policy, such as pi0-style decoders in driving.

  Higher-order few-step solvers are explicitly *not* temperature-like (Heun-5). So "Your ODE Solver Is a Temperature Knob" is true only for first-order Euler and velocity caching. The title should say so.

### W8. Inconsistent data subsets and references make the paper's own numbers contradict each other.

- **Data section vs Table 1.** The Data section states 200 clips with six seeds. Table 1 uses clips 100-199 (100 clips) for Alpamayo-1.5 and 100-159 (60 clips) for R1 and Alpamayo-2. Finding 1 uses 200 clips × 2 seeds. Finding 3 uses 200 × 6. The safety analysis uses 99 + 59 + 59 clips. None of this is stated.
- **Reference solver.** The Setup says Heun32 is the reference. In the evidence, Finding 1 and §1 flips use Heun160, the commitment probe and the minADE_k analysis use Heun16, and Table 1 and the safety analysis use Heun32. Convergence of the reference is a TODO ("Heun-80 and Heun-160 agree to within TODO").
- **Contradictory ADE1 values for the same configurations** (Finding 5 vs Table 1 / artifacts):

  | Configuration | Finding 5 | Table 1 / `temperature_report_v1` |
  |---|---:|---:|
  | Euler-10 | 2.06 | 2.34 |
  | Converged flow | 2.20 | 2.465 (Heun32 tau=1) |
  | Euler-1 | 1.84 | 2.03 |

  These are different subsets, but a reader sees them as contradictions.
- **minADE6 gap.** The Euler-10 minADE6 "degradation" of +0.05 m in Finding 5 has a CI of [-0.023, +0.153] in the log, so it is not significant. On the Table 1 subset the gap is +0.146 m. The draft presents the former as reproducing FlashDrive's minADE6 degradation, without the CI.

### W9. Data hygiene and external validity.

- **Possible train/test contamination.** The log states: "We did not independently verify whether these clips were held out from training." Alpamayo checkpoints are trained on NVIDIA data closely related to PhysicalAI-AV. The paper does not mention this.
- **Single human future.** ADE/minADE to a single human future cannot distinguish "better" from "more mean-like". The authors acknowledge this, but it undermines every "accuracy gain" reinterpretation in both directions.

### W10. "Temperature" is a misnomer that hides the main conceptual subtlety.

- Scaling initial noise in a flow model is not sampling from p^{1/T}. For multimodal targets it pulls z toward the origin. The origin's preimage lies in whichever mode is "central" under the flow map, not necessarily the most probable one.
- So the paper "explains" one heuristic contraction (Euler) by another heuristic contraction (noise scaling) that has the same central bias. The draft should define precisely what is being matched. It should also discuss whether the recommended "explicit temperature" is itself a well-posed knob for decision distributions. The Glow-style reduced-temperature literature is the right precedent to cite here.

---

## 4. Questions for the authors

1. For the same noise z, what is the paired trajectory distance and decision disagreement between Euler-k(z) and Heun32(tau_eff · z)? How does it compare with the disagreement between Heun32(tau · z) and Heun32(tau' · z) for |tau - tau'| = 0.05?
2. Please provide clip-clustered tests for all safety comparisons. How many unique clips drive each discordant count? I count 5 clips for the 0:12 result and 1 clip for 0:2.
3. Why does Euler-1 (the conditional mean, tau = 0) collide more than Euler-2/4 and Heun32 tau=0.4? Is the collision rate monotone in tau below 0.4?
4. How do you reconcile the opposite signs of the stop-frequency shift under explicit temperature (up) and under Euler/cache (down)? Can you give per-class matched-temperature residuals with CIs?
5. What are the exact decision thresholds, and how do flip rates and the matched-temperature conclusions change under ±25% threshold perturbations?
6. Where does "99.95% of 408,769 flips" come from? Please release the script and artifact.
7. Is tau_eff stable per context? Please show the distribution of per-clip tau_eff and its dependence on the policy's entropy and on the spread of the context.
8. Does Prop. 3.1's condition h·Lip(v)<1 hold for Alpamayo's learned field at the released step size, near t=1?
9. With the reasoning trace sampled rather than fixed, what fraction of decision variance comes from language versus action noise? Does solver temperature still move the policy-level decision distribution?
10. For FlashDrive: how much of the reported minADE1 change is the checkpoint (W4A8 + streaming fine-tuning + DFlash) at matched tau, and how much is the cache?
11. Is the anisotropy predicted by the Gaussian analysis (contraction depends on per-direction s) visible in Alpamayo along principal directions? If so, in what sense is Euler "almost exactly" isotropic noise scaling?
12. Are the PhysicalAI-AV test clips disjoint from the Alpamayo training data?

---

## 5. Missing experiments / baselines I would require

1. **Closed loop with real power.** At least 150-200 scenes × 3 seeds in AlpaSim (the v2 plan), scene-clustered analysis, and the explicit-temperature arm using the official sampler parameter (Euler-10 · tau). Ideally on two Alpamayo generations.
2. **Official FlashDrive at matched tau_eff at scale** (Adendo 11, ≥60-100 clips × 4 windows × 6 seeds), separating checkpoint effects from cache effects. Without this, remove the FlashDrive reinterpretation from the abstract.
3. **Coda's own setup** (their policy, task suite and metric), comparing few-step against accurate + explicit temperature. The SmolVLA null should be reported in the main text, not hidden in the appendix.
4. **Paired per-noise equivalence test** (Q1), and TOST equivalence with pre-registered margins for every Table 1 cell.
5. **Language-branch ablation:** multiple reasoning samples per clip.
6. **Safety with a map** (drivable area, lane adherence), plus time-to-collision or near-miss rates as a denser secondary metric. Report the mode-averaging failure (Euler-1) explicitly.
7. **More acceleration families:** DPM-Solver/UniPC-style multistep, stochastic samplers, and at least one distilled / consistency / shortcut few-step policy. Show whether tau_eff is meaningful outside first-order RK.
8. **At least one non-NVIDIA stochastic flow driving policy** whose flow is genuinely stochastic, so that "generality" does not rest on one family.
9. **Reference convergence:** Heun32 vs Heun80/160 per metric (flips, ADE, spread). Use one reference throughout.
10. **Threshold sensitivity** for decision classes, and a decision definition read from the reasoning trace / meta-action.
11. **Contamination check** for the test clips.

---

## 6. Novelty assessment against prior work

- **One-step Euler returns the conditional mean.** This is not novel. It follows directly from the rectified-flow / flow-matching minimizer (Liu et al. 2023; Lipman et al. 2023). It is the stated motivation of the one-step / consistency / mean-velocity literature and of mode-averaging analyses such as GMFlow (Chen et al. 2025). The draft presents it as "the mechanism", which is fine, but it should be credited as known.
- **Euler contracts Gaussian samples on any grid.** Not novel, as the authors concede: it is in the proof of Prop. 2 of Claustre et al. 2026 (arXiv:2609.25839). The draft's sentence "we state it as a contraction because that is the property that reaches decisions" is honest. Still, the Contributions list should not count it.
- **Few-step samplers lose rare modes / collapse toward the mean.** This was shown empirically for images in SubFlow (arXiv:2604.12273) and GMFlow. The draft's extension is "decision-level" and "direction of flips". The direction claim is currently an unreproduced toy statistic plus a TODO proposition (W6).
- **Reduced temperature via noise scaling.** Well known: Glow-style reduced-temperature sampling, and the Alpamayo sampler itself documents tau<1. The genuinely new piece is the *operational* tau_eff matching protocol and its application to decide whether an acceleration "gain" is real. I consider this the paper's real contribution. It is a solid evaluation-methodology contribution. It is not, by itself, an ICML-level theoretical one.
- **FlashDrive (arXiv:2608.12932).** The draft implies that FlashDrive misattributes its minADE1/minADE6 pattern. That is overclaiming until the official stack is evaluated (W1). FlashDrive's fine-tuned, quantized checkpoint plausibly changes the field itself, not just the sampler.
- **Coda (arXiv:2609.21216).** The draft's notes say Coda "does not discuss temperature". If so, offering a mechanism is a fair contribution. But the claim that the Coda effect "disappears at matched temperature" is untested, and the authors' own closed-loop robotics data did not reproduce the effect.
- **Entropic time (Stancevic et al. 2025), speciation (Biroli et al. 2024; Raya & Ambrogioni 2023).** These are cited appropriately. The negative result on decision-information schedules is a real (if modest) contribution relative to them.
- **Citation hygiene.** Several bib entries carry "TODO" authors or verification notes: `alpamayo2025`, `coda2026`, `subflow2026`, `qwendrive2026`, `alpasim2026`, `physicalai2026`. For `alpamayo2025`, the title is the R1 report but it is cited for the Alpamayo family and the 10-step sampler. All of these must be fixed.

**Overall novelty: moderate.** The phenomenon is a natural consequence of known facts. The value lies in the measurement protocol, the driving-specific evidence and the reinterpretation of published claims, and the last of these is currently unsupported.

---

## 7. Numbers in the draft that do not match (or cannot be traced to) the artifacts

| # | Draft claim | Evidence | Status |
|---|---|---|---|
| 1 | Data: "200 clips ... six noise seeds per clip" (§4) | Table 1 uses 100 clips (A1.5, `temperature_report_v1`, `clips: 100`) and 60 clips (R1, A2, `clips: 60`); Finding 1 uses 200 × 2 seeds; safety uses 99/59/59 clips | Inconsistent / undisclosed |
| 2 | "408,769 flipped samples ... 99.95% closer to the global mean" (§3.3) | `mode_collapse_theory_v1_20261022/summary.json`: 733,868 total flips (Euler 289,245; Euler+cached 405,784); no distance-to-mean metric in the artifact or `source_snapshot.py` | Not reproducible |
| 3 | Finding 5: ADE1 Euler-1 1.84, Euler-10 2.06, converged 2.20 | Same configs in `temperature_report_v1`: 2.03 / 2.34 / 2.465 (Heun32 tau=1). The 1.84/2.06/2.20 numbers are from a different 200-clip run (log §5) with a different reference | Contradicts Table 1 without explanation |
| 4 | Finding 5: Euler-10 vs "converged flow" minADE_k gap −0.18 to +0.05 m | Log §5: vs **Heun16**, k=6 CI [−0.023, +0.153], so not significant; Table 1 subset gives +0.146 m | Reference mislabelled; CI omitted |
| 5 | Finding 3: lower temperature reduces stopping | Explicit temperature (Heun32, tau 1 to 0.4) *raises* stop: 3.7 to 4.2% (A1.5), 1.7 to 2.2% (R1), 1.7 to 2.5% (A2) | Sign contradiction |
| 6 | Finding 4 / contribution: "0:12, p<10^-3"; "10:0, p=0.002" | Counts reproduce exactly from `rows.json`, but the 12 discordant pairs come from 5 unique clips (6 from `context_127`); 0:2 comes from 1 clip | Counts correct; p-values invalid (clustering) |
| 7 | Setup: Heun32 is the accurate reference | Finding 1 and log §1 use Heun160; Finding 3 and minADE_k use Heun16 (`commitment_physicalai_v2` manifest: `reference: heun16`) | Inconsistent |
| 8 | Abstract: "6.1% vs 6.1%" as the example | Correct for R1, but Alpamayo-2 Euler-10 is 9.7 vs 7.7% and R1 Euler-4 is 10.8 vs 12.9% | Cherry-picked |
| 9 | Intro: FlashDrive cache at tau_eff ≈ 0.5 | Only the re-implemented cache (shrinkage ratio 0.50, 200 × 2) plus a 1-clip official smoke (`range [100,101]`) | Unverified at scale (TODO acknowledged) |
| 10 | Finding 3: "raises accelerating by 1.25 points" | Log CI [+0.1, +2.6]; not reported, unlike the other two effects | Minor omission |
| 11 | "Euler-4 halves collisions" | 0.69% vs 1.61% (57% reduction) | Fine, but see #6 |

**Verified as matching:**
- All Table 1 cells, after recomputing the interpolation.
- Finding 6 Qwen numbers: spread 0.037-0.038 m, Euler-10 spread 0.16 m, minADE6 1.42 vs 1.56 (RL).
- Finding 3 entropy and decelerate CIs, and the 0% vs 19% flip split with rho = 0.76 (log §2/§5).
- The schedule-matched tau_eff values (`schedule_matched_report_v1`).

---

## 8. Score and confidence

- **Overall score: 4 / 10** (reject in current form; below the acceptance threshold).
  - The open-loop phenomenon is real and well measured.
  - The paper's distinctive claims are TODOs or not supported at the stated strength: decision-level equivalence, safety, reinterpretation of FlashDrive/Coda, and generality.
  - With the missing experiments done and the claims scoped honestly, this could become a 6-7 evaluation-methodology paper.
- **Confidence: 4 / 5.** I checked the numbers directly against the authors' artifacts. My remaining uncertainty is about the cited 2026 works, which I could only assess through the authors' own descriptions.

---

## 9. What would change my score

1. **Closed-loop AlpaSim with adequate power** (at least 150 scenes × 3 seeds, scene-clustered tests) that shows Euler-4 ≈ Heun32 at tau_eff and ≠ Heun32 at tau=1: **+1 to +2**. A null or reversed result reported honestly, with the claims scoped down: neutral.
2. **Official FlashDrive at matched tau_eff at scale**, separating checkpoint from cache, either confirming or refuting the reinterpretation: **+1**. Otherwise, remove the claim from the abstract.
3. **Paired per-noise equivalence evidence** (Q1), TOST with CIs for Table 1, and an honest treatment of the minADE6 residual and the stop-sign contradiction: **+1**.
4. **Safety analysis redone** with clustering, a map / drivable area, and the Euler-1 non-monotonicity discussed. Remove p<10^-3 from the abstract.
5. **Rigorous mixture result** (or an honest conjecture), with a reproducible artifact for every toy statistic. Correct the Lip(v) applicability of Prop. 3.1.
6. **A language-branch ablation** showing that solver temperature still matters when the reasoning is sampled.
7. **One non-Alpamayo stochastic flow driving policy and one distilled/consistency acceleration.** Alternatively, retitle and scope to "first-order samplers in Alpamayo-family policies".
8. **Unify the data subsets and the reference solver**, and fix the bibliography.
