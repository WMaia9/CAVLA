# Review of cited papers and of the Qwen-Drive boundary claim

Date: 2026-10-26. Read-only review; no repo file was modified apart from this report.
Scratch analyses (numpy only, run on the existing `artifacts/qwendrive_probe_{rl,sft}_v1_20261022/plans.npz`) are in
`/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/` (`qspread.py`, `qbf16.py`, `qulp.py`).

---

## 1. Qwen-Drive in depth

### 1.1 What the planning head is (code and paper evidence)

| Item | Qwen-Drive-1.0 (release code + tech report arXiv:2609.00111) |
|---|---|
| Generative family | Flow matching on the linear path `tau_t=(1-t) tau_0 + t tau_1`, `tau_0 ~ N(0,I)` (paper Eq. 7; `planning_expert.py:15-21`). |
| Parameterization | **x-prediction (clean endpoint)**. The network outputs `tau1_hat`; velocity is `(tau1_hat - x_t)/max(1-t, 0.1)` (`planning_expert.py:330-377`, `:417-418`; `docs/model.md:85-94`). |
| Training loss | `L_fm` = squared error between induced velocity and `tau_1 - tau_0`, i.e. `||tau1_hat - tau_1||^2/(1-t)^2`. This is an x-prediction MSE (its minimizer is the posterior mean `E[tau_1 | x_t, c]`) weighted by `1/(1-t)^2`, plus Huber penalties on 1st and 2nd temporal differences, weights 2e-4 and 2e-5 (paper Eq. 8). |
| **Training time distribution** | **`t~ = Beta(1.5, 1.0)`, `t = min(t~, 0.9)`** (paper §2.1, "Planning Expert"). The network never sees `t > 0.9`. About 14.6% of training samples sit exactly at `t = 0.9`. |
| What it is conditioned on | Noisy waypoints (linear + Fourier features), flow time, ego-history poses, history velocity and acceleration, waypoint index (fused into the query, `planning_expert.py:289-291, 346-357`). AdaLN gets time + nav command + ego status (`:359-362`). It cross-attends to the VLM's KV cache (`:242-246`). Noise enters only through the waypoint tokens. |
| Trajectory normalization | **Fixed range scaling, not z-scoring**: x/165 m, y/25 m, heading/(pi/2) (`configuration_qwen_drive.py:119`, `trajectory.py:58-65`, `docs/model.md:70-77`). One unit of noise is 165 m longitudinally and 25 m laterally. |
| Default sampler | 10 uniform Euler steps, `noise_init_std=1.0`, `noise_seed=42` (sample k uses seed 42+k), `min_one_minus_t=0.1` (`configuration_qwen_drive.py:121-124`, `modeling_qwen_drive.py:178-195, 294-307`). No guidance and no clipping at inference. The paper's RL rollouts clip `tau1_hat` to [-1,1]; the release `sample()` does not. |
| Consequence of the floor | At step 9, `remaining = max(0.1, 0.1) = 0.1 = dt`, so the last update **lands exactly on `tau1_hat(x_0.9, t=0.9)`** (`planning_expert.py:417-418`). The release sampler is therefore 9 exact-form Euler steps on [0, 0.9] followed by one Tweedie-style jump to the denoiser's prediction at the training cap. |
| dtype | Expert in bf16. The output is `out_proj(...)` in bf16, then `.float()` (`planning_expert.py:377`). **All probe outputs lie on the bf16 grid in normalized units.** We checked this: 99.7-100% of xy values are bf16-representable after dividing by the scale. The x quantum averages **0.15 m** over the 50 waypoints and is 0.32 m beyond x = 41 m. |
| SFT objective | Stage 3: VLM frozen, expert trained with `L_plan` on mixed samples with and without reasoning text. It "reproduce[s] a single recorded future per scene" (paper §2.2). |
| RL objective | Stage 4: group-relative advantage (GRPO-style, no value function, Shao et al. 2024), discounted policy gradient (gamma = 0.6), G = 8 rollouts per scene. **The rollouts in a group share the initial noise.** Stochasticity is injected only on Euler transitions 7-9, with sigma = 0.03, in the span of the first 6 cosine modes, plus a restoring score drift (Eqs. 9-11). Rewards: PDMS/RFS plus displacement; displacement only for PAI-AV (App. A). No KL term is mentioned. RL is "reward-optimized only on reasoning-conditioned rollouts" (README; `docs/cookbook.md:29-31`). |
| Authors' own diversity remark | "the narrow gap between average ADE and minADE suggests that our candidates remain concentrated around similar motions" (paper §3.3, PAI-AV). The release table shows ADE3 − minADE3 = 0.04 m (`docs/evaluation.md:51-53`). |

### 1.2 Alpamayo action expert, point by point

| | Alpamayo (R1/1.5; `/mnt/datasets/Explainability/alpamayo/src/alpamayo_r1`) | Qwen-Drive |
|---|---|---|
| Output space | 64 × (accel, curvature) controls, dt 0.1 s, unicycle-integrated to a trajectory (`action_space/unicycle_accel_curvature.py:39-48, 98-100`) | 50 × (x, y, heading) positions at 10 Hz |
| Normalization | **z-score per channel** (accel_std 0.681, curvature_std 0.0261 from the HF config; `unicycle_accel_curvature.py:113-118`). The data has about unit scale, comparable to the noise. | Range scaling by 165/25 m. The data occupies a tiny fraction of the noise scale. |
| Prediction target | **velocity** (`action_out_proj` → "noise/vector field", `models/alpamayo_r1.py:318-321`) | clean endpoint (x-pred) |
| Sampler | `x=randn`, `linspace(0,1,k+1)`, `x += dt*v` (`diffusion/flow_matching.py:111-122`). The grid runs to t = 1. | Euler with x→v conversion and a floor; ends at the t = 0.9 prediction |
| Training t range | Not in the inference repo (no training code); presumably the full [0,1] | [0, 0.9] (Beta(1.5,1) capped) |
| RL | RL is on reasoning (per the R1 paper). The action expert's spread is ≈ 2.2 m across seeds. | RL on the expert with shared initial noise |
| Seed spread (our probes) | ≈ 2.2 m | 0.04 m (reference), 0.16 m (release) |

### 1.3 Why Qwen-Drive is nearly deterministic: verdict on each candidate cause

1. **Collapse by construction (main cause): the t ≤ 0.9 cap plus range normalization plus x-prediction MSE.**
   With range scaling, the conditional spread of the future given scene and history is a few metres, i.e. `s ≈ 0.005-0.02` in normalized x (and similar in y). The trained denoiser at the largest trained time sees the signal `0.9·tau_1` buried in residual noise `0.1·z`, whose std is 16.5 m in x and 2.5 m in y. The within-conditional signal-to-noise ratio `t·s/(1-t)` is ≈ 0.05-0.2 at t = 0.9 and smaller everywhere else in the training range. The MSE-optimal x-predictor is `E[tau_1|x_t,c]`. For a Gaussian conditional, its sensitivity to the noise is `t s²/((1-t)² + t² s²) ≈ t s²/(1-t)²`, which is about 3e-3 at t = 0.9. **Even a perfect network, sampled with the release sampler, returns essentially the conditional mean.** The output deviation is about `t s²/(1-t)`, i.e. ≈ 0.055·s_m² metres in x. For s_m ≈ 1.5-2 m this predicts ≈ 0.1-0.2 m, the size of the measured 0.16 m. The flow from t = 0.9 to 1, the only segment where the conditional spread would be resolved, is never trained. Qwen-Drive is a flow-matching model whose sampler is, in effect, **a denoised regressor onto the conditional mean.**
2. **Mean-regression loss.** Yes, but as part of item 1. The x-pred MSE targets the posterior mean at every t, and the cap stops the posterior from ever becoming sharp. The temporal Huber regularizers push further toward smooth, mean-like plans.
3. **RL sharpening: not the cause.** SFT is equally deterministic: reference spread 0.037 (SFT) vs 0.038 m (RL); E10 0.160 vs 0.161 m. RL shares the initial noise within each group (paper §2.2), so the reward never sees the noise-to-output map and cannot sharpen or broaden it.
4. **"Noise ignored by the network."** This is a symptom, not a cause. Ignoring the noise is the Bayes-optimal behaviour given item 1. The measured spreads sit at the bf16 output resolution. Under the reference, 74-76% of waypoints are bit-identical in x across seeds, and most of the rest differ by 1-2 bf16 ulps (`qulp.py`). The "0.04 m reference spread" is therefore mostly quantization-level jitter, not a distribution.
5. **Low noise scale: no.** `noise_init_std = 1.0`. Relative to the normalized data, the noise is very large, which is what drives item 1.
6. **Trajectory normalization: yes, essential to item 1.** Alpamayo z-scores its actions, so `s ≈ O(1)` and the Gaussian contraction factor is about 0.82-0.88. Qwen's range scaling makes `s ≪ 1-t` throughout the trained range.

### 1.4 Why the release 10-step sampler has MORE spread than our "reference"

Our probe output (60 clips × 6 seeds; `qspread.py`):

| sampler | last field eval at t = | residual noise 1−t | spread RL / SFT (m) | minADE6 RL |
|---|---|---|---|---|
| E1 | 0 | 1 | 0.055 / 0.047 | 1.67 |
| E2 | 0.5 | 0.5 | 0.045 / 0.043 | 1.62 |
| E4 | 0.75 | 0.25 | 0.083 / 0.077 | 1.53 |
| **E10 (release)** | **0.9 (= training cap)** | 0.1 | **0.161 / 0.160** | **1.42** |
| E64 ("reference") | 0.984 (**outside training range**) | 0.016 | 0.038 / 0.037 | 1.56 |
| E64, τ = 0.4 | 0.984 | | 0.030 / 0.029 | 1.56 |

With x-prediction and the floor, the output of E_k is `tau1_hat(x_{(k-1)/k}, (k-1)/k)`. Its noise sensitivity is ≈ `t/(1-t)·s²`, which grows with the time of the last evaluation. That is what we see for k = 2 → 4 → 10, after subtracting a ≈ 0.04 m floor: the spread roughly triples per step, matching the predicted ratios 1 : 3 : 9. Beyond t = 0.9 the ideal denoiser would keep becoming *more* sensitive: an ideal E64 would give spread on the order of the true conditional spread, i.e. metres. Instead the spread **collapses** to the quantization floor. That happens because E64 evaluates the network at six untrained times, t ∈ (0.906, 0.984], where the time embedding is extrapolated.

So the "released sampler adds spread" does not come from integration error. **The released sampler is the trained sampler. Our Euler-64 "reference" integrates into a time range the model never saw, so it is not an accurate solver of anything the model learned.** The release's extra spread is the expected Tweedie-step residual at t = 0.9. Its minADE6 advantage (1.42 vs 1.56) is real for the model as designed. Our docs read it as "numerical diversity" (RESULTADOS_DECISAO…:130); that interpretation should be dropped.

### 1.5 Is main.tex accurate about Qwen-Drive? Is our sampling valid?

- **Finding `find:qwen` numbers**: verified against the artifacts (0.037-0.038 m, 0.16 m, minADE6 1.42 vs 1.56 RL). The *interpretation* "the solver adds spread … the temperature view does not apply" is misleading. Under the model's own trained time range the release sampler is the accurate one. Our reference is out of distribution.
- **Reference-solver mismatch**: §2 Notation defines the reference as **Heun-32 (64 NFE)**. For Qwen-Drive we used **Euler-64 with floor 1/64** (`src/cavla/qwendrive_temperature_probe.py:83-88`; manifest `reference: euler64_t1.0`). Either state this or rerun.
- **τ scaling of the initial noise is valid in principle but uninformative here.** The output depends on z only through residual noise at the last evaluation, so τ ∈ [0.4, 1] has nothing measurable to scale once the reference sits beyond the cap. The informative test is the release grid or a fine Euler grid on [0, 0.9] plus the final jump at 0.9, run at τ·z. Theory predicts spread ∝ τ there, so the temperature picture may hold in a degenerate, near-mean regime.
- **Off-label use of planner-rl**: the probe runs **planner-rl in DIRECT_PLANNING** (`with_reasoning=False`, probe line 80). The release says planner-rl must be used in REASONING_PLANNING (`docs/cookbook.md:29-31`). main.tex says "SFT and RL planners" with no caveat. Only SFT is on-label.
- **No reasoning trace** in the Qwen runs, contrary to §4 Policies ("the reasoning trace is shared across samplers").
- **Seed labelling bug**: plans are keyed `s9201…s9206` and the manifest lists seeds 9201-9206. The noise actually comes from `model.config.noise_seed` = 42…47 (probe line 88). This is harmless for spread but wrong for provenance.
- **Nav command** is derived from the logged future heading (a future leak; the same for all configs, so spread comparisons are fine, ADE is not).
- **Possible train/test overlap**: Qwen-Drive trains on PAI-AV (paper §2.3.3) and states that the standard 644 split overlaps training data. Check which split file (`default` vs `alpamayo_aligned`) our clips come from.
- **"The released policies use forward Euler with 10 uniform steps [alpamayo2025, qwendrive2026]"**: true, but for Qwen add "x-prediction, with the last step landing on the t = 0.9 prediction".
- Contribution 4 ("the solver adds spread rather than removing it") should be rewritten.

Suggested replacement for the finding: *"Qwen-Drive is trained only for t ≤ 0.9 with range-normalized positions, so its sampler returns approximately the denoised conditional mean (seed spread 0.16 m, about one bf16 ulp; candidates 'concentrated around similar motions' per its authors). Fewer steps cannot lower a temperature that is already near zero, and integrating past t = 0.9 leaves the trained range. The temperature view is vacuous rather than violated."*

Cheap experiments that would settle it, in priority order:
- E_k on [0, 0.9] for k ∈ {3, 5, 10, 20, 40}, each followed by the jump at 0.9. Prediction: spread is nearly constant in k and proportional to τ.
- E10·τ.
- Rerun in fp32 to remove the bf16 floor.
- planner-rl in reasoning mode.

---

## 2. Bibliography check (every entry in references.bib)

| key | exists | metadata | main.tex claim | action |
|---|---|---|---|---|
| lipman2023flow | yes, ICLR 2023, arXiv 2210.02747 | correct | correct | – |
| liu2023rectified | yes, ICLR 2023, arXiv 2209.03003 | correct | correct (v(z,0)=E[x1\|c]−z holds for independent coupling) | – |
| black2024pi0 | yes, arXiv 2410.24164 | ok ("and others") | **not cited in main.tex** | Cite in Background: robot FM policies with 10-step Euler, or drop. |
| alpamayo2025 (TODO) | yes, **arXiv 2511.00088**, v1 30 Oct 2025, v2 7 Jan 2026 | Title is missing "**in the Long Tail**". Authors: Yan Wang, Wenjie Luo, Junjie Bai, … Boris Ivanovic, Marco Pavone (42, "NVIDIA:" prefix). | Ok. The paper notes R1 = "Alpamayo 1". | Replace (BibTeX below). No arXiv report exists for Alpamayo 1.5 or Alpamayo 2 Super; cite the HF model cards (1.5 released 19 Mar 2026 on Cosmos-Reason2; 2-Super weights 4 Aug 2026). |
| flashdrive2026 | yes, arXiv 2608.12932 (13 Aug 2026) | authors correct. **Venue "CoRL 2026" unverified**: arXiv has no venue, and no CoRL listing was found. | "reuses the velocity on steps 3–6 of an 8-step Euler schedule" matches the official code (`scripts/infer.py:39-40` `DIFFUSION_STEPS=8`, `CACHE_STEPS=[3,4,5,6]`; `flashdrive/diffusion.py:35-39`). The steps are **0-indexed** (4th-7th steps; fresh steps 0, 1, 2, 7). The paper only says "four of eight". | Change to @misc/arXiv unless acceptance is confirmed. Write "steps 3–6 (0-indexed)". |
| probeflow2026 | yes, arXiv 2603.17850 (v1 18 Mar 2026, v3 12 Oct 2026) | authors correct; no venue | "adapt the number of steps" correct (per-chunk step schedule via a lookahead consistency probe) | – |
| adavla2026 | yes, arXiv 2608.29208, **IROS 2026** | correct | correct: adaptive steps from trajectory curvature, plus MLP pruning; π0.5/X-VLA/SmolVLA | – |
| coda2026 | yes, arXiv 2609.21216 (18 Sep 2026) | authors correct; no venue | "Coda corrects a few-step endpoint": correct (trained residual corrector on a frozen few-step policy). Note the title, "Fewer Steps, Better Actions", is a direct target of our argument; it is a robotics paper. | Consider discussing explicitly. |
| qwendrive2026 (TODO) | yes, **arXiv 2609.00111** (31 Aug 2026) | Should be the tech report: Zhou, Zhao, Yang, Li, Zhong, Bai, Chu, Chen, Li, Tang, Wang, Yang, Zhang, Liu, Liang, Bai | see §1.5 | Replace (BibTeX below). |
| stancevic2025entropic | yes, arXiv 2504.13612, **NeurIPS 2025** poster (earlier ICLR'25 DeLTa workshop) | authors Stančević, Handke, Ambrogioni correct | "places steps by information about the output": fair (equal information per step) | – |
| claustre2026gaussian | yes, arXiv 2609.25839 (22 Sep 2026); 5 authors, matches | correct | "implicit in the proof of Proposition 2" is **correct**. App. A.7 shows `y_k ≤ φ(t_k)` for convex φ with stable steps, and our `‖u(t)‖=√((1-t)²+t²s²)` is convex. Their setting is centered commuting Gaussians, and their "straight" (geodesic) path is integrated *exactly* by Euler. | Say "for the independent-coupling path, whose scale is convex", to avoid implying Euler contracts on every Gaussian path. |
| subflow2026 | yes, arXiv 2604.12273 | correct | correct (diversity degradation of few-step FM; "averaging distortion" from MSE) | – |
| chen2025gmflow | yes, ICML 2025, arXiv 2504.05304 | correct | Loose. GMFlow attributes few-step error to discretization and motivates a Gaussian-mixture velocity distribution, i.e. it implicitly blames the mean/single-Gaussian prediction. | Rephrase: "attributed to predicting only the mean (a single Gaussian) of the velocity distribution". |
| biroli2024dynamical | yes, Nat. Commun. 2024, arXiv 2402.18491 | correct | "speciation" correct | add volume 15, art. 9957 |
| raya2023ssb | yes, NeurIPS 2023 | correct | correct | – |
| dauner2024navsim | yes, NeurIPS 2024 D&B | correct | not cited in main.tex (only in a TODO) | ok to keep |
| alpasim2026 (TODO) | GitHub software citation, **2025 (October)**, title correct, "NVIDIA" + 16 named contributors (incl. Yulong Cao, Maximilian Igl, Michael Watson); url github.com/NVlabs/alpasim. No arXiv paper found. | year wrong | not cited yet | Rename to alpasim2025 and copy the repo's BibTeX verbatim. |
| physicalai2026 (TODO) | HF dataset; **initial release v25.10 (Oct 2025)**; the card has no BibTeX. Qwen-Drive cites it as "NVIDIA (2025b), Physical AI autonomous vehicles dataset". | **year should be 2025** | "test split" ok: two split files exist (`default` test 61,599 clips; `alpamayo_aligned` test 28,317) | State which split file. |

Main-text check of our own closed form: "For s_d ≥ 0.3 the factor of E10 lies between 0.82 and 0.88" is **wrong as stated**. The factor is symmetric, f(s) = f(1/s): it is 0.879 at s = 1, 0.862 at s = 2, and 0.617 at s = 10. Write **"for 0.3 ≤ s_d ≤ 3.3"**. The Fig. 2 values for s = 0.5 (0, 0.40, 0.68, 0.86) check out.

Replacement BibTeX:
```bibtex
@article{alpamayo2025,
  title  = {{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
  author = {{NVIDIA} and Wang, Yan and Luo, Wenjie and Bai, Junjie and Cao, Yulong and others and Ivanovic, Boris and Pavone, Marco},
  journal= {arXiv preprint arXiv:2511.00088}, year = {2025}}
@article{qwendrive2026,
  title  = {{Qwen-Drive-1.0}: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving},
  author = {Zhou, Xin and Zhao, Zongchuang and Yang, Zhibo and Li, Mingsheng and Zhong, Humen and Bai, Shuai and Chu, Du and Chen, Ruizhe and Li, Zhaohai and Tang, Jun and Wang, Qiuyue and Yang, Mingkun and Zhang, Jiazhao and Liu, Dayiheng and Liang, Dingkang and Bai, Xiang},
  journal= {arXiv preprint arXiv:2609.00111}, year = {2026}}
@misc{physicalai2025,
  title = {{PhysicalAI-Autonomous-Vehicles} Dataset}, author = {{NVIDIA}}, year = {2025},
  howpublished = {\url{https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles}}, note = {Version 25.10}}
@article{flashdrive2026, ... journal = {arXiv preprint arXiv:2608.12932}, year = {2026}}  % unless CoRL acceptance is confirmed
```
Also add HF model-card @misc entries for Alpamayo-1.5-10B and Alpamayo2-Super, since the paper uses three generations and only R1 has a report.

No entry is fabricated. All 17 exist. Corrections: alpamayo2025 title, FlashDrive venue, two years (PhysicalAI, AlpaSim), the qwendrive tech report, and the GMFlow phrasing.

---

## 3. Missing related work (verify final metadata before adding)

**Temperature, truncation and noise scaling**
- Kingma & Dhariwal, *Glow*, NeurIPS 2018: reduced-temperature sampling of flows, the direct ancestor of our τ·z.
- Brock et al., *BigGAN*, ICLR 2019: truncation trick, a fidelity/diversity trade-off from shrinking the latent.
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, 2022.
- Sadat et al., *CADS*, ICLR 2024: diversity loss from guidance.
- Kynkäänniemi et al., *Applying guidance in a limited interval…*, NeurIPS 2024.

**Few-step / discretization bias in variance**
- Song et al., *DDIM*, ICLR 2021: deterministic few-step sampling.
- Bao et al., *Analytic-DPM*, ICLR 2022: optimal reverse variance; few-step variance bias.
- Karras et al., *EDM*, NeurIPS 2022: Euler vs Heun and S_noise; essential since our reference is Heun.
- **Ning et al., *Elucidating the Exposure Bias in Diffusion Models*, ICLR 2024**: epsilon scaling shrinks the prediction norm, a closely related "scale-like" fix.
- Song et al., *Score SDE*, ICLR 2021: final Tweedie denoising step, which is exactly Qwen's floor-jump at t = 0.9.
- Lu et al., *DPM-Solver*, NeurIPS 2022.
- Few-step generators: Frans et al., *Shortcut models* (ICLR 2025); Geng et al., *MeanFlow* (2025).
- Li & He, *Back to Basics: Let Denoising Generative Models Denoise* (x-prediction, arXiv 2511.13720, 2025). Relevant to Qwen's parameterization.

**Mode collapse and diversity in driving and robot planners**
- Liao et al., *DiffusionDrive*, CVPR 2025 (arXiv 2411.15139): argues vanilla diffusion policies collapse modes in driving; truncated 2-step diffusion from anchors.
- Xing et al., *GoalFlow*, CVPR 2025: goal-conditioned flow matching for multimodal trajectories.
- Chi et al., *Diffusion Policy*, RSS 2023.
- Jiang et al., *MotionDiffuser*, CVPR 2023.
- Yang et al., *Diffusion-ES*, CVPR 2024.
- ReCogDrive, already in a TODO.

**RL on flow policies and diversity**
- Liu et al., *Flow-GRPO*, NeurIPS 2025 (arXiv 2505.05470): ODE→SDE for RL, which Qwen's Stage 4 mirrors.
- Yue et al., *Does RL Really Incentivize Reasoning…?*, NeurIPS 2025 (arXiv 2504.13837): RL improves pass@1 but shrinks pass@k, the analogue of ADE1 vs minADE6.

---

## Summary (≤200 words)

**Qwen-Drive verdict.** The planner is flow matching with x-prediction. It is trained only on t ≤ 0.9 (Beta(1.5,1) capped; paper §2.1), with positions divided by 165/25 m. The 10-step sampler's floor makes it land on the t = 0.9 denoiser prediction (`planning_expert.py:417-418`). At that noise level the residual noise (16.5 m in x) swamps the conditional spread (metres), so even a perfect network returns ≈ the conditional mean. SFT is equally collapsed, so RL is not the cause; RL also shares the noise within each group. Measured spreads are at bf16 output resolution. Our Euler-64 "reference" evaluates untrained times t > 0.9, which is why its spread collapses. The release sampler does not "add spread"; our reference is out of distribution. The boundary claim should be reframed as "the temperature is already ≈ 0, so the view is vacuous". Also: planner-rl was run off-label in direct mode, the reference is Euler-64 rather than the stated Heun-32, and seeds are mislabelled.

**Citations.** None fabricated. Fix: alpamayo2025 (title, arXiv 2511.00088), qwendrive2026 (arXiv 2609.00111), physicalai and AlpaSim years (2025), FlashDrive venue (arXiv only), GMFlow phrasing. Also, "s_d ≥ 0.3" should read 0.3–3.3.

**Top missing.** EDM, Analytic-DPM, exposure bias (Ning 2024), Glow/BigGAN temperature, DiffusionDrive, Flow-GRPO, Yue et al. 2025.
