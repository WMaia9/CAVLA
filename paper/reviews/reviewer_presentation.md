# Presentation Review: "Your ODE Solver Is a Temperature Knob"

Reviewer focus: writing, visual design, results reporting. Benchmarked against papers known for clarity: DDPM, EDM (Karras et al.), Scaling Laws (Kaplan et al.), Rethinking Generalization (Zhang et al.), and Emergent Abilities Mirage (Schaeffer et al.).

Draft reviewed: `paper/main.tex` (209 lines) and `paper/main.pdf` (6 pages, of which 4.6 are main body, with no figures yet).

**Overall.** The paper has one sharp, falsifiable idea, "cutting steps = lowering temperature," and a natural unit of evidence: the matched-temperature comparison. That is the profile of papers that win on clarity. The draft does not yet *show* the idea. There is no plot, the one table hides the baseline that makes the effect visible, and terms for the reference solver and for the samplers vary from paragraph to paragraph. The Mirage paper is the closest model. It has one reframing claim, one main device (change the metric and the effect disappears), and every figure is a before/after pair that shows that disappearance. Here the device is "compare at matched τ_eff and the effect disappears." Every main figure should show that disappearance visually.

---

## 1. MUST-FIX

### M1. Build: wrong fonts, no bold anywhere, suppressed running head
- The PDF was built with Tectonic (XeTeX). `pdffonts` shows Latin Modern CID fonts. The log reports `Font shape TU/ptm/b/n undefined`, so **every `\textbf`, section heading, `\paragraph` heading, and the "Finding" labels render in regular weight**, as pages 1–4 show. ICML requires Times (the `.sty` loads `times`) and Type-1 fonts, so this also risks desk-reject formatting checks.
- The running head reads **"Title Suppressed Due to Excessive Size"**. The `.sty` check (`\ht\titrun>6.25pt` or `\wd>\textwidth`) fails under the substituted font.
- **Fix.** Build with `pdflatex` + `bibtex` (`latexmk -pdf main.tex`). Then check that `pdffonts main.pdf` lists only Type 1 fonts (NimbusRomNo9L/Times) and that the running head shows. If the head is still suppressed, shorten it with `\icmltitlerunning{ODE Solvers as Temperature Knobs}`.
- The title contains `\\`, which triggers a hyperref warning. Use `\icmltitle{Your ODE Solver Is a Temperature Knob:\texorpdfstring{\\}{ }How Accelerating ...}`.

### M2. The abstract is two paragraphs, about 7 sentences, and has a cryptic number
The inline TODO splits the abstract into two paragraphs. ICML asks for one paragraph of 4–6 sentences. "6.1% vs. 6.1% flipped decisions" means nothing to a reader who has not seen Table 1. The abstract also says "every Euler-type sampler we tested pulled *all* of its samples toward that mean across three generations", but Finding 1 only shows this for Alpamayo-1.5 (the R1/A2 versions are TODO). The abstract must not outrun the evidence. Replacement (6 sentences, with the closed-loop sentence as an optional slot):

> Flow-matching action decoders make vision-language-action (VLA) driving policies slow at inference, and recent accelerations cut integration steps, cache velocities, or adapt step schedules while reporting near-unchanged trajectory error. We show that, for first-order samplers, these accelerations are not neutral approximations: they act almost exactly as a lower sampling temperature, because one Euler step returns the conditional mean and Euler on any grid contracts samples toward it. We define a sampler's *effective temperature* $\tau_{\mathrm{eff}}$ by matching its seed spread to an accurate solver with scaled initial noise, and find that the released 10-step samplers of three generations of NVIDIA's Alpamayo policies all run at $\tau_{\mathrm{eff}}\approx0.84$ on real PhysicalAI-AV driving data. An accurate solver run at the matched temperature reproduces these samplers' decision changes (6.1% of Alpamayo-R1 decisions flip under both), their reduction in log-replay collisions, and the pattern in which single-sample error improves while best-of-six error degrades, previously attributed to fine-tuning. [Closed-loop: *In AlpaSim, ... at matched temperature.*] Accelerated flow policies should therefore report $\tau_{\mathrm{eff}}$ and be compared at matched effective temperature.

If the closed-loop sentence is added, merge sentences 1 and 2 to stay within 6.

### M3. One name for each object, defined once and used everywhere
The text currently uses several names for the same thing:

| Concept | Names in the draft | Use |
|---|---|---|
| reference solver at τ=1 | "converged flow", "exact flow", "accurate solver", "Heun with 32 steps", "Heun32", $A_1$, "τ=1" | **"the reference solver" $A_1$** (Heun-32, 64 NFE), and "reference at temperature τ" $A_\tau$. Keep "exact flow" for the mathematical object in §3 only |
| cache method | "FlashDrive-style caching", "FlashDrive's cached Euler", "velocity cache of FlashDrive", "FlashDrive-style cache" | **"FD-cache"** (our reimplementation), defined once in §4. Say "FlashDrive" only for the published system |
| sampler family | "first-order samplers", "Euler-type", "Euler-type sample", "cheaper samplers" | **"first-order samplers"** (Euler-$k$ and FD-cache), defined in §4 |
| single-sample error | "single-sample ADE", "ADE$_1$", "minADE$_1$", "single-sample minADE" | **ADE** (one sample) in our results; write minADE$_1$ only when quoting FlashDrive, and say once that they are the same |
| contraction | "contraction", "shrinkage ratio", "distance ratio", "independent shrinkage measurement" | **contraction ratio $r$** = $\|S(z)-m\|/\|A_1(z)-m\|$, as a displayed equation |
| model | "Alpamayo-2-Super" (§4) vs "Alpamayo-2" | choose one; give the full name in §4 and the short name after |
| step count | "Euler-8" (Finding 1) is not in the sampler list $k\in\{1,2,4,6,10\}$ | add 8 to the list or remove it from Finding 1. Same for Heun-5 and RK4 from the results doc |
| reference for flips | Heun-32 (§4) vs Heun160 (results doc) | state one reference and the convergence check |

Add a short "Notation" paragraph at the end of §2 (3 lines) and a `\newcommand` for each term (`\Aref`, `\Atau{}`, `\euler{k}`, `\fdcache`) so the terms cannot drift.

### M4. Numbers that contradict each other on the page
- Finding 5 gives Euler-10 ADE = 2.06 m and converged = 2.20 m. Table 1 gives Alpamayo-1.5 Euler-10 ADE = 2.34 m. A reader will assume one of them is wrong. If they come from different subsets (2-seed vs 6-seed, 200 vs 436 clips), each number needs its $n$ stated where it appears, and ideally the paper should use one evaluation set throughout the main text.
- Intro: "FlashDrive ... at about 0.5" is a **contraction ratio** (Finding 1: 0.50), not a τ_eff. The paper's own point is that these two quantities are different and only agree empirically, so do not mix them.
- Contributions say "Euler-4 **halves** collisions ... (0:12 discordant **scenes**)". Finding 4 says "clip–seed **units**" and never says "halves" with matching rates. Use one unit and give the rates with CIs.
- Finding 4: "lowering τ ... reduces collisions from 1.61% to 0.84%". Lowering τ *to what*? State the value.
- Hook: "brakes in 20% ... 17%". This must be a real number from the paper, or the hook reads as invented (see S1).

### M5. Table 1 hides the effect it is supposed to explain
Each cell shows only sampler / matched-reference. The reader cannot see the size of the effect being explained, because the τ=1 reference value is missing. "2.19 / 2.19" is only impressive when set beside $A_1$ = 2.4x. The "Flips" column has no reference ("flips relative to what?"), and its % sign is attached only to the second number ("10.8 / 9.1%"). Redesign (table*, booktabs, see §4 below for the full template):

- Add an $A_1$ row per model (the baseline the sampler moves away from).
- For each metric, show **Δ vs $A_1$** (the sampler's effect) and **Δ vs $A_{\tau_\mathrm{eff}}$** (the residual), each with a 95% CI.
- Add a final column, **"explained by temperature"** = $1-|\Delta_{\text{matched}}|/|\Delta_{A_1}|$, with CI. This gives a headline number like "~94% of flips explained", which the outline already has but the paper never states.
- Bold a residual only when its CI excludes 0. That marks the genuine numerical effects, which is what a reader should look for.

### M6. No displayed equations for the paper's two key objects
EDM and DDPM put their central definitions in numbered display equations that readers cite. Here τ_eff (Def. 3.2) and the Euler contraction product are inline. Display them:

```latex
\begin{equation}
  r_{\mathrm{Euler}} \;=\; \frac{\|x^{\mathrm{Euler}}_1 - m\|}{\|x^{\mathrm{exact}}_1 - m\|} \;=\; \prod_{i=0}^{k-1}\cos\theta_i \;\le\; 1
  \label{eq:contraction}
\end{equation}
\begin{equation}
  \tau_{\mathrm{eff}}(S) \;=\; \tau \ \text{ such that }\ \operatorname{Spr}(A_\tau)=\operatorname{Spr}(S),
  \qquad \operatorname{Spr}(S)=\E_{c}\,\E_{z\neq z'}\|S(z;c)-S(z';c)\|
  \label{eq:taueff}
\end{equation}
```

Rename the spread $\sigma(S)$ to $\operatorname{Spr}(S)$ (or $D(S)$). §3.2 already uses $s$ for the Gaussian standard deviation, and $\sigma$ will be read as a standard deviation. Also define $u(t)$ and $\phi_t$ in §3.2. As written, "$\phi_t=\|u(t)\|$ with $u(t)=(1-t,ts)$" is opaque. Say that $x_t-m=(1-t)z+ts\epsilon$, so the per-coordinate standard deviation is $\|u(t)\|$.

### M7. Statistical reporting is uneven
Award-level reporting gives an effect size, a 95% CI, the unit of resampling, $n$, and the named test for every headline number. Current issues:
- Table 1 has no CIs (there is a TODO). Use a cluster bootstrap over clips (seeds nested within clips), 10k resamples, and state it once in §4 "Statistics".
- Finding 1: "400/400". Add the exact binomial lower bound (≥99.1%, 95%) and say what a unit is (clip × seed).
- Finding 3 gives CIs for stop/decelerate/entropy but not for "accelerating +1.25 points". It also does not give base rates. Write "decelerate: 21.3% → 18.8% (−2.5 pp [−3.8, −1.2])". Use **pp** for percentage points consistently.
- Finding 4: name the test ("exact McNemar test on $b{:}c$ discordant pairs"), explain the `b:c` notation once, and give the rates with CIs, not only discordant counts.
- Spearman ρ=0.76: say what the unit is (clips, $n=200$) and what the variables are (decision entropy across seeds vs. flip rate).
- p-value format: use either "$p<10^{-3}$" or "$p=0.002$", and use exact values throughout.

### M8. Bibliography placeholders render in the PDF
"TODO (2026c;a)" appears in the running text five times (ProbeFlow, AdaVLA, Coda, SubFlow). `alpamayo2025` has a TODO note. Fill in the authors before any internal circulation. A reviewer who sees "TODO (2026b)" in a citation stops trusting the numbers. In Related Work, `\citet{flashdrive2026,probeflow2026,...}` produces "Li et al. (2026); TODO (2026c;a)" as the subject of a sentence. Write the names instead: "FlashDrive~\citep{}, ProbeFlow~\citep{}, AdaVLA~\citep{}, and Coda~\citep{}". "Real-time chunking methods" has no citation.

### M9. The title overclaims relative to the paper's own result
"Your ODE Solver Is a Temperature Knob" is memorable, so keep its structure. But Finding 1 shows that second- and fourth-order solvers do **not** contract (ratios 1.00, 1.05), and Qwen-Drive is a counterexample. A reviewer will cite this against the title. Options, in order of preference:
1. **Fewer Steps, Lower Temperature: Accelerating Flow-Matching Driving Policies Changes What They Decide**
2. **Your Euler Sampler Is a Temperature Knob: How Accelerating Flow-Matching Driving Policies Changes What They Decide**
3. Keep the current title and put "first-order" in the first sentence of the abstract and in the thesis sentence (already partly done).

The subtitle is good: "changes what they decide" is the driving-specific hook and matches the lab focus.

---

## 2. SHOULD-FIX

### S1. Opening hook and thesis sentence
The current hook is a good *shape*: concrete, about driving, about decisions. But it uses hypothetical numbers. Replace it with the paper's own measured fact, in the style of Zhang et al. ("we show that ... networks easily fit random labels"):

> Replacing the released 10-step sampler of Alpamayo-1.5 with a velocity cache of the kind FlashDrive reports as near-lossless makes the policy's single-sample trajectory error *better*, from 2.06 to 1.84\,m. It also makes the policy decelerate in 2.5 fewer of every 100 scenes and stop less often, and it makes its decisions 0.24 bits more certain. Trajectory error says the cheaper policy is the better one; its decision distribution says it is a different, less cautious one.

Thesis sentence (place it as the first sentence of paragraph 2, and in italics or as a one-line display if you want an EDM-style "key claim"):

> *For first-order samplers, cutting integration steps is, to within measurement error, the same as lowering the sampling temperature.*

Then give the mechanism in two sentences, and immediately give the protocol: "This suggests a simple control. Compare every accelerated sampler with an accurate solver at the same effective temperature." The current paragraph 3 ("Seeing samplers as temperatures resolves...") is the Mirage-style payoff. Keep it, but list the three puzzles as an enumerated (i)/(ii)/(iii) with forward references to Figs. 3–5.

Use one spelling convention. The draft uses British "centimetres", "manoeuvres" while ICML and the rest of the text are US. Use "centimeters" and "maneuvers".

### S2. Contribution list: fewer, sharper, every bullet with a number and a pointer
There are five bullets, two contain TODOs, and "four driving VLAs" conflicts with "three generations" in the abstract. Award papers use 3–4 bullets, each one sentence plus one number. Proposed:

- **Mechanism and theory (§3).** One Euler step returns the conditional mean; Euler contracts Gaussian samples toward it on any grid (Eq. 1); for reflection-symmetric decisions no explicit Runge–Kutta step flips the decision (Prop. 3.1), and in solvable mixtures 99.95% of numerical flips move toward the mean.
- **A protocol: effective temperature (§3.4).** τ_eff, measured from seed spread alone, agrees with an independent contraction measurement (0.84 vs 0.84) and lets any sampler be compared with an accurate solver at matched temperature.
- **Evidence on real driving data (§5).** On PhysicalAI-AV, the released samplers of Alpamayo-1.5, -R1, and -2 share τ_eff≈0.84, and matched temperature reproduces their accuracy, decision flips, and log-replay collision changes (Table 1, Figs. 3–4). Qwen-Drive, whose flow is nearly deterministic, is a counterexample in which the solver *adds* diversity.
- **Reinterpreting published gains (§5).** The minADE$_1$-better / minADE$_6$-worse pattern reported for FlashDrive and the "fewer steps, better actions" effect follow from temperature alone (Fig. 5).

Move "Safety consequences" into bullet 3. Do not list a result as a contribution while it is a TODO. If the closed-loop result lands, it becomes part of bullet 3.

### S3. "Finding" boxes: keep the device, tighten it
The device works. It is the paper's equivalent of the boxed "observations" in Scaling Laws §1.1. But:
1. **Six is too many** for 8 pages, and Findings 1 and 2 both say "contraction/τ_eff is shared". Merge them into **"Every first-order sampler contracts, and the released samplers share τ_eff≈0.84."** Target 4–5 findings.
2. **Each finding must carry its key number and point to exactly one figure or table**, e.g. "Finding 3. *Lowering temperature removes cautious decisions: FD-cache cuts decelerations by 2.5 pp (Fig. 4).*" A reader skimming only the boxes, which is what reviewers do, should get the paper.
3. **Repeat the findings in the intro.** Either replace the contributions with "Our findings in brief" or add `\cref{find:...}` labels (add `\label` support; the counter already uses `\refstepcounter`) so contributions can point to "Finding 2".
4. **Visual weight.** A full `\fbox` around italic text is heavy and looks like a TODO box. Use a light left rule instead:
   ```latex
   \usepackage[most]{tcolorbox}
   \newtcolorbox{findingbox}{enhanced,breakable,boxrule=0pt,frame hidden,
     borderline west={1.5pt}{0pt}{black!70},colback=black!4,
     left=4pt,right=4pt,top=2pt,bottom=2pt,before skip=6pt,after skip=4pt}
   \newcommand{\finding}[1]{\refstepcounter{finding}\begin{findingbox}
     \textbf{Finding \thefinding.}~#1\end{findingbox}}
   ```
   Use upright text (italic sentences of 15+ words are tiring), with only the key number in bold.
5. Finding titles should be causal only where the evidence is causal. Finding 6's "this spurious diversity *is what improves* its best-of-six error" should become "coincides with" or "accounts for" if you have the matched control, and Finding 4's "Temperature *explains* the safety effect" should become "Temperature accounts for the log-replay collision reduction".

### S4. Section flow and 8-page budget
The body is currently ~4.6 pages with no figures, so there is room. Proposed budget (double-column ICML pages):

| Section | Pages | Contents |
|---|---|---|
| 1 Introduction | 1.25 | hook, thesis, mechanism, 3 puzzles, 4 contributions, **Fig. 1** (single column, page 1–2) |
| 2 Background and Notation | 0.5 | flow-matching decoder, accelerations, explicit τ; notation paragraph |
| 3 Samplers as Temperatures | 1.25 | 3.1 one step = mean; 3.2 Gaussian contraction (Eq. 1); 3.3 symmetry and mixtures (Prop. 3.1); 3.4 τ_eff (Eq. 2); **Fig. 2** (single column) |
| 4 Experimental Setup | 0.6 | policies, data, samplers (with NFE and latency), metrics, **statistics** paragraph |
| 5 Results | 3.0 | 4–5 findings; **Fig. 3** (figure*, ~1/3 page), **Table 1** (table*), **Fig. 4**, **Fig. 5** (single column each), optional Table 2 |
| 6 Related Work | 0.45 | three paragraphs, as now; or move it to after the intro if reviewers need the positioning against FlashDrive/Coda early. Two sentences of positioning in the intro are enough |
| 7 Discussion and Limitations | 0.55 | recommendations as a 3-item checklist; limitations as a paragraph; one sentence on when the effect does not apply (higher-order solvers, near-deterministic flows) |
| **Total** | **7.6** | 0.4 page slack for float placement |

Other flow fixes:
- The heading of §3.3 "Mixtures: decisions flip toward the mean" announces a result, but the proposition beneath it is about symmetry *preventing* flips, and the toward-the-mean claim is empirical or conjectural. Retitle it "Mixtures: Which Decisions Can Flip" and label the toward-mean statement as an empirical observation (or a Conjecture environment) until it is proved.
- Move the Qwen-Drive counterexample to the end of §5 under a short heading "When the Mechanism Does Not Apply". Counterexamples presented as a boundary are a strength. Mirage and Rethinking Generalization both make the boundary explicit.
- The appendix "negative result: decision-information schedules" is the strongest *motivation* for the protocol. Give it one sentence plus a number in §5 (not only in Related Work): "A schedule that passed a pre-registered confirmation (−0.88 pp flips [−1.28, −0.52], 436 clips) shows no gain at matched temperature (App. B)." Pre-registration is rare and earns trust, so show it.

### S5. Heading capitalization (ICML: content words capitalized at all levels)
The template states that subsection headings capitalize content words. Fix:
- `One Euler step returns the conditional mean` → `One Euler Step Returns the Conditional Mean`
- `Gaussian targets: contraction on any grid` → `Gaussian Targets: Contraction on Any Grid`
- `Mixtures: decisions flip toward the mean` → `Mixtures: Which Decisions Can Flip`
- `Effective temperature` → `Effective Temperature`
- Section 2 is too long for a heading and wraps to two lines. Use `Background and Notation`.
- `\paragraph{}` run-in heads may stay in sentence case, but use one style throughout.

### S6. Setup section: add what readers need to read every number
- **Samplers paragraph**: give a small table or list with **NFE and measured latency (ms)** for each sampler. The whole motivation is speed, and readers will want τ_eff vs. speedup. This could be the x-axis of a panel in Fig. 3.
- **Flip definition**: "a flip is a (clip, seed) unit whose decision class under sampler $S$ differs from that of $A_1$ from the *same* initial noise $z$." Without this, "6.1%" has no meaning.
- **Statistics paragraph**: cluster bootstrap over clips, 95% CIs, exact McNemar for paired binary outcomes, and the $n$ for each analysis (200×6, 436×6, 1,302 units).
- State whether the 436 confirmation clips are used for any main-text number or only for App. B.

### S7. Section 3.2 needs one sentence of intuition before the algebra
EDM's style: say what the math will show, then show it. Add before the algebra: "For a Gaussian target the exact flow rotates the noise into the data direction; an Euler step moves along the chord of that arc instead, and every chord is shorter than its arc." That gives the geometric picture of $\prod\cos\theta_i$. It could be an inset in Fig. 2.

---

## 3. FIGURE AND TABLE PLAN

Global conventions (use them in every figure and state them once in the Fig. 1 caption):
- **Color (Okabe–Ito, colorblind-safe):** reference/explicit-temperature curve = **black line with a gray 95% band**; Euler-$k$ = a **single blue hue, darker = fewer steps** (Euler-1 darkest; the ordinal reading is "more contraction"); FD-cache = **vermillion (#D55E00)**; higher-order solvers (Heun-5, RK4) = **bluish green (#009E73)**; human log = dashed dark gray.
- **Markers:** filled marker = the sampler; hollow marker of the same color = $A_{\tau_\mathrm{eff}(S)}$ (its matched-temperature twin). A thin connector between them visually shows "the residual".
- **Models:** circle = Alpamayo-1.5, square = R1, triangle = Alpamayo-2; Qwen-Drive = diamond (gray, appendix or inset).
- **CIs:** 95% cluster-bootstrap over clips. Error bars on markers, a shaded band on curves. State $n$ (clips × seeds) in every caption.
- **Sizes:** single-column figures at 3.25 in wide, figure* at 6.75 in. 8-pt fonts in plots (matching 9-pt captions), no titles inside panels (the caption carries the claim), panel letters (a)(b)(c) in bold at top-left.
- **Captions:** start with the bold claim, then one sentence on how to read the plot, then $n$/CI. Example: "**Every first-order sampler lies on the temperature curve.** Black: reference solver with initial noise ×τ; markers: samplers placed at their measured τ_eff. Higher-order solvers (green) fall off the curve. 200 clips × 6 seeds; bars are 95% cluster-bootstrap CIs."
- Export as vector PDF from matplotlib (`pdf.fonttype=42`), with Times/Nimbus in the plots to match the text.

### Figure 1 (teaser), single column, page 1 right column or top of page 2
**Claim: Fewer solver steps behave like a lower sampling temperature, and that changes decisions.**
- **(a) Left, ~55% width: BEV trajectory plot of one real ambiguous clip** (yield vs. proceed, or decelerate vs. keep). Six seeds per configuration: reference $A_1$ (black, thin), Euler-10 (mid-blue), Euler-4 (dark blue), Euler-1 (a single dark-blue star = the conditional mean, the same for all seeds), human log (dashed gray). Use the same noise seeds so the reader can follow each seed contracting toward the star. Equal aspect ratio, ego at origin, x = longitudinal (m), y = lateral (m). Color the endpoint of each trajectory by its decision class (small ring), so the reader sees one or two seeds change class.
- **(b) Right, ~45% width: "the temperature curve".** x-axis = effective temperature τ (0.4–1.0, reversed is *not* needed). y-axis = **share of cautious decisions (stop + decelerate, %)**, the driving-relevant quantity, not ADE. Black line with band = reference at explicit τ. Blue markers = Euler-1/2/4/6/10, vermillion = FD-cache, each placed at its measured τ_eff. Mark Euler-10 with "released" and FD-cache with "accelerated". The reader sees that samplers move along the same curve toward fewer cautious decisions.
- Alternative for (b), if cautious share is noisy: y = decision flip rate vs $A_1$ (%). Do not use ADE for the teaser, because ADE *improves* and that point belongs in Fig. 5.

### Figure 2 (mechanism), single column, in §3
**Claim: Euler contracts toward the mean, and flips go toward the majority mode.**
- **(a)** A 1-D two-mode mixture (30/70 weights): exact flow paths from $t=0$ to $1$ (gray thin) and Euler-4 paths from the same noises (blue). Highlight in vermillion the paths that end in the other mode, all minority → majority. x = $t$, y = $x_t$. The density of $p(x_1)$ is drawn as a sideways histogram at $t=1$.
- **(b)** Gaussian contraction ratio $\prod\cos\theta_i$ (Eq. 1) vs. number of steps $k$ (log x-axis), for $s\in\{0.25,1,4\}$ (three line styles), with the empirical Alpamayo contraction ratios (0.64 at k=4, 0.84 at k=10) overlaid as markers. This links theory to the real model in one panel and is very persuasive.
- Optional inset: the arc vs. chord geometric picture from S7.

### Figure 3 (main result), figure*, top of a page in §5
**Claim: Every first-order sampler lies on the explicit-temperature curve; higher-order solvers do not.**
- Four panels in a row, shared x-axis = **τ** (explicit for the curve, τ_eff for markers), 0.4–1.0:
  (a) seed spread (m), the definition panel where markers are on the curve by construction, so the panel is a calibration check. Better: x = τ_eff from spread and y = the **contraction ratio $r$** (independent measurement), with the identity line. This validates the protocol.
  (b) ADE (m) to human log.
  (c) minADE$_6$ (m).
  (d) decision flip rate vs $A_1$ (%).
- Overlay the three Alpamayo models by marker shape, in one panel per metric (not one panel per model). If clutter is too high, use 3 rows (models) × 3 cols (metrics) small multiples at 6.75 in × 4 in. Include Heun-5 and RK4×2 in green to show they sit *off* the curve in (d) (~10% flips, mostly lateral, at τ_eff≈1). This makes the "first-order" scope visible and protects the title.
- Optional fifth panel, or a twin top axis: **NFE or speedup**, to connect to the acceleration story (Euler-10 = 10 NFE, Euler-4 = 4, ref = 64).

### Figure 4 (decisions), single column
**Claim: Lower temperature removes cautious maneuvers, and matched temperature reproduces the shift.**
- **Forest plot (paired effects with CIs).** Rows = decision classes (stop, decelerate, keep, accelerate | left, right, lane shift, straight) plus a "decision entropy (bits)" row in a separate block. x = change vs. $A_1$ (pp), vertical line at 0. For each row, three markers vertically jittered: Euler-10 (blue, filled), FD-cache (vermillion, filled), $A_{\tau_\mathrm{eff}}$ for FD-cache (vermillion, hollow). The filled/hollow pairs overlapping is the visual argument.
- Optional right panel (or appendix): flip rate vs. policy uncertainty (decision entropy across six seeds, binned 0 / (0,0.5] / (0.5,1] / >1 bits) as bars with CIs, annotated "0 flips in 47% of clips where all seeds agree". This is Finding 3's second half and is very readable.

### Figure 5 (reinterpreting published gains), single column
**Claim: Single-sample and multi-sample metrics give opposite verdicts, and temperature alone produces the reversal.**
- x = $k$ in minADE$_k$ (1–6). y = Δ minADE$_k$ vs. $A_1$ (m), horizontal line at 0. Lines with CI bands: Euler-10 (blue), FD-cache (vermillion), and the matched explicit-temperature references (same colors, dashed). The curves cross zero. Annotate FlashDrive's reported numbers (−0.13 m at k=1, +0.08 m at k=6) as two black "×" markers with a label, "reported by FlashDrive (attributed to fine-tuning)". This is the Mirage move: the same data, a different reading.
- Small inset or separate panel (or appendix): Qwen-Drive spread vs. τ, flat for the reference and higher for Euler-10, labeled as the counterexample.

### Optional Figure 6 / Table 2 (safety)
If space allows, a single-column **dot plot of log-replay collision rate (%)** with CIs for $A_1$ (τ=1), $A_{0.84}$, $A_{0.65}$, Euler-10, Euler-4, and FD-cache, with McNemar $b{:}c$ annotated on brackets between paired bars (Euler-4 vs $A_1$: 0:12; Euler-4 vs $A_{0.65}$: 0:2). When the AlpaSim closed-loop grid finishes, add a second column (closed-loop incident rate, 60 scenes) with the same rows. That turns it into Table 2 (see below).

### Table 1 redesign (table*, booktabs)
```latex
\begin{table*}[t]
\caption{\textbf{Temperature accounts for most of each sampler's effect.}
For each model, the first row is the reference solver at $\tau{=}1$ ($A_1$).
For each sampler $S$ we report its value, its change from $A_1$, and the
residual change from the reference at matched temperature $A_{\tau_\mathrm{eff}(S)}$.
Residuals whose 95\% CI excludes zero are in bold. Flips: share of (clip, seed)
units whose decision class differs from $A_1$ under the same noise.
PhysicalAI-AV test, 200 clips $\times$ 6 seeds; cluster-bootstrap 95\% CIs.}
\label{tab:matched}
\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}ll c rrr rrr rr@{}}
\toprule
 & & & \multicolumn{3}{c}{ADE (m)} & \multicolumn{3}{c}{minADE$_6$ (m)} & \multicolumn{2}{c}{Flips (\%)} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(l){10-11}
Model & Sampler & $\tau_\mathrm{eff}$ & value & $\Delta_{A_1}$ & $\Delta_{\text{match}}$
      & value & $\Delta_{A_1}$ & $\Delta_{\text{match}}$ & value & $\Delta_{\text{match}}$ \\
\midrule
Alpamayo-1.5 & $A_1$ (ref.) & 1.00 & x.xx & --- & --- & x.xx & --- & --- & --- & --- \\
             & Euler-10     & 0.84 {\scriptsize[.82,.86]} & 2.34 & $-$0.xx & +0.02 & 1.22 & ... & \textbf{+0.06} & 5.0 & +0.3 \\
             & Euler-4      & 0.65 & 2.19 & ... & ... & ... & ... & ... & 10.8 & +1.7 \\
\addlinespace
Alpamayo-R1  & ... \\
\bottomrule
\end{tabular}
\end{table*}
```
- Put CIs either in a `\scriptsize` bracket under the value or as ± in a second line. Do not use slash pairs.
- Use percentages without % signs inside cells. Put units in the header.
- Align numbers on decimals (`siunitx` `S` columns, with `table-format=+1.2`).
- No vertical rules, no bold headers. Use `\addlinespace` between models instead of a horizontal rule.
- If space is tight, move the minADE$_6$ block to the appendix and keep ADE plus flips plus the "explained" column.

### Table 2 (safety, when closed loop lands), single column
Rows: $A_1$, $A_{0.84}$, $A_{0.65}$, Euler-10, Euler-4, FD-cache. Columns: log-replay collisions (%) [CI] | McNemar $b{:}c$ vs $A_1$ | $b{:}c$ vs matched | AlpaSim incidents (%) [CI] (60 scenes). Bold only the rows where the matched comparison is non-significant *and* the τ=1 comparison is significant. Or better, bold nothing and let the $b{:}c$ columns speak.

---

## 4. HOW THE RESULTS SHOULD READ (reporting style)

Use one paragraph template per finding, as in Scaling Laws and Mirage:
1. **Claim** (the box).
2. **Number with CI and n**, comparing sampler vs. $A_1$ (the effect).
3. **The same number vs. $A_{\tau_\mathrm{eff}}$** (the control). This is the paper's signature move and must appear in *every* finding, including Findings 3 and 5, which currently lack the matched comparison in text.
4. **Pointer to one figure or table.**
5. **One sentence of scope or caveat** (which model, which data, open-loop vs. closed-loop).

Example rewrite of Finding 3's paragraph:
> On Alpamayo-1.5, FD-cache lowers the share of decelerations by 2.5 pp [−3.8, −1.2] and of stops by 0.75 pp [−1.6, −0.2], raises accelerations by 1.25 pp [CI], and reduces decision entropy by 0.24 bits [−0.30, −0.19] relative to $A_1$; the released Euler-10 already reduces entropy by 0.14 bits [CI]. The reference solver at FD-cache's effective temperature produces shifts of [x, y, z], indistinguishable from FD-cache (\cref{fig:decisions}). Numerical flips occur only where the policy is itself uncertain: none of the 47% of clips whose six seeds agree flips, against 19% [CI] of the remaining clips (Spearman $\rho=0.76$ between per-clip decision entropy and flip rate, $n=200$).

Honest limitations, written as Rethinking Generalization and Mirage do, which means stating what the result *does not* show:
> Our evidence is open-loop and log-replay except for [closed loop]. Decision classes are read from planned trajectories with fixed thresholds, so a flip is a change in the planned maneuver, not necessarily in vehicle behavior. Log-replay agents do not react. The mechanism applies to first-order samplers: second- and fourth-order solvers show no contraction (Finding 1), and for near-deterministic flows such as Qwen-Drive the solver adds rather than removes diversity. Human trajectories are single samples, so we cannot test whether lower temperature is better calibrated, only that it changes the decision distribution.

---

## 5. POLISH

- **Spacing hacks:** remove `\!\approx\!` (abstract, contributions). Use `$\tau_{\mathrm{eff}}\approx0.84$`.
- **Discordant-pair notation "0:12"**: define it at first use ("$b{:}c$ = pairs where only the first / only the second configuration collides") and typeset it as `$0{:}12$`.
- **Sampler names:** use a macro `\newcommand{\euler}[1]{Euler-#1}` and write "Heun-32" with a hyphen everywhere ("Heun32", "Heun160", and "Heun-80" all appear across the text and the docs).
- **Units:** use a thin space before units (`2.19\,m`) consistently. Finding 5 does this and Finding 6 partly. "pp" for percentage-point differences; "%" only for rates.
- **Hyphenation and compounds:** "single-sample", "best-of-six", "log-replay" (adjective) vs. "log replay" (noun); "closed-loop" (adjective). Check each occurrence.
- **"e.g., 6.1% vs. 6.1%"**: never use "vs." between identical numbers without labels. Write "6.1% under both".
- **Proposition 3.1** uses "$h\,\mathrm{Lip}(v)<1$". Define $h$ (step size) in the statement, and write `\operatorname{Lip}`.
- **"for almost every $z$"**: say "for Lebesgue-almost every $z$" once, or "with probability one over $z$".
- **Footnote/notice:** keep `\printAffiliationsAndNotice{}`. The anonymous footnote is fine.
- **Underfull boxes** (lines 59, 83, 126, 129, 133, 178, 188) are mostly caused by inline TODO boxes breaking paragraphs and by `\paragraph` + short lines. Recheck them after M1 and after the TODOs are filled. Expect most to disappear. In `main.bbl` lines 73–79 and 113–118 the cause is long URLs or arXiv IDs. Load `\usepackage{xurl}` or add `\urlstyle{same}`.
- **Hanging periods** after inline TODOs (visible on pages 2 and 4: a lone "." on its own line) come from `\todo[inline]` mid-sentence. Use `\todo[inline]` only between paragraphs, or switch to a non-inline margin note while drafting.
- **Appendix:** `\onecolumn` is allowed. Keep App. A–D titles in title case ("Solvable Mixture Models" is correct).
- **Impact statement:** good and short. Change "manoeuvres" to "maneuvers".
- **Keywords:** add "effective temperature" and "decision flips".
- **"NVIDIA's Alpamayo models"** in the abstract: possessive brand phrasing is fine. Cite the model card or report in §2.
- **Cross-references:** use `\cref` for findings once labels exist ("\cref{find:shared}"), and give Finding a `\crefname{finding}{Finding}{Findings}`.
- **Consistent "effective temperature"**: always lowercase in running text, always with the symbol at first use per section, and never "effective-temperature" except as a compound adjective ("effective-temperature matching").
