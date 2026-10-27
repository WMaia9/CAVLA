# Presentation Review, Round 2: "When Fewer Steps Mean Lower Temperature: Mean Collapse in Accelerated Flow-Matching Driving Policies"

Reviewer focus: writing, visual design, and results reporting, benchmarked against DDPM, EDM, Scaling Laws, and "Are Emergent Abilities a Mirage?".
Draft reviewed: `paper/main.tex` (= `versions/main_v1.tex`, 194 lines) and `paper/main.pdf` (5 pages, tectonic, T1). Previous review: `reviews/reviewer_presentation.md`. Numbers were checked against `artifacts/matched_stats_v1_20261026`, `gaussian_contraction_v1_20261026`, `temperature_report_*_v1_20261022`, and the raw plans in `temperature_probe_*_v1_20261022/plans.npz`.

**Overall.** This revision is a real improvement. The build is fixed, the tables are generated from artifacts, the macros exist, the hook uses a real measured number, and the thesis now carries its own boundary ("like temperature, except the tail"). That boundary is what makes Mirage-style papers credible. Three things now limit the paper:
1. **The headline "tail" claim is presented with more weight than its data can carry.** The abstract leads its second half with the stopping residual. In the raw plans, every stopping decision across all three models and all samplers comes from **2–4 of the 60 clips**: `context_113` (ego stationary) and `context_141` (ego at 0.8 m/s), plus one plan each in `context_127` (R1) and `context_124` (A2). A bootstrap over clips cannot give a stable interval when two clips carry the signal; the stopping CIs whose upper bound is exactly `+0.0` show this. This is now the main presentation risk. It needs counts in the text, not percentages, and wording proportional to the evidence.
2. **"Reproduced" is claimed while the paper's own pre-declared equivalence test mostly fails.** §4 declares margins, but the artifact's `equivalent_within_margin` flag is true for only 3 of the 6 ADE₁ cells (R1 E10, A2 E10; A1.5 only at E6) and 1 of the 6 flip cells (R1 E10). The text never reports the test it declares.
3. **There are no figures yet.** The body (~3.4 pages including TODO boxes) leaves about 3 pages unused. A 5-page ICML submission reads as thin. The figure specs in §5 below are meant to fill that space with evidence, not padding.

---

## 1. Status of Round-1 Items

| Item | Status | Note |
|---|---|---|
| M1 build/fonts/bold/running head | **Fixed** | `pdffonts`: NimbusRomNo9L Regu/Medi/ReguItal + CM math, all Type 1C embedded. Bold renders (Medi). Running head "When Fewer Steps Mean Lower Temperature" shows. `\texorpdfstring` used. |
| M2 abstract one paragraph, 4–6 sentences | **Partly** | 5 sentences + TODO, one paragraph. New typo bug: `\pp` eats the following space ("−0.8 ppfor", also "pp(95% CI" in Finding 3). The key term "mean collapse" from the title is absent from the abstract. See §3. |
| M3 one name per object, macros | **Partly** | Notation paragraph and macros added. The draft still mixes "Euler-4" (abstract, intro, tables, teaser) with $E_4$ (findings), "accurate solver" (abstract, contributions, §4) with "reference solver", and "single-sample error" with ADE₁. `\Atau` hard-codes τ, so $A_{\taueff}$, $A_{0.4}$ are typed by hand. See §4. |
| M4 numbers from different samples on the page | **Regressed in a new form** | Table 1 (contraction) is 200 clips × 2 seeds, Table 2 is 60 × 6, and Finding 3's "base stopping rates 1.7–3.7%" mixes the 60-clip set (1.7% for all three models) with the 100-clip Alpamayo-1.5 report (3.7%). Finding 1's "all 400 samples of $E_4$, $E_8$ and $E_{10}$" reads as 400 total; it is 400 per sampler (1,200). |
| M5 Table shows effect vs residual | **Partly** | Effect and residual columns exist. But the reader still cannot see the baseline, and the eye cannot find "residual ≈ 0" among 54 bracketed numbers. See §4 for a redesign. |
| M6 displayed equations | **Partly** | Eq. (1) is displayed. τ_eff and Spr are not displayed or numbered, and Eq. (1) uses the nonstandard $\|\cdot\|_d$ with undefined θ. |
| M7 statistics | **Partly** | Clip bootstrap and 95% CIs added. The equivalence test is declared but not reported. The pooled stopping numbers in Finding 3/abstract (−0.40, −0.80) are not in any artifact; they are typed by hand. Distinct-clip counts are missing for the tail. |
| M8 bibliography TODOs | **Not fixed** | "(TODO, 2026c;a;b)" still renders in Background and Related Work, and four bib entries print "TODO verify authors". This is still the single fastest way to lose a reviewer. |
| M9 title overclaims | **Fixed** | First-order scope is in the abstract, and the subtitle names the boundary. See §3 for the remaining title note. |
| S1 hook | **Fixed** | Uses real numbers (0.28 m, one in nine). |
| S2 contributions | **Partly** | 5 bullets, one is a TODO, and bullets 1–2 have no number. |
| S3 Finding boxes | **Mostly** | Left rule, labels, `\crefname` done. Remaining problems: Finding 4 is empty; `\parbox` cannot break across columns; intro never points to findings. |
| S4 flow/budget | **Partly** | §4 "Protocol" is a 6-line section, which gives 8 numbered sections. Qwen boundary is a finding (acceptable). |
| S5 headings | **Fixed** | Title case for sections; run-ins consistently sentence case. |
| S6 setup: NFE/latency, flip def., stats | **Partly** | Flip definition and stats added. NFE/latency still missing, which matters because the paper's premise is speed. |
| S7 one sentence of intuition before algebra | **Not done** | |
| Polish: hanging period after inline TODO | **Not fixed** | Limitations: "within a clip [TODO]. Stopping is rare" puts "." alone on a line (p. 4). |
| Polish: keywords | **Not done** | Add "mean collapse", "effective temperature". |

---

## 2. MUST-FIX

### M1. Report the tail result at the resolution of its data
Stopping plans (out of 360 = 60 clips × 6 seeds per model), recomputed from `plans.npz` with `trajectory_decision_gate.decisions`:

| Config | Alpamayo-1.5 | R1 | Alpamayo-2 | Pooled (of 1080) |
|---|---|---|---|---|
| $A_1$ | 6 (113:5, 141:1) | 6 (113:4, 127:1, 141:1) | 6 (113:4, 124:1, 141:1) | 18 |
| $A_{0.8}$ | 6 | 7 | 7 | 20 |
| $A_{0.6}$ | 7 | 7 | 8 | 22 |
| $A_{0.4}$ | 7 | 8 | 9 | 24 |
| $E_{10}$ | 6 | 4 | 5 | 15 |
| $E_6$ | 5 | 4 | 5 | 14 |
| $E_4$ | 5 | 4 | 4 | 13 |
| $E_2$ | 6 | 5 | 3 | 14 |
| $E_1$ | 6 | 1 | 0 | 7 |

The direction is consistent and the mechanism is visible: on `context_141` the reference's minority "stop" seed (extent 0.4–1.0 m) gets pushed by Euler toward the conditional mean (E₁ ≈ 10 m for every seed). On the stationary clip `context_113`, E₁ returns a 1–3 m "creep", which is the mean of {stay, go}. That is a strong illustration of *mean collapse*. But it is 2 clips. Required changes:
- **Always give counts and distinct clips** next to the percentage: "Euler-4 stops in 13 of 1,080 plans, against 18 for $A_1$ and an interpolated 21.6 for the matched-temperature reference; the difference comes from 2 of 60 clips."
- **Compute the pooled stopping numbers in the artifact** (and in `make_tables.py`), not by hand. Use the clip as the unit; with ≤4 informative clips, also report an exact permutation/sign test over clips. Do not put a bootstrap CI that collapses to `[…, +0.0]` in the abstract.
- **Scale the wording to the evidence until more stop-candidate clips are run.** Recommended: select clips by a pre-registered criterion (ego speed < 2 m/s, or any reference seed stopping) from the 436 confirmation clips or the full test split, and rerun. Until then, the abstract should say "in scenes where the policy hesitates between stopping and going" and the finding title should be descriptive, not universal ("Euler sampling removes minority stopping decisions that temperature keeps").
- The table's stopping CIs `+0.0 [+0.0, +0.0]` (A1.5 E10) and the `[…, +0.0]` upper bounds should be replaced by counts (see M3).

### M2. Report the equivalence test you declared, or remove it
§4 declares margins (ADE₁/minADE₆ 0.05 m, flips 2 pp, stopping 1 pp), and the artifact computes `equivalent_within_margin`. The text then asserts "reproduced" without reporting the test. A careful reviewer will rerun it and find:
- ADE₁: equivalent in R1-E10 and A2-E10 only (and A1.5-E6, not in the table).
- Flips: equivalent in R1-E10 only. A2-E10 has residual +2.0 pp [−0.0, +4.2], which **exceeds** the 2 pp margin.
- minADE₆: never equivalent.

With 60 clips the test is underpowered, which is fine to say. Recommended wording for Finding 2 (replace the whole body):

> $\taueff(E_{10})$ is 0.83, 0.84 and 0.83 for the three Alpamayo generations (95\% intervals of width 0.03--0.06; \cref{tab:matched}). Against $\Aref$, $E_{10}$ lowers ADE$_1$ by 0.13--0.19\,m and changes 4.7--9.7\% of decisions. The reference at $\taueff$ accounts for 92--121\% of the ADE$_1$ change and 79--100\% of the flips: every residual interval contains zero, and the pre-declared equivalence margins are met in 3 of 6 ADE$_1$ cells and 1 of 6 flip cells, a test that 60 clips leave underpowered (\cref{app:protocols}).

Add `fraction_explained` (already in `summary.json`) and the equivalence flag to the generator. Generate *all* in-text numbers through a `tables/numbers.tex` macro file (e.g. `\newcommand{\nTauEtenAtwo}{0.83}`) so the abstract and findings can never drift from the artifacts.

### M3. Table 2 (`tab_matched`) needs a redesign so that "effect vs residual" is visible at a glance
Current problems seen in the render (p. 4, zoomed): 54 bracketed CIs at a scaled `\scriptsize` (~5.5 pt after `\resizebox`, below a legible size); ASCII hyphens for negatives ("-0.28") next to true plus signs ("+0.06"), so columns do not align and minus is too short; "−0.00" and "[-0.0," negative zeros; flips labelled "pp" for an effect that is a rate; no baseline values; the stop *effect* (the half of the tail argument where temperature raises stops) is missing; and numbering puts it as Table 2 although the intro cites it first.

**Proposed layout:** show *absolute values*, with the sampler column placed between its two comparators. The reader then sees "sampler ≈ matched, both ≠ A₁" without reading signs or CIs. Move the full effect/residual/CI table to the appendix.

```latex
\begin{table*}[t]
  \caption{\textbf{At matched effective temperature the Euler sampler and the reference agree on accuracy and decision flips, but not on stopping.}
  For each model and Euler sampler $E_k$: the reference solver at $\tau{=}1$ ($\Aref$), the sampler, and the reference at the sampler's effective temperature ($A_{\taueff}$).
  Read each block left to right: $E_k$ moves away from $\Aref$ and lands on $A_{\taueff}$, except for stops, where $A_{\taueff}$ stops \emph{more} than $\Aref$ and $E_k$ stops \emph{less}.
  Stops are counts of 360 plans (60 clips $\times$ 6 seeds); all stopping plans come from 2--4 clips per model.
  $^\dagger$: 95\% residual interval excludes zero; $^{\equiv}$: equivalent within the pre-declared margin. Same 60 PhysicalAI-AV test clips for every model; clip bootstrap. Full intervals in \cref{tab:matched-full}.}
  \label{tab:matched}
  \centering\footnotesize\setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}ll c ccc cc ccc c@{}}
    \toprule
    & & & \multicolumn{3}{c}{ADE$_1$ (m)} & \multicolumn{2}{c}{Decision flips (\%)} & \multicolumn{3}{c}{Stops (of 360)} & minADE$_6$ (m) \\
    \cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}\cmidrule(l){12-12}
    Model & $S$ & $\taueff$ & $\Aref$ & $S$ & $A_{\taueff}$ & $S$ & $A_{\taueff}$ & $\Aref$ & $S$ & $A_{\taueff}$ & residual \\
    \midrule
    Alpamayo-1.5 & $E_4$    & 0.64 & 2.86 & 2.58 & 2.58 & 10.6 & 9.1  & 6 & 5 & 6.6 & $+0.06$ \\
                 & $E_{10}$ & 0.83 & 2.86 & 2.73 & 2.70 & 4.7  & 4.3  & 6 & 6 & 6.0 & $+0.11$ \\ \addlinespace
    Alpamayo-R1  & $E_4$    & 0.61 & 2.47 & 2.10 & 2.15 & 10.8 & 12.9 & 6 & 4 & 7.0 & $\phantom{+}0.00$ \\
                 & $E_{10}$ & 0.84 & 2.47 & 2.28 & 2.30$^{\equiv}$ & 6.1 & 6.1$^{\equiv}$ & 6 & 4 & 6.6 & $+0.02$ \\ \addlinespace
    Alpamayo-2   & $E_4$    & 0.62 & 2.93 & 2.61 & 2.56 & 14.7 & 13.3 & 6 & 4 & 8.0 & $+0.08$ \\
                 & $E_{10}$ & 0.83 & 2.93 & 2.78 & 2.76$^{\equiv}$ & 9.7 & 7.7 & 6 & 5 & 6.7 & $+0.02$ \\
    \bottomrule
  \end{tabular}
\end{table*}
```
(The values above are the point estimates in `matched_stats_v1_20261026/summary.json`, with $\Aref = $ value − effect and $A_{\taueff} = $ value − residual. The $^\dagger$ marks are left for the generator to place.)
- Design rules: **no `\resizebox`**. This 12-column layout fits `\textwidth` at `\footnotesize` natively. Use math-mode minus (`$-0.28$`, or `siunitx` `S[table-format=+1.2]`). Never print a signed zero. Shade the $S$ column very lightly (`\columncolor{black!5}`, via `colortbl`) so the eye anchors on it.
- Optionally add a final column "explained (%)" for flips (90/86, 100/119, 79/90), but cap or footnote values >100% ("overshoot") rather than printing 119%.
- **Numbering:** `\input{tables/tab_matched}` *before* `tab_contraction` so the main table is Table 1 (the intro cites it first). Better still, fold the 3-row contraction table into Fig. 2(b) of the mechanism figure (§5, F4) and cite the numbers in Finding 1. That saves ~0.25 column.
- Add $E_6$ rows to the appendix table; they exist in the artifact.

### M4. Bibliography placeholders (carried over, still blocking)
"(TODO, 2026c;a;b)" renders on pages 2 and 3, and the bib prints "TODO verify authors" for AdaVLA, Coda ("Fewer steps, better actions"), ProbeFlow, and SubFlow, plus "TODO verify" on NVIDIA/Qwen entries. Fill in the authors. Write names in prose ("ProbeFlow~\citep{probeflow2026} and AdaVLA~\citep{adavla2026}") so a missing name is never the grammatical subject.

### M5. One sample, one reference, everywhere in the main text
- Table 1 (contraction) is Alpamayo-1.5, 200 clips × 2 seeds; everything else is 60 × 6. The 60-clip plans contain everything needed (E₁ output = conditional mean, $A_1$, $E_4$, $E_{10}$, and 6 seeds for the per-clip variance), so recompute the contraction on the same 60 clips for all three models. That also resolves the R1/A2 TODO in Finding 1 without new GPU time. Note: $E_8$ is not on that grid; drop it or report it only in the appendix.
- Finding 3 "base stopping rates of 1.7–3.7%": on the 60-clip set the rate is 1.7% (6/360) for every model. 3.7% is the 100-clip Alpamayo-1.5 report. Use 6/360.
- Finding 5 (Qwen) numbers come from `temperature_report_qwendrive_rl` (60 clips, reference Euler-64). Say so, and say once in Setup that the Qwen reference differs (the TODO already notes this).
- State $n$ in every table/figure caption.

### M6. Abstract: fix bug, add the key term, show effect before residual
Current: 5 sentences + TODO; "ppfor" bug; "mean collapse" missing; the reader never learns the size of the effect that temperature explains, so "reproduces" has no weight. Replacement (one paragraph, 6 sentences; sentence 5 is the slot for FlashDrive/closed loop, and should be deleted rather than left as a TODO if the result does not land):

> Flow-matching driving policies are accelerated by integrating with fewer Euler steps or by caching velocities, and these changes are reported as preserving, or even improving, trajectory accuracy. We show that a first-order sampler pulls every sample toward the conditional mean, by an amount that a Gaussian model with no fitted parameters predicts to within 0.04, and that this mostly acts as a lower sampling temperature. On the PhysicalAI-AV test split, the released 10-step samplers of three Alpamayo generations share an effective temperature of 0.83--0.84; an accurate solver run at that temperature reproduces their 0.13--0.19\,m single-sample gain and 79--100\% of the 5--10\% of driving decisions they change. What temperature does not reproduce we call \emph{mean collapse}: where the policy hesitates between stopping and going, Euler sampling drags the minority stop toward the mean and removes it, whereas a lower temperature keeps it. [FlashDrive / AlpaSim sentence.] Accelerated flow policies should therefore be compared at matched effective temperature, with rare decisions checked separately.

Notes: (i) no stopping CI is given in the abstract (see M1); add a count once more clips are in. (ii) "79–100%" must come from the generator (flip `fraction_explained` for $E_{10}$: 0.90, 1.00, 0.79). (iii) Avoid "every" unless the contraction is measured on all three models (M5).

### M7. `\pp` macro and inline-TODO artifacts
- `\newcommand{\pp}{\,pp}` swallows the following space ("ppfor", "pp(95%"). Use `\usepackage{xspace}` and `\newcommand{\pp}{\,pp\xspace}`, or always write `\pp{}`.
- Move `\TODO{multi-seed language ablation}` out of the middle of the Limitations sentence (it leaves "." alone on a line, p. 4). Move the abstract TODO out of the `abstract` environment into a comment so the first page renders cleanly for internal readers.

---

## 3. SHOULD-FIX

### S1. Title
The current title is good: it is conditional ("When…"), names the mechanism, and is driving-specific. Two small points:
- "Mean … Mean" in two adjacent clauses is a slight stumble when read aloud. This is acceptable; do not trade clarity for it.
- If (and only if) the stopping evidence grows to a pre-registered clip set, this alternative is stronger and more driving-specific: **"Fewer Steps, Lower Temperature, Fewer Stops: Mean Collapse in Accelerated Flow-Matching Driving Policies."** With the current 2-clip support, keep the present title.

Keywords: replace with `flow matching, autonomous driving, vision-language-action models, ODE solvers, sampling temperature, effective temperature, mean collapse`.

### S2. Make the thesis sentence unmissable
The thesis appears mid-paragraph in the intro. Set it as the closing sentence of paragraph 2, in italics:

> *For first-order samplers, most of what fewer steps change is what a lower sampling temperature changes; the exception, which we call mean collapse, is that rare decisions are pulled to the conditional mean and disappear.*

Then add one sentence of payoff and protocol: "This suggests a simple control: compare every accelerated sampler against an accurate solver at the same effective temperature (\cref{sec:protocol})."

### S3. Contributions → "Findings in brief"
Five bullets, one is a TODO, and two have no number. Replace them with four bullets that each point to a Finding:
- **Mechanism (§3, \cref{find:contraction}).** One Euler step returns the conditional mean, and Euler contracts Gaussian targets on any grid (Eq. 1). This closed form, with no fitted parameters, predicts the measured contraction of $E_{10}$ (0.86 vs 0.84) and explains why released samplers land at $\taueff\approx0.83$.
- **Protocol (§3.4).** The effective temperature $\taueff$ is read from seed spread alone, and every effect is split into a temperature part and a residual.
- **Temperature explains the average (\cref{find:matched}).** For three Alpamayo generations it accounts for 92–121% of the single-sample gain and 79–100% of decision flips (Table 1).
- **But not the tail (\cref{find:tail,find:qwen}).** Euler removes minority stopping decisions that temperature keeps, and its best-of-six coverage is worse than temperature predicts. Qwen-Drive, whose flow is nearly deterministic, bounds the claim.

Add the FlashDrive/closed-loop bullet only when the result exists.

### S4. Notation: one table of names, enforced by macros

| Object | Use everywhere | Remove |
|---|---|---|
| $k$-step Euler | `$E_k$` (macro `\Ek{4}` → $E_4$); spell "Euler with $k$ steps ($E_k$)" once in §2 and "Euler-4" only in the abstract | "Euler-4" in tables/findings/figure legends, "Euler sample" |
| reference at temperature τ | `\At{\tau}`, `\At{\taueff}`, `\At{0.4}` with `\newcommand{\At}[1]{A_{#1}}`; `\Aref`=`\At{1}` | "accurate solver" after §2 (abstract may keep it once), "reference solver at τ=1" |
| spread | $\Spr(S)$, displayed and numbered (below) | |
| contraction ratio | $\rho_k$ (measured) and $\hat\rho_k=\prod_i\cos\theta_i$ (predicted) | "contraction factor", "measured contraction" without a symbol |
| single-sample error | ADE$_1$ | "single-sample error" after the abstract |
| flip rate | "flip rate" (%) | "decision-flip rate", "rate of changed driving decisions", "pp" for a rate |
| stop count | "stopping plans" (count of 360) | "stopping rate" when the base is 6 |

Fix Eq. (1): the coordinate-wise norm $\|\cdot\|_d$ is nonstandard, and θ is undefined. Replacement:
```latex
\begin{equation}
  \hat\rho_{k,d} \;=\; \frac{|E_k(z)_d - m_d|}{|\Aref(z)_d - m_d|}
  \;=\; \prod_{i=0}^{k-1}\cos\theta_{i,d}\;\le\;1,
  \qquad \theta_{i,d}=\angle\bigl(u_d(t_i),u_d(t_{i+1})\bigr),
  \label{eq:contraction}
\end{equation}
```
Display the effective temperature as an equation and cite it as Eq. (2) in captions:
```latex
\begin{equation}
  \Spr(S)=\E_c\,\E_{z\neq z'}\bigl\|S(z;c)-S(z';c)\bigr\|,\qquad
  \taueff(S)=\tau \ \text{ such that }\ \Spr(\At{\tau})=\Spr(S).
  \label{eq:taueff}
\end{equation}
```
State what happens at the grid edge: $E_1$ and $E_2$ fall below $\tau=0.4$ (the artifact clips them to 0.40). Either extend the reference to $\tau\in\{0.3,0.2,0.1\}$ (cheap and needed for the figures) or mark them "$<0.4$".

Also explain the apparent paradox in §3.2. The text says the $E_{10}$ factor is 0.82–0.88 for $s_d\in[0.3,2]$, but the measured per-coordinate conditional std has median 0.11, where the factor is ~0.62. The aggregate ratio is dominated by high-variance coordinates. Say so in one sentence, or a reader will think the "why 0.83" explanation contradicts the data.

### S5. S7 from round 1: one sentence of intuition before the algebra
Insert before "For $p(x_1\mid c)=\dots$":
> For a Gaussian target, the exact flow carries each noise sample along an arc from $z$ to its data point; an Euler step follows the chord of that arc instead, and every chord ends inside the circle, so each step loses a factor $\cos\theta_i$ of the deviation from the mean.

### S6. Finding boxes
- Replace `\parbox` with a breakable box. A finding that falls at a column end will otherwise leave a large gap once figures are added:
```latex
\usepackage[most]{tcolorbox}
\newtcolorbox{findingbox}{enhanced,breakable,frame hidden,boxrule=0pt,
  borderline west={1.5pt}{0pt}{black!70},colback=white,
  left=5pt,right=0pt,top=1pt,bottom=1pt,before skip=5pt,after skip=5pt}
\newcommand{\finding}[2]{\begin{findingbox}\refstepcounter{finding}\label{#1}%
  \textbf{Finding \thefinding.}~#2\end{findingbox}}
```
- Each finding ends with exactly one pointer to a figure or table. Currently Finding 3 points to nothing for stopping, and Finding 5 points to nothing.
- Finding 4 is empty. If FlashDrive at matched temperature is not done by submission, delete the finding rather than shipping a title with no evidence. Move the minADE$_k$ crossover into Finding 3, where the minADE₆ residual already appears.
- Retitle Finding 3: "*Temperature does not explain the tail: Euler sampling removes minority stopping decisions that a lower temperature keeps.*" Body: give counts (M1).
- Finding 1 says "Heun-5 and RK4 show no contraction" with no number. Add "(ratios 1.00 and 1.05)" and the sample.

### S7. Structure and section count
- Merge §4 "Matched-Temperature Protocol" (6 lines) into §3 as §3.4 "Effective Temperature and the Matched Comparison". The method then sits with the theory that motivates it, and the paper has 7 sections.
- Add to Setup a 4-row **samplers table** with NFE and measured latency per call (ms) for $E_1, E_4, E_{10}$, FD-cache, and Heun-32. The paper's premise is speed, but no speed number appears anywhere.
- Related Work: the last sentence of "Critical windows" (the pre-registered negative result) is your best motivation for the protocol. Give it a number in the intro or Results: "a schedule that passed a pre-registered confirmation (−0.88 pp flips [−1.28, −0.52], 436 clips) showed no gain at matched temperature (\cref{app:schedules})."
- The Safety paragraph is now honest. Keep it short and give the counts by clip as it does. Do not let the AlpaSim TODO slip into the abstract until it exists.

### S8. Contraction "parameter-free"
The predicted factor uses each clip's estimated conditional variance, so "parameter-free" invites the objection "you fed it the variance". Use "with no fitted parameters", and say once what it is fed.

---

## 4. POLISH

- Negative numbers in tables: math minus. No "−0.00" or "[−0.0,". Format with `f'{x:+.2f}'.replace('-','$-$')` and map `-0.00`→`0.00`.
- Units: flips are rates (%); stops are counts; "pp" only for differences of rates.
- "with intervals of width about 0.05" → give the range (0.03–0.06 for $E_{10}$).
- "0.83, 0.84 and 0.83 for the three Alpamayo generations": name them in order (1.5, R1, 2).
- "Alpamayo-2-Super" in Setup vs "Alpamayo-2" in tables: write "Alpamayo-2-Super (Alpamayo-2 below)".
- Prop. 3.1: define $h$ as the step size; write `\operatorname{Lip}`; "for Lebesgue-almost every $z$".
- Background: "Released policies use forward Euler with 10 uniform steps" needs a source per policy (model card).
- Teaser placeholder caption is good in structure (bold claim first). Keep that pattern for every caption: bold claim, then how to read, then $n$/CI.
- Impact statement: "Accelerations that silently reduce stopping decisions could pass accuracy checks" is good. Keep it.
- After adding figures, recheck underfull boxes at lines 64, 87, 108, 133, 155 (mostly TODO-induced) and the bbl URL lines (`\usepackage{xurl}`).
- The appendix uses `\onecolumn` (fine). Add `\label{tab:matched-full}` for the full CI table.

---

## 5. FIGURE SPECIFICATIONS (ready to implement)

Global style (state once in the Fig. 1 caption):
- matplotlib, `pdf.fonttype=42`, font Nimbus Roman / Times at 8 pt, line width 1.0, no in-panel titles, bold panel letters at top-left.
- Colors: reference/explicit-temperature $A_\tau$ = **black line + gray 95% band**; Euler $E_k$ = **blue sequential, darker = fewer steps** (E₁ `#08306B`, E₂ `#2171B5`, E₄ `#4292C6`, E₆ `#6BAED6`, E₁₀ `#9ECAE1` edge-darkened); FD-cache = vermillion `#D55E00`; higher-order (Heun-5) = bluish green `#009E73`; human log = dashed gray `#666666`.
- Model by marker: Alpamayo-1.5 ○, R1 □, Alpamayo-2 △; Qwen-Drive ◇ (appendix).
- CIs: 95% bootstrap over clips (2,000 resamples, seed 20261026, as in `matched_stats`); recompute for the curves (the temperature reports store point estimates and their own CI triples). **Use the same 60 clips (`context_100`–`context_159`) for all three models.** The Alpamayo-1.5 report has 100 clips, so restrict it.
- Data: per-clip plans `artifacts/temperature_probe_v1_20261022/plans.npz` (A1.5), `temperature_probe_r1_v1_20261022`, `temperature_probe_a2_v1_20261022`; keys `context_XXX_s92NN_{config}`, arrays (64, 3), 0.1 s steps; ego speed from `artifacts/physicalai_test_contexts_v1_20261022/records.json`; decisions from `cavla.trajectory_decision_gate.decisions`. Aggregates: `temperature_report_{,r1_,a2_}v1_20261022/summary.json` (`rows[config]` with `ade1`, `minade6`, `spread`, `decision_entropy`, `stop`, `decelerate`, `flip_vs_reference`); `matched_stats_v1_20261026/summary.json`; `gaussian_contraction_v1_20261026/summary.json` (`steps`, `curve`).
- Pre-requisite run (cheap): reference Heun-32 at $\tau\in\{0.3,0.2,0.1\}$ for all three models, so $E_1$ and $E_2$ can be placed on the curve rather than clipped at 0.4.

### F1. Teaser (single column, 3.25 × 2.6 in, top of p. 1 right column, where the placeholder is now)
**Claim:** *Fewer Euler steps act like a lower temperature, except that they pull a hesitant stop toward the mean.*

- **(a) top, three mini-panels sharing a y-axis (each ~1.0 × 1.2 in): one real clip, `context_141`, Alpamayo-2 (ego at 0.77 m/s).** x = time (s, 0–6.4). y = distance traveled along the plan (m), i.e. the cumulative norm of `diff(plan[:, :2])`, 0–12 m. Panels: "$\Aref$" (6 seeds, black), "$A_{0.6}$, lower temperature" (6 seeds, black), "$E_4$, fewer steps" (6 seeds, blue `#4292C6`). In all three panels, $E_1$ (the conditional mean, identical for all seeds) is a thick dark-blue dashed line, and the human log is gray dashed. Shade y ≤ 1.5 m light red and label it "stop" (the decision threshold is plan extent ≤ 1.5 m). Corner annotation with stop counts: $\Aref$ 1/6, $A_{0.6}$ 3/6, $E_4$ 0/6. From the data, final extents are $\Aref$ {8.6, 6.6, 0.8, 10.7, 9.2, 3.7}, $A_{0.6}$ {7.8, 5.4, 0.6, 8.9, 1.0, 0.4}, $E_4$ {6.5, 4.9, 2.1, 6.5, 2.6, 2.6}, $E_1$ ≈ 7.5. The reader sees temperature push the hesitant seeds *down into* the stop band while Euler pushes them *up toward the mean line*.
- **(b) bottom (3.25 × 1.1 in): stopping plans vs temperature, pooled over the three models (of 1,080).** x = τ (0.1–1.0); black line and markers = $A_\tau$ at τ ∈ {1, 0.9, …, 0.4, 0.3, 0.2, 0.1}. Measured so far: 18 (τ=1), 20 (0.8), 22 (0.6), 24 (0.4); compute 0.9/0.7/0.5 and the extension. Blue markers = $E_k$ at their $\taueff$: $E_{10}$ 15 (0.83), $E_6$ 14 (~0.73), $E_4$ 13 (~0.62), $E_2$ 14 and $E_1$ 7 (need extended grid). Vertical bars = clip-bootstrap 95% CI. Arrow from the curve to each Euler marker labelled "mean collapse". Right-margin text: "all stops from 4 clips".
- **Caption:**
  > **Fewer Euler steps act like a lower temperature, except on a hesitant stop.** (a) One PhysicalAI-AV test clip (Alpamayo-2, ego at 0.8\,m/s), six shared noise seeds; y is distance traveled along each plan, and the shaded band is the stop decision (extent $\le1.5$\,m). Lowering the temperature of an accurate solver ($A_{0.6}$) moves hesitant seeds into the stop band; four Euler steps ($E_4$) move them toward the conditional mean (dashed, $E_1$) and none stops. (b) Plans that stop, pooled over three Alpamayo generations (60 clips $\times$ 6 seeds each): the reference stops more as $\tau$ falls, while Euler samplers placed at their effective temperature stop less. Bars: 95\% clip-bootstrap intervals; stops come from 4 of 60 clips.

### F2. Main result: the temperature curve (figure*, 6.75 × 2.0 in, top of the Results page)
**Claim:** *Placed at its effective temperature, every Euler sampler lands on the explicit-temperature curve for accuracy and decision flips, but below it for coverage.*

- Four panels in a row, shared x = τ (0.1–1.0; explicit τ for curves, $\taueff$ for markers). y is the **change relative to $\Aref$** of each model, so the three models share one scale:
  - (a) ΔSpr / Spr(A₁) (relative spread). This is the calibration panel: markers fall on the curve by construction. Overlay the *independent* contraction measurement as hollow markers at x = $\rho_k$ (from `gaussian_contraction_v1`: 0.64/0.80/0.84, and recomputed for R1/A2 per M5). Agreement between filled and hollow markers validates the protocol. Alternative: x = $\taueff$, y = $\rho_k$, with the identity line.
  - (b) ΔADE₁ (m). Curves fall from 0 to about −0.4 to −0.5; the Euler markers sit on them (residual ≤ 0.05 m).
  - (c) Decision flips vs $\Aref$ (%). Curves rise from 0 to 15–23%. Euler markers sit on them, and $E_1$/$E_2$ sit **above** (E₁ residual +11 to +15 pp: mean collapse at extreme compression). Add **Heun-5 at τ=1 (green ◇, Alpamayo-1.5 only): 11.3% flips at $\taueff\approx1$**, far off the curve. This shows the "first-order" scope.
  - (d) ΔminADE₆ (m). Curves rise. Euler markers sit **above** the curve (residual +0.02 to +0.11 m), which is the coverage cost.
- Encoding: one black curve per model (solid ○ A1.5, dashed □ R1, dotted △ A2) with a gray band; Euler markers in the blue ramp, shape = model; thin gray connector from each Euler marker vertically down/up to the curve (the residual made visible).
- Top twin axis on (b) or a small label on each marker: NFE (1, 2, 4, 6, 10 vs 64 for the reference), which ties the figure to acceleration.
- **Caption:**
  > **At its effective temperature, each Euler sampler lands on the explicit-temperature curve for accuracy and flips, not for coverage.** Curves: accurate solver (Heun-32) with initial noise scaled by $\tau$, change relative to $\tau{=}1$, one line per Alpamayo generation, 95\% clip-bootstrap bands. Markers: Euler with $k$ steps placed at $\taueff$ (Eq.~\ref{eq:taueff}); vertical connectors show the residual. (a) Spread, with the independently measured contraction $\rho_k$ (hollow) agreeing with $\taueff$. (b) Single-sample error falls with temperature and Euler follows. (c) Flips follow the curve except at one or two steps; a second-order solver (Heun-5, green) flips 11\% of decisions at no temperature change. (d) Best-of-six error is worse than temperature predicts. 60 identical PhysicalAI-AV test clips $\times$ 6 seeds per model.

### F3. Tail / mean-collapse figure (single column, 3.25 × 2.4 in, next to Finding 3)
**Claim:** *Temperature and Euler move rare decisions in opposite directions: temperature shrinks plans toward each sample's own mode, Euler toward the global mean.*

- **(a) Strip plot of plan extent per seed for the two decisive clips.** x = plan extent (m, log scale 0.01–20), with a vertical dashed line at the 1.5 m stop threshold and the left side labelled "stop". Rows (top to bottom) = $\Aref$, $A_{0.8}$, $A_{0.6}$, $A_{0.4}$ | gap | $E_{10}$, $E_6$, $E_4$, $E_2$, $E_1$. Dots = 6 seeds × 3 models (shape = model, small vertical jitter). Two side-by-side columns: `context_113` (ego stationary) and `context_141` (ego at 0.8 m/s).
  - From the data: on 113, temperature collapses extents to ~0.01 m (all stop) while $E_1$ gives 0.9–3.2 m "creep" (A2: all 6 seeds ≈2.9–3.3 m, classed keep). On 141, $\Aref$ has one seed per model at 0.4–1.0 m, $A_{0.4}$ has 1–3, and $E_1$ puts every seed at 7.2–11 m.
  - Color the dots that fall on the stop side of the line in black, and the rest in their sampler color.
- **(b) Right, narrow (0.9 in): pooled stop counts** as two short dot columns: $A_{\taueff(E_k)}$ (interpolated) vs $E_k$ for k = 10, 6, 4, with lines joining each pair. Optional if F1(b) is kept; otherwise move F1(b) here and leave the teaser with (a) only.
- **Caption:**
  > **Mean collapse on the tail.** Plan extent (log scale) for every seed of the two test clips that contain stopping decisions in all three Alpamayo generations; plans left of the dashed line are classified as stops. Lowering the temperature of the accurate solver (top rows) shrinks plans toward their own mode, so stops become more frequent. Fewer Euler steps (bottom rows) move every seed toward the one-step conditional mean ($E_1$): the stationary vehicle in clip 113 begins to creep and the hesitant stop in clip 141 disappears. All stopping plans in our 60-clip sample come from these and two other clips; the effect is consistent across models, but its prevalence requires a larger stop-rich sample (\cref{app:protocols}).

### F4 (recommended, fills the page budget). Mechanism figure (single column, 3.25 × 2.0 in, in §3)
- **(a)** A 1-D two-mode mixture (weights 0.2/0.8, as in `mean_attraction_v1_20261026`): exact flow paths (thin gray) and $E_4$ paths from the same noises (blue). Paths that change class are drawn in vermillion; they all go minority → majority and end closer to the mean. x = t, y = $x_t$; a sideways histogram at t = 1. Text inset: "99.95% of 408,769 flips end closer to the mean".
- **(b)** Predicted $\hat\rho_k$ vs conditional std $s$ (log x, 0.03–3) for k = 4 and 10 (`gaussian_contraction_v1.curve`), with the measured medians and IQRs (0.64/0.84) as markers at the clip-weighted effective $s$. This replaces Table 1.
- Caption claim: **"Euler contracts toward the mean by a predictable factor, and flips go to the majority mode."**

---

## 6. PAGE BUDGET (8 pages, ICML two-column)

Current body: ~3.45 pages, of which ~0.35 is TODO boxes and ~0.25 is the teaser placeholder.

| Section | Now | Target | What to add |
|---|---|---|---|
| 1 Intro (+ F1 teaser) | 1.0 | 1.3 | thesis sentence, findings-in-brief, real teaser (0.4) |
| 2 Background and Notation | 0.45 | 0.5 | sampler table with NFE/latency (0.15) |
| 3 Mean Collapse + Protocol (merged) | 0.75 | 1.35 | intuition sentence, Eq. (2), proof sketch, **F4** (0.4) |
| 4 Setup | 0.3 | 0.4 | Qwen reference, decision thresholds, stop-candidate sample |
| 5 Results | 1.1 | 2.9 | Table 1 redesign (0.25), **F2** figure* (0.35), **F3** (0.35), FlashDrive finding (0.25), closed-loop table if ready (0.3), counts/equivalence text |
| 6 Related Work | 0.35 | 0.45 | named citations |
| 7 Discussion and Limitations | 0.3 | 0.5 | limitation on stop-sample size; when the mechanism does not apply |
| **Total** | **~3.45** | **~7.4** | 0.6 page slack for floats |

Even with all three requested figures and F4, the paper reaches ~6.5 pages without the closed-loop and FlashDrive results. Fill any remaining space with evidence from the appendix: the K-mode mixture experiment (App. A), the pre-registered schedule negative result (App. B), and the SmolVLA/LIBERO transfer (App. C, as a two-line paragraph). Do not add prose padding.
