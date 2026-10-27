# Presentation Review, Round 3: "When Fewer Steps Mean Lower Temperature: Euler Sampling of Flow-Matching Driving Policies"

Reviewer focus: writing, visual design, and results reporting, benchmarked against DDPM, EDM, Scaling Laws, and "Are Emergent Abilities a Mirage?".
Draft reviewed: `paper/main.tex` (197 lines, = `versions/main_v3.tex`) and `paper/main.pdf` (6 pages: body ends at about 3.6 pages, references on pp. 4–5, appendix stubs on p. 6). Previous reviews: `reviewer_presentation.md` (round 1) and `reviewer_presentation_v1.md` (round 2). I checked the numbers against `artifacts/paper_numbers_v1_20261026/summary.json` and the per-clip plans.

**Overall.** The narrower thesis is now the paper's best feature. It says that Euler behaves like temperature at the level of distributions, and that Heun is the negative control. This has the Mirage shape: one reframing claim, one device (compare at matched τ_eff), and one control that shows the device can fail. Table 2 now reads at a glance. The abstract is clean. The paper still *shows* nothing, because there is no figure, and that is what this round fixes. Three ready-to-run figure scripts are below and in `paper/figures/`. I also found one number bug, where a macro range hides the Alpamayo-2 value (§2, N1). One claim is slightly false as worded (N2).

---

## 1. Status of Round-2 Items

| Item (round 2) | Status | Note |
|---|---|---|
| M1 tail reported at data resolution | **Fixed** | The tail claim is now a "Rare decisions" paragraph with counts and 4 clips, and is no longer a finding. Wording: "6, 6 and 6 of 360 plans" reads like a typo. Write "6 of 360 plans for each model". |
| M2 report the equivalence test | **Partly** | The text says honestly that the test is underpowered, but it never says which cells pass. Add one clause: "the 2-point margin is met for Alpamayo-R1 $E_{10}$ only". The sentence "flip residuals are within ±2 points" is also not quite true (N2). |
| M3 table redesign (absolute values A₁ \| S \| A_τeff) | **Done; reads at a glance** | See §3 for the remaining fixes: a repeated A₁ column, an unused † footnote, numbering, and no ADE residual interval. |
| M4 bibliography placeholders | **Partly** | In-text citations now name the methods. But the reference list still prints "TODO verify title, authors and arXiv id" (Alpamayo), "TODO verify citation" (PhysicalAI-AV), and "TODO cite the technical report" (Qwen-Drive). |
| M5 one sample, one reference | **Mostly** | All main numbers now use 60 clips × 6 seeds with Heun-32. Two exceptions: (i) the "Gaussian" column of Table 1 comes from `gaussian_contraction_v1` (200 clips, Heun-160, action space, Alpamayo-1.5 only), while the caption says "60 clips × 6 seeds" for the whole table; (ii) the Qwen-Drive numbers are hand-typed and come from another protocol (TODO). Either recompute the Gaussian prediction on the 60-clip plans or label the column's sample in the caption. |
| M6 abstract | **Fixed** | One paragraph, 5 sentences, no TODO inside. See §4 for an optional tightening. |
| M7 `\pp` bug and inline TODOs | **Fixed / partly** | `\pp` is gone. The TODOs no longer split sentences. Four TODO boxes remain inside the body (§5). |
| S1 title | **Fixed** | The new title fits the narrower thesis exactly. Keep it. |
| S2 thesis sentence | **Partly** | Intro ¶1–2 carry the thesis, but there is still no single sentence that a reader can quote. See §4. |
| S3 contributions → findings in brief | **Partly** | Now 4 bullets, which is good. Only bullet 2 has a number, and none of them points to a Finding. |
| S4 notation | **Mostly** | Eq. (1) now defines θ, Heun-$k$ NFE are stated, and Spr is defined. Still mixed: "accurate solver" (abstract, intro, Findings 2–3, Discussion) vs. "reference solver" (Notation, tables). `A_{\taueff}` is still typed by hand because `\Atau` hard-codes τ. |
| S5 intuition sentence before the algebra | **Not done** | The new mechanism figure, panel (a), now carries this picture. Add one sentence that points to it (§4). |
| S6 finding boxes | **Not fixed** | `\parbox` still cannot break. Page 3 shows ~1 cm gaps between Findings 1/2 and 2/3 that will get worse once figures float. Finding 4 still has no pointer. |
| S7 structure | **Not done** | §4 "Matched-Temperature Protocol" is still a 7-line section, so the paper has 8 sections. **No NFE/latency number appears anywhere**, although the paper's premise is speed. |
| Polish: signed zeros, math minus | **Fixed** | |
| Polish: Prop. 3.1 $h$, `\operatorname{Lip}`, "Alpamayo-2-Super" vs "Alpamayo-2", `xurl` | **Not done** | |

---

## 2. New Must-Fix Issues

### N1. A macro range hides the Alpamayo-2 value (wrong number in two places)
The contributions bullet and Finding 2 both say temperature accounts for "**90–100%** of the decision flips of $E_{10}$". This is typeset as `\AOneFiveETenExplFlip--\ROneETenExplFlip`. The Alpamayo-2 value is **79** (`\ATwoETenExplFlip`), so the true range is **79–100%**. The other hand-paired ranges happen to be right today: the abstract's 0.83–0.84, ADE 92–121, $E_4$ 86–114 and 86–119, and the contraction ranges. They will break silently when the data change. **Fix:** have `paper_numbers.py` emit `...Min`/`...Max` macros (e.g. `\ETenExplFlipMin`, `\ETenExplFlipMax`, `\ETenTauMin`, `\ETenTauMax`) and never build a range from two named models.

### N2. "Flip residuals are within ±2 points" is false at two cells
From `summary.json`, the residuals are Alpamayo-2 $E_{10}$ **+2.04** and Alpamayo-R1 $E_4$ **−2.05**. The table rounds them to 2.0. Write "within about 2 points (largest +2.0 and −2.1)", or simply give the range "between −2.1 and +2.0 points". Readers who recompute will find this.

### N3. "Heun samples do not contract" overstates Table 1
Heun-5 has median ratio 0.96 with 69% of samples contracted. Heun-3 has **0.90** (70%), and it is not in Table 1 although it is in Table 2. The right comparison is at equal budget: Heun-3 (6 NFE) 0.90 vs. $E_6$ 0.75–0.78, and Heun-5 (10 NFE) 0.96 vs. $E_{10}$ 0.84–0.86. Retitle Finding 1 "*Euler samples contract toward the conditional mean; Heun samples at the same budget barely do*". Show this at equal NFE in the mechanism figure (panel b, below), which also lets you drop Table 1.

### N4. "Fraction explained" above 100% printed without comment
Values of 121%, 119%, 115% and 114% appear in the contributions and in Finding 2. A reader will stop at "accounts for 121%". Either say once "values above 100% mean the reference at $\taueff$ moves further than the sampler (overshoot)", or report the residual in metres or points instead of a percentage. The new main figure shows the residuals directly as gray ticks, which is the clearer encoding.

### N5. The abstract's "with the same budget"
Heun-5 uses 10 evaluations, the same as $E_{10}$. Heun-3 uses 6. Write "while Heun samplers with the same number of evaluations do not lie on the temperature curve". Better still, name Heun-5 alone. Equal NFE is the strongest form of the control, and the paper never says it outright.

---

## 3. Table 2 (main table): Does It Read at a Glance?

**Yes.** Laying out A₁ | S | A_τeff left to right lets the eye see "S has left A₁ and landed on A_τeff" for every Euler row, and Heun-3/5 jump out (17.8 vs 1.3). This is the most improved element of the draft. Remaining fixes, all in `make_tables.py`:

1. **A₁ is repeated in every row** (2.86 four times). Print it once per model in the first row and leave the cell blank below it. Or move A₁ into the model label: "Alpamayo-1.5 ($\Aref$ ADE$_1$ 2.86 m)". Blank cells help the eye more than repeated numbers.
2. **The † footnote is dead.** The caption defines "$^\dagger$ τ_eff at the edge of the measured range", but no cell carries the mark, because $E_4$/$E_{10}$ are all in range. Remove it, or add the $E_2$ rows where it applies.
3. **Numbering.** `tab_contraction` is input first, so it becomes Table 1, yet the intro's first citation is to the matched table (Table 2). Recommended: delete `tab_contraction` from the body (the mechanism figure, panel b, shows the same numbers at equal NFE) and move it to the appendix. The matched table then becomes Table 1.
4. **The ADE₁ residual has no interval.** Flips get `S−A_τeff [CI]`, but ADE does not. Either add the same column for ADE (the table* has room at `\small`), or state in the caption that every ADE residual interval contains zero, if that is true. Note that for Heun-3 it is +0.15 m and is not small.
5. **Heun ADE moves the other way.** Heun-3/5 *raise* ADE₁ (2.96 and 2.91 vs. 2.86). One clause in Finding 3 should say this. It strengthens the control, because Heun is not a cooler sampler at all.
6. `\scriptsize` CIs at `\small` render at about 6.5 pt. That is legible, so keep them.

---

## 4. Abstract, Thesis, and Section Flow

### Abstract (current: one paragraph, 5 sentences, 163 words, no TODO)
It passes the ICML form. Two optional improvements: (i) give the size of "most" (the reader has no number to hold on to); (ii) split sentence 4 (63 words). Suggested text (6 sentences, ~165 words; macros as in N1):

> Flow-matching driving policies are made faster by integrating with fewer Euler steps, and the change is usually reported as preserving or improving accuracy. We show that, at the level of the distributions they produce, few-step Euler samplers behave like an accurate solver run at a lower sampling temperature. One Euler step returns the conditional mean, and more steps contract samples toward it by an amount that a closed-form Gaussian factor matches at the median. On \NClips{} PhysicalAI-AV test clips, the released 10-step samplers of three Alpamayo generations share an effective temperature of \ETenTauMin--\ETenTauMax, and an accurate solver at that temperature reproduces most of their change in single-sample error and \ETenExplFlipMin--\ETenExplFlipMax\% of the driving decisions they change. Heun samplers with the same number of field evaluations change as many decisions without lowering the temperature. Accuracy gains from fewer steps should therefore be reported against an accurate solver at matched effective temperature.

### Thesis sentence
End intro ¶1 or ¶2 with one quotable italic line:
> *For first-order samplers, fewer steps change the policy's output distribution the way a lower temperature does; second-order samplers do not.*

### Intuition sentence (round-1 S7, still missing), before "For $p(x_1\mid c)=\dots$"
> For a Gaussian target, each Euler step moves along the tangent of the exact path instead of following it, and so loses a factor $\cos\theta_i$ of the deviation from the mean (\cref{fig:mechanism}a).

### Section flow and length (current page fractions → target with figures)

| Section | Now | Target | Notes |
|---|---|---|---|
| 1 Introduction | 0.85 | 1.25 | + teaser (0.4). Contributions: add a number to bullets 1, 3, 4 and `\cref{find:…}` to each |
| 2 Background and Notation | 0.45 | 0.6 | + samplers mini-table: NFE and ms per call for $E_1,E_4,E_{10}$, Heun-3/5, Heun-32 (0.15) |
| 3 Why Euler Steps Act Like Temperature | 0.6 | 1.0 | + mechanism figure (0.35), intuition sentence |
| 4 Matched-Temperature Protocol | 0.25 | → merge into §3 as §3.4 | 7 numbered sections |
| 5 Setup | 0.3 | 0.35 | say where the Qwen-Drive reference differs |
| 6 Results | 1.0 | 2.3 | + main figure* (0.45 for two rows, 0.6 for three), Table 1 (0.25); drop the contraction table |
| 7 Related Work | 0.35 | 0.4 | |
| 8 Discussion and Limitations | 0.35 | 0.45 | |
| **Body** | **~3.6** | **~6.4** | leaves 1.6 pages for the FlashDrive / closed-loop results if they land |

The paper is now short rather than long. Do not pad it. With the three figures it reaches ~5.5 pages of evidence, and the slack belongs to FlashDrive at matched temperature and the 436-clip confirmation.

---

## 5. Remaining TODO Placement

| TODO | Where | Recommendation |
|---|---|---|
| Teaser figure + caption | §1 | **Replace now** with `fig_teaser.pdf` (§6) |
| FlashDrive / AlpaSim contributions | after the contributions list | Move to a `%` comment. A reader should never see a promised contribution |
| Pre-registered 436-clip confirmation | inside Finding 2's box | Move out of the box, to a comment or to Limitations ("a pre-registered confirmation is under way") |
| Full proof / Lipschitz near t=1 | after Prop. 3.1 | Keep as a comment and add "(proof sketch; full proof in \cref{app:…})". The Lipschitz caveat belongs in the proposition's scope sentence |
| Qwen-Drive unification | inside Finding 4's box | Until the numbers are on the 60-clip sample, downgrade Finding 4 to a "Boundary case" paragraph. A finding box with a TODO inside reads as unfinished evidence |
| FlashDrive finding | end of Results | Comment until the run lands |
| Language ablation; ReCogDrive/DiffusionDrive | Limitations | Comment |
| Appendices A–D | p. 6 | Fill App. D (thresholds, seeds, splits) before anyone outside reads it, because Setup points there for the decision thresholds |
| Bib "TODO verify" notes (3) | references | Fix. They print in the PDF |

---

## 6. Figures: Scripts, Outputs, and LaTeX

All scripts live in `paper/figures/`. They share `figstyle.py`, which holds the Okabe–Ito colors, Times New Roman 8 pt text with STIX math, `pdf.fonttype=42`, and the data loaders. They recompute per-clip metrics with `cavla.paper_numbers.load`, so every curve uses the generator that produced `summary.json`. Sampler positions and τ_eff intervals are read from `summary.json`. Curve bands are 95% clip bootstraps (2,000 resamples, seed 20261026, the same as the generator). Run from `paper/`:

```bash
cd /mnt/datasets/CAVLA/paper
PY=/mnt/datasets/Explainability/alpamayo/ar1_venv/bin/python
PYTHONPATH=/mnt/datasets/CAVLA/src $PY figures/make_fig_teaser.py                 # -> figures/fig_teaser.pdf
PYTHONPATH=/mnt/datasets/CAVLA/src $PY figures/make_fig_curves.py                 # -> figures/fig_curves.pdf (3 rows)
PYTHONPATH=/mnt/datasets/CAVLA/src $PY figures/make_fig_curves.py --rows ade1,flip --out fig_curves_2row.pdf
PYTHONPATH=/mnt/datasets/CAVLA/src $PY figures/make_fig_mechanism.py              # -> figures/fig_mechanism.pdf
```
Each script takes about 10 s. All PDFs are vector with embedded TrueType fonts (Times New Roman, STIX). Natural sizes: teaser 3.07 × 2.20 in, mechanism 3.12 × 1.76 in, curves 5.83 × 4.0 in (3 rows) or 5.83 × 3.06 in (2 rows). I checked each one inside a scratch build of the paper at column and text width (main.tex untouched).

Color and marker conventions, the same in all three: black line with a gray band = reference $A_\tau$; blue = Euler $E_k$ (a one-hue ramp in the mechanism figure, darker = fewer steps); vermillion diamond = Heun; star = $E_1$ (conditional mean); circle/square/triangle = Alpamayo-1.5/R1/2.

### Figure 1 (teaser): `fig_teaser.pdf`, single column, replaces the placeholder
(a) Clip `context_101`, Alpamayo-1.5, ego at 10.5 m/s, six shared seeds. Five reference seeds turn left and one goes straight. Under $E_4$ every seed moves toward the one-step mean (star), and the straight seed becomes a left turn (gray arrow). $E_4$ changes 4 of 6 decisions on this clip. (b) Flips vs τ for Alpamayo-1.5: $E_2/E_4/E_6/E_{10}$ at τ_eff = 0.41/0.64/0.75/0.83 sit on the curve (14.7/10.6/7.2/4.7%). Heun-3/5 at τ_eff 0.95/0.98 flip 17.8/9.7%, against 1.3/0.6% on the curve. The script's `--clip` option exists if you prefer `context_106` (a quieter clip at 3.8 m/s with the same straight-to-left flip).

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_teaser.pdf}
  \caption{\textbf{Fewer Euler steps behave like a lower temperature; Heun steps do not.}
  (a)~One PhysicalAI-AV test clip (Alpamayo-1.5, ego at 10.5\,m/s), bird's-eye view, six shared noises.
  Four Euler steps ($E_4$, blue) move every sample of the reference solver ($\Aref$, black) toward the
  one-step conditional mean ($E_1$, star); the one seed that goes straight under $\Aref$ turns left.
  (b)~Share of decisions that differ from $\Aref$ under the same noise. Black: the reference solver started
  from $\tau z$ (95\% clip-bootstrap band). Euler samplers placed at their effective temperature fall on the
  curve; Heun-3 and Heun-5 change 17.8\% and 9.7\% of decisions at almost no change in temperature.
  \NClips{} clips $\times$ 6 seeds; bars are 95\% intervals.}
  \label{fig:teaser}
\end{figure}
```

### Figure 2 (mechanism): `fig_mechanism.pdf`, single column, in §3 right after Eq. (1)
(a) Closed form for one coordinate ($s=0.5$): the exact deviation $\|u(t)\|$ and the Euler polylines for $k=1,2,4,10$, ending at factors 0, 0.40, 0.68, 0.86. (b) Measured median contraction vs. NFE for the three models (60 × 6), with the Gaussian band for $s\in[0.3,1]$ ($E_{10}$: 0.82–0.88) and the Gaussian prediction (hollow squares, 0.67/0.82/0.86 at $k=4/8/10$). Heun-3 (6 NFE) sits at 0.90 and Heun-5 (10 NFE) at 0.96, above the Euler points at the same budget. This replaces Table 1 and fixes N3.

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_mechanism.pdf}
  \caption{\textbf{Euler contracts samples toward the conditional mean by the Gaussian factor of \cref{eq:contraction}.}
  (a)~One coordinate of a Gaussian target ($s=0.5$): deviation from the mean path under the exact flow (black)
  and under $k$ Euler steps (blue, darker $=$ fewer steps). Each step loses a factor $\cos\theta_i$, so $E_k$ ends
  at 0, 0.40, 0.68 and 0.86 of the exact deviation for $k=1,2,4,10$.
  (b)~Median ratio of each sample's distance to $E_1$ over the reference sample's, same noise, against field
  evaluations; three Alpamayo generations, \NClips{} clips $\times$ 6 seeds. Band: the Gaussian factor for
  $s\in[0.3,1]$; squares: its prediction from Alpamayo-1.5's conditional action variance (median and IQR;
  200 clips, action space). Heun samplers (vermillion) barely contract at the same budget.}
  \label{fig:mechanism}
\end{figure}
```
If you recompute the Gaussian prediction on the 60-clip plans, drop "200 clips, action space" from the caption.

### Figure 3 (main result): `fig_curves.pdf`, figure*, top of the Results page
Columns are models. Rows: (a) ΔADE₁ vs. $\Aref$, (b) decisions changed (%), (c) ΔminADE₆ vs. $\Aref$. The ADE rows use the **paired** per-clip change. Absolute ADE bands were dominated by differences in clip difficulty and hid the comparison (I tried both). Gray ticks join each marker to the curve at the same τ, so the tick is the residual of Table 2. Out-of-range samplers ($E_1$; $E_2$ for R1/A2) are omitted, and the caption says so.

**Two-row or three-row.** Row (c) shows that Euler sits slightly *above* the curve in best-of-six error (residual +0.00 to +0.11 m, larger for Alpamayo-1.5). The text currently does not mention minADE₆ at all. My recommendation: use the 2-row version (`fig_curves_2row.pdf`, 3.06 in tall) in the body, and put the 3-row version in the appendix with one honest sentence in Finding 2: "Best-of-six error is slightly worse than temperature predicts (residual 0.00–0.11 m; \cref{fig:curves-full})". Leaving minADE₆ out entirely invites the question from reviewers who know the round-1 story.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics{figures/fig_curves_2row.pdf}
  \caption{\textbf{Placed at its effective temperature, each Euler sampler lands on the explicit-temperature curve; Heun samplers do not.}
  Black: the reference solver started from $\tau z$, as a change from $\tau{=}1$, with a 95\% clip-bootstrap band.
  Blue: Euler with $k$ steps at its effective temperature (\cref{sec:protocol}; horizontal bars are 95\% intervals);
  vermillion: Heun-3 and Heun-5. Gray ticks join each sampler to the curve at the same $\tau$; their length is the
  residual of \cref{tab:matched}. (a)~Single-sample error falls with temperature, and Euler follows it.
  (b)~Euler changes as many decisions as the reference at the same temperature; Heun changes 10--18\% of decisions
  at $\taueff\approx1$. $E_1$, and $E_2$ for Alpamayo-R1 and -2, have $\taueff<0.4$ and are not shown.
  \NClips{} PhysicalAI-AV test clips $\times$ 6 seeds per model.}
  \label{fig:curves}
\end{figure*}
```
(Natural width 5.83 in keeps the 8 pt text. `width=\textwidth` scales it to about 9.3 pt, which is also acceptable.)

### Cross-references to add
- Intro ¶1: "(\cref{fig:teaser,tab:matched})".
- Finding 1 → `\cref{fig:mechanism}` (replaces `\cref{tab:contraction}`).
- Findings 2 and 3 → `\cref{fig:curves}` and `\cref{tab:matched}`.

---

## 7. Polish (short)
- One name: replace "accurate solver" with "reference solver" after the abstract (Findings 2–3 and the Discussion still use both).
- Add `\newcommand{\At}[1]{A_{#1}}` and use `\At{\taueff}` everywhere.
- Finding boxes: switch to the breakable `tcolorbox` given in round 2. The gaps on p. 3 come from this.
- "Alpamayo-2-Super" in Setup → "Alpamayo-2-Super (Alpamayo-2 below)".
- Prop. 3.1: define $h$, use `\operatorname{Lip}`, and write "for Lebesgue-almost every $z$".
- Keywords: add "effective temperature".
- Definition 4.1: mention that $\Spr$ is computed per clip and averaged (the Notation paragraph defines it per trajectory set).
- Related Work "Critical windows": give the number (−0.88 points [−1.28, −0.52]) where you cite the negative result. It is the best motivation in the paper.
