# Presentation Review, Round 4: `versions/main_v4.1.tex` / `main_v4.1.pdf`

Reviewer focus: story, figures, tables, notation, typography. The benchmark is ICML/NeurIPS outstanding-paper writing.
Draft: 8 PDF pages in total. The body ends at about p. 5, column 1 (≈4.6 pages of main text), references run p. 5–6, and the appendix is p. 7–8.
Previous round: `reviewer_presentation_v3.md`.
Scratch builds that test every proposal below (main.tex untouched): `/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/pres_round4/build/test2.tex` / `test2.pdf`.

**Overall.** The three figures landed well. The teaser and the mechanism figure each carry an argument on their own, and Fig. 3 reads at a glance. The main problems now are about **story and placement, not content**:

1. The title, abstract sentence 2, the teaser caption and the Table 1 caption still say "Euler *is* temperature". The paper's own Shrink_k control, together with the Discussion, now says "Euler is *a contraction*; temperature is one visible knob that matches its rates". A reader meets the headline first and the caveat last (p. 5, Discussion). This is the single biggest presentation risk, because a reviewer who finds the caveat will feel it was buried.
2. The Shrink_k rows sit in Table 1 but are **never discussed in Results**. They are also not in any figure.
3. Table 1 floats to p. 5, after Related Work. Its first citation is on p. 1, and Findings 2–3 cite it on p. 3. The "5. Results" heading is also stranded at the bottom of p. 3, column 1, with no text under it.

All three can be fixed with the material below. I tested each fix in the scratch build.

---

## P0: Must fix before anyone outside reads it

### P0-1. Align title, abstract and captions with the contraction framing
The current abstract states the strong claim (sentence 2), then retracts it in sentence 5 ("any contraction toward the mean of the same size would pass it"). It also prints "79–121%", and the ">100%" is not explained until §4. Proposed abstract (≈175 words; uses existing macros except where noted):

> Flow-matching driving policies are accelerated by integrating with fewer Euler steps, and the change is usually reported as preserving or improving accuracy. We show that such comparisons are confounded by a hidden change in sampling temperature. One Euler step returns the conditional mean, and more steps contract samples toward it by an amount that a closed-form Gaussian factor matches at the median; a lower sampling temperature produces the same contraction explicitly. On \NClips{} PhysicalAI-AV test clips, the released 10-step samplers of three Alpamayo generations share an effective temperature of \AOneFiveETenTau--\ROneETenTau. Running an accurate reference solver at that temperature reproduces their gain in single-sample error and their rate of changed driving decisions to within \FlipResidMaxAbs{} points. Heun samplers with the released budget fail the same test, changing \AOneFiveHeunFiveFlipResid--\AOneFiveHeunThreeFlipResid{} points more decisions. A post-hoc shrinkage toward the mean also passes, so the test identifies contraction rather than temperature as such; temperature is the explicit, tunable knob that reproduces it. We propose that accuracy gains from fewer steps be reported against a reference solver at matched effective temperature, and we give the protocol and its clip-level statistics.

Title options, in order of preference. The current title asserts the identity that the Shrink control qualifies.
- **"Fewer Steps, Hidden Temperature: Evaluating Few-Step Flow-Matching Driving Policies at Matched Spread"**
- "Few-Step Euler Is a Contraction Toward the Mean: Matched-Temperature Evaluation of Flow-Matching Driving Policies"
- Keep the current title only if the abstract adopts the wording above. "Mean" in "When Fewer Steps Mean Lower Temperature" can then be read as "amount to at the rate level".

Captions that still claim more than the evidence supports:
- Fig. 1 title: "Fewer Euler steps behave like a lower temperature; Heun steps do not." → **"Fewer Euler steps contract samples toward the mean, as a lower temperature does; Heun steps change decisions without contracting."**
- Fig. 3 title: if Shrink is added (P0-2), use **"At matched spread, Euler and a post-hoc shrinkage land on the explicit-temperature curve; Heun does not."**
- Table 1 title: see the proposal file (P0-3).

### P0-2. Give Shrink_k its own Finding in Results and put it in Fig. 3
The control that defines the paper's framing appears only in the Discussion. Move its core into Results as a Finding, placed after Finding 3 (Heun), so that the two controls sit side by side: one shows the test can fail, the other shows what it cannot tell apart. Then shorten "What the test identifies" in the Discussion to its interpretive half.

Drop-in Finding (macros exist in `numbers_tables.tex`):
```latex
\finding{find:shrink}{\emph{A contraction without an integrator passes too.} Moving each reference sample linearly toward $E_1$ until its spread matches $E_k$'s (Shrink$_k$) gives flip residuals of \ShrinkFlipResidMin{} to \ShrinkFlipResidMax{} points, all with intervals containing zero, and ADE$_1$ on the temperature curve (\cref{fig:curves,tab:matched}). Rate-level matching therefore identifies a contraction toward the conditional mean of the right size, not temperature as such. The two contractions differ per sample: Shrink$_k$ moves every sample by the same factor (median \ShrinkContrFour{} at $E_4$'s spread), whereas Euler moves most samples less and a few more (\cref{find:contraction}).}
```
Then the Discussion paragraph "What the test identifies" can keep only its last two sentences, starting "What the evidence supports …".

**New figure script (written):** `paper/figures/make_fig_curves_shrink.py` → `figures/fig_curves_shrink_2row.pdf` (3-row output with `--rows ade1,flip,minade6 --out fig_curves_shrink.pdf`). It is identical to `make_fig_curves.py` except for two changes: (i) it reads `paper_numbers_v2_20261026/summary.json`, the file Table 1 is generated from (v2 ⊇ v1, and I checked that every Euler/Heun entry is identical); (ii) it draws Shrink_4 and Shrink_10 as hollow black squares, each at its own τ_eff, with a residual tick, plus a legend entry "Shrink_k (control)". Every square sits on the curve, which makes the identification caveat visible at a glance.
```bash
cd /mnt/datasets/CAVLA/paper
PYTHONPATH=/mnt/datasets/CAVLA/src /mnt/datasets/Explainability/alpamayo/ar1_venv/bin/python figures/make_fig_curves_shrink.py --rows ade1,flip --out fig_curves_shrink_2row.pdf
```
Caption sentence to add to Fig. 3: "Hollow squares: Shrink$_k$, each reference sample moved linearly toward $E_1$ to $E_k$'s spread (no integrator); it also lands on the curve (\cref{find:shrink})."
Separately, point `figstyle.NUMBERS` at v2 so that all figures and tables use one file. Today the figures read v1 and the tables read v2. The numbers agree, but a single source is the right invariant.

### P0-3. Table 1: placement and layout (6 rows per model)
**Readability.** Yes, it is still readable, but it is harder than it needs to be. Shrink rows are interleaved with Euler rows (E4, Shrink4, E10, Shrink10), so the eye has to skip every other line to follow the Euler story. A₁ = 2.86 is printed six times (this was a round-3 item, still open). The table* is only ~55% of the text width, so wide white margins frame it. It floats to p. 5, after Related Work.

**Proposal (written, tested):** `paper/figures/make_tab_matched_proposal.py` → `paper/figures/tab_matched_proposal.tex`. This is a **single-column** `table` at `\footnotesize`:
- A model sub-header row ("**Alpamayo-1.5**  A₁: ADE₁ 2.86 m") replaces both the Model column and the repeated A₁ column.
- Rows are grouped by role: Euler (black), then Shrink_k (gray, the control that passes), then Heun (black, **bold** residual, the control that fails), with `\addlinespace` between groups.
- Residuals are signed (+1.5, matching the text's "+16.5").
- Uses `\At{\taueff}` throughout.
- It fits within 3.25 in (checked in the scratch build) and is fully readable at 200 dpi.

Use: replace `\input{tables/tab_matched}` with `\input{figures/tab_matched_proposal}`, or port the generator logic into `tables/make_tables.py`.

**Placement (tested).** Move the `figure*` block for Fig. 3 from §5 to directly after `\end{figure}` of Fig. 2 (source on p. 2). Two-column floats go to the top of the next page, so Fig. 3 lands at the top of p. 3. Keep the table `\input` where it is (just before Finding 2). Result in `test2.pdf`:
- p. 3: Fig. 3 on top, then §4 and the "5. Results" heading, with Finding 1 under it. The widowed heading is gone.
- p. 4: Table 1 at the top of column 1, beside Findings 2–3, which cite it.
- The body still ends on p. 5.

### P0-4. Table 1 shows "[0.0, 4.4]" for an interval that the text says excludes zero
Alpamayo-2 $E_{10}$ has a flip residual of +2.0 with interval [0.03, 4.4], which prints as **[0.0, 4.4]**. Finding 2 says "one interval excludes zero". A reader comparing the two sees a contradiction. Fix it in the generator: print the CI at 2 decimals when |bound| < 0.05, or mark the cell with `$^{*}$` and add "$^{*}$interval excludes 0" to the caption.

### P0-5. Shrink_k's τ_eff differs from E_k's
Shrink_4 is "moved … to E_4's spread", yet its τ_eff is 0.66 against 0.64 for E_4 (R1: 0.65 vs 0.61). Readers will ask why. Add one clause to the caption or the Finding: "matched per clip, so the pooled τ_eff differs slightly from $E_k$'s". Use whatever the true reason is. The alternative is to match pooled spread so that the two coincide.

---

## P1: Notation and naming

| Item | Status | Fix |
|---|---|---|
| "reference solver" as the one name | **Done.** "accurate" survives only as an adjective ("an accurate reference solver", abstract and intro ¶1), which is fine. | — |
| `\At{\taueff}` / `A_{\taueff}` | Text OK. The table header uses a hand-typed `A_{{\taueff}}` in `make_tables.py`, which renders the same. | Use `\At{\taueff}` in the generator (done in the proposal). |
| Name of the curve | "explicit-temperature curve" (Fig. 3, Table 1), "temperature curve" (Fig. 4, Finding 3), "reference at the same temperature" (Fig. 1) | Define it once in §4: "the *temperature curve* $\tau\mapsto M(\Atau)$". Then use "temperature curve" everywhere. |
| Heun "do not" vs "barely" contract | Finding 1 title "Heun samples do not"; Fig. 2 and Finding 3 "barely"; Discussion "without contracting". Heun-5 median ratio 0.96, 69% contracted. | Use "barely" everywhere: Finding 1 title "…; Heun samples barely do", Discussion "Heun fails because it changes decisions while barely contracting". |
| Fig. 2(b) y-label "contraction ratio" | Higher value = *less* contraction, which reads backwards | Use "distance to $E_1$ (ratio to $\Aref$)" or "$\|S-E_1\|/\|\Aref-E_1\|$". |
| Fig. 2(a) y-label $\|x_t-tm\|/\|z\|$ | Uses symbols that the text does not define; the text uses $u_d(t)$, $m_d$ | Use "deviation from mean path (rel.)" or $\|u(t)\|$ to match §3. |
| "Alpamayo-2-Super" (Policies) vs "Alpamayo-2" elsewhere | Open since round 2 | "Alpamayo-2-Super (Alpamayo-2 below)". |
| Prop. 3.1: $h$ undefined, `\mathrm{Lip}` | Open since round 2 | "whose step sizes $h$ satisfy $h\operatorname{Lip}(v(\cdot,t))<1$". |
| Text-mode minus | Finding 2: "decision-entropy residuals lie between -0.03 and 0.03" prints a hyphen | `$\EntropyResidMin$`, or have the macro emit `$-$`. |
| Finding 1 "per-model values in Table 2" | Table 2 is now in the appendix | "(per-model values in \cref{app:extra})". |
| Discussion "90–93%" | Hand-typed | Use the existing `\AOneFiveETenFracContracted`-style macros, or a Min/Max pair. |
| Appendix Table 2 floats above the "A. Additional Results" heading (p. 7) | — | Put `\input{tables/tab_contraction}` after the first sentence of App. A, or use `[h!]`. |

## P2: Finding boxes and TODO boxes

- **Finding boxes now work.** The `framed` leftbar breaks across columns (Finding 2 spans p. 3–4 cleanly), and the round-3 gaps are gone.
- **TODO boxes inside Findings (2, 3, 4) break the visual unit.** The leftbar runs through an orange box. In the v4.1 PDF, Finding 4's bar continues onto p. 4 with *only* a TODO under it. For any version shown to co-authors or reviewers, move every in-Finding TODO to a `%` comment right after the `\finding{...}` call. Proposed toggle, so drafts still show them:
  ```latex
  \newif\ifdraft \drafttrue   % \draftfalse for any external version
  \newcommand{\TODO}[1]{\ifdraft\todo[inline,color=orange!25]{TODO: #1}\fi}
  ```
  Even with the toggle, put TODOs *after* the `\finding{}` and not inside its argument, so a draft render never shows an orange box inside a claim.
- **Finding 4 (Qwen-Drive)** still has hand-typed numbers from another protocol, and a TODO inside it. As in round 3: downgrade it to a "Boundary case" paragraph until the numbers are on the 60-clip sample. A Finding box should hold only finished evidence. This also makes contribution bullet 4 honest ("A boundary" → cite the paragraph).
- Contribution TODO (FlashDrive/AlpaSim) right under the contributions list: comment it out. A reader should never see a promised contribution.

## P3: Abstract and intro vs. the new framing

The intro does not yet match the framing either:
- ¶1 ends "obtained by keeping the reference solver and shrinking the initial noise". Good.
- ¶2 says "the same place a lower sampling temperature puts it". Good.
- The methodological message and the Shrink caveat are missing until p. 5. Add a short ¶3 before the teaser/contributions:

> Our claim is about evaluation, not about what Euler "really is". At the level of rates over clips, a few-step Euler sampler cannot be told apart from the reference solver at a lower temperature, nor from any other contraction toward the mean of the same size (\cref{find:shrink}). Temperature is the explicit and tunable member of that family, so it is the right baseline. Gains that it reproduces should not be credited to the sampler. Second-order samplers at the same budget change decisions without contracting and fail the comparison (\cref{find:heun}), so the test is not vacuous.

- Contribution bullet 2 prints "79–121%". Replace it with the residual: "flip residuals within \FlipResidMaxAbs{} points". The >100% problem from round 3 (N4) persists wherever the fraction appears before §4 defines it: abstract, contributions, Finding 2. Finding 2 can keep the fraction, because §4 explains it.
- Contribution bullet 1 could carry its strongest number: "(Heun-5 at the released budget: +\AOneFiveHeunFiveFlipResid{} points)".

## P4: Page budget

The body is ≈4.6 of 8 pages, so there is no pressure. For ICML, a 4.6-page main text reads as short. Do not pad it. Spend the room (≈3 pages) on, in order:
1. The FlashDrive matched-temperature Finding when it lands (≈0.4 page, with a row in Table 1 or a marker in Fig. 3).
2. The per-sample view that separates Euler from Shrink (≈0.35 page). A single-column histogram of per-sample contraction ratio $\|S-E_1\|/\|\Aref-E_1\|$ for $E_4$ vs Shrink_4 (Alpamayo-1.5) would *show* the one place where Euler and shrinkage differ. It is the natural visual for "rate-level evaluation cannot tell them apart; sample-level can". It is not written yet, because the per-sample shrink ratios are not in `summary.json`.
3. The "Distribution, not trajectory" paragraph needs its number ("disagree on X% of individual decisions"). Without it, "sizable share" is the vaguest phrase in the paper.
4. Pull the App. C negative result (−0.88 points [−1.28, −0.52], then zero at matched temperature) into a short Results paragraph. It is the paper's best motivating example, and today it is a TODO stub in the appendix.

## P5: Figure polish (small)

- Fig. 3(b), Alpamayo-1.5: Heun-3's gray residual tick crosses Heun-5's τ_eff bar and label. Offset the "Heun-5" label to the left (`xytext=(-6,3)`), or draw Heun ticks at zorder below the error bars.
- Fig. 3(a): the Heun τ_eff bars extend past τ = 1.0 to the axis edge. Fine, but `xlim` 1.05 would give the markers room.
- Fig. 1(b): the label "A_τ" under the curve is good. Keep it; it is the only in-plot name for the curve.
- Fig. 4 (appendix) repeats rows (a)–(b) of Fig. 3. Showing only row (c) would save ½ page of appendix, but this is optional.
- All figures: fonts match the body (Times, 8 pt) and are vector. No action.

---

## Files written this round (new names only; nothing existing was modified)
- `paper/figures/make_fig_curves_shrink.py` → `paper/figures/fig_curves_shrink_2row.pdf` (Fig. 3 with the Shrink_k control; reads v2 numbers)
- `paper/figures/make_tab_matched_proposal.py` → `paper/figures/tab_matched_proposal.tex` (single-column grouped Table 1)
- Scratch build showing the new placement: `/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/pres_round4/build/test2.pdf`. It has Fig. 3 moved to after Fig. 2, the proposed table, and the Shrink figure.
