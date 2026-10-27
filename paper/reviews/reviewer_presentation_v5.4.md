# Presentation Review, Round 5.4: `versions/main_v5.4.tex` / `main_v5.4.pdf`

**Reviewer focus:** clarity, structure, figures and tables, and whether promoting FlashDrive to a Finding worked.
**Previous round:** `reviewer_presentation_v5.2.md`.
**Draft:** 8 pages. The body ends on p. 5. Table 1 is still on p. 5. The appendix (p. 7–8) is still mostly TODO stubs.

**Score: 6.5/10** for presentation (6 at v5.2). The promotion worked where it mattered most. FlashDrive is now a boxed Finding (`find:flash`, Finding 5, l. 191). It sits in the recommended slot, after Shrink and before Qwen-Drive. The abstract (l. 70) and contribution 2 (l. 98) now carry it. The abstract sentence "behaves like four Euler steps" is the most memorable line in the paper. What stops a higher score: the Finding box mixes up its sample description, the result has still not reached the intro, Related Work, Recommendations, Table 1 or Fig. 3, and most of the v5/v5.2 structural items are still open.

---

## 1. Did the FlashDrive promotion work?

**What worked**
- **Visibility.** A reader who skims the Findings boxes now meets the published-method result. The Findings now run mechanism → matched → Heun fails → Shrink caveat → published method → boundary, which is the order proposed in v5.2.
- **Abstract framing (l. 70).** The new comparison point is plain 8-step Euler (\FDGainEEight = 0.12 m), not $\Aref$. This is the fairer comparison and the easier one to read. "Behaves like four Euler steps ($\taueff$ 0.64)" is concrete and easy to quote.
- **v5.2 items fixed:** a $\taueff$ interval (`\FDTauCI`); the 90% interval explained ("as in our equivalence test"); "just exceeds the 0.05 m margin"; the comparison with $E_4$ (`\FDVsEFour`), which turns the old "$\taueff$ equals $E_4$'s" remark into a measured difference.

**What did not work, or got worse**
1. **The sample description contradicts itself (l. 191).** The text says "using the same \FDClips{} clips and seeds, with 4 consecutive 10 Hz inputs per clip". §4 (l. 159) defines the main sample as one frame at 5.1 s. So this is *not* the same sample, and the numbers show it: $E_4$ has $\taueff$ 0.63 here against 0.64 in Table 1, and $E_{10}$ has 0.84 against 0.83. A careful reader will see these small mismatches and lose trust. (v5.2 described the sample as "further clips". The two versions must agree with the data. Check which is true.) Suggested wording: "on the same \FDClips{} clips and six seeds, but at four consecutive 10 Hz prediction times per clip (FlashDrive's streaming input) instead of the single frame of §4; $E_4$ and $E_{10}$ are re-run on this sample, and clips are resampled jointly."
2. **"We re-implemented this cache inside FlashDrive's own inference code"** reads as if the authors wrote the cache themselves. That weakens the "their own code" argument that makes the result persuasive. If the released cache was used unchanged, say "We ran FlashDrive's released cache in its own inference code". If the authors did re-implement it, say that it was checked against the released outputs.
3. **"original Alpamayo-1.5 checkpoint"** is still there (v5.2 item 1.1). Write "the released (unquantized) Alpamayo-1.5 checkpoint". "Original" only makes sense next to the l. 200 TODO.
4. **The title overclaims relative to the body and §7.** "A published acceleration *is* a lower temperature" is stronger than what the box shows: equivalence was not established, only ADE$_1$ was tested, and §7 (l. 214) says the claim is "at the level of rates". Suggested title: "*A published velocity cache acts like a lower temperature.*"
5. **"indistinguishable from $E_4$"** sounds like an equivalence claim, but no equivalence test was run on it. Write "matches $E_4$ (ADE$_1$ difference −0.001 [−0.033, 0.032] m)" and let the interval speak.
6. **No flip rate.** The paper's primary endpoint is decision flips, and this Finding reports only ADE$_1$. Either report the flip residual or add a clause explaining why it is absent. (Carried over from v5.2 item 1.7.)
7. **The last sentence is still a log entry:** "We had predicted 0.5–0.6 … the measured value is higher." Either delete it or make it a point: what prediction, from where (the Gaussian factor of Eq. 1 with the cached schedule?), and what follows from the miss. As written, it raises an unanswered question at the end of the paper's best Finding.
8. **The abstract hedges in a misleading way (l. 70).** "The reference at that temperature is within \FDResidHi\,m of it" quotes the upper bound of a 90% interval as if it were a distance. That bound exceeds the margin that §4 defines. Write instead: "…; the reference at that temperature matches its ADE$_1$ to within 0.01 m (90% interval up to 0.051 m)". Or drop the clause, since "behaves like four Euler steps" already carries the point.
9. **The description is duplicated.** "FlashDrive reuses the velocity on steps 3–6 of an 8-step Euler schedule" appears word for word at l. 112 and l. 191 (v5.2 item 1.11). Keep it at l. 112, and give the sampler a symbol there (e.g. $F_8$) so that the Finding, the table and the figure can refer to it.
10. **$E_8$ is used but not defined as a baseline.** The abstract's headline gain is relative to $E_8$, which appears in no table or figure. Give $E_8$'s $\taueff$ in the box so the reader sees the step from $E_8$ to the cache in temperature, not only in metres. `\FDGain` (0.28 m vs $\Aref$) is now unused; either use it or delete it from the generator.
11. **The box is long.** Finding 5 is 9 sentences, more than 20 lines of the column on p. 4. Items 5, 7 and 9 together would cut about 4 lines.

**Propagation still missing (v5.2 §1 list):**
- **Intro l. 79:** after "inherit this temperature shift", add "; for FlashDrive we measure it (\cref{find:flash})". The intro makes the claim and the Finding is its evidence, but there is still no link between them.
- **Related Work l. 205:** "Our protocol separates the part of such gains…" should name the FlashDrive result.
- **Recommendations l. 212:** "cached sampler" now has an example; cite `find:flash`.
- **Table 1 / Fig. 3:** add a FlashDrive row (a separate sample, footnoted; flips "—" if unavailable) and an $F_8$ marker on the Alpamayo-1.5 ADE$_1$ panel. A Finding with no table or figure behind it breaks the paper's own rule (l. 30–31: "one claim, its number, and the table or figure that supports it").
- **l. 200 TODO:** split it. The optimized-checkpoint run belongs under Finding 5; the minADE$_k$ crossover belongs under Finding 2.

## 2. Highest-value remaining edits (ranked)

| # | Line(s) | Edit | Why |
|---|---|---|---|
| 1 | 191 | Fix the sample description ("same clips" vs 4 frames; E4/E10 $\taueff$ mismatches with Table 1) | Internal inconsistency; costs trust |
| 2 | 182 / `tab_matched` | Move Table 1 up so it appears with §4/§5 (it is cited at l. 77 and in Findings 2–4 but prints on p. 5), and add the FlashDrive row | The main evidence table trails its citations |
| 3 | 43, 102, 146, 184, 187, 194, 196, 200, 215; App. B–E | `\ifdraft` toggle around `\TODO`; there are still 8 orange boxes in the body and 4 stub appendices. Related Work (l. 207) cites App. C, which is a stub | Unsubmittable as rendered |
| 4 | 70, 189, 214, `tab_matched` caption | Use "uniform shrinkage" throughout. The paper says "linear" at l. 70, 189 (title) and 214 and in the Table 1 caption, but "uniform" in the body at l. 189 | Same concept, two names (open since v5) |
| 5 | 98 | "\ExplMin–\ExplMax\%" (79–121%) in contribution 2 appears before §4 explains fractions above 100%, and §4 then says not to read fractions as evidence. Replace with "flip residuals within \FlipResidMaxAbs{} points" | Contradicts the paper's own reading guidance |
| 6 | 191 | Title "is" → "acts like"; "indistinguishable" → "matches (interval)"; drop or justify "We had predicted…" | Overclaim and log-entry tone in the headline Finding |
| 7 | 70 | Rephrase "within \FDResidHi\,m" (§1 item 8); the abstract is 235 words, so cut the Heun range or the shrinkage clause | Accuracy and length |
| 8 | 79, 205, 212 | Cite `find:flash` | Propagation |
| 9 | 168 / Fig. 3 | Add the Shrink$_k$ control and an $F_8$ marker (`figures/fig_curves_shrink_v3_2row.pdf` exists) | The defining control is still not shown in any figure |
| 10 | 189 | Finding 4 still has six bracketed intervals. Move the `\SigDiff…Ten` numbers to a per-sample figure (`figures/fig_signature.pdf`) or a table | Density; "ratios vary across samples" has no figure behind it |
| 11 | 193–194 | Qwen-Drive Finding 6: hand-typed numbers from a different sample, with a TODO under it. Demote it to a boundary paragraph next to "The one-step limit" | Weakest box among the Findings; the contribution 4 bullet can stay |
| 12 | 169, `tab_matched` caption | Define "temperature curve" once in §4 and drop "explicit-" | Terminology |
| 13 | 110 / 158 | "Alpamayo-2-Super (Alpamayo-2 below)" | Name drift |
| 14 | 144 | Define $h$ ("step sizes $h$") | Undefined symbol |
| 15 | 180 | "Heun samples do not" → "barely do" (Heun-5 median 0.96; the body at l. 186 says "barely contract") | Title vs body |
| 16 | 184 | Check that `\EntropyResidMin` (−0.03) prints with a minus sign, not a hyphen (the macro is plain `-0.03` in text mode) | Typography |
| 17 | 213 | Quantify "a sizable share" | Vague |
| 18 | 77 / 110 | Echo "shrinking the initial noise" in the Notation for $\Atau$ | Connects the intro intuition to the formal definition |

## 3. Summary for the authors
Promoting FlashDrive was the right call. It now gives the abstract a concrete, quotable consequence ("an 8-step cache behaves like four Euler steps"), and the Finding is in the right place. Before submission, fix the sample description in the box, which currently says "same clips" while using different frames and slightly different $\taueff$ values. Soften the title and "indistinguishable", and either remove the prediction sentence or give it a point. Then finish the propagation: a Table 1 row, a Fig. 3 marker, and one citation each at l. 79, 205 and 212. After that, the largest presentation gains are structural and have been open since v5: Table 1 placement, the TODO toggle and stub appendices, "uniform" vs "linear", the 79–121% in contribution 2, and Shrink in Fig. 3.
