# Presentation Review, Round 5.2: `versions/main_v5.2.tex` / `main_v5.2.pdf`

**Reviewer focus:** clarity, structure, figures and tables, and how the new FlashDrive paragraph fits.
**Previous round:** `reviewer_presentation_v5.md` (written against v5; v5.1 changed the abstract, Finding 4 and "What the test identifies").
**Draft:** 8 pages. The body ends on p. 5, col. 2, references are on p. 5–6, and the appendix is on p. 7–8 (mostly TODO stubs).

**Score: 6/10** for presentation. It was about 5.5 at v5. The abstract and Finding 4 are now honest about what the per-sample evidence shows. The FlashDrive result is the most persuasive single piece of evidence for the paper's recommendation, but it is written as a lab-notebook paragraph at the end of Results. It has no table row, no figure marker, and no mention in the abstract, intro or contributions.

---

## 1. The new paragraph "A published acceleration" (l. 198–199)

**Fit.** This is the only place where the paper applies its protocol to a *published* acceleration, using that method's own code. It is the result that turns "a caution about Euler" into "your reported gain is a temperature change". Right now it is the last paragraph of §5, after "The one-step limit" and "Rare decisions", which are two boundary or caveat paragraphs. A reader skimming the Findings boxes will miss it. The intro already claims (l. 79) that "Accelerations that build on first-order integration … inherit this temperature shift", and this paragraph is the evidence for that claim. Yet nothing points from the claim to it.

**Clarity problems within the paragraph.**
1. **"the original Alpamayo-1.5 checkpoint"**: the paper says "Alpamayo-1.5" everywhere else. "Original" only makes sense next to the TODO's "optimized checkpoint (W4A8…)". Write "the released (unquantized) Alpamayo-1.5 checkpoint". Do the same in any later sentence about the optimized one.
2. **"on \FDClips{} further clips (4 windows each)"**: "further" is ambiguous. It could mean 60 *new* clips or 60 *more* of the same kind. "Windows" is a unit that §4 never defines; §4 uses six noise seeds at a fixed 5.1 s frame. Also state how many noise seeds were used, and that the bootstrap resamples clips (not windows). Suggested wording: "on \FDClips{} PhysicalAI-AV test clips disjoint from those above, four prediction times per clip and N seeds each, resampled by clip".
3. **The interval level switches without warning.** Table 1 and all figures use 95% intervals. Here the paragraph says "(90% interval)" with no reason given. Add "(90% interval, the equivalence test of §4)".
4. **"The upper bound sits at our 0.05 m margin"**: the bound is 0.051, which is just *above* the margin. Say "just exceeds", or the sentence reads as if equivalence nearly held.
5. **"almost all of the gain is a temperature effect"**: quantify it, as Finding 2 does. 1 − 0.008/0.28 ≈ 97%. Add a macro `\FDExpl` from the same generator rather than computing it by hand.
6. **$\taueff$ has no interval.** Every other $\taueff$ in the paper carries one. Add `\FDTauLo`/`\FDTauHi`.
7. **Only ADE$_1$ is reported.** The paper's primary endpoint is the flip rate (Findings 2–4, the abstract, the margins). A reader will ask whether FlashDrive also passes on flips. Report the flip residual, or say in one clause why it is absent (e.g. FlashDrive's code path does not expose a per-noise reference).
8. **"E$_{10}$ again has $\taueff$ \FDETenTau"**: "again" refers back to Finding 2's 0.83, two columns earlier. Make that link explicit: "($\taueff$ 0.84, as in \cref{find:matched}: the estimate transfers to new clips)". This is a small *replication* of the main finding on held-out clips, and it is currently given half a sentence.
9. **The last two sentences** ("We had predicted … The measured value is higher.") read like a log entry. Either drop them or make them a point. FlashDrive's $\taueff$ (0.64) coincides with $E_4$'s (0.61–0.64, Table 1). In other words, the cache with 8 steps behaves like 4 Euler steps, which is a vivid, citable sentence. Suggested replacement: "Its $\taueff$ equals that of $E_4$ (Table 1): reusing four of eight velocities contracts like halving the step count."
10. **The sampler has no symbol.** Give it one, such as $F_8$ ("FlashDrive's cache"), so that it can sit in Table 1 and Fig. 3.
11. The description in parentheses repeats l. 112 almost word for word. Drop it here, or drop it at l. 112.

**Drop-in replacement** (new macros: `\FDExpl`, `\FDTauLo`, `\FDTauHi`, `\FDSeeds`, and optionally `\FDFlipResid`):
```latex
\finding{find:flashdrive}{\emph{A published acceleration's gain is a temperature effect.}
FlashDrive's velocity cache ($F_8$; \cref{sec:background})~\citep{flashdrive2026}, run through its own code on the released Alpamayo-1.5 checkpoint, lowers ADE$_1$ by \FDGain\,m relative to $\Aref$ on \FDClips{} test clips disjoint from those above (four prediction times and \FDSeeds{} seeds per clip; clips resampled). Its $\taueff$ is \FDTau{} [\FDTauLo, \FDTauHi], the same as $E_4$'s (\cref{tab:matched}), and the reference solver at that temperature reproduces \FDExpl\% of the gain: the ADE$_1$ residual is \FDResid\,m (90\% interval). The upper bound just exceeds our 0.05\,m margin, so equivalence is not established. On the same clips the released $E_{10}$ has $\taueff$ \FDETenTau{} (residual \FDETenResid\,m), replicating \cref{find:matched} on new data.}
```
Place it **directly after Finding 4 (Shrink) and before the Qwen-Drive material**. Ideally, also demote Qwen-Drive to a paragraph, as proposed in v5 item 6. The Findings sequence then runs: mechanism → matched → Heun fails → Shrink caveat → published method. Move "The one-step limit" and "Rare decisions" after it as boundary paragraphs.

**Propagate the result** (it is currently mentioned only at l. 198):
- **Abstract (l. 70)**: before "We therefore propose…", add: "Applied to FlashDrive, a published velocity cache, the protocol attributes \FDExpl\% of its single-sample gain to an effective temperature of \FDTau."
- **Intro l. 77/79**: after "inherit this temperature shift", add "; for FlashDrive we measure it directly (\cref{find:flashdrive})".
- **Contributions (l. 98)**: add the FlashDrive result to bullet 2, and delete the TODO at l. 102 for the FlashDrive part.
- **Related Work (l. 204)**: change "Our protocol separates the part…" to "…; for FlashDrive that part is \FDExpl\% (\cref{find:flashdrive})."
- **Recommendations (l. 211)**: "cached sampler" now has a concrete example, so cite it.
- **Table 1**: add an Alpamayo-1.5 sub-block row "FlashDrive $F_8$ (new clips)" with ADE$_1$ columns. Mark the flip columns "—" if flips are unavailable, and add a footnote that this row uses a different clip sample.
- **Fig. 3(a), Alpamayo-1.5 panel**: one marker for $F_8$ at $\taueff=0.64$ (e.g. a hollow diamond). The ADE$_1$ change is on a different sample, so either plot it in a small inset or mark it in the caption as "different clips". At minimum, put it in Table 1.

## 2. Structure and story (carried over or new)

- **Abstract (l. 70)**: v5.1 fixed the conflict between the per-sample and rate statements. "The match holds for rates over clips, not for the same noise, and a linear shrinkage toward the mean passes it too" is now accurate. Remaining issue: the abstract says "linear shrinkage", while Finding 4's title and the Discussion say "uniform shrinkage" (l. 189 says "linear" in the title and "uniform" in the body). Pick **uniform** throughout, because it says what matters, namely one factor.
- **Contribution bullet 2 (l. 98)** still prints "79–121%" before §4 explains fractions above 100%. Replace it with residuals (v5 item 1 drop-in), or add "(above 100\%: temperature overshoots)".
- **Intro l. 77, "shrinking the initial noise"**: this is the clearest one-line definition of temperature in the paper. Echo it in §2's Notation ("$\Atau$ … started from $\tau z$, i.e. shrunken initial noise") so that readers connect the two.
- **Finding 4 (l. 189)** is now correct but dense: six bracketed intervals in one box, about 11 lines in the column. Move the four `\SigDiff…Ten` numbers to a small table or to the caption of a per-sample figure (`figures/fig_signature.pdf`, written in round 5 and still unused). Keep only the $E_4$ numbers in the box. Without a figure, the claim that "Euler's ratios vary across samples" is asked to be taken on trust.
- **"What the test identifies" (l. 213)** is now consistent with Finding 4. Good.
- **"Distribution, not trajectory" (l. 212)**: "a sizable share" is still unquantified.

## 3. Figures and tables

- **Table 1 placement**: it now lands at the top of p. 5, after Findings 1–4 and Fig. 3 on p. 4. That is better than v5, where it sat behind Related Work, but it is still cited first at l. 77 (p. 1) and in Findings 2–4 on p. 3–4. The v5 placement proposal (item 4: `\input` right after `\label{sec:protocol}`, single-column grouped version `figures/tab_matched_v5_proposal.tex`) still applies. The interval bug ([0.03, 4.4]) is fixed in v5.2.
- **Table 1 caption** still says "explicit-temperature curve", and so do the Fig. 3 caption and title (l. 169). Define "temperature curve" once in §3 or §4 and use only that phrase.
- **Table 1 caption** says "Shrink$_k$ … each $A_1$ sample moved linearly toward $E_1$ to $E_k$'s spread". Add "by one factor per model", to match Finding 4.
- **Fig. 3** still omits Shrink$_k$, which is the control that defines the paper's framing (v5 item 3; `figures/fig_curves_shrink_v3_2row.pdf` exists).
- **Fig. 1 caption (l. 89)** uses "95% clip-bootstrap band", and Fig. 3 uses the same phrase. That is consistent. The FlashDrive paragraph's 90% interval is the only outlier; see 1.3.

## 4. Line-level edits

| Line | Edit |
|---|---|
| 43 | `\newif\ifdraft` toggle around `\TODO`. The body still shows 8 orange boxes (l. 102, 146, 184, 187, 192, 194, 199, 214), and App. B–E are stubs. |
| 70 | Use "uniform shrinkage"; add the FlashDrive sentence (§1). |
| 98 | Remove "79–121%" or explain it; add FlashDrive. |
| 102 | Drop the FlashDrive part of the TODO once the paragraph becomes a Finding. |
| 110 | "Alpamayo-2-Super (Alpamayo-2 below)" is still missing (l. 158 introduces -Super, and the rest of the paper uses "Alpamayo-2"). |
| 112 | Give FlashDrive's sampler a symbol ($F_8$) here; remove the duplicate description at l. 198. |
| 144 | Define $h$: "whose step sizes $h$ satisfy". |
| 180 | Finding 1 title "Heun samples do not" → "barely do" (Fig. 2b: Heun-5 median 0.96). |
| 184 | `$\EntropyResidMin$` so that "−0.03" prints with a minus sign, not a hyphen. |
| 189 | Title: "linear" → "uniform"; slim the numbers (§2). |
| 191–192 | Finding 5 (Qwen-Drive) holds hand-typed numbers from a different protocol and has a TODO under it. Demote it to a paragraph (v5 item 6). |
| 198 | Replace with the Finding drop-in in §1; move it before l. 191. |
| 199 | Keep the minADE$_k$ crossover TODO separate from FlashDrive, because it belongs to Finding 2. |
| 204 | Cite the FlashDrive result. |
| 206 | Related Work still cites App. C as evidence while App. C is a TODO. |

## 5. Summary for the authors
The paper reads more honestly than v5. The new FlashDrive result is the strongest evidence that the recommendation matters in practice, but it is presented as an afterthought. Promote it to a Finding with a symbol, a $\taueff$ interval, an explained fraction and (if available) a flip residual. Then give it one row in Table 1 and one sentence each in the abstract, intro, contributions and Related Work. Finally, close the remaining v5 items: Table 1 placement, Shrink in Fig. 3, the per-sample figure, the draft toggle and the Qwen paragraph.
