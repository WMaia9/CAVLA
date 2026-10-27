# Presentation Review, Round 5.8: `versions/main_v5.8.tex` / `main_v5.8.pdf`

**Reviewer focus:** clarity, structure, figures, notation, flow, precision of claims. Special attention to the new grid bound (Gaussian paragraph, l. 125), the rewritten Qwen-Drive paragraph "A second family" (l. 213), and contribution 4 (l. 100).
**Previous round:** `reviewer_presentation_v5.4.md` (score 6.5).
**Draft:** 9 pages; body now ends on p. 6 (Discussion runs over), Table 1 still on p. 5, appendix still TODO stubs.

**Score: 6.5/10** (unchanged). The FlashDrive Finding is now honest and well-scoped, and the new bound is a real addition to §3. But the new bound is stated without its scope, and the Qwen-Drive paragraph as rewritten seems to contradict it: Qwen's $E_{10}$ has $\taueff=0.98$, above the "any grid" cap of 0.883. The Qwen paragraph is also the densest and least precise text in the paper. Most of the structural items open since v5 are still open.

---

## 1. Changes since v5.4

**Improved**
- Finding 5 title is now "behaves like" (was "is"). "Indistinguishable" has been replaced by the $E_4$ difference with its interval (`\FDVsEFour`). The sample is now described explicitly (4 consecutive 10 Hz frames; "$\taueff$ values therefore differ slightly from Table 1"). This resolves the v5.4 #1 trust problem.
- $E_8$, $E_4$ and $E_{10}$ residuals are now given in the box, with which ones pass the margin. That makes the comparison fair and readable. The "all intervals are 90%" sentence removes the 90/95 confusion.
- The intro (l. 79) and contribution 2 (l. 98) now cite `find:flash`.
- The abstract no longer quotes the 0.051 upper bound as a distance (v5.4 #8).
- Qwen-Drive is no longer a weak Finding box (v5.4 #11 done). It is now a Discussion paragraph.
- The new grid bound $\prod_i\cos\theta_i\le\cos(\pi/2k)^k$ (l. 125) is a clean, quotable result. "Choosing where the steps go … cannot remove it" is exactly the sentence a schedule-design reader needs.

**Regressed or new problems**
- The abstract grew from 235 to 252 words.
- The body spilled onto p. 6.
- The Qwen paragraph adds a second set of results with no table or figure behind it (see §3).
- Contribution 4 is now harder to parse than the v5.7 "A boundary" version (see §3).

## 2. The new bound (l. 125)

1. **The scope is not stated, and Qwen-Drive appears to violate it.** "On any grid the factor is at most $\cos(\pi/2k)^k$ … 0.883 for $E_{10}$". At l. 213 Qwen-Drive's released 10-step sampler has $\taueff=0.98$. A reader will see a contradiction. The bound's premise is that the angles sum to $\pi/2$, which holds only when Euler integrates the velocity field over the whole interval $[0,1]$. Qwen integrates on $[0,0.9]$ and then jumps to a predicted endpoint, so the premise fails. Say this in both places:
   - l. 125: "…for Euler on the full interval $[0,1]$…"
   - l. 213: "Its sampler stops at $t=0.9$, so the angles sum to less than $\pi/2$ and the cap of \cref{eq:contraction} does not apply."

   As written, the paper's two new pieces of text undercut each other.
2. **Attribution is ambiguous.** "The inequality is implicit in the proof of Proposition 2 of Claustre et al." comes right before the new cap, so a reader cannot tell whether the cap is also from Claustre et al. or is new. Reorder:
   - (i) the closed form, which is ours, used as a prediction;
   - (ii) "$\le1$ is implicit in [Claustre, Prop. 2]";
   - (iii) "We add a grid-independent cap:" followed by the bound.

   If the cap is a contribution, put it in contribution 3 (l. 99) and consider making it a displayed equation, e.g. Eq. 2.
3. **The paragraph order is now broken.** "Coordinates outside this range contract more…" (last sentence) refers to "$0.3\le s_d\le3.3$", which is now three sentences earlier, past the cap. Move that sentence up so it directly follows the 0.82–0.88 sentence. The paragraph is also too long for one paragraph at about 14 lines on p. 2. Split it at "The angles $\theta_{d,i}$ sum to $\pi/2$" with a run-in such as "**No grid removes the contraction.**".
4. **The wording is backwards.** "Choosing where the steps go can reduce the contraction toward this cap" reads as moving the contraction toward the cap. Write: "A better grid can raise the factor up to this cap but not past it."
5. **A missed link.** The cap is the theoretical reason the decision-information schedule's gain disappeared at matched temperature (l. 205, App. C). Add one clause: "…cannot remove it, consistent with the negative schedule result of \cref{app:schedules}." This ties §3 to Related Work and makes the cap do argumentative work.
6. **Equality needs a qualifier.** Equality requires equal angles, which depend on $s_d$. So no single grid attains the cap for all coordinates at once. Add "(per coordinate)".
7. **Notation drift.** The equation uses $\theta_{d,i}$, while contribution 3 (l. 99) and the Fig. 2 caption (l. 132) use $\theta_i$. Either say "dropping $d$" once or use $\theta_{d,i}$ everywhere.
8. **"on any grid" appears twice** (the bold run-in and twice in the body). Once in the run-in is enough.

## 3. Qwen-Drive "A second family" (l. 213) and contribution 4 (l. 100)

1. **The paragraph mixes four kinds of content.** It contains model facts, a reference-solver change, run configuration (reasoning mode, float32) and results, all in 9 sentences with 8 numbers and no table. Move the configuration (Heun-64 check, float32, reasoning mode) to App. E. Put the Qwen rows ($\Aref$, $E_{10}$, $E_4$, $A_{\taueff}$; ADE$_1$, flips) as a block in Table 1 or a small appendix table, and cite it. This is the paper's own "one claim, its number, its table" rule (l. 30).
2. **The reference solver is redefined without being renamed.** $\Aref$ is defined at l. 110 as Heun-32 to $t=1$. For Qwen it is Heun-32 on $[0,0.9]$ plus an endpoint jump. Name it ($\Aref^{0.9}$) or say "our protocol with the reference truncated at $t=0.9$". Otherwise $\taueff$ for Qwen is not the same quantity as in Definition 1.
3. **The claims are imprecise.**
   - "Lowering this model's temperature changes ADE$_1$ by 0.001 m": this gives no interval and no temperature. State the $\tau$ (0.42?) and the interval.
   - "$E_4$ is worse … by 0.041 [0.022, 0.062] m": the interval excludes zero but straddles the 0.05 m margin. Say so in the same words as Finding 5 uses for FlashDrive.
   - "decision flips (4.4% … 1.8% at matched temperature) are too rare to test": the 2.6-point gap exceeds the 2-point flip margin. Calling it "too rare" therefore reads as selective. Report the residual and its interval, as for every other sampler.
4. **Over-interpretation.** "Fewer steps … pay only their discretization cost" infers the cause of the residual. Write "fewer steps … add error that temperature does not explain".
5. **Macro inconsistency.** l. 213 uses `$\tau_{\text{eff}}$` twice instead of `\taueff`. It renders the same, but the source should use the macro.
6. **The key explanation comes too late.** The paragraph's real point is that seed spread (0.14 m) is a tenth of ADE$_1$ (1.55 m), so temperature has little to change. Lead with that: "Qwen-Drive's seed spread is a tenth of its error, so temperature cannot move its ADE$_1$; fewer steps still map to temperature, but only add error."
7. **Contribution 4 (l. 100).**
   - "With the reference kept inside the trained time range" is an implementation detail that is meaningless at this point in the paper. Cut it.
   - "Temperature does not change its error" is stated as a fact without a number.
   - "Fewer steps only cost accuracy" rests on one model and one comparison ($E_4$).

   Suggested: "**A second model family.** On Qwen-Drive, whose seed spread is a tenth of its error, step count still maps to effective temperature, but lowering temperature leaves ADE$_1$ unchanged, so $E_4$ only adds error (\cref{sec:discussion})."
8. **Inconsistent framing across sections.**
   - Limitations (l. 214) says "one planner of a second family … has an error that does not depend on temperature".
   - The abstract does not mention Qwen-Drive at all, although it is listed as a contribution.

   Either add one abstract clause ("a second family, Qwen-Drive, maps steps to temperature but gains nothing from it") or demote contribution 4 to a sentence under contribution 2.

## 4. Items carried over from v5.4

| # | Line(s) | Status | Edit |
|---|---|---|---|
| 1 | 182 / Table 1 | **Open** | Table 1 still prints on p. 5, after all the Findings that cite it. Move `\input{tables/tab_matched}` before Finding 1 or use `[!t]`/`\afterpage`. |
| 2 | 102, 146, 184, 187, 194, 198, 213, 214; App. B–E | **Open** | Use an `\ifdraft` toggle for `\TODO`; there are 8 orange boxes in the body. Related Work (l. 205) still cites App. C, which is a stub. |
| 3 | 70, 189, 212, Table 1 caption | **Open** | "Linear" (abstract, Finding 4 title, l. 212, "moved linearly") vs "uniform shrinkage" (l. 189 body). Pick one term. |
| 4 | 98, 184 | **Open** | "79–121%" in contribution 2, while l. 154 says not to read fractions as evidence. Replace with "flip residuals within 2.0 points". |
| 5 | 191 | **Open** | "We implemented this cache ourselves inside FlashDrive's inference code" still undercuts "their own code". Say how it was validated against the released implementation, or use the released code. |
| 6 | 70, 191 | **Open** | "original Alpamayo-1.5 checkpoint" → "released (unquantized)". |
| 7 | 112 / 191 | **Open** | "reuses the velocity on the 4th to 7th of 8 Euler steps" is still written twice. Define $F_8$ at l. 112 and use it in the box. |
| 8 | 191 | **Partial** | The box still ends on "Our pre-registered guess of $\taueff$ 0.5–0.6 … was too low." Either give the source of the guess and what follows from it, or cut it. Cutting also helps with the box length (still about 10 sentences). |
| 9 | 191 | **Partial** | "Decision flips for this run are pending": the paper's primary endpoint is still missing from the Finding that the abstract headlines. |
| 10 | 70 | **New wording issue** | "Its error residual … is 0.008 m, close to but not within our equivalence margin": the point estimate *is* within the margin; the interval is not. Write "(90% interval up to 0.051 m, just outside our 0.05 m equivalence margin)". |
| 11 | 203, 210 | **Open** | Related Work and Recommendations still do not cite `find:flash`. |
| 12 | Fig. 3 / Table 1 | **Open** | There is still no FlashDrive marker or row, and no Shrink$_k$ in Fig. 3. |
| 13 | 198 | **Open** | The TODO still combines the FlashDrive optimized checkpoint and the minADE crossover. Split it. |
| 14 | 169, Table 1 caption | **Open** | "explicit-temperature curve": define "temperature curve" once in §4. |
| 15 | 158 | **Open** | Write "Alpamayo-2-Super (Alpamayo-2 below)". |
| 16 | 144 | **Open** | $h$ is still undefined in Proposition 3.1. |
| 17 | 180 vs 186 | **Open** | Finding 1 says Heun samples "do not" contract, Finding 3 says "barely". Use "barely" in both. |
| 18 | 184 | **Open** | `\EntropyResidMin` renders "-0.03" with a hyphen (confirmed in the PDF). Use `$-$0.03`. |
| 19 | 211 | **Open** | Quantify "a sizable share". |
| 20 | generator | Minor | `\FDGain` and `\FDResidHi` are now unused. Prune them. |
| 21 | 70 | Regressed | The abstract is 252 words. Cut the Heun range or the shrinkage clause. |

## 5. Top three fixes
1. **Scope the bound and reconcile it with Qwen-Drive** (l. 125, l. 213). Add "Euler on $[0,1]$" to the cap. Explain that Qwen's truncated sampler ($\taueff(E_{10})=0.98 > 0.883$) escapes it, and name Qwen's truncated reference differently from $\Aref$.
2. **Make Qwen-Drive a supported, precise claim** (l. 100, l. 213). Give a table block, intervals for the 0.001 m temperature effect and for the flip residual (whose 2.6-point gap exceeds the margin), lead with "spread is a tenth of error", move the configuration to the appendix, and rewrite contribution 4 as in §3.7.
3. **Fix the structural items open since v5** (Table 1 placement, TODO toggle and stub appendices, "linear"/"uniform", 79–121% in contribution 2), plus the abstract wording on the FlashDrive margin (§4 #10).
