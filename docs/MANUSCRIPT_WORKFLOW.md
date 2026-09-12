# Manuscript-to-submission workflow — IMDC 2026 paper (2026-09-12)

The ordered checklist for turning the current draft into a submission-ready manuscript for PLOS
Computational Biology, executed one step at a time as the user asked. Each step closes specific
items from `docs/MANUSCRIPT_GAP_AUDIT.md` (the diagnosis) using the exact checklist in
`docs/EPIFORGE_COMPLIANCE.md` where relevant. `docs/PAPER_PLAN.md` remains the reference for
overall structure/outline/writing-style rules (§0 there is binding for every edit in every step
below) — this document does not repeat it, only sequences the remaining work.

**How to use this:** work one step at a time; do not skip ahead. Each step ends with a compile +
verification gate (matching this project's own established discipline: run the fast test suite
and both `pdflatex` passes before calling a step done, the same way every code change in this
repo is verified before being called complete). Re-read `MANUSCRIPT_GAP_AUDIT.md` §3 (literature)
close to actual submission — it is a dated snapshot, not a standing fact.

## Step 1 — Density pass on the abstract and Author Summary
- **Reframed 2026-09-12** (see `MANUSCRIPT_GAP_AUDIT.md` §2): venue is undecided, so this is not a
  trim-to-a-limit task. Tighten both for density: cut redundant clauses and throat-clearing,
  shorten sentences, keep every one of the four numbered findings and every number in the
  abstract. In the Author Summary specifically, plain-language every named technical term ("WIS,"
  "conformal recalibration," "normalized WIS") per EPIFORGE item 15's nonspecialist-reader intent,
  since that's a completeness/accessibility improvement independent of length.
- Gate: recompile, zero pdflatex errors. Re-run the word count out of curiosity, not as a
  pass/fail check; report it either way.

## Step 2 — Related-work and positioning
- Verify exact bibliographic details (volume, pages, DOI) for the four sources in
  `MANUSCRIPT_GAP_AUDIT.md` §3 before writing `\bibitem` entries — do not trust search-snippet
  formatting as final.
- Add all four as `\bibitem` entries and `\citep{}` them at the locations specified in §3.
- Write the one-paragraph novelty/positioning addition from `MANUSCRIPT_GAP_AUDIT.md` §4
  (differentiating this paper from Wu et al. 2025's larger multi-country sweep).
- Gate: recompile (two `pdflatex` passes for citation resolution), confirm no `undefined
  citation` warnings, re-run the em-dash/cliché-vocabulary grep from `PAPER_PLAN.md` §0 on the
  new prose.

## Step 3 — EPIFORGE upgrade — **DONE 2026-09-12**
- S5 Text now carries the full 19-item checklist in the guideline's own numbering and wording,
  grouped by its five categories, as a three-column table (#, item, location).
- Locations are given by **section name, not section number**, deliberately: the SI is a separate
  document, so cross-file `\ref` is unavailable, and hardcoded numbers are exactly the fragility
  flagged in `MANUSCRIPT_GAP_AUDIT.md` §7. Section names survive reordering.
- Compliance-claim sentence added, naming the guideline and the item count (19).
- Item 16's provenance-manifest interpretation is now stated explicitly in the S5 Text preamble
  rather than left implicit, and items 18/19's "if applicable" condition is answered explicitly
  (both apply; addressed for the 2024 season).
- Item 18 strengthened in the **main text**: Discussion, Robustness under regime shift now states
  the concrete public-health recommendation (prefer a recalibrated ensemble to the single
  best-scoring model, because the season a leaderboard-topping model fails in is by construction
  the season with no precedent in the training record).
- Also fixed: the SI's opening paragraph listed only S1-S7 while the document contains S8 and S9.
- Verified: two pdflatex passes on both documents, zero errors, all 19 rows present, and a
  baseline compile of the pre-edit SI (`git show HEAD:...`) confirmed the new table introduced
  **zero** new overfull boxes (all four are pre-existing at identical magnitudes).

## Step 4 — Statistical-rigor additions — **DONE 2026-09-12** (both approved and built)
- **Diebold-Mariano justification** added to Methods, Scoring, with `diebold1995` cited: states
  that DM is the more common choice and why a block bootstrap over state-season units fits this
  panel better (DM assumes one long weakly-dependent error series; we have 26 short,
  cross-sectionally correlated state series with one season an order of magnitude harder).
  Preempts the reviewer question without needing to run DM tests.
- **PIT calibration figure** built (`scripts/make_paper_pit_fig.py`, new SI section S9 Text +
  S7 Fig). Two implementation details that a naive version would get wrong, both documented in
  the script docstring and the SI text:
  - The nine quantile levels cut **unequal-width** bins (25 points at the centre, 2.5 at the
    tails), so panels plot observed/expected share, not raw counts, which would make the wide
    central bins look over-full by construction.
  - Counts are discrete and tie against their own quantiles (the all-zero-median sparse case), so
    it uses the **non-randomized PIT for discrete data** (an observation tied against $k$
    quantiles contributes $1/(k+1)$ to each bin it could occupy). A naive assignment would charge
    every tie to the bottom bin and manufacture a spike there.
- The figure earned its place rather than ticking a box: it shows three things the four-level
  reliability curve cannot. The GRU's overconfidence is symmetric (both tails over-full, centre at
  half share); **every** model over-populates the topmost bin, which is the 2024 under-prediction
  restated distributionally; and conformal recalibration cuts that topmost-bin excess from 4.4x to
  2.4x nominal while leaving the rest flat.
- Near-empty lowest bins for the mechanistic model (0.01) and conformal ensemble (0.00) are an
  expected non-negativity artifact (lower quantiles clipped at zero, so a zero observation ties
  rather than falling below), disclosed in the SI text so it does not read as a defect.
- SI sections renumbered: the glossary moved S9 to S10. Verified safe first: the main text makes
  **zero** references to SI section numbers, so all affected references were internal.
- Gate met: fast test suite 113 passed; two pdflatex passes on both documents, zero errors, zero
  undefined citations; the four SI overfull boxes are unchanged pre-existing ones, no new ones.
- **Created two new hardcoded cross-references** (SI S9 Text cites main-text "Fig 8" and "Fig 9B").
  Verified correct against the current figure order at time of writing. Add to the Step 7 recheck.

## Step 5 — Full SI audit — **DONE 2026-09-12**

Completed: all 17 em-dashes fixed (uniform `\textbf{term} —` glossary separator, replaced with a
colon, which reads better than a comma for a glossary); the 83.6pt equation overflow eliminated by
splitting the WIS and interval-score definitions into an `aligned` block; the S2 Table overflow cut
from 44.4pt to 6.9pt (`\footnotesize` plus removing a redundant "(chosen)" label that duplicated
what the caption already says); "Results Section~2.11" re-verified as still correct.

**Two substantive defects the numbers check surfaced, neither of which was on the audit list:**
1. **S2 Table claimed to score "every member subset" but showed 8 of 11.** Recomputing the full
   sweep from `final_scored.csv` reproduced all 8 published rows exactly and revealed three
   missing subsets (Climatological+XGBoost+Mechanistic 1174/0.567/0.416/0.358;
   XGBoost+GRU+Mechanistic 1204/0.581/0.398/0.336; Climatological+XGBoost 1207/0.583/0.389/0.310).
   All three added, so the completeness claim is now true and checkable. None of them changes any
   claim in the surrounding text, which was re-verified against the full 11.
2. **A clause inverted a fact.** The text described XGBoost+GRU as "the worst composition of all on
   the reported ordinary-season metric, 0.365 notwithstanding." Its 0.365 is the *best*
   ordinary-season value in the table; it is the worst on *all-season* normalized WIS (0.601). The
   clause also glided past a real tension with "notwithstanding": that subset beats the deployed
   composition on fold 1 (0.282 vs 0.306), which is the criterion the same paragraph argues should
   govern. Rewritten to state the tension plainly and answer it: the composition was fixed in
   advance by structure (one baseline, one gradient-boosted, one deep model) rather than selected
   from the table, and a 0.024 margin picked out of eleven candidates on a single fold is the
   search artifact this discipline exists to discount. The honest version strengthens the
   paragraph's own argument instead of hedging around it.

Verified: S4 Table reproduced exactly from current results; S2 Table's 8 original rows reproduced
exactly; zero em-dashes; zero errors; remaining overfull boxes are 0.43pt, 6.9pt, and 3.97pt, all
imperceptible and comparable to the main text's 1.76pt.

### Original step definition (kept for reference)
- Read `imdc_paper_SI.tex` end to end (already done once for this workflow's own drafting, but
  re-read after Steps 1-4 land, since they touch it).
- Fix all 17 em-dash instances in S9 Text per `MANUSCRIPT_GAP_AUDIT.md` §6 — rewrite each as
  comma/colon/two-sentence per the project's own style rule, don't just delete the dash.
- Recheck the hardcoded "Results Section~2.11" cross-reference in S9 Text against the *then-current*
  subsection count (a fresh `grep -n "^\\subsection{" paper/imdc_paper.tex` count) — it may have
  shifted if Steps 2-4 added or removed any main-text subsection.
- Verify every number in the SI's tables (S2, S4) is still consistent with the latest
  `results/metrics/*.csv` — this project has already caught real staleness bugs this way twice
  (the September data-refresh resync, and today's presentation resync); treat SI numbers with the
  same suspicion as main-text numbers, not as a lower-stakes appendix.
- **Fix two pre-existing overfull boxes found during Step 3** (measured against a baseline compile
  of the pre-edit file, so these are confirmed pre-existing, not introduced):
  - **83.6pt (~2.9 cm) past the margin at the WIS display equation in S1 Text.** This is the
    largest overflow in either document and sits in the first SI section a reader reaches. The
    equation puts the WIS definition and the interval-score definition on one line; splitting it
    across two lines (or an `aligned`/`split` environment) is the obvious fix.
  - **44.4pt (~1.5 cm) at the S2 Table** (ensemble composition sweep). Likely the long member
    names in column 1; shorten the labels or reduce the font one step.
  - For contrast, the main text's only overfull box is 1.76pt at the ablation table, which is
    imperceptible and needs no action. These two do not meet that bar.
- Gate: recompile the SI, zero em-dashes (`grep -c "—"` returns 0), zero pdflatex errors, and both
  overfull boxes above resolved or consciously accepted with a reason.

## Step 6 — Figures and tables final polish
- Re-open every figure referenced in the main text and SI (not yet done in the audit — see
  `MANUSCRIPT_GAP_AUDIT.md` §9) and check: colorblind-safe palette, consistent styling across all
  ~13+ figures (same font, same color mapping for the same model across figures), captions
  self-contained (a reader should understand a figure without re-reading the Results text).
- Confirm figure file formats meet PLOS's post-acceptance requirements (checked at this step so it
  isn't a scramble later, even though initial submission is format-free per
  `MANUSCRIPT_GAP_AUDIT.md` §1): typically TIFF/EPS/PDF at specified minimum resolution, separate
  files rather than embedded-only.
- Gate: visual review of each figure at full size (not the thumbnail scale of a compiled page);
  no automated gate here, this is a human/visual-judgment step.

## Step 7 — Full numbers-consistency pass
- **Cross-reference recheck (accumulating list).** The SI is a separate document, so it cannot
  `\ref` into the main text and every main-text pointer in it is a hardcoded string. Verify each
  against the then-current numbering: SI S10 Text glossary cites "Results Section~2.11"; SI S9
  Text cites "Fig 8" (Calibration and its correction) and "Fig 9B" (WIS decomposition). All three
  were correct when written; they are exactly what goes stale when a section or figure is added.
- Cross-check every number that appears in the abstract, author summary, results tables, and
  in-text citations of specific figures against the current `results/metrics/*.csv` files, treating
  this exactly like the September resync and the presentation resync earlier this session: a
  systematic re-derivation, not a spot check.
- Specifically re-verify the new log-linear-pooling ablation numbers (added 2026-09-12) are still
  current if any upstream model or ensemble composition changes between now and this step.
- Gate: a written list of every number checked and its source file, not just "looks fine" — matches
  this project's own established verification discipline (see `superpowers:verification-before-completion`).

## Step 8 — Prose and style pass
- Full read-through for grammar, flow, and the `PAPER_PLAN.md` §0 style rules (no em dashes, no
  AI-cliche vocabulary, declarative and quantitative sentences, active voice, past/present tense
  discipline).
- Re-run the em-dash and cliché-vocabulary greps on the *entire* paper directory (main + SI) as a
  final gate, not just the sections touched in earlier steps.
- Gate: both greps return 0 hits across `paper/*.tex`.

## Step 9 — Cover letter
- Draft a cover letter for PLOS Computational Biology highlighting the five-point contribution
  list already in the Introduction, why PLOS Comp Biol specifically (methods rigor +
  reproducibility fit, per `PAPER_PLAN.md` §2's own reasoning), and suggested/excluded reviewers if
  the venue's submission system asks for them (a decision for the user, not something to invent).

## Step 10 — Co-author review (human step, not mine)
- Circulate the near-final draft to Easha and Helal for review and sign-off (ICMJE authorship
  criteria: substantial contribution, drafting/revising, final approval, accountability). Schedule
  this explicitly rather than treating it as implicit — it has not happened yet as of this
  document's writing.

## Step 11 — Submission packaging
- Prepare the post-acceptance formatting package early (figures as separate files, SI as a
  separate PDF, cover letter, any reviewer suggestions) even though PLOS's initial submission is
  format-free, so this isn't a scramble after a provisional accept.
- Decide with the user whether to post a preprint (medRxiv/bioRxiv) before or alongside submission
  — consistent with this exact community's norms (`araujo2026` itself was first posted to
  medRxiv); not required, a decision only the user can make.

## Step 12 — Anticipated reviewer questions (optional prep, do close to submission)
- Draft short, honest answers to the questions this audit itself surfaces a reviewer is likely to
  ask: novelty versus Wu et al. 2025's broader multi-country sweep (answered by Step 2's
  positioning paragraph, but worth having a fuller answer ready for a review response); the
  single-team scope; fold-4's partial resolution and what happens if the true 2026-27 forecast
  score, once resolved, contradicts a headline finding.

---

**Explicitly deferred, not forgotten** (from `MANUSCRIPT_GAP_AUDIT.md` §9): a
similarity/plagiarism check and a reference-by-reference DOI audit of the ~35 pre-existing
citations are real remaining gaps this workflow does not yet have a dedicated step for — add one
before Step 11 if time allows, or explicitly accept the risk and note why.
