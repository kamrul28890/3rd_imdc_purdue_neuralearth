# Manuscript gap audit — IMDC 2026 paper (2026-09-12)

Companion to `docs/PAPER_PLAN.md` (the original draft-from-scratch plan, now substantially
executed) and `docs/MANUSCRIPT_WORKFLOW.md` (the ordered execution checklist this audit feeds).
This document is a dated snapshot: what the *existing* draft (`paper/imdc_paper.tex` +
`paper/imdc_paper_SI.tex`) is missing or getting wrong relative to top-journal and EPIFORGE 2020
standards, found by (a) re-reading the current draft in full and (b) live web research on the
target venue's actual current guidelines and the current (2025-2026) related-work landscape,
since a plan written from memory goes stale exactly the way our own backtest data did. Findings
are triaged MAJOR (blocks a credible submission) / MINOR (polish, not blocking). Re-run the
research portion (Sec. 3) again close to actual submission — this is a snapshot, not a fact valid
forever, the same caveat this project already applies to its own ablation findings.

## 1. Target venue — a working structural reference, not a locked decision

**Correction (2026-09-12, user directive):** the submission venue is not yet decided. Treat PLOS
Computational Biology (`PAPER_PLAN.md` §2's recommendation) as a reasonable *working structural
template* only, since it's IMRaD with Vancouver numbered references, which is close to a
generalist default. Do **not** enforce its specific numeric limits (word counts, figure counts,
format) as blocking gates before a venue is actually chosen. Verified live (2026-09-12) against
`journals.plos.org/ploscompbiol` for reference, kept here so the numbers exist when a venue
decision is eventually made:
- Research articles are IMRaD with Vancouver (numbered) references — **already matches**
  (`\usepackage[numbers,sort&compress]{natbib}`), venue-independent enough to keep regardless.
- Initial submission is format-free; PLOS's own guidance flags "abstracts over 300 words" as the
  single most common first-submission defect *for that journal specifically* — not a universal
  rule, and not binding here until PLOS Comp Biol (or any other venue with its own limit) is
  actually chosen.
- Requires a Data Availability Statement meeting PLOS's specific policy language — present in our
  draft (`\section*{Data and code availability}`) but not checked against PLOS's exact phrasing;
  defer to the final formatting pass once a venue is chosen (Workflow step 6/11), not now.

## 2. Word counts — a density note, not a compliance gate

**Reframed (2026-09-12):** the abstract (333 words) and Author Summary (237 words) are not "over
a limit" in any binding sense right now, since no venue is fixed. The applicable standard is the
one in `PAPER_PLAN.md` §0: dense and compact, not verbose, but never lossy. Reread both for
tightenable prose (redundant clauses, throat-clearing, anything sayable in fewer words without
losing a finding, a number, or a nuance) rather than for what to cut. If a density pass happens to
also land under 300/150 words, that's a byproduct, not the goal — and if it doesn't, that's fine
too, as long as every sentence is earning its place. Re-apply the actual PLOS-style numeric limits
(or whatever venue is chosen) only at the final formatting step (Workflow step 11), never as a
drafting-stage constraint.

## 3. MAJOR — related-work completeness (found via live literature search, 2026-09-12)

`PAPER_PLAN.md` §5 sketched a ~30-40 reference plan written before this exact literature existed
or before we'd found it. A live search turned up four items **not currently cited anywhere** in
`imdc_paper.tex` (checked by `grep`, confirmed absent):

1. **Wu S, et al. Ensemble approaches for short-term dengue fever forecasts: a global evaluation
   study. *PNAS* 122, e2422335122 (2025).** Multi-country (Brazil, Colombia, Malaysia, Mexico,
   Thailand, Iquitos-Peru, San Juan-PR), 180+ locations, 1-3 month horizon, retrospective +
   prospective; finds ensembles consistently rank among top performers. This is the single most
   important missing citation: it is in *PNAS*, in the exact same challenge lineage (already
   cited by our own `araujo2026` as its ref. 23), and a reviewer in this specific niche will
   almost certainly expect it. **Action:** cite in Introduction (ensemble-forecasting precedent,
   alongside `cramer2022`/`reich2019`) and add one differentiating sentence in Discussion (see
   §4 below) — they operate at a shorter horizon (1-3 months vs. our 16-67 weeks), across
   countries rather than within one, and do not examine the raw-vs-normalized-WIS metric
   disagreement or use conformal recalibration, which is where our contribution sits.
2. **Freitas LP, et al. A statistical model for forecasting probabilistic epidemic bands for
   dengue cases in Brazil. *Infect Dis Model* 10, 1479-1487 (2025).** Same country, same research
   community (a co-author of `araujo2026`), directly on probabilistic dengue-band forecasting in
   Brazil. **Action:** cite in Introduction alongside `araujo2026`, or in Discussion near the
   existing M6/simple-baseline point.
3. **A 2026 Sierra Leone monthly-dengue comparison (NB-GLM vs. INGARCH-NB vs. Renewal-NB
   mechanistic vs. BiLSTM-NB deep-sequence), *PLOS Global Public Health* (2026); PubMed ID
   41894525.** Independent, very recent (accepted 2026), different continent and surveillance
   system, architecturally close to our own GRU-negative-binomial model (BiLSTM + NegBin head).
   Their finding — a simple autoregressive count model (INGARCH-NB) beats the deep-sequence model
   on distributional accuracy, though the deep model is preferred when tail reliability matters —
   is an independent, cross-continental echo of this paper's own "no single model dominates" and
   "simple methods are hard to beat" findings, on a near-identical model architecture. **Action:**
   cite in Discussion §"Simple methods and the limits of complexity," alongside `makridakis2020`
   and the `araujo2026` M6 point already added there (2026-09-12 session).
4. **A systematic review and network meta-analysis of dengue forecasting models, medRxiv, 59
   studies 2014-2024 (posted 2026-02-19).** Ranks models by RMSE-based network meta-analysis —
   i.e. most of the literature it surveys compares models on **point accuracy**, not probabilistic
   calibration. **Action:** cite in the Introduction to sharpen the novelty claim: most prior
   dengue-forecasting comparisons (per this review) use point-accuracy metrics; we contribute a
   leakage-controlled, fully probabilistic (WIS/CRPS-based), dual-metric comparison. This
   strengthens exactly the kind of "why does this matter beyond one challenge" framing a
   single-team paper needs.

**Verify exact bibliographic details (volume/page/DOI) for all four before drafting the
`\bibitem` entries** — the details above come from search-result snippets, not primary-source
confirmation of every field.

## 4. MAJOR — sharpen the novelty/positioning claim against Wu et al. 2025

`imdc_paper.tex`'s Limitations already says "the comparison is single-team, so it does not
capture the full diversity of approaches a multi-team sprint would" — an honest limitation, but
with Wu et al. 2025 now a known, citable, much larger (180+ location) multi-country ensemble
study, a reviewer's natural next question is "what does a single-country, single-team study add
beyond that." The paper does not yet answer this explicitly anywhere. **Action:** add one
paragraph (Introduction, near the contributions list, or as the opening of Discussion) stating
what this paper adds that a broader multi-country sweep does not: (a) a fully leakage-audited,
automated-test-covered backtest harness at a depth few forecasting papers document; (b) the
explicit metric-disagreement finding (raw magnitude-weighted vs. scale-free normalized WIS pick
different winners) which is not examined by Wu et al. or `araujo2026`; (c) conformal
recalibration of the *combined ensemble's* intervals as a cheap post-hoc fix, absent from both;
(d) a two-disease (dengue + chikungunya) comparison showing the best model is disease-specific;
(e) a fully public, tested, reproducible pipeline with a provenance manifest. This is a
one-paragraph fix with outsized effect on how a reviewer reads the paper's significance.

## 5. MINOR — EPIFORGE 2020: the SI's checklist table is a paraphrase, not the real 19 items

`imdc_paper_SI.tex` S5 Text maps 12 informally-grouped categories ("Study purpose and forecasting
objectives," "Data sources and provenance," ...) to manuscript locations. The actual published
EPIFORGE 2020 checklist (Pollett et al., *PLOS Medicine* 18(10):e1003793, 2021) has **19 numbered
items** in 5 categories (fetched verbatim 2026-09-12; full list in `docs/EPIFORGE_COMPLIANCE.md`).
A paper that cites `pollett2021` and claims EPIFORGE alignment should map to the *actual* 19 items
by number — that is what "EPIFORGE-compliant" means to an editor checking the claim, and PLOS
Comp Biol reviewers in this space (per `araujo2026`'s own precedent of citing but not fully
itemizing EPIFORGE) will likely expect item-level granularity given we explicitly invoke the
guideline. **Action:** replace S5 Text's table with the full 19-item version (drafted in
`docs/EPIFORGE_COMPLIANCE.md`, ready to paste in during Workflow step 4).

## 6. MINOR — style-rule violation isolated to the SI

`PAPER_PLAN.md` §0 bans literal em dashes. `grep -c "—" paper/imdc_paper.tex` returns **0**
(verified 2026-09-12, and again after every edit this session) — the main text is clean. The same
check on `paper/imdc_paper_SI.tex` returns **17**, concentrated in S9 Text (the glossary), which
reads as written before the style rule was locked in or drafted separately from the main-text
passes. **Action:** rewrite S9 Text's 17 em-dash instances as commas/colons/two sentences,
matching main-text convention, during the SI audit (Workflow step 5).

## 7. MINOR — one fragile hardcoded cross-reference

SI's S9 Text glossary ends with a plain-text reference, "Results Section~2.11 of the main text."
Verified against the current subsection count (2026-09-12): this is *currently correct* (the 11th
Results subsection, "Onset, peak, and total burden..."), but it is a hardcoded number, not a
LaTeX `\label`/`\ref`, in a two-file (main + SI) document — exactly the kind of thing that goes
silently stale the next time a subsection is added or reordered (as already happened once this
session with the new log-linear-pooling ablation paragraph, which did not shift this number only
because it landed inside an existing subsection rather than adding a new one). **Action:** either
add a `\label` in the main text and reference it properly (requires compiling both files together
or hand-syncing, since they're separate documents — check which PLOS Comp Biol expects: SI as a
genuinely separate PDF, in which case cross-file `\ref` isn't possible and a hardcoded number with
a recheck step is the actual best option), or at minimum add this exact recheck to the
pre-submission numbers-consistency pass (Workflow step 7) every time either file changes.

## 8. OPTIONAL — statistical-rigor additions found in comparable current literature

Not blocking, but worth a decision (ask the user, don't assume):

- **Diebold-Mariano tests.** The 2026 Sierra Leone comparison (item 3 above) uses DM tests for
  pairwise forecast comparison; we use a block-bootstrap over state-season units with paired
  differences instead (`sec:significance`). Our approach is arguably more appropriate here (DM
  assumes a single long series with mild dependence; we have short, highly heterogeneous,
  cross-sectionally correlated panels), and the paper already justifies the block-bootstrap
  choice. **Recommendation:** add one sentence in Methods naming DM as the more common
  alternative and briefly stating why block-bootstrap fits this panel structure better — this
  preempts a reviewer asking "why not DM" rather than requiring us to actually run DM tests.
- **PIT histograms.** A standard calibration diagnostic in this exact sub-literature (used by
  item 3 above), complementary to but distinct from our existing coverage/reliability figure
  (`paper_coverage.png`, referenced Results §2.6). **Recommendation:** consider as an optional new
  SI figure (needs a new script converting quantile forecasts to PIT values plus a histogram per
  model) — real but moderate implementation effort, lower priority than the MAJOR items above.

## 9. Explicitly NOT yet audited (so this document doesn't overclaim completeness)

- Figure accessibility (colorblind-safe palettes, consistent styling across all ~13 figures) —
  not checked this pass.
- Reference-by-reference DOI/page-number accuracy for the existing ~35 citations — not checked;
  only the 4 new ones above were researched.
- A full read-through for typos/grammar/flow, separate from the style-rule grep checks above.
- Co-author review: Easha and Helal have not seen the current draft state; ICMJE authorship
  criteria and their sign-off are a human step, not something this audit can verify.
- Any similarity/plagiarism check.
- The `.tex`'s figure files (SI's S1-S6 Figs and the main text's referenced PNGs) have not been
  individually re-opened and visually reviewed in this pass, only confirmed to exist and be
  dated after the last relevant data refresh.

## 10. Summary triage

**MAJOR (do before anything else, in this order):** §3+§4 related-work and positioning (one
editing pass: add citations, then write the differentiation paragraph that cites them) -> §5
EPIFORGE 19-item upgrade.

**Density pass (ongoing, not a one-time gate):** §2 — tighten the abstract/Author Summary and,
over time, every section, for compactness without content loss. Not "done" the way a citation
add is done; revisit whenever a section is touched.

**MINOR (do during the full-draft polish pass, not urgent):** §6 SI em-dash cleanup, §7 hardcoded
cross-reference recheck, §8 optional statistical additions (decide, don't assume yes).

**Not yet known:** everything in §9 — scope these explicitly into the workflow rather than
silently skipping them.
