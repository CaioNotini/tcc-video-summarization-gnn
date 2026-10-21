```
Role: Evidence record for T02. Saves what was added to the Overleaf bibliography and body text, and the verification behind it.
Entry: docs/status.md
Related to: [status.md](../status.md), [phases.md](../phases.md), [references.md](../references.md)
```

# T02 evidence: Overleaf bibliography and in-text citations

Done on 2026-10-20, in the Overleaf project linked from `docs/references.md`, under the scope
rule recorded in [CLAUDE.md](../../CLAUDE.md) rule 15: edits start limited to the references
section, and go further only when the researcher explicitly extends the authorisation in the
conversation. Both the bibliography-only pass and the later in-text-citation pass were done this
way, confirmed separately.

## What was done

Ten `\bibitem` entries were added to `thebibliography` in `tcc_artigo_sbc.tex`, after the five
existing ones, with their author lists, titles and venues verified against primary sources
(arXiv, publisher pages, IEEE Xplore, JMLR, PMLR) rather than taken from memory or from the
project notes as-is:

| # | Key | Work |
|---|-----|------|
| 6 | `videosage2024` | VideoSAGE (Chaves & Tripathi) |
| 7 | `vognet2023` | VOGNet (Zhang, Wu & Song) |
| 8 | `sumgraph2020` | SumGraph (Park et al.) |
| 9 | `rsgn2022` | RSGN (Zhao et al.) |
| 10 | `dmon2023` | DMoN (Tsitsulin et al.) |
| 11 | `dmondpr2025` | DMoN-DPR (Salehi & Giannacopoulos) |
| 12 | `diffpool2018` | DiffPool (Ying et al.) |
| 13 | `mincutpool2020` | MinCutPool (Bianchi, Grattarola & Alippi) |
| 14 | `videosummarysurvey2021` | Video summarisation survey (Apostolidis et al.) |
| 15 | `asap2020` | ASAP (Ranjan, Sanyal & Talukdar) |

Two corrections were made to what `docs/references.md` previously recorded, found during
verification:

- **VideoSAGE** has a published venue (CVPR 2024 Workshops, SG2RL), not only the arXiv preprint.
- **RSGN**'s official publication year is 2022 (IEEE TPAMI v. 44, n. 5), not 2021; 2021 was the
  early-access date.

One correction was made to a number produced by an earlier research pass before it was used: a
first search reported MinCutPool's page range as 874-883 in one pass and 2729-2738 in another;
the PMLR proceedings page (`proceedings.mlr.press/v119/bianchi20a.html`) confirms 874-883, which
is what was used.

`\begin{thebibliography}{9}` was changed to `{99}` so the numbering column fits two-digit entries.

The document was recompiled in Overleaf after the edit: it now produces 4 pages (previously 3),
with no compile error, and all 15 entries render correctly, numbered [1] to [15], checked by
reading the compiled PDF preview in the editor.

`docs/references.md` was updated to match: the per-work entries for VideoSAGE, VOGNet, SumGraph,
RSGN, DMoN/DMoN-DPR and DiffPool/MinCutPool now carry their verified bibliographic data and, for
DiffPool and MinCutPool, a technical summary contrasting each with SLC pooling. Two new entries
(the survey and ASAP) were added. The "Gaps to close in T02" section marks all four items done.

## In-text citations (second pass, same day, after explicit authorisation)

The researcher explicitly authorised going beyond the references section to insert `\cite{}`
markers in the body text, so that entries [6] to [15] are actually cited, not just listed. Eight
edits were made to `tcc_artigo_sbc.tex`:

- Introduction, end of the HELMNet-family paragraph: added a clause citing VideoSAGE, SumGraph
  and RSGN as recent graph-based video summarisation approaches ([6, 8, 9]).
- Motivation paragraph: cited the video summarisation survey ([14]) to support the claim about
  "most of the video summarization literature".
- Specific objectives, SLC bullet: contrasted SLC pooling with DiffPool and MinCutPool ([12, 13]).
- Specific objectives, training-objective bullet: cited VOGNet and RSGN for graph reconstruction
  ([7, 9]) and DMoN/DMoN-DPR for diversity-preserving regularisation ([10, 11]).
- Specific objectives, readout bullet: cited ASAP for the scale-aware readout ([15]) and VOGNet
  for the node-level attention ([7]).
- Specific objectives, evaluation bullet: cited VideoSAGE for the rank-correlation metrics ([6]).

One mistake was made and corrected during this pass: the ASAP citation was first typed inside the
word "scale-aware" (word-jump navigation miscounted a hyphenated word as two words), producing
"scale-~\cite{asap2020}aware". This was caught by re-reading the inserted text, the wrong insertion
was selected and deleted, and the citation was placed correctly after "mechanism,".

The document was recompiled again: still 4 pages, no error, and the PDF preview shows every new
citation rendering as the expected bracketed number (e.g. "[6, 8, 9]" after the video-graph
sentence, "[12, 13]" after the SLC bullet).

## Result

All fifteen bibliography entries are now actually cited in the thesis body text (`docs/references.md`
has the full list with their reference numbers). T02's acceptance criterion in
[phases.md](../phases.md) ("at least ten works are cited in the thesis text") is met: fifteen are
cited. T02 is recorded as validated in [status.md](../status.md), linked to this file.
