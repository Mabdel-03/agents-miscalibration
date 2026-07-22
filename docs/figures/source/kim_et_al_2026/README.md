# Kim et al. topology figure acquisition note

Reference:

- Yubin Kim et al., "Towards a Science of Scaling Agent Systems",
  arXiv:2512.08296v3, last revised 2026-04-08.

Inspection performed for the documentation upgrade:

- The arXiv source bundle contains topology figure PDFs under `imgs/`:
  `sas.pdf`, `mas_independent.pdf`, `mas_decentralized.pdf`,
  `mas_centralized.pdf`, `mas_hybrid.pdf`, and
  `architecture_comparison_combined.pdf`.
- The arXiv page lists the article under the arXiv non-exclusive distribution license.
  That license grants arXiv permission to distribute the article; it is not a clear
  permission for this repository to redistribute extracted figure assets.

Decision:

- Do not commit copied/extracted Kim et al. figure PDFs or rasters.
- Use repo-specific editable Mermaid diagrams in `docs/figures/*.mmd`, cite Kim et al.
  for the topology framing, and explicitly state that Hybrid is not implemented in this
  harness.
