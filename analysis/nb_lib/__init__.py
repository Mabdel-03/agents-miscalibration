"""Shared helpers for the full_sweep_v1 analysis notebooks.

Modules:
  data    — cached-parquet loaders, categorical orders, derived columns, canonical filters
  ingest  — raw cells/ -> analysis/cache/*_v1.parquet (item, agent, deduped-cell tables)
  calib   — calibration estimators beyond the package's 15-bin plug-in ECE
  boot    — hierarchical (question -> seed) bootstrap and resampling utilities
  stats   — model-fitting wrappers (FE logit, fractional logit, ANOVA, GBM, recipes)
  plots   — repo plotting conventions (palette, savefig, reliability diagrams)

Notebooks bootstrap with:
    import sys; sys.path.insert(0, "/orcd/data/tpoggio/001/mabdel03/agents_scaling/analysis")
    from nb_lib import data, plots, stats, calib, boot
"""
