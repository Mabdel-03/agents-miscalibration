"""Inference layer of the study package (WP2: client + content-addressed request store).

* :mod:`agents_scaling.study.inference.client` — endpoint pool over the serving registry and
  the vLLM chat client that turns a :class:`~agents_scaling.study.types.RequestSpec` into a
  :class:`~agents_scaling.study.types.RequestRecord` (§3.6, §4.3, §10.4).
* :mod:`agents_scaling.study.inference.store` — ``<run_root>/requests`` first-writer-wins
  store; the only path through which cells obtain records (§3.6 aliasing, §10.4).
* ``tokens.py`` (WP1) — envelope helpers.
"""
