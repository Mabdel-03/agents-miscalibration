"""Resource accounting of the study package (WP4).

* :mod:`agents_scaling.study.resources.oracle` — the analytic dense-Qwen3 FLOP oracle
  ``F(L, T)`` (spec §6.5, §10.2; architecture §4).
* :mod:`agents_scaling.study.resources.broker` — the per-episode reserve/debit ledger with
  the frozen stop reasons (spec §6.5 steps 1–4, §10.2 "admission is a transaction").
* :mod:`agents_scaling.study.resources.profile` — the outcome-blind B0 profiler that writes
  ``FROZEN.yaml`` (spec §6.5, §10.2; corrections P1-1, P1-2).
"""
