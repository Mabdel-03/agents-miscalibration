"""Evaluator identity of the study (WP5) — the ONLY package allowed to read protected data.

Spec §3.3 (exact MC scoring; frozen isolated judge for HLE short answers; official tests in
a fresh environment for code; ambiguous judge output is never scored correct), §10.6
(protected references and correctness joins are mounted only in the evaluator identity).
Corrections P1-4 (import-graph firewall: nothing outside this package imports
``study.data.protected``), P1-5 (one frozen code-string rule shared with grouping),
P1-10 (MC-based judge audit), §4 item 11 (judge outputs live under ``<run_root>/eval/``).

Modules
* :mod:`.hle_judge` — MC letter scoring, the cais/hle judge request/parse, per-item judge logic.
* :mod:`.audit` — judge-vs-letter FP/FN audit, the 2-pp differential bound, human audit export.
* :mod:`.bcb` — the apptainer BigCodeBench evaluator (one exec per call, in-container loop).
* :mod:`.bcb_container_driver` — the stdlib-only script executed inside the container.
"""
