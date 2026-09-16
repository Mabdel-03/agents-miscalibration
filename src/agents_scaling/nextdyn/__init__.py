"""``agents_scaling.nextdyn`` — task_conditioned_dynamics_v1 (spec v1.0, 2026-09-15).

The study asks why communication helps some agent teams and harms others, especially when
the correct answer starts in the minority.  The repo already holds the direct predecessor,
ARD36 (``agents_scaling.ard``), whose mechanisms this package imports unchanged and whose
*vocabulary* (N=5, 3 rounds, a closed Protocol enum, one bank that doubled as roots,
``decision_id == request_id``) it re-implements parametrically.  Those frozen constants are
exactly what produced ARD36's data defects; the seven rules below make each one structural.

Contract document: ``docs/nextdyn/00_contract.md``.  Plan:
``~/.claude/plans/please-flesh-out-the-velvet-brooks.md`` (the plan wins over every design
note where they disagree).

Seven structural rules (plan §1.1; pinned by ``tests/nextdyn/test_guards.py``)
==========================================================================

R1  Every fact lives in exactly one module.  Table columns + keys are the row dataclasses in
    ``tables.py`` (schemas derived from ``dataclasses.fields``); token contract in
    ``contract.TokenContract``; output shape in ``schema.PublicOutput``; frozen manifest in
    ``freeze.FrozenManifest`` (same class writes and verifies); dependency facts in
    ``deps.py``; DAG in ``plan.py``.
    Prevents: D-05, D-08, ``latency`` vs ``latency_s``.  Mechanism: consumers import the
    dataclass; dict-key access to item files forbidden by AST test;
    ``test_every_row_dataclass_is_the_only_declaration_of_its_table_columns``.

R2  Every declared knob is consumed, checked **statically**: ``config.py`` compiles YAML into
    frozen dataclasses (unknown keys rejected) and declares ``CONSUMERS: {leaf key path ->
    consuming module or slurm template}``;
    ``test_every_config_leaf_has_a_declared_consumer_that_reads_it`` walks the package AST for
    attribute reads; template-consumed keys are checked by a render test.  No runtime
    read-recording.  Behavioural companions: peak in-flight == n, items_per_cell honoured,
    deadline_bearing honoured, packet cap honoured.
    Prevents: D-03 ``max_inflight``.  Mechanism: AST walk.

R3  Every producer has a consumer test in the same change: ``test_pipeline_dry_run.py``
    drives every cell kind on 4 synthetic items through the fake vLLM, materializes, asserts
    keys, runs ``analyze`` in test mode.  A WP is not done until its kind runs there.
    Prevents: "both ends, no middle".  Mechanism: ``TableSchema.producer/consumers``
    importable names.

R4  ``cell_kind`` is the first component of every id.
    Prevents: D-11.  Mechanism: ``ids.episode_id`` / ``ids.bank_draw_id`` refuse the other
    kind.

R5  Nothing sent to vLLM is outside the request identity; nothing about the condition is
    inside the seed; no template names a condition or describes the graph.
    Prevents: contract rule 3; D4.  Mechanism:
    ``test_request_cache_invalidates_on_model_prompt_seed_or_parent_change``,
    ``test_no_template_contains_a_protocol_label``, byte-equality tests of aliasing pairs.

R6  Rows carrying physical ids are built at one site: ``telemetry.decision_row()`` is the
    only ``DecisionRow`` constructor.
    Prevents: D-02 recurrence.  Mechanism: AST test.

R7  Gold never enters the generation import closure: ``grading``, ``construct``,
    ``items_build``, ``datasets``, ``inventory``, ``e0``, ``analysis`` are WRITER_SIDE; runtime
    reads only ``items.py`` / ``bundles.py``.
    Prevents: contract rule 2.  Mechanism: import-closure walk
    (``tests/ard/test_wp4_runner.py:298`` pattern).

Work packages (plan §5; disjoint files; parallel after WP0)
===========================================================

===  ============================================================================  ================
WP   Owns                                                                          Depends on
===  ============================================================================  ================
WP0  errors, contract, config, locks, ids, tables, kinds, deps, cli, __init__;      —
     config skeletons with null lock fields; compute.yaml
WP1  decoding, adapters, inference, continuation, telemetry; fake-server           WP0
     fixtures; serving profiles + sampler injection
WP2  graphs, protocols, prompts, templates/, schema, packets, states, episode      WP0, WP1 sigs
WP3  families, datasets, items, items_build, inventory, graders/, grading, banks,  WP0
     construct, bundles; datasets.lock.json
WP4  cells, plan, runner, run_one, dispatch, fleet, status, materialize, freeze,   WP0–WP3, WP5
     gates, __main__; slurm/nextdyn_*; scripts/nextdyn_*                           interfaces
WP5  mod97, checker, receiver, natural_packets, selectors, audit                   WP1, WP2
WP6  capture                                                                       WP2, WP4
WP7  outcomes, stats, calib, panels, analysis, power, figures, release, e0/*;      WP0
     analysis.yaml, figure_manifest.yaml, legacy_sources.yaml
===  ============================================================================  ================

``ARD_IMPORTS`` below is the exact allowlist of ``agents_scaling.ard`` names this package may
import (plan §1.11, decision D2).  ``study/`` is imported freely; anything from ``ard`` that
is not listed here must be vendored with a parity test, never imported.
"""

from __future__ import annotations

from typing import Final

#: The exact dotted names nextdyn may import from ``agents_scaling.ard`` (plan §1.11).
#: A module name admits ``import that.module`` and ``from that import module``; a
#: ``module.attr`` name admits only that attribute.  Checked by
#: ``tests/nextdyn/test_guards.py::test_ard_imports_are_exactly_the_allowlist``.
ARD_IMPORTS: Final[frozenset[str]] = frozenset({
    "agents_scaling.ard.episode.dispatch_batch",
    "agents_scaling.ard.packets.pseudonym_map",
    "agents_scaling.ard.graders.math_equiv",
    "agents_scaling.ard.graders.trace",
    "agents_scaling.ard.graders._boxed.extract_boxed",
    "agents_scaling.ard.graders.GoldStore",
    "agents_scaling.ard.stats.analysis_seed",
    "agents_scaling.ard.stats.Resampler",
    "agents_scaling.ard.stats.paired_contrast",
    "agents_scaling.ard.stats.mean_ci",
    "agents_scaling.ard.stats.noninferiority_bound",
    "agents_scaling.ard.reference_metrics",
    "agents_scaling.ard.resources.parse_nvidia_smi_csv",
    "agents_scaling.ard.resources.capture_sacct_gpu_seconds",
    "agents_scaling.ard.resources.attribute_gpu_seconds",
    "agents_scaling.ard.calibration_panel.within_item_auc",
    "agents_scaling.ard.calibration_panel.tie_rate",
    "agents_scaling.ard.neural.geometry.team_metrics",
    "agents_scaling.ard.neural.geometry.agent_metrics",
    "agents_scaling.ard.neural.anchors",
    "agents_scaling.ard.neural.capture.consolidate_shards",
    "agents_scaling.ard.legacy_e0.Emitter",
    "agents_scaling.ard.legacy_e0.NETWORK_MARKERS",
})

__all__ = ["ARD_IMPORTS"]
