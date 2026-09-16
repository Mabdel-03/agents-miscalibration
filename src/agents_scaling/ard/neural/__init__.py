"""ARD36 neural capture and readout (WP6).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §14 (E6 —
neural/uncertainty instrumentation), plan §5.6 and the neural panel of §5.8.

The modules, in the order the run uses them:

* :mod:`~agents_scaling.ard.neural.panel` — the frozen family/split-balanced 144-item
  capture panel (``contract.CAPTURE_TOTAL_ITEMS``) and its sha256 manifest, drawn from a
  declared seed before anything is captured.
* :mod:`~agents_scaling.ard.neural.anchors` — the four spec §14.2 capture sites of one
  decision, resolved from tokenizer byte spans and typed when missing, plus the
  ``feature_time`` / ``last_allowed_round`` timing labels.
* :mod:`~agents_scaling.ard.neural.capture` — the decision-manifest-driven replay runner
  over the reused ``study.neural`` engine/replay/storage stack, with the two-anchor
  full-vocabulary entropy computed in the same pass and stored as a scalar.
* :mod:`~agents_scaling.ard.neural.geometry` — the ``neural_geometry`` and ``neural_agent``
  tables: ``study.neural.geometry.spectrum_metrics`` plus the spec §14.3 additions.
* :mod:`~agents_scaling.ard.neural.probes` — regularized linear probes for ``CURRENT`` and
  ``EXPOSURE_VALUE`` with their controls and baselines on identical folds and rows.

Nothing here reaches ``agents_scaling.benchmarks``, ``datasets`` or any protected path:
gold never enters a runtime module (contract §0 rule 2), and the probe targets arrive as
already-graded values on the evaluator side.
"""

from __future__ import annotations

from agents_scaling.ard.neural import anchors, capture, geometry, panel, probes

__all__ = ["anchors", "capture", "geometry", "panel", "probes"]
