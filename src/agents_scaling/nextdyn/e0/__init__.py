"""E0: the legacy reanalysis (spec 7.1-7.4; plan §2.6, D10).

E0 is the part of the study that is *about* the three runs that came before it -- ARD36, the
study_v4 atlas and the Run0 full sweep -- and its whole value is that it says which of their
published claims survive a recomputation and which cannot be checked at all.  Two rules make
that possible and both are structural here rather than editorial:

* **Every read is a bounded named path.**  ``registry.py`` resolves paths only through
  ``configs/nextdyn/legacy_sources.yaml``; there is no directory walk anywhere in this package.
  A path that is not in the registry cannot be read, and a registered path that is not on disk
  produces a ``BLOCKED_MISSING_SOURCE`` row.
* **A missing source is never an observation.**  ``BLOCKED_MISSING_SOURCE`` and ``unavailable``
  are separate statuses: the first means the input is not here, the second means the input is
  here but no code exists that could have produced the published number.  Neither is ever
  written as ``fail``, because "we could not check this" and "this did not reproduce" are
  different findings.

Modules: :mod:`registry` (bounded paths), :mod:`inventory` (what exists and what it hashes to),
:mod:`protocol_map` (what each legacy run's protocol actually was), :mod:`ard36`, :mod:`atlas`,
:mod:`run0` (the three claim legs), :mod:`contrasts` (the spec 7.3 strata and standardised
delta), :mod:`bcb_inputgen` / :mod:`bcb_repair` (spec 7.4), :mod:`claims` (the reproduction
table) and :mod:`main` (the ``audit-legacy`` verb).
"""

from __future__ import annotations

__all__: list[str] = []
