"""Strict wire-format parsers of the study package (WP3).

* :mod:`agents_scaling.study.parse.candidate` — the §3.5 complete-candidate contract
  (one-fence salvage, strict JSON, six-key schema, RFC 8785 canonical bytes).
* :mod:`agents_scaling.study.parse.coordinator` — the CEN_FLAT ``coordinator_action`` and
  ``subtask_result`` contracts (handoff schemas) with the §4.2 cross-record rules.

No parser here ever repairs model text (§3.5 "No repair model"); every failure is a typed
code and the opportunity stays in the denominator.
"""
