# Agent membership and information design — v4 handoff

Start with [agents_scaling_local_future_state_reversal_experiment_spec_v4_0.md](agents_scaling_local_future_state_reversal_experiment_spec_v4_0.md). This standalone specification replaces the earlier
reversal-centered plan while preserving its bounded content-use assay. It adds
truthful vote awareness, identical reset-sampling controls, three native orchestration
families, Recursive Language Models, complete-episode pass@K/resource accounting,
membership/checkpoint panels, calibration and causal internal readouts.

This package contains a specification and implementation aids. **It is not a working
HPC experiment harness, and no research experiment has been run by this package.**
The original proposal and v3 files are preserved.

- `configs/`: declarative scientific/runtime/HPC defaults and unresolved freeze inputs.
- `prompts/`: literal role/forecast templates; resolve bounded placeholders and hash rendered requests.
- `schemas/`: selected strict wire formats; cross-record semantics require runtime tests.
- `metrics_reference.py` and `tests/`: small reference logic, not the deployment selector/evaluator.
- `scheduler/`: nonexecutable site-rendered job template; no account/GPU allocation is guessed.
- `rlm_literature_audit.md`: inspected primary sources, overlap and unresolved reproduction details.
- `validate_package.py`, `VALIDATION_REPORT.json`, `MANIFEST.json`: package checks and checksums.

Coding agents should implement the work packages and acceptance evidence in §11,
beginning with plan validation, public/protected data separation, request identity
and a complete-cost broker. Resolve every required execute input, profile actual
BF16 memory/throughput, finish development power/tuning and freeze manifests before
confirmation. Do not treat a template, parsed schema or passing metric fixture as
evidence that native recursion, GPU hooks or cluster execution already works.

The study tests a plausible residual question. It does not certify that no prior or
unpublished study overlaps it, and it does not promise an ICLR acceptance outcome.

Local package checks: `python validate_package.py` and
`python -m unittest discover -s tests -v` from this directory. JSON Schema and YAML
validation require `jsonschema` and `PyYAML`; use a pinned environment when implementing.
