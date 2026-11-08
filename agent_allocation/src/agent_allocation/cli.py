"""``allocation`` CLI root (lead-owned). Verbs resolve lazily so a module that has not landed yet reports
``blocked`` (exit 2) instead of a traceback. Every verb is ``fn(argv: list[str]) -> int``."""
from __future__ import annotations

import importlib
import sys

VERBS: dict[str, str] = {
    "validate-config": "agent_allocation.config_cli:cmd_validate_config",
    "prepare-data": "agent_allocation.manifest:cmd_prepare_data",
    "doctor": "agent_allocation.doctor:cmd_doctor",
    "plan": "agent_allocation.plan:cmd_plan",
    "canary": "agent_allocation.live:cmd_canary",
    "pilot": "agent_allocation.live:cmd_pilot",
    "gates": "agent_allocation.gates:cmd_gates",
    "freeze": "agent_allocation.freeze:cmd_freeze",
    "run": "agent_allocation.live:cmd_run",
    "status": "agent_allocation.runtime.status:cmd_status",
    "stop": "agent_allocation.runtime.control:cmd_stop",
    "grade": "agent_allocation.grading.worker:cmd_grade",
    "seal": "agent_allocation.live:cmd_seal",
    "analyze": "agent_allocation.reporting.cli:cmd_analyze",
    "audit": "agent_allocation.reporting.audit:cmd_audit",
    "shard-plan": "agent_allocation.shards:cmd_shard_plan",
    "merge-shards": "agent_allocation.analysis.merge:cmd_merge_shards",
    "calibrate": "agent_allocation.calibrate:cmd_calibrate",
    "passk-run": "agent_allocation.passk.run:cmd_passk_run",
    "passk-analyze": "agent_allocation.passk.analyze:cmd_passk_analyze",
    # dynorch_v1 (WP-D, docs/dynorch/00_contract.md section 8)
    "dynorch-plan": "agent_allocation.dynorch.cli:cmd_dynorch_plan",
    "dynorch-run": "agent_allocation.dynorch.cli:cmd_dynorch_run",
    "dynorch-tables": "agent_allocation.dynorch.cli:cmd_dynorch_tables",
    "dynorch-gates": "agent_allocation.dynorch.cli:cmd_dynorch_gates",
    "dynorch-reconcile": "agent_allocation.dynorch.cli:cmd_dynorch_reconcile",
    "dynorch-doctor": "agent_allocation.dynorch.cli:cmd_dynorch_doctor",
    "dynorch-invalidate": "agent_allocation.dynorch.cli:cmd_dynorch_invalidate",
}
EXIT_BLOCKED = 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: allocation <verb> [options]\nverbs: " + ", ".join(VERBS))
        return 0
    verb, rest = argv[0], argv[1:]
    target = VERBS.get(verb)
    if target is None:
        print(f"unknown verb {verb!r}; expected one of: {', '.join(VERBS)}", file=sys.stderr)
        return EXIT_BLOCKED
    module_name, func_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name):
            print(f"blocked: {verb} is not implemented yet ({module_name})", file=sys.stderr)
            return EXIT_BLOCKED
        raise
    return int(getattr(module, func_name)(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
