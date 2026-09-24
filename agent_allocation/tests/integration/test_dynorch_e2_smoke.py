"""E2 spawn-disabled controls (H1n, H3n) through the REAL proxy stack over the fake upstream ($0; plan §1 E2, §9
Gate 5 prerequisite). The arms had unit coverage of their argv/config only; this pins what the real CLIs offer the
proxy when spawning is disabled, and that the proxy's fail-closed tool surface (plan §5) admits it: the first request
carries no spawn tool, nothing is refused, and a direct answer ends the run graded."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_allocation.api.fake import FakeReply
from agent_allocation.dynorch.harness.codex import CodexAdapter
from agent_allocation.dynorch.harness.openclaude import OpenclaudeAdapter
from tests.integration import dynorch_smoke_support as S
from tests.integration.dynorch_smoke_support import RunScript, SmokeStack

CODEX_BIN = Path(S.CFG["dynorch"]["codex_prefix"]) / "bin" / "codex"
FINAL = FakeReply(content="<rationale>Direct.</rationale>\n<final>42</final>", think_tokens=120)
SPAWN_NAMES = {"Agent", "spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent", "sessions_spawn"}


@pytest.fixture(scope="module")
def base():
    loaded = S.load_base()
    if loaded is None:
        pytest.skip("base run root not readable here")
    return loaded


def _adapters(harness: str):
    if harness == "codex":
        wall = S.CFG["dynorch"]["wall_timeout_seconds"]
        return {"codex": CodexAdapter(model=S.CFG["primary_model"], prefix=str(S.CFG["dynorch"]["codex_prefix"]),
                                      wall_base=int(wall["base"]), wall_per_rung=int(wall["per_rung"]))}
    return {"openclaude": OpenclaudeAdapter(model=S.CFG["primary_model"])}


@pytest.mark.timeout(300)
@pytest.mark.parametrize("arm,harness", [("H1n", "openclaude"), ("H3n", "codex")])
async def test_spawn_disabled_arm_answers_directly(tmp_path: Path, base, arm: str, harness: str):
    if harness == "codex" and not CODEX_BIN.is_file():
        pytest.skip("codex is not installed in the scratch prefix")
    marker = S.home_marker()
    item = S.dev_item(base, "aime")
    async with SmokeStack(tmp_path, adapters=_adapters(harness)) as stack:
        spec = stack.specs(base, arm=arm, rung=4, items=[item], stage="ext_e2")[0]
        stack.scripts[spec.run_key] = RunScript(root=[FINAL])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    first = docs[0]["harness_request"] if docs else {}
    names = {t.get("name") for t in first.get("tools") or []}
    common = S.common_checks(stack, spec, record, marker)
    checks = {
        "a_request_reached_the_proxy": bool(docs),
        "no_spawn_tool_offered": not (names & SPAWN_NAMES),
        "offered_tools_within_allowlist": names <= set(spec.allowed_tools),
        "nothing_refused": not record.refused_calls,
        "run_done_graded": record.status == "done" and record.parse_status == "valid" and record.correct in (True, False),
        "one_call_no_spawn": record.calls == 1 and record.spawns == 0,
        **common[0],
    }
    failed = S.record_case(f"e2/{arm}", checks, detail={**common[1], "offered_tools": sorted(n for n in names if n),
                                                        "allowed_tools": list(spec.allowed_tools),
                                                        "status": record.status, "error": record.error,
                                                        "refused": record.refused_calls})
    assert not failed, (failed, sorted(n for n in names if n), spec.allowed_tools, record.status, record.error,
                        record.refused_calls)
