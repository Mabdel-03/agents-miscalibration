"""One deterministic mocked four-benchmark run through the lead's orchestration entry point, then resume and reconcile.

Entry point (docs/00_contract.md section 6, Q1):
    live.run_offline(paths, config, items, transport, *, extension_item_ids=(), code_grader=None) -> Mapping
It writes manifests from ``items`` and runs generate -> extract -> grade -> select -> seal -> analyze -> audit; a
second call on the same paths is the resume path.

The transport answers from the rendered prompt only (no node ids, no hidden state), so the run is deterministic:
  ROOT (base budget)            -> wrong answer          DEEP (budget above base) -> correct answer
  PRIVATE / CP update           -> correct answer        PEER update              -> repeats its own previous final
  POOLED synthesis              -> correct answer
Consequences checked independently: PEER n=4 r=4 delivers 0 and PRIVATE_TEAM n=4 r=4 delivers 1 on every item
(primary_i = -1); on extension items CP h=3 is 1, CP h=0 (= PEER round 1) is 0, PRIVATE_TEAM r=1 and r=4 are 1
(primary_ii = +1); POOLED is correct although its four input roots are all wrong (synthesis may exceed coverage).
"""
from __future__ import annotations

import copy
import gzip
import json
import re
from hashlib import sha256

import httpx
import pytest

from agent_allocation.contracts import CheckpointView, GradeResult, GradeStatus, InventoryItem, ParseStatus

from tests.integration import adversarial
from tests.integration import independent_recompute as ir
from tests.integration import reference_impl as ref
from tests.integration.conftest import optional_module
from tests.integration.dispatch_harness import fast_config
from tests.integration.openrouter_stub import completion_payload

pytestmark = [pytest.mark.timeout(600)]

E2E_ITEMS = ("math500/adv/1", "mmlu_pro/adv/1", "bbh_logic4/adv/1", "bbh_logic4/adv/2", "mbpp_plus/adv/1")
EXTENSION_ITEMS = ("math500/adv/1", "bbh_logic4/adv/2")
ANSWERS = {                                                    # item -> (correct, wrong)
    "math500/adv/1": ("4", "5"),
    "mmlu_pro/adv/1": ("C", "A"),
    "bbh_logic4/adv/1": ("B", "A"),
    "bbh_logic4/adv/2": ("False", "True"),
    "mbpp_plus/adv/1": ("def add_two(a, b):\n    return a + b", "def add_two(a, b):\n    return a - b"),
}
FINAL_RE = re.compile(r"<final>(.*?)</final>", re.S)


class PromptOnlyModel:
    """Deterministic stand-in for the provider: every reply is a pure function of the request body."""

    def __init__(self, tasks):
        self.tasks = tasks
        self.completions: list[dict] = []

    def _item(self, text: str) -> str:
        found = [t.item_id for t in self.tasks if t.prompt in text]
        assert len(found) == 1, f"request does not identify exactly one task: {found}"
        return found[0]

    def reply(self, body: dict) -> str:
        user = "\n".join(m.get("content", "") for m in body["messages"] if m.get("role") == "user")
        item = self._item(user)
        correct, wrong = ANSWERS[item]
        if "OTHER CANDIDATE SOLUTIONS" in user:
            own = user.split("YOUR PREVIOUS WRITTEN SOLUTION", 1)[-1].split("OTHER CANDIDATE SOLUTIONS", 1)[0]
            finals = FINAL_RE.findall(own)
            answer = finals[-1].strip() if finals else wrong
        elif "INDEPENDENT CANDIDATE SOLUTIONS" in user or "YOUR PREVIOUS WRITTEN SOLUTION" in user:
            answer = correct
        else:
            answer = correct if body["reasoning"]["max_tokens"] > 1024 else wrong
        return f"<rationale>Deterministic mock.</rationale>\n<final>\n{answer}\n</final>"

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            if not request.url.path.endswith("/chat/completions"):
                return httpx.Response(404, json={"error": {"code": 404, "message": "not available offline"}})
            body = json.loads(request.content)
            assert "seed" not in body and body["provider"]["allow_fallbacks"] is False
            self.completions.append(body)
            content = self.reply(body)
            payload = completion_payload(content, generation_id=f"gen-e2e-{len(self.completions)}",
                                         reasoning_tokens=min(300, body["reasoning"]["max_tokens"]))
            return httpx.Response(200, json=payload, headers={"x-openrouter-cache-status": "MISS"})
        return httpx.MockTransport(handle)


def stub_code_grader(candidate, private, paths) -> GradeResult:
    """Labelled stub (never executes model code): used only when the sandboxed private grader is unavailable."""
    if candidate.parse_status is not ParseStatus.VALID:
        return GradeResult(candidate.candidate_id, GradeStatus.INVALID_OUTPUT, None, False, "qa-stub-code-v1",
                           private.private_hash, detail_code="stub_invalid")
    return GradeResult(candidate.candidate_id, GradeStatus.GRADED, "a + b" in (candidate.final_answer or ""), True,
                       "qa-stub-code-v1", private.private_hash, detail_code="stub_code_grader")


def _snapshot(paths, subdirs=("raw", "parsed", "selections")) -> dict[str, str]:
    return {paths.relative(p): sha256(p.read_bytes()).hexdigest()
            for sub in subdirs for p in sorted((paths.run_dir / sub).rglob("*")) if p.is_file()}


VERDICT_FIELDS = ("candidate_id", "status", "semantic_correct", "operational_success", "grader_version",
                  "private_eval_hash", "detail_code")


def _verdicts(paths) -> dict[str, tuple]:
    out = {}
    for p in sorted(paths.grades_dir.rglob("*.json")):
        data = json.loads(p.read_text())
        out[paths.relative(p)] = tuple(data.get(k) for k in VERDICT_FIELDS)
    return out


@pytest.fixture()
def e2e(run_paths, cfg, monkeypatch):
    live = optional_module("agent_allocation.live")
    if not hasattr(live, "run_offline"):
        pytest.skip("waiting on live.run_offline (Q1)")
    monkeypatch.setenv("OPENROUTER_API_KEY", adversarial.fake_api_key())   # must never be read into artifacts
    items = {t.item_id: (t, p) for t, p, _ in adversarial.load_items()}
    inventory = [InventoryItem(public=items[i][0], private=items[i][1]) for i in E2E_ITEMS]
    model = PromptOnlyModel([items[i][0] for i in E2E_ITEMS])
    config = fast_config(cfg)
    config = copy.deepcopy(config)

    def run():
        return live.run_offline(run_paths, config, inventory, model.transport(), extension_item_ids=EXTENSION_ITEMS,
                                code_grader=stub_code_grader)
    return run_paths, run, model, config


def test_mocked_four_benchmark_run_generates_selects_grades_exports_resumes_and_reconciles(e2e, canaries):
    paths, run, model, config = e2e
    summary = run()
    expected_nodes = 125 * len(E2E_ITEMS) + 13 * len(EXTENSION_ITEMS)

    # --- generate: every semantic node dispatched exactly once --------------------------------------------------
    recomputed = ir.load(paths.run_dir)
    job_nodes = {nid for nodes in recomputed.jobs.values() for nid in nodes}
    assert len(job_nodes) == expected_nodes, summary
    assert len(model.completions) == expected_nodes, f"{len(model.completions)} requests for {expected_nodes} nodes"
    assert set(recomputed.parsed) == job_nodes

    # --- select / seal / grade / export: the independent hook agrees and every check passes ---------------------
    payload = ir.independent_recompute(paths.run_dir)
    failing = {c["name"]: c["detail"] for c in payload["checks"] if c["passed"] is False}
    assert not failing, failing
    assert payload["contrasts"][ir.PRIMARY_I] == pytest.approx(-1.0, abs=1e-12)
    assert payload["contrasts"][ir.PRIMARY_II] == pytest.approx(1.0, abs=1e-12)
    assert (paths.reports_dir / "AUDIT.md").is_file() and any(paths.tables_dir.rglob("*")), "analysis exports missing"
    assert summary.get("audit_rc") == 0, summary.get("audit_rc")
    assert summary.get("stubbed_code_grader") is True

    # --- one full item per benchmark, traced by the separate script --------------------------------------------
    for item in ("math500/adv/1", "mmlu_pro/adv/1", "bbh_logic4/adv/1", "mbpp_plus/adv/1"):
        trace = ir.recompute_item(paths.run_dir, item)["partitions"]["main"]
        n = 138 if item in EXTENSION_ITEMS else 125
        assert trace["nodes"] == trace["parsed"] == trace["raw_verified"] == trace["graded"] == n, (item, trace["raw_problems"])
        cps = trace["checkpoints"]
        assert cps["peer/4/4/current"]["delivered_correct"] == 0 and cps["private_team/4/4/current"]["delivered_correct"] == 1
        assert cps["sample/25/current"]["delivered_correct"] == 0 and cps["deep/16/current"]["delivered_correct"] == 1
        assert cps["peer/5/4/current"]["nominal_generations"] == 25
        if item in EXTENSION_ITEMS:
            assert cps["pooled/4/current"]["delivered_correct"] == 1 and cps["sample/4/current"]["coverage"] == 0

    # --- no gold, private test, reference code or key in any public artifact -------------------------------------
    public_hits = ref.scan_tree_for_canaries(paths.run_dir, canaries,
                                             exclude_dirs=[paths.private_dir, paths.grades_dir, paths.grade_cache_dir])
    assert public_hits == [], public_hits
    key_only = {"fake_api_key": canaries["fake_api_key"]}
    assert ref.scan_tree_for_canaries(paths.run_dir, key_only) == []
    assert ref.scan_tree_for_canaries(paths.local_state_dir, key_only) == []
    assert all(canaries["fake_api_key"] not in json.dumps(body) for body in model.completions)

    # --- resume: zero new requests, artifacts byte-identical, same numbers -------------------------------------
    before, verdicts = _snapshot(paths), _verdicts(paths)
    run()
    assert len(model.completions) == expected_nodes, "resume re-dispatched stored successful nodes"
    assert _snapshot(paths) == before, "resume rewrote raw/parsed/selection artifacts"
    assert _verdicts(paths) == verdicts, "resume changed a private verdict"
    again = ir.independent_recompute(paths.run_dir)
    assert again["contrasts"] == payload["contrasts"]

    # --- reconcile: one settlement per successful attempt, collection cost equals the usage we billed ----------
    events = ir.load(paths.run_dir).journal
    settles = [e for e in events if e.kind.value == "settle"]
    assert len(settles) == expected_nodes
    assert ref.LedgerChecker(ref.Decimal(str(config["budget"]["credit_ceiling_usd"]))).replay(events).violations == []


def test_mocked_run_survives_a_crash_after_raw_write_and_resumes_without_redispatch(e2e, monkeypatch):
    paths, run, model, _ = e2e
    state_mod = optional_module("agent_allocation.storage.state")
    original = state_mod.StateStore.record_result
    calls = {"n": 0}

    def crash_once(self, result, candidate):
        calls["n"] += 1
        if calls["n"] == 40:
            raise RuntimeError("injected crash: raw written, terminal commit lost")
        return original(self, result, candidate)

    monkeypatch.setattr(state_mod.StateStore, "record_result", crash_once)
    try:
        run()
    except Exception:
        pass
    monkeypatch.setattr(state_mod.StateStore, "record_result", original)
    served_ok_before = len(model.completions)
    run()
    bodies = [json.dumps(b, sort_keys=True) for b in model.completions]
    expected_nodes = 125 * len(E2E_ITEMS) + 13 * len(EXTENSION_ITEMS)
    recomputed = ir.load(paths.run_dir)
    assert set(recomputed.parsed) == {nid for nodes in recomputed.jobs.values() for nid in nodes}
    assert len(model.completions) == expected_nodes, (
        f"{len(model.completions) - expected_nodes} extra requests after resume (first pass sent {served_ok_before})")
    failing = {c["name"]: c["detail"] for c in ir.independent_recompute(paths.run_dir)["checks"] if c["passed"] is False}
    assert not failing, failing
    assert len(bodies) == expected_nodes


@pytest.mark.xfail(strict=True, reason="L-1: live.run_offline resume re-grades every stored candidate and rewrites its "
                                       "current grade file (only cpu_seconds changes); the spec 10.4 grade cache "
                                       "(item, code hash, evaluator version, environment digest) is not consulted")
def test_resume_reuses_stored_grades_byte_for_byte(run_paths, cfg):
    live = optional_module("agent_allocation.live")
    if not hasattr(live, "run_offline"):
        pytest.skip("waiting on live.run_offline (Q1)")
    items = {t.item_id: (t, p) for t, p, _ in adversarial.load_items()}
    ids = ("math500/adv/1",)
    inventory = [InventoryItem(public=items[i][0], private=items[i][1]) for i in ids]
    model = PromptOnlyModel([items[i][0] for i in ids])
    config = fast_config(cfg)
    live.run_offline(run_paths, config, inventory, model.transport(), code_grader=stub_code_grader)
    before = _snapshot(run_paths, ("grades",))
    live.run_offline(run_paths, config, inventory, model.transport(), code_grader=stub_code_grader)
    assert _snapshot(run_paths, ("grades",)) == before


@pytest.mark.xfail(strict=True, reason="L-2: run_offline returns stubbed_code_grader=True but no report under reports/ "
                                       "labels the code verdicts as stubbed (contract section 6, Q1 decision)")
def test_stubbed_code_grader_is_labelled_in_reports(run_paths, cfg):
    live = optional_module("agent_allocation.live")
    if not hasattr(live, "run_offline"):
        pytest.skip("waiting on live.run_offline (Q1)")
    items = {t.item_id: (t, p) for t, p, _ in adversarial.load_items()}
    ids = ("mbpp_plus/adv/1",)
    inventory = [InventoryItem(public=items[i][0], private=items[i][1]) for i in ids]
    model = PromptOnlyModel([items[i][0] for i in ids])
    live.run_offline(run_paths, fast_config(cfg), inventory, model.transport(), code_grader=stub_code_grader)
    text = "\n".join(p.read_text(errors="replace") for p in run_paths.reports_dir.rglob("*.md")).lower()
    assert "stub" in text
