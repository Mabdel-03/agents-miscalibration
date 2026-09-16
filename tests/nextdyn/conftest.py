"""Shared fixtures for ``tests/nextdyn`` (plan §6).

* ``tmp_run_root`` — a run root with the nextdyn directory skeleton (cells, registers,
  plan/root_teams, pools/<pool>/servers, tables, protected).
* ``study_cfg`` / ``compute_cfg`` / ``analysis_cfg`` — the committed configs, ``study.yaml`` in
  ``mode: test`` (R=2) and ``analysis.yaml``'s ``test`` profile.
* ``stub_tokenizer`` / ``real_tokenizer_32b`` / ``register_fake_server`` / ``study_config`` —
  re-registered from ``tests/study/conftest.py`` so WP1 can drive the fake vLLM server here.
* HF is forced offline exactly as ``tests/study/conftest.py`` does (importing it does so).
* ``pytest_collection_modifyitems`` records every collected test name on the pytest config
  (attribute ``nextdyn_collected_test_names``) so ``test_spec18_index.py`` can report which
  spec-18 names are not yet collected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("HF_HOME", "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tests.study.conftest import (  # noqa: E402,F401  (re-registered fixtures)
    StubTokenizer,
    real_tokenizer_32b,
    register_fake_server,
    stub_tokenizer,
    study_config,
)

from agents_scaling.nextdyn import config as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "nextdyn"
RUN_ROOT_SUBDIRS = (
    "cells", "registers", "plan/root_teams", "plan/manifests", "pools/deadline/servers",
    "pools/low/servers", "pools/audit/servers", "tables", "protected", "requests", "status",
)

COLLECTED_ATTR = "nextdyn_collected_test_names"


def pytest_collection_modifyitems(session, config, items):
    names = getattr(config, COLLECTED_ATTR, None)
    if names is None:
        names = set()
        setattr(config, COLLECTED_ATTR, names)
    for item in items:
        names.add(getattr(item, "originalname", None) or item.name)


@pytest.fixture
def tmp_run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run_root"
    for name in RUN_ROOT_SUBDIRS:
        (root / name).mkdir(parents=True)
    return root


@pytest.fixture(scope="session")
def study_cfg() -> C.StudyConfig:
    return C.load_study(CONFIG_DIR / "study.yaml", mode="test")


@pytest.fixture(scope="session")
def compute_cfg() -> C.ComputeConfig:
    return C.load_compute(CONFIG_DIR / "compute.yaml")


@pytest.fixture(scope="session")
def analysis_cfg() -> C.AnalysisConfig:
    return C.load_analysis(CONFIG_DIR / "analysis.yaml", profile="test")


# =============================================================================== WP1 (inference)
# Fake-server fixtures for the generation path: a fake vLLM 0.21 endpoint registered on a
# nextdyn serving profile, a dispatch context wired to it, and the small token contract the
# WP1 tests use (a 16-token thinking cap makes a forced boundary cost sixteen tokens instead
# of two thousand).  Everything is injected -- pool, tokenizer, cost function -- so the whole
# path runs offline, which is what spec 18.5 asks for.

from contextlib import contextmanager  # noqa: E402
from typing import Any, Iterator  # noqa: E402

from agents_scaling.nextdyn import contract as K  # noqa: E402
from agents_scaling.nextdyn import ids as I  # noqa: E402
from agents_scaling.nextdyn import inference as INF  # noqa: E402
from agents_scaling.study.inference.store import RequestStore  # noqa: E402
from agents_scaling.study.resources.oracle import FlopOracle, make_cost_fn  # noqa: E402
from tests.study.fake_vllm import FakeTokenizer, FakeVllmServer  # noqa: E402

#: The 8B lane: one profile, one served name, one pinned cell from the committed study.yaml.
WP1_PROFILE = "nd1-8B"
WP1_MODEL = "primary"
WP1_SPLIT_ROLE = "pilot"
WP1_ITEM = "ndtestitem000001"
WP1_VARIANT = "ndtestitem000001.v0"


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """The whitespace tokenizer the client and the fake server must *share* (decode is
    instance-local, so a report decoded through another instance comes back as ``<unk>``)."""
    return FakeTokenizer()


@pytest.fixture(scope="session")
def nd_contract(study_cfg) -> K.StudyContract:
    return C.contract_of(study_cfg)


@pytest.fixture
def nd_tokens() -> K.TokenContract:
    """A small token contract: same shape as the study's, sixteen thinking tokens wide."""
    return K.TokenContract(thinking_cap=16, final_cap=32, served_window=4096, reserve=8,
                           packet_cap=64, rationale_cap=32, claims_max=4, claim_cap=16)


@pytest.fixture
def nd_cost_fn():
    """The analytic FLOP oracle off the frozen arch table -- never the ``None`` placeholder,
    and never a read of the HF cache (``FlopOracle.from_table``)."""
    return make_cost_fn({size: FlopOracle.from_table(size) for size in ("8B", "32B")})


@pytest.fixture
def nd_serve(tmp_run_root, fake_tokenizer):
    """``with nd_serve(**knobs) as srv:`` -- a started, registered fake nd1-8B endpoint."""

    @contextmanager
    def _serve(profile: str = WP1_PROFILE, **kwargs: Any) -> Iterator[FakeVllmServer]:
        from agents_scaling.serving.profiles import get_serving_profile

        spec = get_serving_profile(profile)
        kwargs.setdefault("tokenizer", fake_tokenizer)
        kwargs.setdefault("served_model_name", spec.served_model_name)
        kwargs.setdefault("max_model_len", spec.max_model_len)
        kwargs.setdefault("responder", nd_json_responder())
        # A short thinking span by default: a test that wants a forced boundary asks
        # for one by raising ``thinking_words`` above the arm's cap.
        kwargs.setdefault("thinking_words", 4)
        server = FakeVllmServer(**kwargs).start()
        server.register(tmp_run_root, profile)
        try:
            yield server
        finally:
            server.stop()

    return _serve


@pytest.fixture
def nd_pool(tmp_run_root):
    """A :class:`NextdynEndpointPool` over the fake registry, unfiltered by default."""

    def _pool(profile: str = WP1_PROFILE, **kwargs: Any) -> INF.NextdynEndpointPool:
        kwargs.setdefault("refresh_min_interval_s", 0.0)
        kwargs.setdefault("probe_timeout", 2.0)
        return INF.NextdynEndpointPool(tmp_run_root, profile, shard=0, **kwargs)

    return _pool


@pytest.fixture
def nd_ctx(tmp_run_root, nd_contract, nd_cost_fn, fake_tokenizer, nd_pool):
    """A :class:`DispatchContext` on the fake endpoint, one tokenizer for every cell."""

    def _ctx(*, cell_id: str = "donor_bank/test", run_id: str = "nd1-test",
             pool: Any = None, **kwargs: Any) -> INF.DispatchContext:
        shared = pool if pool is not None else nd_pool()
        return INF.DispatchContext(
            store=RequestStore(tmp_run_root / "requests"),
            pool_for=lambda profile: shared,
            cost_fn=nd_cost_fn,
            tokenizer_for=lambda name: fake_tokenizer,
            contract=nd_contract,
            cell_id=cell_id,
            run_id=run_id,
            **kwargs,
        )

    return _ctx


def nd_key(contract: K.StudyContract, model: K.ModelCell, tokens: K.TokenContract, *,
           kind: I.CellKind = I.CellKind.DONOR_BANK, item: str = WP1_ITEM,
           variant: str = WP1_VARIANT, draw: int = 0) -> I.CompositeKey:
    """A bank-draw composite key: the simplest key every generation kind shares."""
    return I.CompositeKey(
        study_id=contract.study_id, cell_kind=kind, source_item_id=item, item_variant_id=variant,
        root_team_id="", protocol_hash="", receiver_model_hash=I.model_hash(model),
        budget_hash=tokens.budget_hash(), replicate_id=0, round=0, agent_slot=draw,
        phase=K.PHASE_BANK)


def nd_messages(task: str = "What is 2 + 2 ? Answer with the integer .") -> tuple[dict, ...]:
    """A neutral prompt naming the D6 output contract and no condition (R5)."""
    return ({"role": "user",
             "content": ("Solve the task and answer with one JSON object with fields "
                         f"reasoning , shareable_claims , final_answer , confidence .\n\nTask: {task}")},)


def nd_request(contract: K.StudyContract, tokens: K.TokenContract, *, model: str = WP1_MODEL,
               messages: tuple[dict, ...] | None = None, top_logprobs: int = 1,
               **key_kwargs: Any) -> INF.DecisionRequest:
    cell = contract.model(model)
    key = nd_key(contract, cell, tokens, **key_kwargs)
    return INF.DecisionRequest(
        messages=messages if messages is not None else nd_messages(),
        model=cell, adapter=cell.adapter, tokens=tokens,
        seed_key=I.seed_key_for(key, split_role=WP1_SPLIT_ROLE),
        purpose=key.phase, top_logprobs=top_logprobs, key=key)


#: The public object of plan D6, spaced so the whitespace tokenizer sees several tokens.
NDJSON_OPEN = '{ "reasoning": "'
NDJSON_TAIL = (' a public rationale ", "shareable_claims": [ "one claim" ], '
               '"final_answer": "4", "confidence": 0.75 }')


def nd_json_responder(answer: str = "4"):
    """A fake-server responder that speaks plan D6's JSON object on both routes.

    On ``/v1/chat/completions`` it returns the whole object; on ``/v1/completions`` (a phase-2
    continuation, recognised by the token-prefix prompt) it returns the object *minus* the
    prefilled opening, so a client that prefilled ``{"reasoning": "`` reassembles exactly one
    object.  The fake server's own continuation detector looks for ARD36's ``<final>`` tag and
    knows nothing about this study's JSON contract, which is why WP1 supplies its own.
    """

    def _respond(payload: dict) -> str:
        tail = NDJSON_TAIL.replace('"4"', f'"{answer}"')
        if "prompt_token_ids" in payload:
            return tail.lstrip()
        return NDJSON_OPEN + tail

    return _respond


def nd_unreported_derivation(text: str = "So m ^ 2 = 1 so m = 1 or m = -1 . Therefore"):
    """The measured ARD36 defect: after a forced boundary phase 1 keeps deriving and never
    opens the object; the continuation is what delivers a report."""
    tail = nd_json_responder()

    def _respond(payload: dict) -> str | None:
        if "prompt_token_ids" in payload:
            return tail(payload)
        return text

    return _respond


#: Commands that change cluster state.  A test that reaches one of these has escaped its
#: sandbox: the WP4 suite once left two 12-hour `asys-nextdyn-*` control jobs running on
#: mit_preemptable because `test_every_runbook_verb_parses` exercised the `run` verb without
#: --dry-run, and nothing noticed for over an hour.  Injected runners are the design (every
#: dispatch entry point takes one); this is the net under it.
SCHEDULER_COMMANDS = ("sbatch", "scancel", "srun", "salloc", "scontrol")


@pytest.fixture(autouse=True)
def _no_test_ever_reaches_the_scheduler(monkeypatch):
    """Refuse `sbatch` and friends for the whole suite.

    Read-only Slurm queries (`squeue`, `sinfo`, `sacct`) are left alone: they are how the
    fleet and status modules observe the cluster, they change nothing, and stubbing them out
    would hide the very code paths those tests exist to cover.
    """
    import subprocess

    real_run, real_popen, real_check = subprocess.run, subprocess.Popen, subprocess.check_output

    def _refuse(cmd):
        argv = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        head = str(argv[0]).rsplit("/", 1)[-1] if argv else ""
        if head in SCHEDULER_COMMANDS:
            raise AssertionError(
                f"a test tried to run {head!r}: tests never change cluster state. "
                "Inject a runner (every dispatch entry point takes one) or pass --dry-run.")

    def run(cmd, *a, **k):
        _refuse(cmd)
        return real_run(cmd, *a, **k)

    def popen(cmd, *a, **k):
        _refuse(cmd)
        return real_popen(cmd, *a, **k)

    def check_output(cmd, *a, **k):
        _refuse(cmd)
        return real_check(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "check_output", check_output)
