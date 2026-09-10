"""WP7 E0: the legacy study_v4 reanalysis (spec §8; plan §5.7).

The suite runs against a synthetic mini run root in the study_v4 layout, so it exercises the
real readers (nested BCB item paths, ``sha256(final_answer)``-keyed HLE judgements, the
``episode.candidates`` stage/slot grid) without touching the shared filesystem.  One bounded
test does read the real run root - it only probes shard directory names - and the full
end-to-end legacy read is opt-in behind ``ARD36_E0_LEGACY=1`` because loading the legacy BCB
grade files costs about ten seconds of shared I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import legacy_e0 as E

PEER_ARM = next(a for a in E.ARMS if a.role == "peer")
PRIVATE_ARM = next(a for a in E.ARMS if a.role == "private")
GOLD = "42"
WRONG = "7"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidate(stage, slot, answer, *, confidence=0.7, valid=True, salt=""):
    body = json.dumps({"answer": answer, "salt": salt, "slot": slot}, sort_keys=True)
    payload = {"final_answer": answer}
    if confidence is not None:
        payload["confidence"] = confidence
    return {
        "stage": stage, "slot": slot, "valid": valid,
        "candidate": payload if valid else None,
        "candidate_id": _sha(f"cid:{stage}:{slot}:{answer}:{salt}"),
        "candidate_sha256": _sha(body) if valid else _sha(f"invalid:{stage}:{slot}"),
        "raw_content_sha256": _sha(body), "vote_key": None, "grouping_mode": None,
        "failure_code": None if valid else "TRUNCATED",
        "request_id": _sha(f"req:{stage}:{slot}:{salt}"),
    }


def _write_item(run_root: Path, cell_id: str, sid: str, candidates, *, nested=False,
                calls=(), packets=()):
    folder = run_root / "cells" / cell_id / "items"
    if nested:
        head, tail = sid.split("/", 1)
        path = folder / head / f"{tail}.json"
    else:
        path = folder / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 4, "split": "main", "source_id": sid,
        "domain": "bcb" if sid.startswith("bcb") else "hle", "status": "complete",
        "episode": {"candidates": list(candidates), "calls": list(calls),
                    "packets": list(packets), "N": K.N_AGENTS, "method": "DEC_ONE_ROUND"},
    }), encoding="utf-8")
    return path


def _write_hle_eval(run_root: Path, sid: str, correct_answers):
    path = run_root / "eval" / "hle" / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    judgements = {}
    for answer in (GOLD, WRONG):
        judgements[_sha(answer)] = {"correct": answer in correct_answers,
                                    "final_answer_sha256": _sha(answer), "judge": "yes"}
    path.write_text(json.dumps({"answer_format": "exactMatch", "judgements": judgements}),
                    encoding="utf-8")


@pytest.fixture
def legacy(tmp_path) -> Path:
    """Two HLE items and one BCB item under ``N.DEC_ONE_ROUND...s000``."""
    run_root = tmp_path / "study_v4"
    cell = f"{PEER_ARM.prefix}.s000"

    # hle-1: C0 = 2.  Slot 0 reprints the same answer with different bytes (identity changes,
    # canonical class does not); slot 1 switches from the correct to the wrong answer.
    root1 = [_candidate("root", 0, GOLD, confidence=0.9),
             _candidate("root", 1, GOLD, confidence=0.8),
             _candidate("root", 2, WRONG, confidence=0.3),
             _candidate("root", 3, WRONG, confidence=0.2),
             _candidate("root", 4, WRONG, confidence=0.1)]
    round1 = [_candidate("round1", 0, GOLD, confidence=0.95, salt="reprint"),
              _candidate("round1", 1, WRONG, confidence=0.7),
              _candidate("round1", 2, WRONG, confidence=0.3),
              _candidate("round1", 3, WRONG, confidence=0.2),
              _candidate("round1", 4, WRONG, confidence=0.1)]
    calls = [{"role": "root", "request_id": _sha(f"r{i}"), "engine_seed": 100 + i,
              "finish_reason": "stop", "prompt_tokens": 300, "completion_tokens": 900,
              "reasoning_tokens": 800, "aliased": True,
              "seed_key": ["hle-1", "main", "Qwen3-32B", 0, 0, "root", i, "stateless_bank"]}
             for i in range(K.N_AGENTS)]
    packets = [{"round": 1, "recipient_slot": s, "sender_slot": o, "kind": "peer"}
               for s in range(K.N_AGENTS) for o in range(K.N_AGENTS) if s != o]
    _write_item(run_root, cell, "hle-1", root1 + round1, calls=calls, packets=packets)
    _write_hle_eval(run_root, "hle-1", {GOLD})

    # hle-2: C0 = 1 and the correct answer is lost at round 1 - current coverage falls to 0
    # while archive coverage stays 1.  Slot 4 reports no confidence at the root.
    root2 = [_candidate("root", 0, GOLD, confidence=0.6),
             _candidate("root", 1, WRONG, confidence=0.9),
             _candidate("root", 2, WRONG, confidence=0.2),
             _candidate("root", 3, WRONG, confidence=0.1),
             _candidate("root", 4, WRONG, confidence=None)]
    round2 = [_candidate("round1", 0, WRONG, confidence=0.8),
              _candidate("round1", 1, WRONG, confidence=0.9),
              _candidate("round1", 2, WRONG, confidence=0.2),
              _candidate("round1", 3, WRONG, confidence=0.1),
              _candidate("round1", 4, WRONG, confidence=0.4, valid=False)]
    _write_item(run_root, cell, "hle-2", root2 + round2)
    _write_hle_eval(run_root, "hle-2", {GOLD})

    # A nested BCB item: one candidate passed, one failed on network access (quarantined).
    bcb_root = [_candidate("root", slot, f"code-{slot}", confidence=0.5)
                for slot in range(K.N_AGENTS)]
    _write_item(run_root, cell, "bcb:BigCodeBench/1", bcb_root, nested=True)
    bcb_path = run_root / "eval" / "bcb" / "eval.w1.s000.jsonl"
    bcb_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"candidate_id": bcb_root[0]["candidate_id"], "status": "pass", "stderr_tail": ""},
            {"candidate_id": bcb_root[1]["candidate_id"], "status": "fail",
             "stderr_tail": "urlopen error [Errno -3] Temporary failure in name resolution"},
            {"candidate_id": bcb_root[2]["candidate_id"], "status": "fail",
             "stderr_tail": "AssertionError: expected 3"}]
    bcb_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return run_root


def _rows(legacy: Path):
    store = E.LegacyStore(legacy)
    return store, E.transition_rows(store, PEER_ARM, PEER_ARM.cell_ids(legacy))


# --------------------------------------------------------------------------- arms


def test_the_matched_pair_is_dec_one_round_and_ind_private_revision():
    assert PEER_ARM.name == "N.DEC_ONE_ROUND"
    assert PRIVATE_ARM.name == "N.IND_PRIVATE_REVISION"
    assert PEER_ARM.n_items == PRIVATE_ARM.n_items == 100
    assert "S_HISTORY" not in {a.name for a in E.ARMS}
    assert "single-agent 64-step chain" in E.S_HISTORY_NOTE


def test_arm_cells_are_probed_not_listed(legacy):
    """The shard names are probed one ``isdir`` at a time; ``cells/`` is never walked."""
    assert PEER_ARM.cell_ids(legacy) == [f"{PEER_ARM.prefix}.s000"]
    assert PRIVATE_ARM.cell_ids(legacy) == []


# --------------------------------------------------------------------------- transitions


def test_both_change_definitions_are_emitted_and_they_disagree(legacy):
    _store, (rows, _reasons, _extra) = _rows(legacy)
    reprint = next(r for r in rows if r["sid"] == "hle-1" and r["slot"] == 0)
    assert reprint["changed_identity"] == 1     # different bytes
    assert reprint["changed_class"] == 0        # same canonical class
    assert reprint["identity_from"] != reprint["identity_to"]
    assert reprint["class_from"] == reprint["class_to"] == GOLD
    switched = next(r for r in rows if r["sid"] == "hle-1" and r["slot"] == 1)
    assert switched["changed_identity"] == switched["changed_class"] == 1
    assert switched["transition"] == "CW"


def test_candidate_sha256_over_splits_relative_to_the_canonical_class(legacy):
    _store, (rows, _reasons, _extra) = _rows(legacy)
    rates = E._change_rates(rows, PEER_ARM.name)
    assert rates["change_rate_candidate_sha256"] > rates["change_rate_vote_key"]


def test_every_row_carries_an_include_or_exclude_reason(legacy):
    _store, (rows, reasons, _extra) = _rows(legacy)
    assert rows, "no transition rows"
    for row in rows:
        assert row["include"] is (row["exclude_reason"] is None)
    assert "invalid_current" in reasons              # hle-2 slot 4 at round 1
    assert reasons["included"] > 0
    # nothing was dropped: one row per (item, slot) transition.  The BCB item has only a
    # root stage, so the two HLE items contribute every transition.
    assert len(rows) == 2 * K.N_AGENTS


def test_transitions_record_the_invalidity_classes(legacy):
    _store, (rows, _reasons, _extra) = _rows(legacy)
    lost = next(r for r in rows if r["sid"] == "hle-2" and r["slot"] == 4)
    assert lost["valid_from"] is True and lost["valid_to"] is False
    assert lost["invalidity_class"] == "valid_to_invalid"
    assert lost["exclude_reason"] == "invalid_current"
    assert lost["failure_code_to"] == "TRUNCATED"


def test_peer_packet_count_travels_with_the_transition(legacy):
    _store, (rows, _reasons, _extra) = _rows(legacy)
    peer_row = next(r for r in rows if r["sid"] == "hle-1")
    assert peer_row["n_packets_round"] == K.N_AGENTS * (K.N_AGENTS - 1)


# --------------------------------------------------------------------------- BCB quarantine


def test_bcb_network_failures_are_quarantined_not_scored_incorrect(legacy):
    store = E.LegacyStore(legacy)
    candidates = json.loads(
        (legacy / "cells" / f"{PEER_ARM.prefix}.s000" / "items" / "bcb:BigCodeBench"
         / "1.json").read_text())["episode"]["candidates"]
    assert store.correct("bcb:BigCodeBench/1", candidates[0]) is True
    assert store.correct("bcb:BigCodeBench/1", candidates[1]) is None   # quarantined
    assert store.correct("bcb:BigCodeBench/1", candidates[2]) is False  # a real failure
    assert len(store.bcb_quarantined) == 1


def test_nested_bcb_item_files_are_read(legacy):
    _store, (_rows_, _reasons, extra) = _rows(legacy)
    assert any(row["sid"] == "bcb:BigCodeBench/1" for row in extra["per_round"])


# --------------------------------------------------------------------------- distributions


def test_root_count_distribution_and_the_two_coverage_series(legacy):
    _store, (_rows_, _reasons, extra) = _rows(legacy)
    dist = E.root_count_distributions(extra["per_round"])[PEER_ARM.name]
    assert dist["C0_histogram"]["2"] == 1      # hle-1
    assert dist["C0_histogram"]["1"] == 1      # hle-2
    assert dist["coverage_current"]["0"] == pytest.approx(1.0)
    assert dist["coverage_current"]["1"] == pytest.approx(0.5)   # hle-2 lost its correct answer
    assert dist["coverage_archive"]["1"] == pytest.approx(1.0)   # the archive keeps it
    assert dist["coverage_current"]["1"] < dist["coverage_archive"]["1"]
    assert "not an iid pass@k" in dist["note"]


def test_forward_trajectories_are_conditioned_on_c0(legacy):
    _store, (_rows_, _reasons, extra) = _rows(legacy)
    dist = E.root_count_distributions(extra["per_round"])[PEER_ARM.name]
    assert dist["forward_trajectories"]["1"]["1"] == {"0": 1.0}   # C0=1 -> C1=0
    assert dist["forward_trajectories"]["2"]["1"] == {"1": 1.0}   # C0=2 -> C1=1


def test_items_without_complete_root_labels_are_counted_not_silently_dropped(legacy):
    _store, (_rows_, _reasons, extra) = _rows(legacy)
    dist = E.root_count_distributions(extra["per_round"])[PEER_ARM.name]
    assert dist["n_items_seen"] == 3
    assert dist["n_items_with_complete_root_labels"] == 2
    assert dist["n_items_excluded_unlabelled_root"] == 1      # the BCB item is partly graded


# --------------------------------------------------------------------------- discrimination


def test_within_item_auc_excludes_missing_confidence_and_never_imputes_half(legacy):
    _store, (_rows_, _reasons, extra) = _rows(legacy)
    panel = E.confidence_discrimination(extra["per_round"])[PEER_ARM.name]["by_round"]["0"]
    assert panel["n_missing_confidence"] == 1
    # hle-1 ranks perfectly (1.0); hle-2 with slot 4 EXCLUDED scores 2/3; the BCB item has one
    # graded-correct and one graded-wrong candidate at equal confidence, so it ties at 0.5.
    # Imputing 0.5 for hle-2's missing slot would have made that item 0.75 and the mean 0.75.
    assert panel["within_item_auc"] == pytest.approx((1.0 + 2 / 3 + 0.5) / 3)
    assert panel["within_item_auc"] != pytest.approx((1.0 + 0.75 + 0.5) / 3)
    assert panel["eligible_mixed_items"] == 3
    assert "conf.fillna(0.5) is not lifted" in \
        E.confidence_discrimination(extra["per_round"])[PEER_ARM.name]["note"]


def test_single_class_items_are_excluded_and_counted(legacy):
    _store, (_rows_, _reasons, extra) = _rows(legacy)
    panel = E.confidence_discrimination(extra["per_round"])[PEER_ARM.name]["by_round"]["1"]
    assert panel["excluded_single_class_items"] >= 1
    assert panel["eligible_mixed_items"] != panel["n_items"]


def test_pooled_auc_is_a_labelled_diagnostic_beside_the_within_item_value(legacy):
    _store, (_rows_, _reasons, extra) = _rows(legacy)
    panel = E.confidence_discrimination(extra["per_round"])[PEER_ARM.name]["by_round"]["0"]
    assert panel["pooled_auc_diagnostic"] is not None
    assert panel["pooled_auc_diagnostic"] != panel["within_item_auc"]
    assert set(panel["calibration"]) >= {"brier", "nll", "ece_primary", "conf_missing_rate"}


# --------------------------------------------------------------------------- call audit


def test_call_audit_recovers_request_identity_seeds_and_stop_reasons(legacy):
    store = E.LegacyStore(legacy)
    audit = E.call_audit(store, PEER_ARM, PEER_ARM.cell_ids(legacy))
    assert audit["n_calls"] == K.N_AGENTS
    assert audit["n_distinct_request_ids"] == K.N_AGENTS
    assert audit["n_distinct_engine_seeds"] == K.N_AGENTS
    assert audit["finish_reasons"] == {"stop": K.N_AGENTS}
    assert audit["seed_namespaces"] == {"stateless_bank": K.N_AGENTS}
    assert "8,192-token single-cap" in audit["note"] or "8,192-token" in audit["note"]


# --------------------------------------------------------------------------- deliverables


def test_run_e0_writes_the_five_deliverables(legacy, tmp_path):
    out = tmp_path / "e0"
    result = E.run_e0(legacy, out, arms=(PEER_ARM,))
    for name in ("legacy_audit.json", "legacy_root_count_distributions.json",
                 "legacy_confidence_discrimination.json", "legacy_discrepancies.md"):
        assert (out / name).exists(), name
    assert (out / "legacy_transitions.parquet").exists()
    audit = json.loads((out / "legacy_audit.json").read_text())
    assert audit["arms"][PEER_ARM.name]["transitions"] == 2 * K.N_AGENTS
    assert audit["arms"][PEER_ARM.name]["include_exclude_reasons"]
    assert any("fillna" in rule for rule in audit["rules"])
    assert any("judge-agreement" in rule for rule in audit["rules"])
    assert result["audit"]["bcb_quarantined"] == 1


def test_the_transitions_parquet_carries_both_definitions(legacy, tmp_path):
    from agents_scaling.ard.analysis import read_table

    out = tmp_path / "e0"
    E.run_e0(legacy, out, arms=(PEER_ARM,))
    rows = read_table(out / "legacy_transitions.parquet")
    assert rows
    assert {"identity_from", "identity_to", "class_from", "class_to", "changed_identity",
            "changed_class", "include", "exclude_reason"} <= set(rows[0])


def test_discrepancy_report_puts_both_definitions_side_by_side(legacy, tmp_path):
    out = tmp_path / "e0"
    E.run_e0(legacy, out, arms=(PEER_ARM,))
    report = (out / "legacy_discrepancies.md").read_text()
    assert "has no generating code" in report
    assert "0.392" in report and "0.085" in report
    assert "0.3948" in report and "0.0902" in report
    assert "0.477" in report and "0.814" in report
    assert "never reconciled" in report
    assert "legacy-only" in report
    assert E.S_HISTORY_NOTE in report


def test_emitter_refuses_a_duplicate_key():
    emitter = E.Emitter("test")
    emitter.emit("a", 1, note="first")
    with pytest.raises(KeyError):
        emitter.emit("a", 2)


def test_selection_pools_reports_both_pool_kinds_separately(legacy):
    from agents_scaling.study.aggregate import _write_parquet

    rows = [{"pool_kind": kind, "selected_correct": True, "oracle_coverage": True,
             "method": "DEC", "selector_id": "VOTE", "source_id": "hle-1"}
            for kind in ("latest_slots", "archive")]
    (legacy / "tables").mkdir(parents=True, exist_ok=True)
    _write_parquet(legacy / "tables" / "selections.parquet", rows)
    out = E.selection_pools(legacy)
    assert set(out["by_pool_kind"]) == {"latest_slots", "archive"}
    assert "different populations" in out["note"]


# --------------------------------------------------------------------------- real store


REAL_ROOT = Path(E.DEFAULT_LEGACY_RUN_ROOT)


@pytest.mark.skipif(not (REAL_ROOT / "cells").exists(), reason="study_v4 run root absent")
def test_real_run_root_shard_probe_finds_the_matched_arms():
    """Bounded: ``Arm.cell_ids`` stats named directories; it never lists ``cells/``."""
    assert len(PEER_ARM.cell_ids(REAL_ROOT)) == 10        # 10 shards x 10 items = 100
    assert len(PRIVATE_ARM.cell_ids(REAL_ROOT)) == 10
    main = next(a for a in E.ARMS if a.role == "main")
    assert len(main.cell_ids(REAL_ROOT)) == 40            # 40 shards x 10 items = 400


@pytest.mark.skipif(not os.environ.get("ARD36_E0_LEGACY"),
                    reason="set ARD36_E0_LEGACY=1 to read the real legacy store (~10 s of "
                           "shared I/O for the BCB grade files)")
def test_real_legacy_shard_reads_end_to_end():
    store = E.LegacyStore(REAL_ROOT)
    cells = PEER_ARM.cell_ids(REAL_ROOT)[:1]
    rows, reasons, extra = E.transition_rows(store, PEER_ARM, cells)
    assert rows and reasons["included"] > 0
    rates = E._change_rates(rows, PEER_ARM.name)
    assert rates["change_rate_candidate_sha256"] >= rates["change_rate_vote_key"]
