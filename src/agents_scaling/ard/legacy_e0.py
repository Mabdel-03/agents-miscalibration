"""E0: reanalysis of the study_v4 run root (spec §8; plan §5.7).

The purpose is to recover cheap dynamics from an existing run and to *choose frozen feature
definitions* from them.  It is explicitly **not** independent confirmation of anything: the
legacy HLE labels are judge-agreement labels, the legacy runs used a single 8,192-token cap
(spec §1.2, so every number here is labelled legacy-only), and the BCB rows that failed on
network access are quarantined rather than scored.

Source layout (all reads bounded to named cell directories - never a walk of ``cells/`` or
``requests/``):

* ``cells/<cell>/items/<sid>.json`` - ``episode.candidates`` as up to 9 stages x 5 slots, plus
  ``episode.calls`` (seed_key, engine_seed, token counts, finish_reason) and ``episode.packets``;
* ``eval/hle/<sid>.json`` - judgements keyed by ``sha256(final_answer)`` exactly as
  ``hle_judge.answer_sha256``, with ``candidate_id``/``candidate_sha256`` aliases;
* ``eval/bcb/*.jsonl`` - one row per graded candidate id;
* ``tables/selections.parquet`` - both pool kinds (``latest_slots`` and ``archive``).

The stage/slot indexing, the switch-row construction and the lone-correct survival logic are
lifted from ``docs/study_v4/atlas/scripts/s3_dynamics.py:46-158``; :class:`Emitter` is lifted
from that directory's ``_common.py``.  Three deliberate departures from the atlas scripts:

1. **The matched peer-versus-private pair is ``N.DEC_ONE_ROUND`` and
   ``N.IND_PRIVATE_REVISION``** (100 items each, root + one round, 20 versus 0 packets), not
   ``A.S_HISTORY`` - which despite its ``N5`` label is a single-agent 64-step chain and is
   relabelled as such in the audit rather than being read as a five-agent arm.
2. **Both change definitions travel together**: ``candidate_sha256`` identity *and* a
   recomputed canonical class from the frozen ``selection/normalize.vote_key`` dispatch.
   ``candidate_sha256`` over-splits (the atlas measured 0.751 versus 0.395 change rate on the
   same transitions), so a single number would be a definition, not a finding.
3. **Missing confidence is missing.**  ``s5_probes.py``'s ``conf.fillna(0.5)`` is *not*
   lifted: a candidate with no confidence is excluded from the discrimination panel and
   counted, because imputing the midpoint invents a ranking that was never reported.

Deliverables (spec §8): ``legacy_audit.json``, ``legacy_transitions.parquet``,
``legacy_root_count_distributions.json``, ``legacy_confidence_discrimination.json`` and the
discrepancy report ``legacy_discrepancies.md``.  Bad transitions are never discarded to make
the counts consistent; every row carries an include/exclude reason.

This module is evaluator-side: it reads correctness labels and must never be imported by a
runtime module.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.ard import analysis as AN
from agents_scaling.ard import contract as K
from agents_scaling.ard.reference_metrics import within_item_auc
from agents_scaling.experiment import io
from agents_scaling.study.aggregate import _write_parquet
from agents_scaling.study.selection.normalize import vote_key

#: The study_v4 run root (plan §9.4: E0 targets this run root only).
DEFAULT_LEGACY_RUN_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4"

#: Stage names in order; a legacy DEC episode runs root + up to eight rounds.
STAGES: tuple[str, ...] = ("root",) + tuple(f"round{i}" for i in range(1, 9))

#: BCB rows whose failure is an environment problem, not a wrong program (spec §8: "existing
#: code execution labels are usable subject to grader and environment checks").
NETWORK_MARKERS: tuple[str, ...] = (
    "Temporary failure in name resolution", "urlopen error", "Network is unreachable",
    "Connection refused", "nltk_data] Error loading",
)


@dataclass(frozen=True)
class Arm:
    """One legacy arm, addressed by shard probe rather than by listing ``cells/``."""

    name: str
    prefix: str
    role: str                      # "main" | "peer" | "private" | "single_agent"
    n_items: int
    max_shards: int = 64
    note: str = ""

    def cell_ids(self, run_root: Path) -> list[str]:
        cells = run_root / "cells"
        return [f"{self.prefix}.s{i:03d}" for i in range(self.max_shards)
                if (cells / f"{self.prefix}.s{i:03d}").is_dir()]


#: The arms plan §5.7 names.  ``A.S_HISTORY`` is carried only so the audit can state what it
#: is; it never enters the peer-versus-private pair.
ARMS: tuple[Arm, ...] = (
    Arm("A.DEC", "A.DEC.32B.N5.B4.Fnat.e0", "main", 400,
        note="400 main items, five agents, up to eight peer-revision rounds."),
    Arm("N.DEC_ONE_ROUND", "N.DEC_ONE_ROUND.32B.N5.B4.F00.e0", "peer", 100,
        note="root + one round, 20 peer packets; the PEER side of the matched pair."),
    Arm("N.IND_PRIVATE_REVISION", "N.IND_PRIVATE_REVISION.32B.N5.B4.F00.e0", "private", 100,
        note="root + one private revision, 0 packets; the PRIVATE side of the matched pair."),
)

S_HISTORY_NOTE = (
    "A.S_HISTORY carries an N5 label but is a single-agent 64-step chain, not a five-agent "
    "team; it is relabelled here and excluded from the peer-versus-private pair."
)

#: Figures the atlas reported, kept for the discrepancy report and never silently reconciled.
ATLAS_CLAIMS: Mapping[str, Any] = {
    "peer_vs_private_change_rate_text": {"peer": 0.392, "private": 0.085,
                                         "provenance": "no generating code in the atlas"},
    "peer_vs_private_change_rate_recomputed": {"peer": 0.3948, "private": 0.0902,
                                               "provenance": "recomputed by final-answer identity"},
    "figure_4a_agreement": {"value": 0.477, "grouping": "candidate_sha256",
                            "pool": "current"},
    "section_7_4_agreement": {"value": 0.814, "grouping": "vote_key", "n_items": 372},
}


# --------------------------------------------------------------------------- Emitter


def _clean(value: Any) -> Any:
    """``docs/study_v4/atlas/scripts/_common.py`` verbatim (numpy-free variant)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if value is None or isinstance(value, str):
        return value
    return str(value)


class Emitter:
    """Collects ``{key: record}`` with provenance (lifted from the atlas ``_common.py``).

    A duplicate key is an error: two analyses that disagree must both be visible.
    """

    def __init__(self, script: str) -> None:
        self.script = script
        self.rows: dict[str, dict[str, Any]] = {}

    def emit(self, key: str, value: Any, *, n: int | None = None, stat: Any = None,
             domain: str | None = None, note: str | None = None,
             source: str | None = None, reason: str | None = None) -> dict[str, Any]:
        if key in self.rows:
            raise KeyError(f"duplicate atlas key: {key}")
        record: dict[str, Any] = {"value": _clean(value), "script": self.script}
        if n is not None:
            record["n"] = int(n)
        if stat is not None:
            record["stat"] = _clean(stat)
        if domain is not None:
            record["domain"] = domain
        if note is not None:
            record["note"] = note
        if source is not None:
            record["source"] = source
        if reason is not None:
            record["reason"] = reason
        self.rows[key] = record
        return record

    def to_dict(self) -> dict[str, Any]:
        return dict(self.rows)


# --------------------------------------------------------------------------- store


class LegacyStore:
    """Bounded reader over one study_v4 run root."""

    def __init__(self, run_root: str | os.PathLike) -> None:
        self.run_root = Path(run_root)
        self._bcb: dict[str, bool] | None = None
        self._bcb_quarantined: dict[str, str] = {}
        self._hle: dict[str, dict[str, Any]] = {}

    # ---- labels -----------------------------------------------------------------
    @property
    def bcb(self) -> dict[str, bool]:
        """``candidate_id -> passed``.  Environment failures are quarantined, not scored."""
        if self._bcb is None:
            table: dict[str, bool] = {}
            folder = self.run_root / "eval" / "bcb"
            if folder.exists():
                for path in sorted(folder.glob("*.jsonl")):
                    with path.open(encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            cid = row.get("candidate_id")
                            if not cid:
                                continue
                            tail = f"{row.get('stderr_tail', '')}{row.get('stdout_tail', '')}"
                            if row.get("status") != "pass" and any(m in tail for m in NETWORK_MARKERS):
                                self._bcb_quarantined[cid] = "bcb_network_access"
                                continue
                            table[cid] = row.get("status") == "pass"
            self._bcb = table
        return self._bcb

    @property
    def bcb_quarantined(self) -> dict[str, str]:
        _ = self.bcb
        return self._bcb_quarantined

    def hle(self, sid: str) -> dict[str, Any]:
        if sid not in self._hle:
            path = self.run_root / "eval" / "hle" / f"{sid}.json"
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            self._hle[sid] = payload
        return self._hle[sid]

    def answer_format(self, sid: str) -> str:
        if sid.startswith("bcb"):
            return "code"
        return str(self.hle(sid).get("answer_format") or "exactMatch")

    def correct(self, sid: str, candidate: Mapping[str, Any]) -> bool | None:
        """``s3_dynamics.py:24-30`` with the BCB environment quarantine added."""
        if sid.startswith("bcb"):
            cid = candidate.get("candidate_id")
            if cid in self.bcb_quarantined:
                return None
            return self.bcb.get(cid)
        judgements = self.hle(sid).get("judgements") or {}
        final_answer = str((candidate.get("candidate") or {}).get("final_answer", ""))
        digest = hashlib.sha256(final_answer.encode("utf-8")).hexdigest()
        for key in (candidate.get("candidate_id"), candidate.get("candidate_sha256"), digest):
            if key in judgements:
                return bool(judgements[key].get("correct"))
        return None

    # ---- items ------------------------------------------------------------------
    def item_files(self, cell_ids: Sequence[str]) -> Iterator[Path]:
        """``s3_dynamics.py:33-37``: flat and one-level-nested item files of named cells."""
        for cell_id in cell_ids:
            items = self.run_root / "cells" / cell_id / "items"
            if not items.exists():
                continue
            yield from sorted(items.glob("*.json"))
            yield from sorted(items.glob("*/*.json"))

    def load_item(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- candidates


def index_by_stage(episode: Mapping[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    """``{stage: {slot: candidate}}`` (``s3_dynamics.py:54-57``)."""
    by_stage: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    for candidate in episode.get("candidates") or []:
        if candidate.get("stage") and candidate.get("slot") is not None:
            by_stage[str(candidate["stage"])][int(candidate["slot"])] = candidate
    return by_stage


def stage_order(by_stage: Mapping[str, Mapping[int, Any]]) -> list[str]:
    return [stage for stage in STAGES if stage in by_stage]


def identity_key(candidate: Mapping[str, Any]) -> str | None:
    """Exact-answer identity: the atlas's cross-round grouping key."""
    return candidate.get("candidate_sha256") or None


def canonical_key(store: LegacyStore, sid: str, candidate: Mapping[str, Any]) -> str | None:
    """Recomputed canonical class through the frozen ``normalize.vote_key`` dispatch.

    The legacy ``vote_key`` field is only populated in the sealed final round, so a per-round
    trajectory cannot use it; recomputing gives both definitions on every transition.
    """
    final_answer = (candidate.get("candidate") or {}).get("final_answer")
    if final_answer is None:
        return None
    try:
        key, _mode = vote_key(store.answer_format(sid), str(final_answer))
    except ValueError:
        return None
    return key


def confidence_of(candidate: Mapping[str, Any]) -> float | None:
    """Self-reported confidence, or ``None``.  Never imputed (plan §5.7)."""
    raw = (candidate.get("candidate") or {}).get("confidence")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _domain(sid: str) -> str:
    return "bcb" if sid.startswith("bcb") else "hle"


# --------------------------------------------------------------------------- transitions


def transition_rows(
    store: LegacyStore, arm: Arm, cell_ids: Sequence[str], *, splits: Sequence[str] = ("main",)
) -> tuple[list[dict[str, Any]], collections.Counter, dict[str, Any]]:
    """Member-level ``(item, slot, round)`` transitions with both change definitions.

    Every candidate pair produces a row.  ``include`` is False with an ``exclude_reason``
    whenever the row cannot enter an aggregate - the row itself is never dropped (spec §8:
    "do not discard bad transitions to make the counts consistent").
    """
    rows: list[dict[str, Any]] = []
    reasons: collections.Counter = collections.Counter()
    per_round_state: list[dict[str, Any]] = []
    for path in store.item_files(cell_ids):
        payload = store.load_item(path)
        if splits and payload.get("split") not in splits:
            reasons["item_wrong_split"] += 1
            continue
        sid = str(payload["source_id"])
        episode = payload.get("episode") or {}
        by_stage = index_by_stage(episode)
        order = stage_order(by_stage)
        if not order:
            reasons["episode_without_candidates"] += 1
            continue
        packets_by_round: collections.Counter = collections.Counter()
        for packet in episode.get("packets") or []:
            packets_by_round[int(packet.get("round", -1))] += 1
        previous: Mapping[int, Mapping[str, Any]] | None = None
        for round_index, stage in enumerate(order):
            members = by_stage[stage]
            valid = {slot: c for slot, c in members.items() if c.get("valid")}
            id_keys = {slot: identity_key(c) for slot, c in valid.items()}
            class_keys = {slot: canonical_key(store, sid, c) for slot, c in valid.items()}
            id_counts = collections.Counter(k for k in id_keys.values() if k)
            class_counts = collections.Counter(k for k in class_keys.values() if k)
            labels = {slot: store.correct(sid, c) for slot, c in valid.items()}
            known = [v for v in labels.values() if v is not None]
            per_round_state.append({
                "arm": arm.name, "sid": sid, "domain": _domain(sid), "round": round_index,
                "stage": stage, "n_members": len(members), "n_valid": len(valid),
                "n_labelled": len(known), "n_correct": sum(1 for v in known if v),
                "labels": {int(s): labels[s] for s in sorted(labels)},
                "confidences": {int(s): confidence_of(valid[s]) for s in sorted(valid)},
                "identity_keys": {int(s): id_keys[s] for s in sorted(valid)},
                "class_keys": {int(s): class_keys[s] for s in sorted(valid)},
                "unlabelled_valid": len(valid) - len(known),
            })
            if previous is not None:
                majority_id = id_counts.most_common(1)[0][0] if id_counts else None
                majority_class = class_counts.most_common(1)[0][0] if class_counts else None
                for slot, candidate in sorted(members.items()):
                    before = previous.get(slot)
                    row = _transition_row(
                        store, arm, sid, slot, round_index, stage, order[round_index - 1],
                        before, candidate, majority_id, majority_class,
                        packets_by_round.get(round_index, 0))
                    reasons[row["exclude_reason"] or "included"] += 1
                    rows.append(row)
            previous = members
    audit = {"n_rows": len(rows), "reasons": dict(reasons)}
    return rows, reasons, {"per_round": per_round_state, "audit": audit}


def _transition_row(
    store: LegacyStore, arm: Arm, sid: str, slot: int, round_index: int, stage: str,
    prev_stage: str, before: Mapping[str, Any] | None, after: Mapping[str, Any],
    majority_id: str | None, majority_class: str | None, n_packets: int,
) -> dict[str, Any]:
    valid_from = bool(before and before.get("valid"))
    valid_to = bool(after.get("valid"))
    id_from = identity_key(before) if before else None
    id_to = identity_key(after) if valid_to else None
    class_from = canonical_key(store, sid, before) if before else None
    class_to = canonical_key(store, sid, after) if valid_to else None
    correct_from = store.correct(sid, before) if before else None
    correct_to = store.correct(sid, after) if valid_to else None
    exclude_reason: str | None = None
    if before is None:
        exclude_reason = "no_previous_member"
    elif not valid_from:
        exclude_reason = "invalid_previous"
    elif not valid_to:
        exclude_reason = "invalid_current"
    elif id_from is None or id_to is None:
        exclude_reason = "missing_identity_key"
    elif correct_from is None or correct_to is None:
        exclude_reason = "unlabelled_member"
    transition = None
    if correct_from is not None and correct_to is not None:
        transition = ("C" if correct_from else "W") + ("C" if correct_to else "W")
    invalidity = None
    if valid_from and not valid_to:
        invalidity = "valid_to_invalid"
    elif not valid_from and valid_to:
        invalidity = "invalid_to_valid"
    elif not valid_from and not valid_to:
        invalidity = "invalid_to_invalid"
    return {
        "arm": arm.name, "arm_role": arm.role, "sid": sid, "domain": _domain(sid),
        "slot": int(slot), "round_from": round_index - 1, "round_to": round_index,
        "stage_from": prev_stage, "stage_to": stage,
        "valid_from": valid_from, "valid_to": valid_to,
        "identity_from": id_from, "identity_to": id_to,
        "class_from": class_from, "class_to": class_to,
        "changed_identity": (None if (id_from is None or id_to is None) else int(id_from != id_to)),
        "changed_class": (None if (class_from is None or class_to is None)
                          else int(class_from != class_to)),
        "to_majority_identity": (None if (id_to is None or majority_id is None)
                                 else int(id_from != id_to and id_to == majority_id)),
        "to_majority_class": (None if (class_to is None or majority_class is None)
                              else int(class_from != class_to and class_to == majority_class)),
        "in_minority_before": (None if (id_from is None or majority_id is None)
                               else int(id_from != majority_id)),
        "correct_from": correct_from, "correct_to": correct_to,
        "transition": transition, "invalidity_class": invalidity,
        "confidence_from": confidence_of(before) if before else None,
        "confidence_to": confidence_of(after),
        "failure_code_to": after.get("failure_code"),
        "n_packets_round": int(n_packets),
        "include": exclude_reason is None,
        "exclude_reason": exclude_reason,
    }


# --------------------------------------------------------------------------- distributions


def root_count_distributions(per_round: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """C0 in {0..5} from the root stage, with forward trajectories and both coverage series.

    Coverage is reported twice and never merged: ``coverage_current`` is "some member of the
    current round is correct", ``coverage_archive`` is "some member of any round up to t is
    correct".  The archive series is cumulative and monotone by construction; reading it as an
    iid pass@k would be the spec §15.4 nonclaim.
    """
    by_item: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = collections.defaultdict(dict)
    for row in per_round:
        by_item[(str(row["arm"]), str(row["sid"]))][int(row["round"])] = row
    out: dict[str, Any] = {}
    for arm in sorted({str(r["arm"]) for r in per_round}):
        c0_hist: collections.Counter = collections.Counter()
        trajectories: dict[int, dict[int, collections.Counter]] = collections.defaultdict(
            lambda: collections.defaultdict(collections.Counter))
        coverage_current: dict[int, list[float]] = collections.defaultdict(list)
        coverage_archive: dict[int, list[float]] = collections.defaultdict(list)
        excluded_unlabelled = 0
        n_items = 0
        for (arm_name, _sid), rounds in by_item.items():
            if arm_name != arm or 0 not in rounds:
                continue
            n_items += 1
            root = rounds[0]
            if root["unlabelled_valid"] or root["n_labelled"] == 0:
                excluded_unlabelled += 1
                continue
            c0 = int(root["n_correct"])
            c0_hist[c0] += 1
            seen_correct = False
            for round_index in sorted(rounds):
                row = rounds[round_index]
                trajectories[c0][round_index][int(row["n_correct"])] += 1
                current = float(row["n_correct"] > 0)
                seen_correct = seen_correct or bool(row["n_correct"] > 0)
                coverage_current[round_index].append(current)
                coverage_archive[round_index].append(float(seen_correct))
        total = sum(c0_hist.values())
        out[arm] = {
            "n_items_seen": n_items,
            "n_items_with_complete_root_labels": total,
            "n_items_excluded_unlabelled_root": excluded_unlabelled,
            "C0_histogram": {str(c): c0_hist.get(c, 0) for c in range(K.N_AGENTS + 1)},
            "C0_distribution": {str(c): (c0_hist.get(c, 0) / total if total else None)
                                for c in range(K.N_AGENTS + 1)},
            "forward_trajectories": {
                str(c0): {str(r): {str(c): counts[c] / max(1, sum(counts.values()))
                                   for c in sorted(counts)}
                          for r, counts in sorted(rounds.items())}
                for c0, rounds in sorted(trajectories.items())},
            "coverage_current": {str(r): (sum(v) / len(v)) for r, v in sorted(coverage_current.items()) if v},
            "coverage_archive": {str(r): (sum(v) / len(v)) for r, v in sorted(coverage_archive.items()) if v},
            "note": ("coverage_current and coverage_archive are separate series; the archive "
                     "series is cumulative and is not an iid pass@k (spec §15.4)."),
        }
    return out


def confidence_discrimination(per_round: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Within-question confidence AUC by arm and round, with the eligible mixed-item count.

    Missing confidences are **excluded**, not imputed: this is the atlas figure of ~0.495
    recomputed under the honest missingness policy.  The pooled AUC is reported beside it as a
    labelled diagnostic only - a pooled number mixes between-item difficulty into a
    within-item ranking claim (spec §15.1).
    """
    out: dict[str, Any] = {}
    for arm in sorted({str(r["arm"]) for r in per_round}):
        by_round: dict[int, dict[str, Any]] = {}
        for round_index in sorted({int(r["round"]) for r in per_round if r["arm"] == arm}):
            aucs: list[float] = []
            excluded_single_class = 0
            n_items = 0
            missing_conf = 0
            pooled_scores: list[float] = []
            pooled_labels: list[int] = []
            for row in per_round:
                if row["arm"] != arm or int(row["round"]) != round_index:
                    continue
                n_items += 1
                scores, labels = [], []
                for slot, conf in row["confidences"].items():
                    label = row["labels"].get(int(slot))
                    if label is None:
                        continue
                    if conf is None:
                        missing_conf += 1
                        continue
                    scores.append(float(conf))
                    labels.append(int(bool(label)))
                if len(scores) < 2:
                    continue
                pooled_scores.extend(scores)
                pooled_labels.extend(labels)
                value = within_item_auc(scores, labels)
                if value is None:
                    excluded_single_class += 1
                else:
                    aucs.append(float(value))
            pooled = within_item_auc(pooled_scores, pooled_labels) if pooled_scores else None
            calibration = AN.calibration_metrics(pooled_scores, pooled_labels)
            by_round[round_index] = {
                "within_item_auc": (sum(aucs) / len(aucs)) if aucs else None,
                "eligible_mixed_items": len(aucs),
                "excluded_single_class_items": excluded_single_class,
                "n_items": n_items,
                "n_missing_confidence": missing_conf,
                "pooled_auc_diagnostic": pooled,
                "calibration": calibration,
            }
        out[arm] = {
            "by_round": {str(r): v for r, v in sorted(by_round.items())},
            "note": ("Within-item AUC over mixed-correctness items only, averaged over items; "
                     "missing confidence is excluded (the atlas probe script's "
                     "conf.fillna(0.5) is not lifted).  pooled_auc_diagnostic is a labelled "
                     "diagnostic, never the within-item claim.  Legacy judge labels only."),
        }
    return out


# --------------------------------------------------------------------------- call audit


def call_audit(store: LegacyStore, arm: Arm, cell_ids: Sequence[str],
               *, splits: Sequence[str] = ("main",)) -> dict[str, Any]:
    """Spec §8 item 6: request hashes, sampler seeds, call counts, tokens, stop reasons."""
    request_ids: set[str] = set()
    engine_seeds: set[int] = set()
    finish: collections.Counter = collections.Counter()
    roles: collections.Counter = collections.Counter()
    namespaces: collections.Counter = collections.Counter()
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    n_calls = aliased = 0
    for path in store.item_files(cell_ids):
        payload = store.load_item(path)
        if splits and payload.get("split") not in splits:
            continue
        for call in (payload.get("episode") or {}).get("calls") or []:
            n_calls += 1
            aliased += bool(call.get("aliased"))
            if call.get("request_id"):
                request_ids.add(str(call["request_id"]))
            if call.get("engine_seed") is not None:
                engine_seeds.add(int(call["engine_seed"]))
            finish[str(call.get("finish_reason"))] += 1
            roles[str(call.get("role"))] += 1
            seed_key = call.get("seed_key") or []
            if isinstance(seed_key, (list, tuple)) and seed_key:
                namespaces[str(seed_key[-1])] += 1
            prompt_tokens += int(call.get("prompt_tokens") or 0)
            completion_tokens += int(call.get("completion_tokens") or 0)
            reasoning_tokens += int(call.get("reasoning_tokens") or 0)
    return {
        "n_calls": n_calls, "n_distinct_request_ids": len(request_ids),
        "n_distinct_engine_seeds": len(engine_seeds), "n_aliased_calls": aliased,
        "finish_reasons": dict(finish), "roles": dict(roles),
        "seed_namespaces": dict(namespaces),
        "prompt_tokens_total": prompt_tokens, "completion_tokens_total": completion_tokens,
        "reasoning_tokens_total": reasoning_tokens,
        "note": ("Legacy runs used a single 8,192-token cap with no separate final-answer "
                 "allowance; every quantity here is labelled legacy-only (spec §1.2)."),
    }


# --------------------------------------------------------------------------- selections


def selection_pools(run_root: Path) -> dict[str, Any]:
    """Both pool kinds from ``tables/selections.parquet`` (``latest_slots`` and ``archive``).

    Mismatched pool definitions are flagged, never reconciled: the two atlas agreement numbers
    differ because they are computed on different pools with different grouping keys.
    """
    path = run_root / "tables" / "selections.parquet"
    rows = AN.read_table(path)
    if not rows:
        return {"status": "absent", "path": str(path)}
    by_pool: dict[str, dict[str, Any]] = {}
    for row in rows:
        pool = str(row.get("pool_kind"))
        entry = by_pool.setdefault(pool, {"n": 0, "selected_correct": 0, "oracle_coverage": 0,
                                          "methods": collections.Counter(),
                                          "selectors": collections.Counter()})
        entry["n"] += 1
        entry["selected_correct"] += bool(row.get("selected_correct"))
        entry["oracle_coverage"] += bool(row.get("oracle_coverage"))
        entry["methods"][str(row.get("method"))] += 1
        entry["selectors"][str(row.get("selector_id"))] += 1
    for entry in by_pool.values():
        entry["methods"] = dict(entry["methods"])
        entry["selectors"] = dict(entry["selectors"])
        entry["selected_accuracy"] = entry["selected_correct"] / entry["n"] if entry["n"] else None
        entry["oracle_rate"] = entry["oracle_coverage"] / entry["n"] if entry["n"] else None
    return {"status": "read", "n_rows": len(rows), "by_pool_kind": by_pool,
            "note": "Both pool kinds are reported separately; they are different populations."}


# --------------------------------------------------------------------------- driver


def _change_rates(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    included = [r for r in rows if r["arm"] == arm and r["include"]]
    by_identity = [r["changed_identity"] for r in included if r["changed_identity"] is not None]
    by_class = [r["changed_class"] for r in included if r["changed_class"] is not None]
    return {
        "n_included_transitions": len(included),
        "change_rate_candidate_sha256": (sum(by_identity) / len(by_identity)) if by_identity else None,
        "n_candidate_sha256": len(by_identity),
        "change_rate_vote_key": (sum(by_class) / len(by_class)) if by_class else None,
        "n_vote_key": len(by_class),
    }


def discrepancy_report(rates: Mapping[str, Any], arms: Sequence[Arm]) -> str:
    """The brief discrepancy report (spec §8; plan §5.7).  Nothing is reconciled silently."""
    peer = next((a.name for a in arms if a.role == "peer"), None)
    private = next((a.name for a in arms if a.role == "private"), None)
    lines = [
        "# E0 discrepancy report (legacy study_v4 reanalysis)", "",
        "Every figure below is **legacy-only**: the legacy runs used a single 8,192-token cap "
        "with no separate final-answer allowance (spec §1.2), and the HLE labels are judge-"
        "agreement labels until independently audited (spec §8).", "",
        "## 1. The 39.2% / 8.5% pair has no generating code", "",
        "The atlas text reports a peer-versus-private answer-change pair of "
        f"{ATLAS_CLAIMS['peer_vs_private_change_rate_text']['peer']:.3f} / "
        f"{ATLAS_CLAIMS['peer_vs_private_change_rate_text']['private']:.3f}; no script in "
        "`docs/study_v4/atlas/scripts/` produces it.  Recomputed by final-answer identity the "
        f"pair is {ATLAS_CLAIMS['peer_vs_private_change_rate_recomputed']['peer']:.4f} / "
        f"{ATLAS_CLAIMS['peer_vs_private_change_rate_recomputed']['private']:.4f}.  This run "
        "recomputes it again on the matched arms, under **both** change definitions:", "",
        "| arm | transitions | change rate (candidate_sha256) | change rate (vote_key) |",
        "|---|---|---|---|",
    ]
    for name in [n for n in (peer, private) if n]:
        row = rates.get(name, {})
        lines.append(
            f"| {name} | {row.get('n_included_transitions', 0)} | "
            f"{_fmt(row.get('change_rate_candidate_sha256'))} | "
            f"{_fmt(row.get('change_rate_vote_key'))} |")
    lines += [
        "",
        "`candidate_sha256` over-splits: two byte-different reports of the same answer are two "
        "changes under identity and none under the canonical class.  Neither number is "
        "\"the\" change rate; both travel together.", "",
        "## 2. Figure 4a agreement versus the §7.4 text", "",
        f"Figure 4a reports {ATLAS_CLAIMS['figure_4a_agreement']['value']:.3f} agreement "
        f"({ATLAS_CLAIMS['figure_4a_agreement']['grouping']} grouping, "
        f"{ATLAS_CLAIMS['figure_4a_agreement']['pool']} pool); §7.4 reports "
        f"{ATLAS_CLAIMS['section_7_4_agreement']['value']:.3f} "
        f"({ATLAS_CLAIMS['section_7_4_agreement']['grouping']} grouping, "
        f"{ATLAS_CLAIMS['section_7_4_agreement']['n_items']} items).  These are **two "
        "definitions on two populations**, reported side by side and never reconciled into a "
        "single agreement number.", "",
        "## 3. Arm labelling", "", f"- {S_HISTORY_NOTE}",
        "- The matched peer-versus-private pair is `N.DEC_ONE_ROUND` (20 packets) versus "
        "`N.IND_PRIVATE_REVISION` (0 packets), 100 items each, root + one round.", "",
        "## 4. Rules applied", "",
        "- Missing confidence is excluded from the discrimination panel, never imputed as 0.5.",
        "- BCB rows that failed on network access are quarantined, not scored as incorrect.",
        "- Unlabelled valid members are counted and reported, not dropped silently.",
        "- No bad transition was discarded to make the counts reconcile.", "",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def run_e0(
    legacy_run_root: str | os.PathLike = DEFAULT_LEGACY_RUN_ROOT,
    out_dir: str | os.PathLike | None = None,
    *,
    arms: Sequence[Arm] = ARMS,
    splits: Sequence[str] = ("main",),
    write: bool = True,
) -> dict[str, Any]:
    """Run the whole E0 reanalysis and write the five deliverables."""
    store = LegacyStore(legacy_run_root)
    out = Path(out_dir) if out_dir is not None else Path(legacy_run_root) / "e0"
    emitter = Emitter("agents_scaling.ard.legacy_e0")
    all_rows: list[dict[str, Any]] = []
    all_per_round: list[dict[str, Any]] = []
    arm_meta: dict[str, Any] = {}
    for arm in arms:
        cell_ids = arm.cell_ids(store.run_root)
        rows, reasons, extra = transition_rows(store, arm, cell_ids, splits=splits)
        all_rows.extend(rows)
        all_per_round.extend(extra["per_round"])
        arm_meta[arm.name] = {
            "role": arm.role, "prefix": arm.prefix, "n_cells": len(cell_ids),
            "expected_items": arm.n_items,
            "items_seen": len({(r["sid"]) for r in extra["per_round"]}),
            "transitions": len(rows),
            "include_exclude_reasons": dict(reasons),
            "calls": call_audit(store, arm, cell_ids, splits=splits),
            "note": arm.note,
        }
        emitter.emit(f"e0.{arm.name}.rows", len(rows), n=len(rows),
                     source=f"{arm.prefix}.s*", note=arm.note)
        for reason, count in sorted(reasons.items()):
            emitter.emit(f"e0.{arm.name}.reason.{reason}", count, n=count,
                         reason=reason,
                         note="include/exclude reason per analysis row (spec §8)")

    rates = {arm.name: _change_rates(all_rows, arm.name) for arm in arms}
    for name, row in rates.items():
        emitter.emit(f"e0.{name}.change_rates", row,
                     note="both change definitions; candidate_sha256 over-splits")
    distributions = root_count_distributions(all_per_round)
    discrimination = confidence_discrimination(all_per_round)
    pools = selection_pools(store.run_root)
    emitter.emit("e0.selection_pools", pools,
                 note="both legacy pool kinds, reported separately")
    emitter.emit("e0.s_history", S_HISTORY_NOTE, reason="arm_relabelled")
    emitter.emit("e0.bcb_quarantined", len(store.bcb_quarantined),
                 n=len(store.bcb_quarantined), reason="bcb_network_access",
                 note="graded BCB rows whose failure was environmental, not a wrong program")
    emitter.emit("e0.legacy_only", True,
                 note="Legacy 8,192-token single-cap runs; judge-agreement labels (spec §1.2, §8)")

    audit = {
        "study_id": K.STUDY_ID,
        "legacy_run_root": str(store.run_root),
        "arms": arm_meta,
        "atlas_claims": dict(ATLAS_CLAIMS),
        "change_rates": rates,
        "selection_pools": pools,
        "bcb_quarantined": len(store.bcb_quarantined),
        "emitted": emitter.to_dict(),
        "rules": [
            "Missing confidence is excluded, never imputed as 0.5 (s5_probes.py's fillna is "
            "not lifted).",
            "Both candidate_sha256 identity and a recomputed vote_key canonical class are "
            "emitted for every transition.",
            S_HISTORY_NOTE,
            "BCB rows failing on network access are quarantined.",
            "Missing-final candidates are invalid and scored 0 in the count series; they are "
            "excluded from the transition aggregates with reason 'invalid_current'.",
            "Bad transitions are retained with an exclude reason, never discarded.",
            "Legacy labels are judge-agreement labels; this is not independent confirmation.",
        ],
    }
    report = discrepancy_report(rates, arms)
    result = {
        "audit": audit, "transitions": all_rows, "root_count_distributions": distributions,
        "confidence_discrimination": discrimination, "discrepancy_report": report,
        "out_dir": str(out),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        io.write_json(out / "legacy_audit.json", audit)
        _write_parquet(out / "legacy_transitions.parquet", all_rows)
        io.write_json(out / "legacy_root_count_distributions.json", distributions)
        io.write_json(out / "legacy_confidence_discrimination.json", discrimination)
        io.atomic_write_text(out / "legacy_discrepancies.md", report)
    return result


def run_e0_cell(runner: Any, item_id: str) -> None:
    """``ArdCellKind.E0`` worker (``kinds.KIND_TABLE``): run E0, then publish the item file."""
    options = dict(runner.ard_cell.options)
    result = run_e0(
        options.get("legacy_run_root", DEFAULT_LEGACY_RUN_ROOT),
        options.get("out_dir") or (Path(runner.run_root) / "e0"),
        splits=tuple(options.get("splits", ("main",))),
    )
    counts = {
        "transitions": len(result["transitions"]),
        "arms": len(result["audit"]["arms"]),
        "out_dir": result["out_dir"],
    }
    runner.event("legacy_e0_complete", **counts)
    task = runner.tasks.get(item_id)
    if task is None:      # the synthetic "ALL" item of an analysis cell
        io.write_json(runner.paths.item(item_id), {
            "schema_version": 1, "cell": runner.ard_cell.to_dict(), "source_id": item_id,
            "family": "", "split": runner.ard_cell.split, "status": "complete",
            "episodes": [], "timing": {"finished_at": runner.clock()},
            "worker": runner._worker_block(), "grades": [],
            "code_version": runner.code_version, "rows": counts,
        })
        return
    runner.write_item(task, ())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.ard.legacy_e0",
                                description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--legacy-run-root", default=DEFAULT_LEGACY_RUN_ROOT)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--split", action="append", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_e0(args.legacy_run_root, args.out_dir,
                    splits=tuple(args.split or ("main",)))
    print(json.dumps({"transitions": len(result["transitions"]),
                      "arms": {name: meta["transitions"]
                               for name, meta in result["audit"]["arms"].items()},
                      "out_dir": result["out_dir"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARMS", "ATLAS_CLAIMS", "DEFAULT_LEGACY_RUN_ROOT", "STAGES", "Arm", "Emitter",
    "LegacyStore", "S_HISTORY_NOTE", "call_audit", "canonical_key", "confidence_of",
    "confidence_discrimination", "discrepancy_report", "identity_key", "index_by_stage",
    "root_count_distributions", "run_e0", "run_e0_cell", "selection_pools", "stage_order",
    "transition_rows",
]
