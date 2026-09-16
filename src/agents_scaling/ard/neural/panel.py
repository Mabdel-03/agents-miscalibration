"""ARD36 E6 capture panel: the frozen item selector that did not exist (WP6).

Spec §14.2 ("capture cohort"), plan §5.6.  ``contract.CAPTURE_TOTAL_ITEMS`` (144) and
``contract.CAPTURE_SPLIT_ITEMS`` (64 train / 16 validation / 64 test) have been in the
frozen contract since WP0 and **were never read by anything**: ``cells.build_e6_capture``
is handed whatever task list ``build_module("E6")`` passes it, which is all 384 natural
items.  This module is the missing selector, and nothing else in the run changes meaning
because of it — it adds a panel, it does not reinterpret one.

Why a panel at all, and what question it is for
-----------------------------------------------
E6 here is authorised for **descriptive geometry only**: participation ratio, effective
rank and displacement/alignment statistics by round, protocol and team state, over the
capture-eligible decisions (8B, ``cohort == natural``, ``contract.CAPTURE_PROTOCOLS`` =
PRIVATE/PEER/BALANCED, rounds 0-3).  That question is answered by *coverage of the design*
— every (protocol, round) cell filled on complete five-agent teams — so the panel balances
**family × split** and takes whole items, never teams chosen on their outcome.

**This panel is not powered to separate a correct minority from its wrong bloc, and must
never be described as if it were.**  Measured on this run's own tables (below): the panel
contains **12 contested items** carrying **48 contested (protocol, round) team-rounds**
(38 once the round-0 roots, which all three protocols share, are collapsed to one per
item).  Contested means ``n_correct in (1, 2)`` on a fully graded five-agent team.  A
held-out probe on that many positives has a clustered ``SE(AUC) ~ 0.083`` and
``MDE80(AUC) ~ 0.73``, so neither a hit nor a null from it would be informative.  The
number is *reported* by :func:`contested_team_rounds`, never optimised — selecting items
because they are contested would both destroy the design balance this panel exists for and
condition the cohort on a measured outcome.

Selection rule
--------------
Within each (split, family) stratum, items are ordered by
``sha256("ARD36_v1|1|capture_panel|<seed>|<item_id>")`` and a prefix of the quota is
taken.  Two properties follow: the panel is a pure function of (item table, seed), so it
can be recomputed and audited by anyone; and it is decided before any capture runs, on
nothing a capture could tell us.

The item table already carries a frozen ``rank`` (the hash order within the stratum used
to draw the 96-item 32B panel, ``items.large_panel``).  A prefix of ``rank`` is *not*
reused here on purpose: it would make the capture panel a near-superset of the 32B panel
by construction, silently coupling two cohorts that the spec treats as independent.  A
separate declared seed keeps them independent, and :data:`DEFAULT_PANEL_SEED` is the only
number anyone has to trust.

Measured budget (run root ``<results>/ard36``, ``tables_v2`` + the request store,
seed 36, 2026-09-13; every figure below is from :func:`panel_budget`, not from the brief)
------------------------------------------------------------------------------------
* **Items** — 144: math500/mmlu_pro 32/32 train, 8/8 validation, 32/32 test.  All 21
  subjects present in the natural cohort appear in the panel.
* **Capture-eligible decisions** — **7,200** on the panel; **19,200** over all 384 items.
  Each item contributes exactly 50: five round-0 roots, *shared byte-for-byte by all three
  protocols* (verified: 5 distinct ``decision_id`` at round 0 per item, not 15), plus
  3 protocols × 3 update rounds × 5 agents.  Every one of the 19,200 has a committed
  record in ``<run_root>/requests`` — there are no capture-eligible decisions that cannot
  be replayed.  (The brief's "~19,019" does not reproduce; see "Contradictions" below.)
* **Replayed tokens** — **9,570,118** on the panel (``len(prompt_token_ids) +
  len(response.token_ids)``, summed over the store records themselves, not estimated from
  ``prompt_tokens + thinking_tokens + final_tokens``, which over a 300-record sample is off
  by a median of +1 and a range of -133..+175 per decision).  Panel sequences: mean 1,329,
  median 1,297, max 3,072 — every replay fits ``contract.MAX_MODEL_LEN`` (16,384), so no
  decision has to be dropped or truncated.  Over all 384 items the total is 25,154,545
  tokens (mean 1,310, median 1,275, max 3,136), i.e. the panel is 38.0% of the replay
  work for 37.5% of the items.
* **Stored bytes at the declared depth** — vectors are fp16 ``[hidden=4096]``
  (``study.neural.storage``), blocks ``(8, 17, 26)`` = ``blocks_for(36,
  contract.CAPTURE_RELATIVE_LAYERS)``, primary block 17:

  =========================  =========  =============  ==============
  layers × anchors           vectors    vector bytes   + ledger/keys
  =========================  =========  =============  ==============
  3 × 4 (all CAPTURE_ANCHORS)   86,400    707.8 MB       ~854 MB
  3 × 2 (PROMPT_END/PRE_FINAL)  43,200    353.9 MB       ~427 MB
  1 × 2 (block 17 only)         14,400    118.0 MB       ~142 MB
  =========================  =========  =============  ==============

  Vector bytes are exact (``n × 4096 × 2``).  The overhead is the ``.jsonl`` ledger at a
  **measured** 1,385 B per row for a row carrying the join keys and the entropy scalars,
  plus the ``.npz`` ``keys`` array at ~312 B per row (numpy UCS-4 over a 78-char key).
  For comparison, the unselected all-384 run at 3 × 4 is 230,400 vectors / 1.89 GB of
  vectors alone — **2.67× the panel**, which is exactly the overrun this module removes.

What this module does not do, and what must be true before E6 runs
------------------------------------------------------------------
It selects *items*.  ``cells.capture_panel_items`` picks the rule up through
:func:`select_capture_panel` (its ``_panel_selector`` hook), and ``build_e6_capture`` also
takes an explicit ``panel=`` list, so a manifest builder never has to re-derive it — but
this module emits no cells and writes no decision manifest of its own.
:func:`capture_decision_ids` returns exactly the ``decision_id`` rows such a manifest has
to carry, and is the cheapest way to check one.

E6 was, at audit time, unrunnable for reasons this panel does not fix: the cell options
carried neither ``cells`` nor ``manifest``, so ``manifest_for_item`` returned empty and the
worker wrote item status ``no_decisions`` **and exited 0** — a GPU cell that looks complete
having captured nothing — and ``options["snapshot_path"]`` was the one hard ``KeyError``,
reached only once a work list was non-empty.  Those are being repaired alongside this
module; a capture must not be launched until a dry run shows a non-empty work list for a
panel item and an options block carrying ``cells``/``manifest`` **and** ``snapshot_path``.

    PYTHONPATH=src python -m agents_scaling.ard.neural.panel --run-root <run_root>

Owned by WP6 (``docs/ard36/00_contract.md`` §1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import items as ARD_ITEMS

__all__ = [
    "CONTESTED_N_CORRECT",
    "DEFAULT_PANEL_SEED",
    "HIDDEN_SIZE_8B",
    "LEDGER_ROW_BYTES",
    "NPZ_KEY_BYTES",
    "NUM_LAYERS_8B",
    "PANEL_PURPOSE",
    "PANEL_SCHEMA",
    "PanelError",
    "STORED_DTYPE",
    "STORED_ITEMSIZE",
    "TABLES_DIRNAME",
    "capture_blocks",
    "capture_decision_ids",
    "contested_team_rounds",
    "estimated_bytes",
    "family_quota",
    "main",
    "panel_budget",
    "panel_manifest",
    "panel_sha256",
    "replayed_tokens",
    "select_capture_items",
    "select_capture_panel",
    "split_family_counts",
]

# --------------------------------------------------------------------------- constants

#: Schema tag of the manifest this module emits.  Bump it if the *meaning* of the selected
#: list changes; the sha256 alone cannot tell a reader that the rule changed under it.
PANEL_SCHEMA: Final[str] = "ard36-capture-panel-v1"

#: Domain separator inside the hash.  Without it, the same (namespace, seed, item_id)
#: triple drawn for another purpose would produce the same order.
PANEL_PURPOSE: Final[str] = "capture_panel"

#: The declared seed.  It is a *contract* number, not a tuning knob: changing it re-draws
#: the cohort, so a run that has captured anything must never move it.
DEFAULT_PANEL_SEED: Final[int] = 36

#: Tables directory this module reads.  ``tables/`` (v1) is frozen and known-defective —
#: duplicate primary keys on three of four joined tables (``ard.keys``) — so the contested
#: count and the decision list are read from the repaired ``tables_v2`` only, and a run
#: root without it is an error rather than a silent fall-back to v1.
TABLES_DIRNAME: Final[str] = "tables_v2"

#: Qwen3-8B ``config.json`` (pinned revision ``contract.REVISIONS["8B"]``): 36 layers,
#: hidden 4096.  Kept as a constant so the byte estimate needs no model snapshot on a
#: CPU-only node; :func:`estimated_bytes` accepts an override.
HIDDEN_SIZE_8B: Final[int] = 4096
NUM_LAYERS_8B: Final[int] = 36

#: ``study.neural.storage`` writes fp16 vectors, whatever the model's compute dtype.
STORED_DTYPE: Final[str] = "float16"
STORED_ITEMSIZE: Final[int] = 2

#: Measured on one representative ``ActivationRow`` carrying the seven ``JOIN_KEYS`` plus
#: the entropy scalars, serialised exactly as ``ShardWriter.flush`` serialises it.
LEDGER_ROW_BYTES: Final[int] = 1385
#: ``np.savez`` stores the ``keys`` array as UCS-4: 4 bytes per character of the longest
#: ``"<snapshot>|<block>|<anchor>"`` key in the chunk (~78 chars here).
NPZ_KEY_BYTES: Final[int] = 312

#: A team-round is contested when its five members are all graded and exactly one or two
#: of them are correct (``ard.minority``: ``minority_correct`` is ``n_correct <= 2``).
CONTESTED_N_CORRECT: Final[tuple[int, ...]] = (1, 2)


class PanelError(ValueError):
    """The panel cannot be drawn or verified from this run root."""


# --------------------------------------------------------------------------- selection


def _digest(seed: int, item_id: str) -> str:
    """The frozen per-item sort key.  Namespaced by study, contract version and purpose."""
    payload = f"{K.SEED_NAMESPACE}|{K.CONTRACT_VERSION}|{PANEL_PURPOSE}|{int(seed)}|{item_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def family_quota(total: int, families: Sequence[str] = tuple(f.value for f in K.FAMILIES)) -> dict[str, int]:
    """Split one split's quota evenly across families, remainder to the earliest families.

    ``CAPTURE_SPLIT_ITEMS`` is even in every split for the two ARD36 families, so the
    remainder branch never fires today; it exists so that a contract edit produces a
    stated, deterministic imbalance rather than an exception or a silent drop.
    """
    if int(total) < 0:
        raise PanelError(f"a split quota cannot be negative (got {total})")
    if not families:
        raise PanelError("family_quota needs at least one family")
    base, remainder = divmod(int(total), len(families))
    return {family: base + (1 if index < remainder else 0)
            for index, family in enumerate(families)}


def select_capture_items(
    run_root: str | os.PathLike,
    *,
    seed: int = DEFAULT_PANEL_SEED,
) -> tuple[str, ...]:
    """The frozen E6 capture panel: ``contract.CAPTURE_TOTAL_ITEMS`` item ids, sorted.

    Balanced over (split, family) by ``contract.CAPTURE_SPLIT_ITEMS`` and
    :func:`family_quota`, drawn by hash order inside each stratum.  The return is sorted by
    ``item_id`` — the *selected set* is the object, and no caller should infer a dispatch
    order from it (``capture.work_list`` imposes the train/val-before-test order the replay
    job actually runs in).

    Fails closed on a stratum that cannot fill its quota and on a total that disagrees with
    the contract, because every downstream number — decisions, tokens, bytes — is quoted
    against a stated cohort size.
    """
    return _select(ARD_ITEMS.load_ard_tasks(run_root).values(),
                   seed=seed, where=str(ARD_ITEMS.items_path(run_root)))


def select_capture_panel(
    tasks: Sequence[ARD_ITEMS.ArdTask],
    *,
    seed: int = DEFAULT_PANEL_SEED,
) -> list[ARD_ITEMS.ArdTask]:
    """The same panel, from an in-memory task list — the hook ``cells.capture_panel_items``
    looks for so that ``build_e6_capture`` picks the panel up without reopening the run root.

    One selection rule, two entry points: a manifest builder already holds ``ArdTask``
    objects and must not re-read (or be given) a run root to know which 144 it may emit.
    """
    by_id = {task.item_id: task for task in tasks}
    if len(by_id) != len(list(tasks)):
        raise PanelError("capture panel: the task list repeats an item_id")
    return [by_id[item_id] for item_id in _select(by_id.values(), seed=seed, where="the task list")]


def _select(
    tasks: Iterable[ARD_ITEMS.ArdTask],
    *,
    seed: int,
    where: str,
) -> tuple[str, ...]:
    """The selection rule itself, shared by both entry points."""
    tasks = list(tasks)
    chosen: list[str] = []
    for split, total in K.CAPTURE_SPLIT_ITEMS.items():
        if split not in K.SPLITS:
            raise PanelError(f"CAPTURE_SPLIT_ITEMS names split {split!r}, not in {K.SPLITS}")
        for family, want in family_quota(int(total)).items():
            stratum = sorted(
                (task for task in tasks
                 if task.split == split and task.family == family),
                key=lambda task: (_digest(seed, task.item_id), task.item_id),
            )
            if len(stratum) < want:
                raise PanelError(
                    f"capture panel: ({family}, {split}) has {len(stratum)} items, needs "
                    f"{want}; {where} is not the frozen natural panel"
                )
            chosen.extend(task.item_id for task in stratum[:want])
    if len(set(chosen)) != len(chosen):
        raise PanelError("capture panel: an item_id was drawn twice")  # pragma: no cover
    if len(chosen) != int(K.CAPTURE_TOTAL_ITEMS):
        raise PanelError(
            f"capture panel: drew {len(chosen)} items but CAPTURE_TOTAL_ITEMS is "
            f"{K.CAPTURE_TOTAL_ITEMS}; CAPTURE_SPLIT_ITEMS sums to "
            f"{sum(K.CAPTURE_SPLIT_ITEMS.values())}"
        )
    return tuple(sorted(chosen))


def split_family_counts(run_root: str | os.PathLike, items: Sequence[str]) -> dict[str, int]:
    """``{"train/math500": n, ...}`` for a selected list — the balance, stated not assumed."""
    tasks = ARD_ITEMS.load_ard_tasks(run_root)
    missing = [item_id for item_id in items if item_id not in tasks]
    if missing:
        raise PanelError(f"{len(missing)} panel items are not in the item table, e.g. {missing[:3]}")
    counter = Counter(f"{tasks[item_id].split}/{tasks[item_id].family}" for item_id in items)
    return dict(sorted(counter.items()))


# --------------------------------------------------------------------------- manifest


def panel_sha256(items: Iterable[str]) -> str:
    """sha256 of the newline-terminated, sorted item list.

    Hashing the *sorted* list makes the digest a property of the selected set, so a caller
    that reorders the panel cannot produce a different hash for the same cohort.
    """
    body = "".join(f"{item_id}\n" for item_id in sorted(items))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def panel_manifest(
    items: Sequence[str],
    *,
    seed: int = DEFAULT_PANEL_SEED,
    run_root: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """The auditable record of one panel: the list, its sha256, and the rule that made it.

    ``run_root`` is optional and adds only the (split, family) balance; the manifest is
    otherwise a pure function of the list, so it can be written next to shards on a node
    that has no item table mounted.
    """
    ordered = tuple(sorted(items))
    if len(set(ordered)) != len(ordered):
        raise PanelError("panel manifest: the item list contains duplicates")
    if not ordered:
        raise PanelError("panel manifest: the item list is empty")
    manifest: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "study_id": K.STUDY_ID,
        "contract_version": K.CONTRACT_VERSION,
        "seed_namespace": K.SEED_NAMESPACE,
        "purpose": PANEL_PURPOSE,
        "seed": int(seed),
        "n_items": len(ordered),
        "expected_n_items": int(K.CAPTURE_TOTAL_ITEMS),
        "split_quota": dict(K.CAPTURE_SPLIT_ITEMS),
        "protocols": [p.value for p in K.CAPTURE_PROTOCOLS],
        "rounds": list(K.ROUNDS),
        "checkpoint": K.CHECKPOINT_SMALL,
        "cohort": K.Cohort.NATURAL.value,
        "relative_layers": list(K.CAPTURE_RELATIVE_LAYERS),
        "blocks": list(capture_blocks()),
        "anchors": list(K.CAPTURE_ANCHORS),
        "prospective_anchors": sorted(K.PROSPECTIVE_ANCHORS),
        "items": list(ordered),
        "sha256": panel_sha256(ordered),
        "nonclaim": ("A family/split-balanced descriptive-geometry panel.  It is NOT powered "
                     "to separate a correct minority from a wrong majority; the contested "
                     "team-round count is reported for transparency, never optimised."),
    }
    if run_root is not None:
        manifest["split_family_counts"] = split_family_counts(run_root, ordered)
    return manifest


# --------------------------------------------------------------------------- budget


def capture_blocks(num_layers: int = NUM_LAYERS_8B,
                   fracs: Sequence[float] = K.CAPTURE_RELATIVE_LAYERS) -> tuple[int, ...]:
    """``blocks_for`` without importing the engine: ``floor(f*(L-1))``, unique, ascending.

    The engine's :func:`study.neural.engine.blocks_for` is the authority and the test
    asserts the two agree; this copy exists so that reading a budget never drags torch and
    transformers into a CPU-only process.
    """
    out = sorted({int(math.floor(float(f) * (int(num_layers) - 1))) for f in fracs})
    if any(block < 0 or block >= int(num_layers) for block in out):
        raise PanelError(f"relative layers {tuple(fracs)} leave the block range for L={num_layers}")
    return tuple(out)


def _tables_dir(run_root: str | os.PathLike, tables: str = TABLES_DIRNAME) -> Path:
    path = Path(run_root) / tables
    if not path.is_dir():
        raise PanelError(f"repaired tables directory missing: {path} (v1 tables/ is not a substitute)")
    return path


def _read_columns(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # local import: keeps this module importable without pandas

    if not path.exists():
        raise PanelError(f"table missing: {path}")
    available = set(pq.read_schema(path).names)
    missing = [name for name in columns if name not in available]
    if missing:
        raise PanelError(f"{path}: missing columns {missing}")
    return pq.read_table(path, columns=list(columns)).to_pylist()


def _is_capture_eligible(row: Mapping[str, Any]) -> bool:
    """8B, natural, a CAPTURE protocol, a contract round — the E6 universe, stated once."""
    return (str(row.get("checkpoint")) == K.CHECKPOINT_SMALL
            and str(row.get("cohort")) == K.Cohort.NATURAL.value
            and str(row.get("protocol")) in {p.value for p in K.CAPTURE_PROTOCOLS}
            and int(row.get("round") or 0) in K.ROUNDS)


def capture_decision_ids(
    run_root: str | os.PathLike,
    items: Sequence[str],
    *,
    tables: str = TABLES_DIRNAME,
) -> tuple[str, ...]:
    """Deduplicated ``decision_id``s the panel would replay, in a stable order.

    Deduplication is the point, not a tidy-up: a round-0 root is one generated decision
    that all three capture protocols alias (spec §14.2 "reuse root captures across
    protocols"), so ``decisions.parquet`` carries it three times and a naive count
    overstates the replay budget by 25% (23,040 rows → 19,200 decisions over 384 items).
    """
    wanted = set(items)
    rows = _read_columns(_tables_dir(run_root, tables) / "decisions.parquet",
                         ("decision_id", "item_id", "round", "protocol",
                          "agent_id", "checkpoint", "cohort"))
    seen: dict[str, tuple[str, int, int]] = {}
    for row in rows:
        if str(row.get("item_id")) not in wanted or not _is_capture_eligible(row):
            continue
        decision_id = str(row.get("decision_id") or "")
        if not decision_id or decision_id in seen:
            continue
        seen[decision_id] = (str(row.get("item_id")), int(row.get("round") or 0),
                             int(row.get("agent_id") or 0))
    return tuple(sorted(seen, key=lambda d: (seen[d], d)))


def replayed_tokens(
    run_root: str | os.PathLike,
    decision_ids: Sequence[str],
    *,
    workers: int = 16,
) -> dict[str, Any]:
    """Exact replayed sequence lengths, read from the committed request records.

    ``prompt_token_ids + response.token_ids`` is what ``replay.teacher_forced_sequence``
    forwards, so this is the real forward-pass cost.  It is deliberately not estimated from
    ``prompt_tokens + thinking_tokens + final_tokens (+2 delimiters)``: measured against
    300 records that estimate is off by -133..+175 tokens per decision, because the
    delimiter and tail accounting differs per parse status.
    """
    from concurrent.futures import ThreadPoolExecutor

    root = Path(run_root) / "requests"

    def measure(decision_id: str) -> tuple[int, int]:
        path = root / decision_id[:2] / f"{decision_id}.json"
        if not path.exists():
            return (-1, -1)
        record = json.loads(path.read_text(encoding="utf-8"))
        return (len(record["prompt_token_ids"]), len(record["response"]["token_ids"]))

    with ThreadPoolExecutor(max(1, int(workers))) as pool:
        measured = list(pool.map(measure, list(decision_ids)))

    absent = [d for d, (p, _c) in zip(decision_ids, measured) if p < 0]
    lengths = sorted(p + c for p, c in measured if p >= 0)
    if not lengths:
        raise PanelError("no request record backs any panel decision; the store is wrong or empty")
    return {
        "n_decisions": len(decision_ids),
        "n_records_missing": len(absent),
        "missing_examples": absent[:5],
        "prompt_tokens": sum(p for p, _c in measured if p >= 0),
        "completion_tokens": sum(c for _p, c in measured if c >= 0),
        "total_tokens": sum(lengths),
        "mean_sequence_tokens": sum(lengths) / len(lengths),
        "median_sequence_tokens": lengths[len(lengths) // 2],
        "max_sequence_tokens": lengths[-1],
        "over_max_model_len": sum(1 for n in lengths if n > K.MAX_MODEL_LEN),
    }


def estimated_bytes(
    n_decisions: int,
    *,
    n_layers: int = len(K.CAPTURE_RELATIVE_LAYERS),
    n_anchors: int = len(K.CAPTURE_ANCHORS),
    hidden_size: int = HIDDEN_SIZE_8B,
    itemsize: int = STORED_ITEMSIZE,
) -> dict[str, Any]:
    """Stored bytes for a capture of ``n_decisions`` at ``n_layers × n_anchors`` sites.

    The vector term is exact.  The two overhead terms are measured per-row constants
    (:data:`LEDGER_ROW_BYTES`, :data:`NPZ_KEY_BYTES`) and are labelled as estimates because
    a missing anchor writes a ledger row with *no vector* — that trades 8 KB of vector for
    a slightly shorter row, so the real total is bounded above by this number.
    """
    if int(n_decisions) < 0 or int(n_layers) < 1 or int(n_anchors) < 1:
        raise PanelError("byte estimate needs a non-negative decision count and >=1 layer/anchor")
    vectors = int(n_decisions) * int(n_layers) * int(n_anchors)
    vector_bytes = vectors * int(hidden_size) * int(itemsize)
    overhead = vectors * (LEDGER_ROW_BYTES + NPZ_KEY_BYTES)
    return {
        "n_decisions": int(n_decisions),
        "n_layers": int(n_layers),
        "n_anchors": int(n_anchors),
        "hidden_size": int(hidden_size),
        "dtype": STORED_DTYPE,
        "n_vectors": vectors,
        "vector_bytes": vector_bytes,
        "vector_mb": vector_bytes / 1e6,
        "ledger_bytes_estimate": overhead,
        "total_bytes_estimate": vector_bytes + overhead,
        "total_mb_estimate": (vector_bytes + overhead) / 1e6,
    }


def contested_team_rounds(
    run_root: str | os.PathLike,
    items: Sequence[str],
    *,
    tables: str = TABLES_DIRNAME,
) -> dict[str, Any]:
    """How many contested team-rounds the panel *happens* to contain.  Reported, not sought.

    Contested = a fully graded five-agent team with ``n_correct`` in
    :data:`CONTESTED_N_CORRECT`.  ``minority_frame.parquet`` has no ``team_fully_graded``
    column; the equivalent is ``n_graded == contract.N_AGENTS``, which is exactly
    ``compact.py``'s ``all(v is not None for v in round_correct)``.

    Two counts are returned because they answer different questions.  ``n_rows`` counts
    (protocol, round) cells and is what a per-protocol table would show.  ``n_distinct``
    collapses round 0, where all three protocols share one root team, and is the honest
    count of independent contested teams — the number any power statement must use.
    """
    wanted = set(items)
    rows = _read_columns(_tables_dir(run_root, tables) / "minority_frame.parquet",
                         ("item_id", "episode_id", "cell_id", "round", "protocol",
                          "checkpoint", "cohort", "n_correct", "n_graded"))
    teams: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("item_id")) not in wanted or not _is_capture_eligible(row):
            continue
        key = (str(row.get("cell_id")), str(row.get("episode_id")), int(row.get("round") or 0))
        teams.setdefault(key, row)

    contested = [row for key, row in teams.items()
                 if row.get("n_graded") is not None
                 and int(row["n_graded"]) == K.N_AGENTS
                 and row.get("n_correct") is not None
                 and int(row["n_correct"]) in CONTESTED_N_CORRECT]
    distinct: set[tuple[str, int, str]] = set()
    for row in contested:
        round_ = int(row.get("round") or 0)
        # Round 0 is one root team aliased into all three protocols: count it once.
        distinct.add((str(row["item_id"]), round_,
                      "" if round_ == 0 else str(row.get("protocol"))))
    by_round = Counter(int(row.get("round") or 0) for row in contested)
    by_protocol = Counter(str(row.get("protocol")) for row in contested)
    return {
        "definition": f"n_graded == {K.N_AGENTS} and n_correct in {CONTESTED_N_CORRECT}",
        "n_team_rounds": len(teams),
        "n_rows": len(contested),
        "n_distinct": len(distinct),
        "n_items": len({str(row["item_id"]) for row in contested}),
        "by_round": {str(k): v for k, v in sorted(by_round.items())},
        "by_protocol": dict(sorted(by_protocol.items())),
        "nonclaim": ("Descriptive only.  A held-out minority-vs-bloc probe on this many "
                     "positives has MDE80(AUC) ~ 0.73; neither a hit nor a null would be "
                     "informative, and no such analysis is promised by this panel."),
    }


def panel_budget(
    run_root: str | os.PathLike,
    items: Sequence[str] | None = None,
    *,
    seed: int = DEFAULT_PANEL_SEED,
    tables: str = TABLES_DIRNAME,
    tokens: bool = True,
) -> dict[str, Any]:
    """Everything the docstring quotes, recomputed from this run root.

    ``tokens=False`` skips the request-store pass (one small JSON read per decision, ~17 s
    for 19,200 records over the shared filesystem) for callers that only need item counts.
    """
    selected = tuple(items) if items is not None else select_capture_items(run_root, seed=seed)
    decisions = capture_decision_ids(run_root, selected, tables=tables)
    blocks = capture_blocks()
    payload: dict[str, Any] = {
        "manifest": panel_manifest(selected, seed=seed, run_root=run_root),
        "n_items": len(selected),
        "n_capture_eligible_decisions": len(decisions),
        "decisions_per_item": len(decisions) / len(selected) if selected else 0.0,
        "contested": contested_team_rounds(run_root, selected, tables=tables),
        "bytes": {
            f"{len(blocks)}x{len(K.CAPTURE_ANCHORS)}": estimated_bytes(
                len(decisions), n_layers=len(blocks), n_anchors=len(K.CAPTURE_ANCHORS)),
            f"{len(blocks)}x{len(K.PROSPECTIVE_ANCHORS)}": estimated_bytes(
                len(decisions), n_layers=len(blocks), n_anchors=len(K.PROSPECTIVE_ANCHORS)),
            f"1x{len(K.PROSPECTIVE_ANCHORS)}": estimated_bytes(
                len(decisions), n_layers=1, n_anchors=len(K.PROSPECTIVE_ANCHORS)),
        },
    }
    if tokens:
        payload["tokens"] = replayed_tokens(run_root, decisions)
    return payload


# --------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agents_scaling.ard.neural.panel",
        description="Draw and hash the frozen ARD36 E6 capture panel (descriptive geometry).")
    ap.add_argument("--run-root", required=True, type=Path,
                    help="an ARD36 run root (read only; this module never writes into it)")
    ap.add_argument("--seed", type=int, default=DEFAULT_PANEL_SEED,
                    help=f"panel seed (default {DEFAULT_PANEL_SEED}; moving it re-draws the cohort)")
    ap.add_argument("--tables", default=TABLES_DIRNAME,
                    help=f"tables directory under the run root (default {TABLES_DIRNAME}; "
                         "the frozen v1 tables/ must not be used)")
    ap.add_argument("--no-budget", dest="budget", action="store_false",
                    help="print the panel and its hash only, without touching the tables")
    ap.add_argument("--no-tokens", dest="tokens", action="store_false",
                    help="skip the exact token measurement over the request store")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the JSON here (the manifest file for a capture run)")
    args = ap.parse_args(argv)

    selected = select_capture_items(args.run_root, seed=args.seed)
    if args.budget:
        payload: dict[str, Any] = panel_budget(args.run_root, selected, seed=args.seed,
                                               tables=args.tables, tokens=args.tokens)
    else:
        payload = {"manifest": panel_manifest(selected, seed=args.seed, run_root=args.run_root)}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
