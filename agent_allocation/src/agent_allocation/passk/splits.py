"""passk_cont_v1 frozen splits (lead-owned): duplicate registry, T200 / X80 / C48 panels, T200 provider assignment,
the legacy-format root manifests, and Gate S.

Everything here is outcome-blind except C48, whose strata are the legacy shallow bank counts (correct out of 25
forced-A roots), exactly as the approved plan specifies. Inputs are the pinned benchmark inventories (offline HF
snapshots, checked against the legacy dataset lock) and the forced_budget_v2 run root, which is only ever read.

Frozen rules (changing any of them needs a new ``SPLITS_VERSION``):

* **Duplicate registry.** Over every eligible item of the four benchmarks: exact duplicates by sha256 of the
  normalized prompt (NFKC, lower case, whitespace runs collapsed), and near duplicates flagged when the char 5-gram
  or the word-trigram Jaccard of the normalized prompts is >= 0.3. The reviewed Omni-MATH pairs
  (:data:`REVIEWED_PAIRS`) become duplicate groups (union-find with the exact duplicates); a reviewed pair must be
  flagged by the scan. Per group: members of legacy main or dev are *used*; if a group holds a used item, every
  other member is excluded from the pool (``partner_of_used``); else, if the members' golds disagree, all members
  are excluded (``conflicting_gold``); else the member with the lowest source row index is the representative and
  the others are excluded (``duplicate_of``). Gold texts are compared (whitespace-free) but never written out.
* **T200.** Pool = eligible - (legacy main U dev) - registry exclusions (legacy reserve items are eligible). Per
  benchmark, :func:`text_manifest._stratified_panels` with the legacy stratum rule, largest-remainder
  apportionment and the explicit salt ``openrouter_passk_cont_v1|<benchmark>|seed=20260925|t200-v1``: T200 is the
  ``main`` draw (quota 20/20/100/60, in the seeded interleave order = the frozen rank order), and the rest of the
  pool, drawn as ``reserve`` in the same call, is ``reserve_pool.json``.
* **X80.** The same stratified draw over T200 with salt ``...|x80-v1`` and quota 8/8/40/24.
* **T200 assignment (1:1).** Per benchmark, T200 items ordered by ``rank_key(salt, "rank", item)`` with salt
  ``...|t200-shard`` alternate mancer, parasail, mancer, ...; with an odd count the last item goes to parasail.
* **C48.** Bank-known L96 items (legacy ``sample_bank_items`` k=1 ``bank_correct`` not null; cross-checked against
  ``candidates.parquet`` main/root/1024). Per benchmark 12 items over the bins 0 / 1-5 / 6-15 / 16-25: each cell
  first takes min(size, 3); the remaining quota goes to cells with spare = size - 3 > 0 by largest remainder of
  spare-proportional shares (ties by ``sha256(C48 prefix + "cell=" + bin)``); within a cell items are ranked by
  ``sha256("openrouter_passk_cont_v1|C48|<benchmark>|seed=20260925|" + item)``. Inclusion probability =
  selected / size of the item's cell. C48 (= V48) is listed in legacy main order.

Run: ``python -m agent_allocation.passk.splits --config configs/passk_cont_v1.json`` (writes under the passk run
root; every file is write-once: an identical rerun is a no-op, a different one refuses).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent_allocation.benchmarks import text_manifest as tm
from agent_allocation.contracts import BENCHMARKS, InventoryItem, PanelDraw, canonical_hash, dump_record, load_record

STUDY = "openrouter_passk_cont_v1"
SPLITS_VERSION = "passk-splits-v1"
REGISTRY_VERSION = "passk-dup-registry-v1"
PROCEDURE = "passk_cont_v1"
PASSK_SEED = 20260925
LEGACY_SEED = 20260917
LEGACY_RUN_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts")
PASSK_RUN_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1")
PREREG_COMMIT = "e781c34"
PREREG_PATH = "agent_allocation/docs/passk_cont/PREREG.md"
ADAPTERS = {
    "aime": "agent_allocation.benchmarks.aime",
    "hmmt": "agent_allocation.benchmarks.hmmt",
    "omni_math_hard": "agent_allocation.benchmarks.omni_math_hard",
    "gpqa_diamond": "agent_allocation.benchmarks.gpqa_diamond",
}

T200_QUOTA = {"aime": 20, "hmmt": 20, "omni_math_hard": 100, "gpqa_diamond": 60}
X80_QUOTA = {"aime": 8, "hmmt": 8, "omni_math_hard": 40, "gpqa_diamond": 24}
C48_PER_BENCHMARK = 12
C48_TARGET_PER_CELL = 3
C48_BINS: tuple[tuple[str, int, int], ...] = (("0", 0, 0), ("1-5", 1, 5), ("6-15", 6, 15), ("16-25", 16, 25))
BANK_SIZE = 25
JACCARD_THRESHOLD = 0.3
CHAR_N = 5
WORD_N = 3
SHARDS = ("mancer", "parasail")

_O = "omni_math_hard/"
#: Omni-MATH pairs found by the scan and reviewed by text + gold (plan: 13 pairs, both members eligible).
REVIEWED_PAIRS: tuple[tuple[str, str], ...] = tuple((_O + a, _O + b) for a, b in (
    ("1476", "3806"), ("1464", "3617"), ("258", "3612"), ("336", "3611"), ("1617", "3563"), ("308", "3566"),
    ("162", "3962"), ("391", "3597"), ("316", "3670"), ("244", "3575"), ("3295", "4379"), ("2408", "4070"),
    ("328", "3724")))
#: Frozen review outcomes the registry must reproduce (asserted in Gate S).
EXPECTED_PARTNER_EXCLUSIONS = {_O + "3611": _O + "336", _O + "162": _O + "3962"}
EXPECTED_CONFLICTING = frozenset({_O + "308", _O + "3566"})
UNKNOWN_BANK_ITEMS = frozenset({"hmmt/2025/18", "omni_math_hard/473"})


class SplitsRefused(RuntimeError):
    """A frozen split artifact would change, or an input does not have its frozen shape."""


# ---------------------------------------------------------------------------------------------------------------
# Salts and keys
# ---------------------------------------------------------------------------------------------------------------

def salt(benchmark: str, tag: str) -> str:
    return f"{STUDY}|{benchmark}|seed={PASSK_SEED}|{tag}"


def c48_prefix(benchmark: str) -> str:
    return f"{STUDY}|C48|{benchmark}|seed={PASSK_SEED}|"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def salts_record() -> dict[str, Any]:
    return {
        "t200": {b: salt(b, "t200-v1") for b in BENCHMARKS},
        "x80": {b: salt(b, "x80-v1") for b in BENCHMARKS},
        "t200_shard": {b: salt(b, "t200-shard") for b in BENCHMARKS},
        "c48": {b: c48_prefix(b) for b in BENCHMARKS},
        "formulas": {
            "t200/x80 rank": "text_manifest.rank_key(salt, 'rank', item_id) = sha256(salt \\x1f 'rank' \\x1f item_id)"
                             " within stratum; text_manifest._stratified_panels (main=quota, dev=0, reserve=rest)",
            "t200_shard rank": "text_manifest.rank_key(salt, 'rank', item_id); alternate mancer/parasail from the "
                               "lowest key; odd count: last item -> parasail",
            "c48 item rank": "sha256(c48_prefix + item_id), ascending",
            "c48 cell tie": "sha256(c48_prefix + 'cell=' + bin), ascending",
        },
    }


# ---------------------------------------------------------------------------------------------------------------
# Write-once files
# ---------------------------------------------------------------------------------------------------------------

def write_once(path: Path, data: bytes, *, mode: int | None = None) -> str:
    """Create ``path`` with ``data`` exactly once. Identical existing bytes are a no-op; different bytes refuse.

    The file appears atomically through a hard link (``os.link`` fails if the target exists), so two concurrent
    writers can never both succeed with different content."""
    path = Path(path)
    digest = sha256_bytes(data)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise SplitsRefused(f"{path} exists with different content (sha256 {sha256_bytes(existing)} != "
                                f"{digest}); frozen split artifacts are write-once")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if mode is not None:
        os.chmod(tmp, mode)
    try:
        os.link(tmp, path)
    except FileExistsError:
        tmp.unlink()
        return write_once(path, data, mode=mode)
    finally:
        if tmp.exists():
            tmp.unlink()
    return digest


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------

def load_inventories(cfg: Mapping[str, Any], lock: Mapping[str, Any] | None = None) -> dict[str, list[InventoryItem]]:
    """The four pinned inventories via the existing adapters; with ``lock`` (the legacy ``locks/datasets.json``),
    each inventory hash must equal the locked one."""
    out: dict[str, list[InventoryItem]] = {}
    for benchmark in BENCHMARKS:
        inventory = list(importlib.import_module(ADAPTERS[benchmark]).load_inventory(cfg, None))
        if lock is not None:
            observed = inventory_hash(inventory)
            expected = lock["sources"][benchmark]["inventory_hash"]
            if observed != expected:
                raise SplitsRefused(f"{benchmark}: inventory hash {observed} != legacy lock {expected}")
        out[benchmark] = inventory
    return out


def inventory_hash(inventory: Sequence[InventoryItem]) -> str:
    return canonical_hash([[i.public.item_id, i.private.private_hash, i.eligible] for i in inventory])


class Legacy:
    """Read-only view of the forced_budget_v2 run root."""

    def __init__(self, root: Path = LEGACY_RUN_ROOT):
        self.root = Path(root)
        self.panels_path = self.root / "manifests" / "panels.json"
        self.exclusions_path = self.root / "manifests" / "exclusions.json"
        self.assignment_path = self.root / "shards" / "assignment.json"
        self.lock_path = self.root / "locks" / "datasets.json"
        self.bank_path = self.root / "merged" / "tables" / "sample_bank_items.parquet"
        self.candidates_path = self.root / "merged" / "tables" / "candidates.parquet"

    def public_path(self, benchmark: str) -> Path:
        return self.root / "manifests" / "public" / f"{benchmark}.jsonl"

    def private_path(self, benchmark: str) -> Path:
        return self.root / "private" / f"{benchmark}.jsonl"

    def panels(self) -> dict[str, PanelDraw]:
        return {b: load_record(PanelDraw, row) for b, row in json.loads(self.panels_path.read_text()).items()}

    def lock(self) -> dict[str, Any]:
        return json.loads(self.lock_path.read_text())

    def assignment(self) -> dict[str, str]:
        return dict(json.loads(self.assignment_path.read_text())["assignment"])

    def lines(self, path: Path) -> dict[str, bytes]:
        """``{item_id: raw line bytes incl. newline}`` in file order."""
        out: dict[str, bytes] = {}
        for line in path.read_bytes().splitlines(keepends=True):
            if line.strip():
                out[json.loads(line)["item_id"]] = line
        return out

    def input_hashes(self) -> dict[str, str]:
        files = [self.panels_path, self.exclusions_path, self.assignment_path, self.lock_path, self.bank_path,
                 self.candidates_path]
        files += [self.public_path(b) for b in BENCHMARKS] + [self.private_path(b) for b in BENCHMARKS]
        return {str(p.relative_to(self.root)): sha256_file(p) for p in files}


def legacy_bank_correct(legacy: Legacy) -> dict[str, int | None]:
    """``{L96 item: correct out of 25 forced-A roots, or None when any root grade is unknown}``, from
    ``sample_bank_items.parquet`` and cross-checked against ``candidates.parquet`` (main / root / 1024)."""
    import pandas as pd

    bank = pd.read_parquet(legacy.bank_path, columns=["partition_id", "item_id", "k", "bank_size", "bank_correct"])
    bank = bank[(bank.partition_id == "main") & (bank.k == 1)]
    if len(bank) != 96 or set(bank.bank_size) != {BANK_SIZE}:
        raise SplitsRefused("sample_bank_items: expected 96 main items with bank_size 25")
    out: dict[str, int | None] = {}
    for item_id, value in zip(bank.item_id, bank.bank_correct):
        out[str(item_id)] = None if pd.isna(value) else int(value)
    cand = pd.read_parquet(legacy.candidates_path,
                           columns=["partition_id", "item_id", "kind", "reasoning_budget", "root_slot", "node_id",
                                    "grade_state", "correct"])
    cand = cand[(cand.partition_id == "main") & (cand.kind == "root") & (cand.reasoning_budget == 1024)]
    for item_id, rows in cand.groupby("item_id"):
        if rows.node_id.nunique() != BANK_SIZE or sorted(rows.root_slot) != list(range(BANK_SIZE)):
            raise SplitsRefused(f"{item_id}: legacy forced-A bank is not slots 0..24")
        unknown = int((rows.grade_state == "unknown_evaluator_error").sum())
        count = None if unknown else int((rows.correct == 1).sum())
        if out.get(str(item_id), "missing") != count:
            raise SplitsRefused(f"{item_id}: bank_correct {out.get(str(item_id))} != candidates count {count}")
    if set(cand.item_id) != set(out):
        raise SplitsRefused("candidates and sample_bank_items disagree on the L96 item set")
    return out


# ---------------------------------------------------------------------------------------------------------------
# Duplicate registry
# ---------------------------------------------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def char_shingles(text: str, n: int = CHAR_N) -> frozenset[str]:
    return frozenset(text[i:i + n] for i in range(max(1, len(text) - n + 1)))


def word_shingles(text: str, n: int = WORD_N) -> frozenset[tuple[str, ...]]:
    words = re.findall(r"\w+", text)
    return frozenset(tuple(words[i:i + n]) for i in range(max(1, len(words) - n + 1)))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def gold_key(gold: str | None) -> str:
    return re.sub(r"\s+", "", gold or "").strip("$")


def gold_relation(gold_a: str | None, gold_b: str | None, *, benchmark: str, timeout_s: float = 10.0) -> str:
    """``equal`` (whitespace-free text equal, or Math-Verify equal both ways), ``one_way`` (Math-Verify equal in
    exactly one direction, e.g. ``\\{n \\ge 13\\}`` vs ``n \\geq 13``), ``conflict`` (not equal either way) or
    ``unknown`` (a timeout / unparseable gold). Only ``conflict`` makes a group's golds conflicting. Option
    benchmarks compare letters. Must run on the main thread (Math-Verify's signal time bound)."""
    if gold_key(gold_a) == gold_key(gold_b):
        return "equal"
    if benchmark == "gpqa_diamond" or gold_a is None or gold_b is None:
        return "conflict"
    from agent_allocation.grading.text import math_equiv

    verdicts = [math_equiv.verify_gold(gold_a, gold_b, timeout_s=timeout_s).correct,
                math_equiv.verify_gold(gold_b, gold_a, timeout_s=timeout_s).correct]
    if all(v is True for v in verdicts):
        return "equal"
    if any(v is True for v in verdicts):
        return "one_way"
    if all(v is False for v in verdicts):
        return "conflict"
    return "unknown"


def _row_order(inventories: Mapping[str, Sequence[InventoryItem]]) -> dict[str, tuple[int, int]]:
    """Source row order: (benchmark position, position in the adapter's source-ordered inventory)."""
    return {item.public.item_id: (BENCHMARKS.index(b), pos)
            for b, inventory in inventories.items() for pos, item in enumerate(inventory)}


def near_duplicate_scan(items: Sequence[InventoryItem], *, threshold: float = JACCARD_THRESHOLD
                        ) -> list[dict[str, Any]]:
    """Every pair of ``items`` that is an exact normalized duplicate or has char-5 / word-3 Jaccard >= threshold."""
    norm = {i.public.item_id: normalize_text(i.public.prompt) for i in items}
    chars = {k: char_shingles(v) for k, v in norm.items()}
    words = {k: word_shingles(v) for k, v in norm.items()}
    hashes = {k: sha256_text(v) for k, v in norm.items()}
    ids = sorted(norm)
    flagged = []
    for x in range(len(ids)):
        a = ids[x]
        for y in range(x + 1, len(ids)):
            b = ids[y]
            exact = hashes[a] == hashes[b]
            # Exact bound: J(A, B) <= min(|A|, |B|) / max(|A|, |B|); skip pairs that cannot reach the threshold.
            ca, cb, wa, wb = len(chars[a]), len(chars[b]), len(words[a]), len(words[b])
            if not exact and min(ca, cb) < threshold * max(ca, cb) and min(wa, wb) < threshold * max(wa, wb):
                continue
            jc = jaccard(chars[a], chars[b])
            jw = jaccard(words[a], words[b])
            if exact or jc >= threshold or jw >= threshold:
                flagged.append({"a": a, "b": b, "char5_jaccard": round(jc, 4), "word3_jaccard": round(jw, 4),
                                "exact_normalized": exact})
    return sorted(flagged, key=lambda f: (f["a"], f["b"]))


def _canon_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pair_relation(by_id: Mapping[str, InventoryItem], a: str, b: str) -> str:
    bench_a, bench_b = by_id[a].public.benchmark, by_id[b].public.benchmark
    if bench_a != bench_b:
        return "conflict" if gold_key(by_id[a].private.gold_answer) != gold_key(by_id[b].private.gold_answer) \
            else "equal"
    return gold_relation(by_id[a].private.gold_answer, by_id[b].private.gold_answer, benchmark=bench_a)


def build_registry(inventories: Mapping[str, Sequence[InventoryItem]], used: Iterable[str], *,
                   reviewed_pairs: Sequence[tuple[str, str]] = REVIEWED_PAIRS,
                   threshold: float = JACCARD_THRESHOLD) -> dict[str, Any]:
    """The duplicate registry over every eligible item (rules in the module docstring).

    Returns ``{"flagged": [...], "groups": [...], "excluded": {item: reason}}``. Reviewed pairs that the scan does
    not flag, or that name an ineligible item, refuse. Flagged pairs that were not reviewed are listed with status
    ``flagged_not_duplicate`` and do not form groups (exact normalized duplicates always do)."""
    used = set(used)
    eligible = [i for b in BENCHMARKS for i in inventories.get(b, ()) if i.eligible]
    by_id = {i.public.item_id: i for i in eligible}
    order = _row_order(inventories)
    flagged = near_duplicate_scan(eligible, threshold=threshold)
    flagged_keys = {(f["a"], f["b"]): f for f in flagged}
    reviewed = {_canon_pair(a, b) for a, b in reviewed_pairs}
    for a, b in sorted(reviewed):
        if a not in by_id or b not in by_id:
            raise SplitsRefused(f"reviewed pair {a}/{b} names an item that is not eligible")
        if (a, b) not in flagged_keys:
            raise SplitsRefused(f"reviewed pair {a}/{b} is not flagged by the scan (threshold {threshold})")
    edges = []
    relation: dict[tuple[str, str], str] = {}
    for f in flagged:
        key = (f["a"], f["b"])
        f["gold_text_equal"] = (gold_key(by_id[f["a"]].private.gold_answer)
                                == gold_key(by_id[f["b"]].private.gold_answer))
        if f["exact_normalized"] or key in reviewed:
            f["status"] = "reviewed_duplicate" if key in reviewed else "exact_duplicate"
            f["gold_relation"] = relation[key] = _pair_relation(by_id, *key)
            edges.append(key)
        else:
            f["status"] = "flagged_not_duplicate"
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=order.get)] = min(ra, rb, key=order.get)
    members_of: dict[str, list[str]] = {}
    for x in list(parent):
        members_of.setdefault(find(x), []).append(x)
    groups = []
    excluded: dict[str, str] = {}
    for members in sorted(members_of.values(), key=lambda m: min(order[x] for x in m)):
        members = sorted(members, key=order.get)
        in_use = [m for m in members if m in used]
        pair_relations = {}
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                key = _canon_pair(members[x], members[y])
                pair_relations[key] = relation.get(key) or _pair_relation(by_id, *key)
        if "unknown" in pair_relations.values():
            raise SplitsRefused(f"duplicate group {members}: gold relation unknown (Math-Verify timeout)")
        consistent = "conflict" not in pair_relations.values()
        if in_use:
            decision, representative = "partner_of_used", None
            reasons = {m: f"partner_of_used:{in_use[0]}" for m in members if m not in used}
        elif not consistent:
            decision, representative = "conflicting_gold", None
            reasons = {m: "conflicting_gold" for m in members}
        else:
            decision, representative = "keep_lowest_row", members[0]
            reasons = {m: f"duplicate_of:{members[0]}" for m in members[1:]}
        excluded.update(reasons)
        groups.append({"group_id": f"dup{len(groups):03d}", "members": members, "used": in_use,
                       "gold_relations": {f"{a}|{b}": r for (a, b), r in sorted(pair_relations.items())},
                       "gold_consistent": consistent, "decision": decision, "representative": representative,
                       "excluded": dict(sorted(reasons.items()))})
    return {"flagged": flagged, "groups": groups, "excluded": dict(sorted(excluded.items()))}


def registry_payload(registry: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for f in registry["flagged"]:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    return {
        "version": REGISTRY_VERSION, "study_id": STUDY,
        "method": {"population": "eligible items of aime, hmmt, omni_math_hard, gpqa_diamond (public prompt)",
                   "normalization": "NFKC, lower case, whitespace runs -> one space, strip",
                   "exact": "sha256 of the normalized prompt",
                   "near": f"char {CHAR_N}-gram or word {WORD_N}-gram (\\w+ tokens) Jaccard >= {JACCARD_THRESHOLD}",
                   "gold_compare": "gold_relation: whitespace-free text equality, else Math-Verify both directions "
                                   "(equal / one_way / conflict / unknown); only conflict counts as conflicting "
                                   "(gold texts are never written here)",
                   "flagged_not_duplicate_review": "the 43 flagged pairs outside the 13 reviewed ones were read on "
                                                   "2026-09-25: shared templates or preambles (functional "
                                                   "equations, GPQA shared stems) with different problems; not "
                                                   "duplicates",
                   "group_rule": "union-find over exact duplicates and reviewed pairs; used (legacy main U dev) "
                                 "member -> exclude the other members; else conflicting golds -> exclude all; "
                                 "else keep the lowest source row"},
        "reviewed_pairs": [list(_canon_pair(a, b)) for a, b in REVIEWED_PAIRS],
        "flagged_counts": dict(sorted(counts.items())),
        "flagged_pairs": registry["flagged"],
        "groups": registry["groups"],
        "excluded_from_pool": registry["excluded"],
    }


# ---------------------------------------------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------------------------------------------

def stratified(items: Sequence[InventoryItem], quota: int, salt_text: str) -> tuple[list[str], list[str]]:
    """``(chosen, rest)``: the legacy stratified draw of ``quota`` items (``main``) and every other item
    (``reserve``), each in seeded interleave order."""
    panels = tm._stratified_panels(items, salt_text, {"main": quota, "dev": 0, "reserve": len(items) - quota})
    if panels["dev"]:
        raise AssertionError("dev must be empty")
    return panels["main"], panels["reserve"]


def strata_counts(items: Sequence[InventoryItem], ids: Iterable[str]) -> dict[str, int]:
    by_id = {i.public.item_id: i for i in items}
    counts: dict[str, int] = {}
    for item_id in ids:
        name = tm._stratum_name(by_id[item_id])
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def t200_assignment(t200: Mapping[str, Sequence[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for benchmark in BENCHMARKS:
        ids = sorted(t200.get(benchmark, ()), key=lambda i: tm.rank_key(salt(benchmark, "t200-shard"), "rank", i))
        paired = 2 * (len(ids) // 2)
        for pos, item_id in enumerate(ids):
            out[item_id] = SHARDS[pos % 2] if pos < paired else "parasail"
    return dict(sorted(out.items()))


def c48_bin(correct: int) -> str:
    for name, lo, hi in C48_BINS:
        if lo <= correct <= hi:
            return name
    raise ValueError(f"bank count {correct} outside 0..{BANK_SIZE}")


def select_c48(bank: Mapping[str, int | None], order: Sequence[str], *, per_benchmark: int = C48_PER_BENCHMARK,
               target: int = C48_TARGET_PER_CELL) -> dict[str, Any]:
    """C48 by the frozen rule. ``bank`` maps L96 items to correct-out-of-25 (None = unknown, never eligible);
    ``order`` is the legacy L96 order the output list follows."""
    strata: dict[str, dict[str, dict[str, int]]] = {}
    inclusion: dict[str, float] = {}
    cells_of: dict[str, str] = {}
    chosen: set[str] = set()
    for benchmark in BENCHMARKS:
        known = [i for i in order if i.split("/", 1)[0] == benchmark and bank.get(i) is not None]
        if not known:
            continue
        if len(known) < per_benchmark:
            raise SplitsRefused(f"C48 {benchmark}: {len(known)} bank-known items < {per_benchmark}")
        prefix = c48_prefix(benchmark)
        cells = {name: sorted((i for i in known if c48_bin(int(bank[i])) == name),
                              key=lambda i: sha256_text(prefix + i)) for name, _, _ in C48_BINS}
        base = {name: min(len(ids), target) for name, ids in cells.items()}
        spare = {name: len(ids) - target for name, ids in cells.items() if len(ids) - target > 0}
        remaining = per_benchmark - sum(base.values())
        if remaining < 0:
            raise SplitsRefused(f"C48 {benchmark}: base allocation exceeds {per_benchmark}")
        extra = tm.largest_remainder(spare, remaining, lambda name: sha256_text(prefix + "cell=" + name)) \
            if spare else {}
        table = {}
        for name, ids in cells.items():
            k = base[name] + extra.get(name, 0)
            table[name] = {"size": len(ids), "selected": k}
            for item_id in ids:
                cells_of[item_id] = name
            for item_id in ids[:k]:
                chosen.add(item_id)
                inclusion[item_id] = k / len(ids)
        strata[benchmark] = table
    items = [i for i in order if i in chosen]
    return {"items": items, "strata": strata, "inclusion_prob": {i: inclusion[i] for i in items},
            "cells": dict(sorted(cells_of.items())),
            "bank_correct": {i: bank[i] for i in order if bank.get(i) is not None}}


def c48_table_rows(strata: Mapping[str, Mapping[str, Mapping[str, int]]]) -> list[str]:
    header = "benchmark        " + "".join(f"{name:>10}" for name, _, _ in C48_BINS) + "     total"
    rows = [header]
    for benchmark, cells in strata.items():
        text = "".join(f"{cells[n]['selected']:>5}/{cells[n]['size']:<4}" for n, _, _ in C48_BINS)
        total = sum(c["selected"] for c in cells.values())
        size = sum(c["size"] for c in cells.values())
        rows.append(f"{benchmark:<17}{text}{total:>5}/{size:<4}")
    return rows


# ---------------------------------------------------------------------------------------------------------------
# The whole build
# ---------------------------------------------------------------------------------------------------------------

def build_splits(cfg: Mapping[str, Any], legacy: Legacy) -> dict[str, Any]:
    """Every frozen split artifact's content (nothing is written). Deterministic in its inputs."""
    lock = legacy.lock()
    inventories = load_inventories(cfg, lock)
    panels = legacy.panels()
    if set(panels) != set(BENCHMARKS):
        raise SplitsRefused("legacy panels.json must hold the four text benchmarks")
    main = {b: list(panels[b].main_item_ids) for b in BENCHMARKS}
    dev = {b: list(panels[b].dev_item_ids) for b in BENCHMARKS}
    used = {i for b in BENCHMARKS for i in (*main[b], *dev[b])}
    registry = build_registry(inventories, used)
    excluded = registry["excluded"]

    t200: dict[str, list[str]] = {}
    reserve_pool: dict[str, list[str]] = {}
    x80: dict[str, list[str]] = {}
    t200_strata: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        inventory = inventories[benchmark]
        eligible = [i for i in inventory if i.eligible]
        pool = [i for i in eligible if i.public.item_id not in used and i.public.item_id not in excluded]
        quota = T200_QUOTA[benchmark]
        if len(pool) < quota:
            raise SplitsRefused(f"{benchmark}: pool {len(pool)} < T200 quota {quota}")
        chosen, rest = stratified(pool, quota, salt(benchmark, "t200-v1"))
        t200[benchmark] = chosen
        reserve_pool[benchmark] = rest
        t200_items = [i for i in pool if i.public.item_id in set(chosen)]
        x80[benchmark], _ = stratified(t200_items, X80_QUOTA[benchmark], salt(benchmark, "x80-v1"))
        t200_strata[benchmark] = {
            "rule": tm._STRATUM_RULE[benchmark], "eligible": len(eligible), "pool": len(pool),
            "pool_strata": strata_counts(pool, (i.public.item_id for i in pool)),
            "t200_strata": strata_counts(pool, chosen), "x80_strata": strata_counts(pool, x80[benchmark]),
            "legacy_reserve_in_t200": [i for i in panels[benchmark].reserve_item_ids if i in set(chosen)],
        }

    bank = legacy_bank_correct(legacy)
    l96 = [i for b in BENCHMARKS for i in main[b]]
    d32 = [i for b in BENCHMARKS for i in dev[b]]
    if set(bank) != set(l96):
        raise SplitsRefused("legacy bank items differ from legacy main")
    unknown = {i for i, v in bank.items() if v is None}
    if unknown != UNKNOWN_BANK_ITEMS:
        raise SplitsRefused(f"bank-unknown L96 items {sorted(unknown)} != frozen {sorted(UNKNOWN_BANK_ITEMS)}")
    c48 = select_c48(bank, l96)
    assignment = t200_assignment(t200)
    t200_flat = [i for b in BENCHMARKS for i in t200[b]]
    splits = {
        "version": SPLITS_VERSION, "study_id": STUDY, "passk_seed": PASSK_SEED, "legacy_seed": LEGACY_SEED,
        "panels": {"L96": l96, "C48": c48["items"], "V48": list(c48["items"]), "T200": t200_flat, "D32": d32,
                   "X80": [i for b in BENCHMARKS for i in x80[b]]},
        "panels_by_benchmark": {"T200": t200, "X80": x80,
                                "C48": {b: [i for i in c48["items"] if i.startswith(b + "/")] for b in BENCHMARKS}},
        "c48": {"strata": c48["strata"], "inclusion_prob": c48["inclusion_prob"], "table": c48_table_rows(c48["strata"]),
                "bins": [name for name, _, _ in C48_BINS], "target_per_cell": C48_TARGET_PER_CELL,
                "eligible": sorted(c48["bank_correct"]), "bank_correct": c48["bank_correct"], "cells": c48["cells"],
                "excluded_unknown": sorted(UNKNOWN_BANK_ITEMS)},
        "t200_strata": t200_strata,
        "assignment_t200": assignment,
        "assignment_t200_counts": {b: {s: sum(1 for i in t200[b] if assignment[i] == s) for s in SHARDS}
                                   for b in BENCHMARKS},
        "duplicate_exclusions": excluded,
        "salts": salts_record(),
        "sha256_inputs": {**{f"legacy/{k}": v for k, v in legacy.input_hashes().items()},
                          **{f"inventory/{b}": inventory_hash(inventories[b]) for b in BENCHMARKS}},
    }
    new_panels = {b: PanelDraw(benchmark=b, dev_item_ids=tuple(dev[b]), main_item_ids=tuple(main[b]),
                               reserve_item_ids=tuple(t200[b]), selection_seed=LEGACY_SEED, procedure=PROCEDURE)
                  for b in BENCHMARKS}
    reserve = {"version": SPLITS_VERSION, "study_id": STUDY,
               "rule": "pool (eligible - legacy main U dev - duplicate-registry exclusions) minus T200, in the seeded "
                       "interleave order of the T200 draw's reserve panel",
               "counts": {b: len(reserve_pool[b]) for b in BENCHMARKS}, "items": reserve_pool}
    return {"splits": splits, "registry": registry, "reserve_pool": reserve, "panels": new_panels,
            "inventories": inventories, "legacy_panels": panels}


# ---------------------------------------------------------------------------------------------------------------
# Manifests (legacy prepare-data format)
# ---------------------------------------------------------------------------------------------------------------

def record_line(record: Any) -> bytes:
    """One manifest row exactly as ``manifest.prepare_data`` writes it."""
    return (json.dumps(dump_record(record), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def panels_text(panels: Mapping[str, PanelDraw]) -> bytes:
    return json.dumps({b: dump_record(d) for b, d in panels.items()}, indent=1, sort_keys=True).encode("utf-8")


def manifest_bytes(built: Mapping[str, Any], legacy: Legacy) -> tuple[dict[str, bytes], dict[str, Any]]:
    """``{relpath: bytes}`` for the public/private manifests plus a verification record.

    Legacy dev + main rows are copied byte-for-byte from the legacy files; T200 rows are produced by the adapter
    path (``dump_record`` of the inventory records). Every one of the 160 legacy rows must regenerate byte-equal
    through that path, which proves the T200 rows (GPQA shuffles included) follow the legacy per-item rule."""
    out: dict[str, bytes] = {}
    verify: dict[str, Any] = {"regenerated_legacy_rows": 0, "regenerated_mismatches": [], "copied_legacy_rows": 0}
    for benchmark in BENCHMARKS:
        by_id = {i.public.item_id: i for i in built["inventories"][benchmark]}
        draw: PanelDraw = built["panels"][benchmark]
        chosen = [*draw.dev_item_ids, *draw.main_item_ids, *draw.reserve_item_ids]
        legacy_count = len(draw.dev_item_ids) + len(draw.main_item_ids)
        for kind, path, attr in (("public", legacy.public_path(benchmark), "public"),
                                 ("private", legacy.private_path(benchmark), "private")):
            legacy_lines = legacy.lines(path)
            legacy_panel = built["legacy_panels"][benchmark]
            expected_order = [*legacy_panel.dev_item_ids, *legacy_panel.main_item_ids, *legacy_panel.reserve_item_ids]
            if list(legacy_lines) != expected_order:
                raise SplitsRefused(f"legacy {kind}/{benchmark} rows are not in dev, main, reserve order")
            for item_id, line in legacy_lines.items():
                verify["regenerated_legacy_rows"] += 1
                if record_line(getattr(by_id[item_id], attr)) != line:
                    verify["regenerated_mismatches"].append(f"{kind}/{item_id}")
            rows = []
            for pos, item_id in enumerate(chosen):
                if pos < legacy_count:
                    rows.append(legacy_lines[item_id])
                    verify["copied_legacy_rows"] += 1
                else:
                    rows.append(record_line(getattr(by_id[item_id], attr)))
            rel = f"manifests/public/{benchmark}.jsonl" if kind == "public" else f"private/{benchmark}.jsonl"
            out[rel] = b"".join(rows)
    return out, verify


# ---------------------------------------------------------------------------------------------------------------
# Writing and Gate S
# ---------------------------------------------------------------------------------------------------------------

def artifact_files(built: Mapping[str, Any], legacy: Legacy) -> tuple[dict[str, tuple[bytes, int | None]], dict]:
    manifests, verify = manifest_bytes(built, legacy)
    files: dict[str, tuple[bytes, int | None]] = {
        "manifests/splits.json": (json_bytes(built["splits"]), None),
        "manifests/duplicate_groups.json": (json_bytes(registry_payload(built["registry"])), None),
        "manifests/reserve_pool.json": (json_bytes(built["reserve_pool"]), None),
        "manifests/panels.json": (panels_text(built["panels"]), None),
        "manifests/waves.json": (json.dumps([], indent=1, sort_keys=True).encode("utf-8"), None),
        "manifests/exclusions.json": (legacy.exclusions_path.read_bytes(), None),
        "locks/datasets.json": (legacy.lock_path.read_bytes(), None),
    }
    for rel, data in manifests.items():
        files[rel] = (data, 0o600 if rel.startswith("private/") else None)
    return files, verify


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prereg_check(repo: Path) -> tuple[bool, str]:
    def show(rev: str) -> bytes | None:
        try:
            return subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{PREREG_PATH}"], capture_output=True,
                                  check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    at_commit, at_head = show(PREREG_COMMIT), show("HEAD")
    on_disk = (repo / PREREG_PATH).read_bytes() if (repo / PREREG_PATH).is_file() else None
    ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"],
                              capture_output=True).returncode == 0
    ok = bool(at_commit) and at_commit == at_head == on_disk and ancestor
    return ok, (f"PREREG.md at {PREREG_COMMIT} sha256 {sha256_bytes(at_commit) if at_commit else None}; "
                f"HEAD equal={at_commit == at_head}; working tree equal={at_commit == on_disk}; ancestor={ancestor}")


def legacy_assignment_check(cfg: Mapping[str, Any], artifacts: Path, legacy: Legacy, built: Mapping[str, Any],
                            local_state: Path) -> tuple[bool, dict[str, Any]]:
    """Run the unchanged ``shards.root_assignment`` on the new root panels.json and compare L96 U D32 with legacy."""
    from agent_allocation.paths import RunPaths
    from agent_allocation.shards import root_assignment

    result = root_assignment(cfg, RunPaths.for_run(artifacts, local_state_dir=local_state))
    old = legacy.assignment()
    items = [*built["splits"]["panels"]["L96"], *built["splits"]["panels"]["D32"]]
    mismatches = [i for i in items if result["assignment"].get(i) != old.get(i)]
    old_counts = json.loads(legacy.assignment_path.read_text())["counts"]
    counts_equal = {p: result["counts"][p] == old_counts[p] for p in ("main", "dev")}
    t200 = built["splits"]["assignment_t200"]
    reserve_by_root = {i: result["assignment"][i] for i in t200}
    detail = {"seed": result["seed"], "shares": {s["id"]: s["share"] for s in result["shards"]},
              "items_compared": len(items), "mismatches": mismatches, "counts_equal": counts_equal,
              "root_assignment_reserve_counts": result["counts"].get("reserve"),
              "t200_1to1_equals_root_assignment_reserve": reserve_by_root == t200,
              "note": "unchanged shards.root_assignment gives the reserve (T200) panel the 3:2 config shares; the "
                      "frozen T200 assignment is splits.json assignment_t200 (1:1), which WP-G's per-panel override "
                      "must reproduce verbatim"}
    return (not mismatches and all(counts_equal.values())), detail


def gate_s_checks(cfg: Mapping[str, Any], built: Mapping[str, Any], legacy: Legacy, files: Mapping[str, Any],
                  verify: Mapping[str, Any], digests: Mapping[str, str], artifacts: Path, repo: Path,
                  local_state: Path) -> list[dict[str, Any]]:
    splits = built["splits"]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    on_disk = {rel: sha256_file(artifacts / rel) for rel in files}
    check("write_once_sha256", on_disk == dict(digests), {"files": dict(sorted(on_disk.items()))})
    lock = legacy.lock()
    check("inventories_match_legacy_lock",
          all(inventory_hash(built["inventories"][b]) == lock["sources"][b]["inventory_hash"] for b in BENCHMARKS),
          {b: inventory_hash(built["inventories"][b]) for b in BENCHMARKS})
    main = set(splits["panels"]["L96"])
    dev = set(splits["panels"]["D32"])
    partners = set(splits["duplicate_exclusions"])
    t200 = splits["panels"]["T200"]
    overlap = sorted(set(t200) & (main | dev | partners))
    check("t200_disjoint_main_dev_duplicates", not overlap and len(set(t200)) == len(t200),
          {"overlap": overlap, "t200": len(t200), "excluded_partners": len(partners)})
    eligible = {i.public.item_id for b in BENCHMARKS for i in built["inventories"][b] if i.eligible}
    check("t200_eligible", set(t200) <= eligible, {"ineligible": sorted(set(t200) - eligible)})
    composition = {b: len(splits["panels_by_benchmark"]["T200"][b]) for b in BENCHMARKS}
    check("t200_composition_20_20_100_60", composition == T200_QUOTA, composition)
    x80 = splits["panels"]["X80"]
    x80_comp = {b: len(splits["panels_by_benchmark"]["X80"][b]) for b in BENCHMARKS}
    check("x80_subset_8_8_40_24", set(x80) <= set(t200) and x80_comp == X80_QUOTA, x80_comp)
    reg = built["registry"]
    partner_ok = all(reg["excluded"].get(k, "").endswith(v) and reg["excluded"][k].startswith("partner_of_used")
                     for k, v in EXPECTED_PARTNER_EXCLUSIONS.items())
    conflict_ok = all(reg["excluded"].get(k) == "conflicting_gold" for k in EXPECTED_CONFLICTING)
    reviewed_gold = {f"{f['a']}|{f['b']}": f["gold_relation"] for f in reg["flagged"]
                     if f["status"] == "reviewed_duplicate"}
    gold_ok = all((v == "conflict") == (set(k.split("|")) == set(EXPECTED_CONFLICTING))
                  for k, v in reviewed_gold.items())
    omni_pool = built["splits"]["t200_strata"]["omni_math_hard"]["pool"]
    check("duplicate_registry_review", partner_ok and conflict_ok and gold_ok and len(reviewed_gold) == 13,
          {"partner_exclusions_ok": partner_ok, "conflicting_ok": conflict_ok, "reviewed_gold_relation": reviewed_gold,
           "omni_pool_after_dedup": omni_pool, "excluded": reg["excluded"],
           "flagged_not_duplicate": sum(1 for f in reg["flagged"] if f["status"] == "flagged_not_duplicate")})
    bank = splits["c48"]["bank_correct"]
    c48 = splits["panels"]["C48"]
    probs = splits["c48"]["inclusion_prob"]
    c48_ok = (set(c48) <= set(bank) and len(c48) == 48 and not set(c48) & UNKNOWN_BANK_ITEMS
              and all(len(splits["panels_by_benchmark"]["C48"][b]) == C48_PER_BENCHMARK for b in BENCHMARKS)
              and set(probs) == set(c48) and all(0 < p <= 1 for p in probs.values())
              and splits["panels"]["V48"] == c48)
    check("c48_subset_bank_known_with_inclusion_probs", c48_ok,
          {"bank_known": len(bank), "c48": len(c48), "table": splits["c48"]["table"]})
    counts = splits["assignment_t200_counts"]
    check("t200_assignment_1to1", all(counts[b]["mancer"] == counts[b]["parasail"] for b in BENCHMARKS)
          and set(splits["assignment_t200"]) == set(t200), counts)
    ok, detail = legacy_assignment_check(cfg, artifacts, legacy, built, local_state)
    check("legacy_assignment_reproduced_l96_d32", ok, detail)
    check("manifest_rows_legacy_byte_identical",
          not verify["regenerated_mismatches"] and verify["regenerated_legacy_rows"] == 320
          and verify["copied_legacy_rows"] == 256, dict(verify))
    covered = set()
    for b in BENCHMARKS:
        for kind in ("manifests/public", "private"):
            covered |= {json.loads(line)["item_id"] for line in (artifacts / f"{kind}/{b}.jsonl").read_text().splitlines()}
    check("manifests_cover_l96_d32_t200", covered == main | dev | set(t200),
          {"rows": len(covered), "expected": len(main | dev | set(t200))})
    ok, detail = prereg_check(repo)
    check("prereg_committed_e781c34", ok, detail)
    return checks


def write_artifacts(cfg: Mapping[str, Any], run_root: Path = PASSK_RUN_ROOT, legacy_root: Path = LEGACY_RUN_ROOT,
                    *, repo: Path | None = None, local_state: Path | None = None) -> dict[str, Any]:
    """Build, write (write-once) and gate. Returns the Gate S record (also written to ``gates/gate_s.json``)."""
    legacy = Legacy(legacy_root)
    built = build_splits(cfg, legacy)
    files, verify = artifact_files(built, legacy)
    artifacts = Path(run_root) / "artifacts"
    # Refuse before writing anything if any existing file differs.
    for rel, (data, _mode) in files.items():
        target = artifacts / rel
        if target.exists() and target.read_bytes() != data:
            raise SplitsRefused(f"{target} exists with different content; frozen split artifacts are write-once")
    private = artifacts / "private"
    private.mkdir(parents=True, exist_ok=True)
    os.chmod(private, 0o700)
    digests = {rel: write_once(artifacts / rel, data, mode=mode) for rel, (data, mode) in files.items()}
    repo = Path(repo) if repo else Path(__file__).resolve().parents[4]
    local_state = Path(local_state) if local_state else Path(run_root) / "logs" / "splits_local_state"
    checks = gate_s_checks(cfg, built, legacy, files, verify, digests, artifacts, repo, local_state)
    body = {"gate": "gate_s", "study_id": STUDY, "version": SPLITS_VERSION,
            "passed": all(c["passed"] for c in checks), "checks": checks,
            "splits_sha256": digests["manifests/splits.json"], "files": dict(sorted(digests.items()))}
    record = {**body, "code_commit": git_head(repo),
              "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    gate_path = Path(run_root) / "gates" / "gate_s.json"
    if gate_path.exists():
        existing = json.loads(gate_path.read_text())
        if {k: existing.get(k) for k in body} != body:
            raise SplitsRefused(f"{gate_path} exists with a different Gate S result; gate records are write-once")
        return existing
    write_once(gate_path, json_bytes(record))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_allocation.passk.splits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default=str(PASSK_RUN_ROOT))
    parser.add_argument("--legacy-root", default=str(LEGACY_RUN_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="build and print a summary; write nothing")
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config

    cfg = load_config(args.config)
    if args.dry_run:
        legacy = Legacy(Path(args.legacy_root))
        built = build_splits(cfg, legacy)
        files, verify = artifact_files(built, legacy)
        splits = built["splits"]
        print(json.dumps({"files": {k: sha256_bytes(v[0]) for k, v in files.items()}, "verify": verify,
                          "c48_table": splits["c48"]["table"], "t200_counts": splits["assignment_t200_counts"],
                          "excluded": splits["duplicate_exclusions"],
                          "flagged": registry_payload(built["registry"])["flagged_counts"]}, indent=1))
        return 0
    record = write_artifacts(cfg, Path(args.run_root), Path(args.legacy_root))
    print(json.dumps({"passed": record["passed"], "splits_sha256": record["splits_sha256"],
                      "failed": [c["name"] for c in record["checks"] if not c["passed"]]}, indent=1))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
