"""DEV SHIM (WP-M, tests only) -- NOT the production reader.

Builds ``draws.parquet``-shaped frames (contract §5.3 columns that the models read) from the REAL read-only legacy
forced_v2 run root, so the E2/E5 models can be developed and validated before WP-A's ``extract.py`` lands. It lives
under ``tests/`` on purpose: nothing in ``passk/models`` may reference ``shards/`` or the legacy run root (seal static
scan). Classes are left to ``passk.models.inputs`` (exact-key fallback: no Math-Verify merges here).

Also provides a pseudo-C48 (12 per benchmark, salted hash order over L96) because ``splits.json`` may not exist yet.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

LEGACY_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts")
A = 1024
CACHE = Path(os.environ.get("PASSK_DEVSHIM_CACHE", "/tmp")) / "passk_wpm_devshim_draws.parquet"


def available() -> bool:
    return (LEGACY_ROOT / "merged" / "tables" / "candidates.parquet").exists()


def _tasks():
    from agent_allocation.contracts import PublicTask, load_record
    tasks = {}
    for path in sorted((LEGACY_ROOT / "manifests" / "public").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                t = load_record(PublicTask, json.loads(line))
                tasks[t.item_id] = t
    return tasks


def legacy_draws(cache: Path | None = CACHE) -> pd.DataFrame:
    """L96 root @A (slots 0-24) + deep r0 @2/4/8/16/25A, labelled, from the legacy run root (read-only)."""
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    from agent_allocation.contracts import Candidate, OutputKind, load_record
    from agent_allocation.selection.keys import option_letters, vote_key
    cands = pd.read_parquet(LEGACY_ROOT / "merged" / "tables" / "candidates.parquet")
    m = cands[(cands.partition_id == "main") & cands.kind.isin(["root", "deep"])].copy()
    m = m[(m.kind == "deep") | (m.reasoning_budget == A)]
    tasks = _tasks()
    rows = []
    for r in m.itertuples(index=False):
        path = LEGACY_ROOT / "shards" / r.shard_id / "parsed" / r.node_id[:2] / f"{r.node_id}.json"
        rec = json.loads(path.read_text()) if path.exists() else {}
        cand = rec.get("candidate")
        vt = vk = opt = None
        alen = None
        if cand is not None:
            c = load_record(Candidate, cand)
            alen = None if c.visible_content is None else len(c.visible_content)
            task = tasks.get(r.item_id)
            if task is not None:
                letters = option_letters(task) if task.output_kind is OutputKind.OPTION else None
                key = vote_key(c, task, letters=letters)
                if key is not None:
                    vt, vk = key.text, key.exact
                    if task.output_kind is OutputKind.OPTION:
                        opt = key.exact
        slot = int(r.root_slot) if r.kind == "root" else 0
        role = "deep_r0" if r.kind == "deep" else ("pool" if slot < 16 else ("bank" if slot < 20 else "diag"))
        gs = r.grade_state
        correct = None if (r.correct != r.correct) else bool(r.correct)
        if gs == "op_invalid_output":
            correct, gs = False, "invalid"
        elif gs.startswith("unknown"):
            correct, gs = None, "unknown"
        rows.append(dict(
            candidate_id=r.node_id, source="legacy", partition="legacy_main", panel_roles=["L96"], draw_role=role,
            item_id=r.item_id, benchmark=r.benchmark, shard=r.shard_id, kind=r.kind,
            allowance_a=int(r.reasoning_budget) // A, replicate=slot, parse_status=r.parse_status,
            vote_text=vt, vote_exact_key=vk, option=opt, answer_len_chars=alen, answer_finish_reason=r.finish_reason,
            latency_s=float(r.latency_seconds), observed_provider=r.observed_provider, absent=False,
            correct=correct, grade_state=gs))
    df = pd.DataFrame(rows)
    if cache is not None:
        try:
            df.to_parquet(cache, index=False)
        except OSError:
            pass
    return df


def pseudo_c48(draws: pd.DataFrame, per_benchmark: int = 12) -> list[str]:
    """Deterministic stand-in for splits.json C48 (salted hash order; NOT the frozen stratified rule)."""
    out = []
    items = draws[["item_id", "benchmark"]].drop_duplicates()
    for b, grp in items.groupby("benchmark"):
        ranked = sorted(grp.item_id, key=lambda i: hashlib.sha256(f"wpm-devshim|{i}".encode()).hexdigest())
        out += ranked[:per_benchmark]
    return sorted(out)


if __name__ == "__main__":  # pragma: no cover
    d = legacy_draws()
    print(d.shape, d.groupby(["kind", "allowance_a"]).size().to_dict())
