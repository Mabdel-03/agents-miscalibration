"""Effective depth: reasoning tokens spent before the reasoning degenerates into a loop (forced runs).

Forcing a generation to its full budget far past the model's natural stopping point can make it recycle text. For
every generation the reasoning text is scanned in 3,000-character windows (step 500); loop onset is the start of the
first window whose zlib compression ratio (compressed / raw bytes, level 9) is below 0.1 -- prose sits near
0.3-0.45, a repeating loop far below 0.1 (``tools/forced_deep_pilot.py``'s ``loop_onset``). Effective tokens of a
generation = onset fraction x ``reasoning_budget`` when a loop is found, else its realized reasoning tokens (the
budget, in a forced run).

A checkpoint's effective depth is the sum over its required (from-scratch ancestor) generations, the same
accounting as its requested reasoning ``C`` -- so ``effective_fraction = effective / C`` says how much of the
iso-compute budget did non-recycled work.

Reasoning text comes from the raw artifact (``storage/raw.py``) via WP-1's
``agent_allocation.api.forced.reasoning_text_from_raw(payload)`` when importable, else :func:`fallback_reasoning_text`
(the forced client's raw layout: ``response_body_text`` = ``{"think": <raw THINK JSON>, "answer": ...}``; the THINK
completion text is the reasoning; a chat-completions body's ``message.reasoning`` also works).

Tables (``analyze`` in forced mode): ``effective_depth_nodes`` (one row per generation) and ``effective_depth``
(one row per current-view checkpoint and item, plus ``effective_depth_summary``: means per checkpoint).
"""
from __future__ import annotations

import gzip
import json
import zlib
from typing import Any, Callable, Mapping

import pandas as pd

from agent_allocation.analysis.load import RunData

WINDOW_CHARS = 3000
STEP_CHARS = 500
LOOP_RATIO = 0.1


def loop_onset(text: str, window: int = WINDOW_CHARS, step: int = STEP_CHARS,
               threshold: float = LOOP_RATIO) -> float | None:
    """Fraction of the reasoning before the first window whose zlib ratio falls below ``threshold`` (a loop)."""
    if not text:
        return None
    for start in range(0, max(1, len(text) - window), step):
        chunk = text[start:start + window].encode()
        if len(zlib.compress(chunk, 9)) / max(1, len(chunk)) < threshold:
            return start / len(text)
    return None


def _json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _completion_reasoning(body: Any) -> str | None:
    """Reasoning text of one raw completion / chat-completion body."""
    body = _json(body)
    if not isinstance(body, Mapping):
        return None
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], Mapping):
        return None
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    for source in (choice, message):
        for key in ("reasoning", "reasoning_content"):
            if isinstance(source.get(key), str) and source[key]:
                return source[key]
    text = choice.get("text")
    if isinstance(text, str):
        return text.split("</think>", 1)[0] if "</think>" in text else text
    return None


def fallback_reasoning_text(payload: Mapping[str, Any]) -> str | None:
    """Local parser of a raw artifact's reasoning (used until WP-1's ``reasoning_text_from_raw`` is importable)."""
    body = _json(payload.get("response_body_text"))
    if isinstance(body, Mapping) and "think" in body:
        think = body["think"]
        if isinstance(think, Mapping) and "body" in think and "choices" not in think:
            think = think["body"]
        return _completion_reasoning(think)
    return _completion_reasoning(body)


def reasoning_text_reader() -> Callable[[Mapping[str, Any]], str | None]:
    try:
        from agent_allocation.api.forced import reasoning_text_from_raw  # type: ignore[attr-defined]
    except ImportError:
        return fallback_reasoning_text

    def read(payload: Mapping[str, Any]) -> str | None:
        try:
            text = reasoning_text_from_raw(payload)
        except Exception:  # noqa: BLE001 -- a foreign/partial artifact: fall back to the local parser
            text = None
        return text if text else fallback_reasoning_text(payload)
    return read


def effective_tokens(onset: float | None, budget: int, realized: int | None) -> float:
    if onset is not None:
        return onset * budget
    return float(realized if realized is not None else budget)


def node_table(run: RunData, *, reader: Callable[[Mapping[str, Any]], str | None] | None = None) -> pd.DataFrame:
    read = reader or reasoning_text_reader()
    rows = []
    specs = {nid: s for part in run.jobs.values() for nid, s in part.items()}
    for node_id, spec in sorted(specs.items()):
        parsed = run.parsed.get(node_id)
        if parsed is None:
            continue
        result = parsed[0]
        text = None
        if result.raw_response_path:
            path = run.paths.run_dir / result.raw_response_path
            if path.is_file():
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    text = read(json.load(handle))
        realized = result.usage.reasoning_tokens if result.usage else None
        onset = loop_onset(text) if text else None
        rows.append({"partition_id": spec.partition_id, "node_id": node_id, "item_id": spec.item_id,
                     "benchmark": spec.benchmark, "kind": spec.kind.value, "reasoning_budget": spec.reasoning_budget,
                     "reasoning_tokens": realized, "reasoning_chars": len(text) if text else None,
                     "text_available": text is not None, "looped": onset is not None, "loop_onset_fraction": onset,
                     "effective_tokens": effective_tokens(onset, spec.reasoning_budget, realized)
                     if text is not None else None})
    return pd.DataFrame(rows)


def checkpoint_table(run: RunData, nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame()
    eff = dict(zip(nodes["node_id"], nodes["effective_tokens"]))
    looped = dict(zip(nodes["node_id"], nodes["looped"]))
    specs = {nid: s for part in run.jobs.values() for nid, s in part.items()}
    rows = []
    for partition in run.partitions:
        for cp in run.policies.get(partition, []):
            if cp.view.value != "current":
                continue
            values = [eff.get(n) for n in cp.required_node_ids]
            known = all(v is not None and not pd.isna(v) for v in values)
            requested = sum(specs[n].reasoning_budget for n in cp.required_node_ids if n in specs)
            total = float(sum(values)) if known else None
            rows.append({"partition_id": cp.partition_id, "policy_id": cp.policy_id, "item_id": cp.item_id,
                         "name": cp.name, "family": cp.family, "requested_reasoning_tokens": requested,
                         "effective_tokens": total,
                         "effective_fraction": (total / requested) if (total is not None and requested) else None,
                         "looped_generations": sum(1 for n in cp.required_node_ids if looped.get(n)),
                         "generations": len(cp.required_node_ids)})
    return pd.DataFrame(rows)


def checkpoint_summary(per_item: pd.DataFrame) -> pd.DataFrame:
    if per_item.empty:
        return pd.DataFrame()
    grouped = per_item.groupby(["partition_id", "name", "family"], sort=True)
    out = grouped.agg(items=("item_id", "nunique"), requested_reasoning_tokens=("requested_reasoning_tokens", "mean"),
                      mean_effective_tokens=("effective_tokens", "mean"),
                      mean_effective_fraction=("effective_fraction", "mean"),
                      mean_looped_generations=("looped_generations", "mean"),
                      items_known=("effective_tokens", "count")).reset_index()
    from agent_allocation.analysis.names import name_sort_key
    order = sorted(range(len(out)), key=lambda i: (str(out.at[i, "partition_id"]), name_sort_key(out.at[i, "name"])))
    return out.iloc[order].reset_index(drop=True)


def effective_depth_tables(run: RunData, *, reader=None) -> dict[str, pd.DataFrame]:
    nodes = node_table(run, reader=reader)
    per_item = checkpoint_table(run, nodes)
    return {"effective_depth_nodes": nodes, "effective_depth": per_item,
            "effective_depth_summary": checkpoint_summary(per_item)}
