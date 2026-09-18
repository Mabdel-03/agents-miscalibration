"""Benchmark adapters. Text adapters: ``aime``, ``hmmt`` (both via ``_matharena``), ``omni_math_hard`` and
``gpqa_diamond`` plus the shared panel selector in ``text_manifest``. The code adapter ``mbpp_plus`` is kept for the
dormant code-grading path; no study benchmark uses it.

Every adapter exposes ``load_inventory(cfg, paths) -> list[InventoryItem]``; nothing here touches the network
or spends money.
"""
from __future__ import annotations

TEXT_BENCHMARKS: tuple[str, ...] = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")
