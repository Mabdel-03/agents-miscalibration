"""Benchmark adapters. Text adapters (agent 04): ``math500``, ``mmlu_pro``, ``bbh_logic4`` plus the shared
panel selector in ``text_manifest``. The code adapter ``mbpp_plus`` belongs to agent 05.

Every adapter exposes ``load_inventory(cfg, paths) -> list[InventoryItem]``; nothing here touches the network
or spends money.
"""
from __future__ import annotations

TEXT_BENCHMARKS: tuple[str, ...] = ("math500", "mmlu_pro", "bbh_logic4")
