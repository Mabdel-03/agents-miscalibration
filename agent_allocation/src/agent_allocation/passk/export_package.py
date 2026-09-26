"""Paper export package for passk_cont_v1: every result as CSV, with a data dictionary and a manifest.

    python -m agent_allocation.passk.export_package --study-root R --out P

``P`` (``R/paper_package/passk_cont_v1_results``) gets ``csv/00_study`` ... ``csv/11_costs``, ``prompts/``,
``prereg/``, ``figures/``, ``code_manifest.txt``, ``DATA_DICTIONARY.md`` and ``MANIFEST.csv``. ``README.md`` (the
write-up) and any other file a write-up stage keeps at the package root are never touched here.

Read-only on the run root (``sealed/``, ``scores/``, ``analysis_inputs/`` and the legacy forced_v2 root are only
read); no network. Re-runnable and deterministic: every run first removes what it owns under ``P`` (the four
directories and the three generated files) and regenerates it; no wall-clock time is written, rows keep a fixed order,
floats are written at full precision. ``MANIFEST.csv`` lists every file under ``P``, including README.md. CSV conventions: UTF-8, one header row, no index column, lists/dicts as JSON strings, booleans as
``true``/``false``, missing values as empty cells.

A source that is absent is recorded as a gap (printed at the end and listed in ``DATA_DICTIONARY.md``); columns
whose meaning could not be taken from the code are marked ``UNDOCUMENTED`` rather than guessed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["CSV_DIRS", "Package", "export", "main", "to_csv_frame", "write_csv"]

STUDY_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1")
DEFAULT_OUT = STUDY_ROOT / "paper_package" / "passk_cont_v1_results"
CODE_ROOT = Path(__file__).resolve().parents[3]                    # .../agent_allocation
PLAN_PATH = Path("/home/mabdel03/.claude/plans/i-want-to-run-sharded-swan.md")
SPEC_REL = Path("expermt_agent_specs/Pass@K-Frontier-Cont/remaining_experiments_spec.md")
MAIN_CHECKOUT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling")
README = "README.md"

CSV_DIRS = ("00_study", "01_e0_audit", "02_draws", "03_pools", "04_e1", "05_e2", "06_e5",
            "07_e8a_llm_allocator", "08_lex_front", "09_selector_comparison", "10_e8b_lex_llm", "11_costs")
TOP_DIRS = ("csv", "prompts", "prereg", "figures")
#: everything the exporter writes (and therefore clears on a re-run); nothing else under the package is touched
OWNED = frozenset(TOP_DIRS + ("code_manifest.txt", "DATA_DICTIONARY.md", "MANIFEST.csv"))
BENCH = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")
PANELS = ("L96", "C48", "V48", "T200", "D32", "X80")
A_TOKENS = 1024
E8B_FALLBACK = "1x16"

#: Commits named in the study history (memory note / git log). Resolved with ``git rev-parse`` at export time.
KEY_COMMITS = (
    ("forced_v2 base the branch was cut from", "dc7eb24"),
    ("PREREG.md committed (day 0, before any generation)", "e781c34"),
    ("build merged (WP-G/A/M handoffs)", "3c0a530"),
    ("pre-launch review fix pass 1", "6fd7c1a"),
    ("launch/seal code commit (E1 common cohort; sealed fits bound to it)", "c8b9b7b"),
    ("E8a addendum + module (before any allocator call)", "9c067a5"),
    ("LEX addendum + c48_25a runner", "d718d17"),
    ("LEX front analysis", "f38a27e"),
    ("E8b addendum + module (before any E8b call)", "d9bd36a"),
    ("selector comparison (selcmp) under the LEX objective", "c8a521b"),
)

#: Timestamps recorded in the run's memory note (EDT = UTC-4), for context only.
MEMORY_NOTE_EVENTS = (
    ("plan approved", "2026-09-25 ~14:00 EDT"),
    ("Gate S passed / build merged", "2026-09-25 15:25 EDT"),
    ("Gate 0 passed; conductor launched (job 23937247)", "2026-09-25 17:07 EDT"),
    ("Gate P seal written (sealed/SEAL_predictions.json)", "2026-09-25 18:42 EDT"),
    ("generation complete (Gate 4 passed v48 + t200)", "2026-09-25 19:57 EDT"),
    ("core study complete: report built (conductor 23963120)", "2026-09-25 20:07 EDT"),
    ("E8a allocator run and scored", "2026-09-25 ~21:00 EDT"),
    ("LEX c48_25a generation + front analysis", "2026-09-25 ~21:43-21:48 EDT"),
    ("E8b decisions finished", "2026-09-25 ~22:18 EDT"),
)

#: OpenRouter key-usage checkpoints recorded in the memory note (the /key endpoint's cumulative usage).
MEMORY_NOTE_KEY_USAGE = (
    ("2026-09-25 14:20 EDT", "owner raised the key limit before launch", 181.32, 400.0, 218.68),
    ("2026-09-25 20:07 EDT", "core study complete (10,572 new generations): 277.58 - 181.32 = 96.26 all-in",
     277.58, 400.0, None),
)

#: How every stage was run (RUNBOOK.md, module docstrings, memory note).
STAGE_COMMANDS = (
    ("environment", "cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation && "
                    "source slurm/env.sh; A=.venv/bin/allocation; PY=.venv/bin/python; CFG=configs/passk_cont_v1.json; "
                    "STUDY=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1; RUN=$STUDY/artifacts"),
    ("Gate S splits", "$PY -m agent_allocation.passk.splits --config $CFG"),
    ("shard plan", "$A shard-plan --config $CFG --run-dir $RUN --write; for s in mancer parasail; do $A passk-run "
                   "--config $CFG --run-dir $RUN --shard $s --partitions smoke --adopt-legacy-lock; done"),
    ("plan check", "$A passk-run --config $CFG --run-dir $RUN --shard <s> --plan-only "
                   "--partitions smoke+c48_extra+e1_fill+t200_diag+v48_bank+t200_bank"),
    ("Gate 0", "$PY -m agent_allocation.passk.gates gate0 --config $CFG --study-root $STUDY --check-key --run-tests "
               "--run-e0"),
    ("conductor (Gate 1 canary+smoke -> main -> analysis chain -> seal -> release -> score -> report)",
     "sbatch --mail-user=$OWNER_MAIL --export=ALL,START_AT=<gate0|gate1|main|step token> "
     "slurm/passk_conductor.sbatch"),
    ("generation (per shard, what the sbatch runs)", "$A passk-run --config $CFG --run-dir $RUN --shard <s> "
     "--partitions <a+b> --allow-paid --max-usd <slice> --account-auto-reload --resume"),
    ("grading (per shard, fenced ids skipped until the seal)", "$A grade --run-dir $RUN/shards/<s> --resume --follow "
     "--config $RUN/shards/<s>/state/shard_config.json"),
    ("E0 audit", "$A passk-analyze --config $CFG --run-root $STUDY --stage e0"),
    ("E0 gate subset", "$PY -m agent_allocation.passk.e0_gate --legacy-root <forced_v2>/artifacts --study-root $STUDY"),
    ("single reader", "$A passk-analyze --run-root $STUDY --stage extract"),
    ("pair cache", "$A passk-analyze --run-root $STUDY --stage pairs"),
    ("E1 lock", "$A passk-analyze --run-root $STUDY --stage e1-lock"),
    ("E2 fit (bwrap fence)", "$A passk-analyze --run-root $STUDY --stage e2-fit --sandbox"),
    ("E5 fit (bwrap fence)", "$A passk-analyze --run-root $STUDY --stage e5-fit --sandbox"),
    ("Gate P seal", "$A passk-analyze --run-root $STUDY --stage seal --sandbox"),
    ("release held-out split", "$A passk-analyze --run-root $STUDY --stage release --release v48 (then t200)"),
    ("primary family + E2/E5 scoring", "$A passk-analyze --run-root $STUDY --stage score"),
    ("report", "$A passk-analyze --run-root $STUDY --stage report  (= docs/passk_cont_report/build.sh)"),
    ("E8a LLM allocator", "$PY -m agent_allocation.passk.e8a run --config $CFG --study-root $STUDY "
                          "--panels L96+T200 --arms A0+A1; "
                          "$PY -m agent_allocation.passk.e8a score --study-root $STUDY"),
    ("LEX 25A draws", "$PY -m agent_allocation.passk.lex gen25 --study-root $STUDY --cap-usd 32; "
                      "$PY -m agent_allocation.passk.lex grade25 --study-root $STUDY"),
    ("LEX front", "$PY -m agent_allocation.passk.lexfront --study-root $STUDY --workers 16"),
    ("selector comparison", "$PY -m agent_allocation.passk.selcmp --study-root $STUDY --n-sim 2000 [--overwrite]"),
    ("E8b LLM selectors told the LEX objective", "$PY -m agent_allocation.passk.e8b run --study-root $STUDY "
                                                 "--panels C48+T200 --arms B0+B1"),
    ("this package", "$PY -m agent_allocation.passk.export_package --study-root $STUDY "
                     "--out $STUDY/paper_package/passk_cont_v1_results"),
)


# =====================================================================================================================
# CSV serialization
# =====================================================================================================================

def _is_missing(v: Any) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    return False


def _native(v: Any) -> Any:
    """numpy scalars / arrays -> Python natives (recursively) for JSON."""
    if isinstance(v, np.ndarray):
        return [_native(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_native(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _native(x) for k, x in v.items()}
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _cell(v: Any) -> Any:
    if isinstance(v, (list, tuple, np.ndarray)):
        return json.dumps(_native(v), ensure_ascii=False)
    if isinstance(v, dict):
        return json.dumps(_native(v), ensure_ascii=False, sort_keys=True)
    if _is_missing(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def infer_type(series: pd.Series) -> str:
    """A human type label for the data dictionary (from the frame before serialization)."""
    dt = series.dtype
    if pd.api.types.is_bool_dtype(dt):
        return "boolean"
    if pd.api.types.is_integer_dtype(dt):
        return "integer"
    if pd.api.types.is_float_dtype(dt):
        vals = series.dropna()
        return "float" if len(vals) else "empty (all missing)"
    vals = [v for v in series.tolist() if not _is_missing(v) and not (isinstance(v, np.ndarray) and False)]
    vals = [v for v in vals if not (isinstance(v, str) and v == "")] or vals
    if not vals:
        return "empty (all missing)"
    kinds = set()
    for v in vals:
        if isinstance(v, (bool, np.bool_)):
            kinds.add("boolean")
        elif isinstance(v, (list, tuple, np.ndarray)):
            kinds.add("JSON list")
        elif isinstance(v, dict):
            kinds.add("JSON object")
        elif isinstance(v, (int, np.integer)):
            kinds.add("integer")
        elif isinstance(v, (float, np.floating)):
            kinds.add("float")
        else:
            kinds.add("string")
    if kinds == {"integer", "float"}:
        return "float"
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed (" + ", ".join(sorted(kinds)) + ")"


_COUNT_COL = re.compile(r"(^|_)(n|count|calls|attempts|missing|rank|components|requests|iter|coef|nodes|index|items|"
                        r"decisions|resamples|primary|unknown|rows|size|selected|dropped|tokens)($|_)")


def intify_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count-like float columns (names such as n_*, *_n_items, rank, n_resamples) whose values are all integers
    become nullable integers, so a count is written ``20`` rather than ``20.0`` next to missing cells."""
    out = df
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_float_dtype(s.dtype) or not _COUNT_COL.search(str(col)):
            continue
        vals = s.dropna().to_numpy(dtype=float)
        if len(vals) and np.all(np.isfinite(vals)) and np.all(vals == np.round(vals)) and np.all(np.abs(vals) < 2**53):
            if out is df:
                out = df.copy()
            out[col] = s.round().astype("Int64")
    return out


def to_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The frame as it is written: bool -> true/false, lists/dicts -> JSON, missing -> empty."""
    out = pd.DataFrame(index=range(len(df)))
    for col in df.columns:
        s = df[col].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(s.dtype):
            out[col] = s.map(lambda v: None if _is_missing(v) else ("true" if v else "false")).astype(object)
        elif pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype):
            out[col] = s
        else:
            out[col] = pd.Series([_cell(v) for v in s.tolist()], dtype=object)
    return out


def write_csv(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = to_csv_frame(df)
    frame.to_csv(path, index=False, lineterminator="\n", na_rep="", encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return int(len(frame))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# =====================================================================================================================
# JSON flattening
# =====================================================================================================================

def flatten_kv(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every scalar leaf as (dotted path, value); lists of scalars stay whole (JSON), lists of dicts are indexed."""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        if not obj:
            out.append((prefix, "{}"))
        for k in obj:
            out += flatten_kv(obj[k], f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        if any(isinstance(x, (dict, list, tuple)) for x in obj):
            for i, x in enumerate(obj):
                out += flatten_kv(x, f"{prefix}[{i}]")
        else:
            out.append((prefix, json.dumps(_native(list(obj)), ensure_ascii=False)))
    else:
        out.append((prefix, obj))
    return out


def kv_frame(obj: Any, *, key: str = "key", value: str = "value") -> pd.DataFrame:
    rows = [{key: k, value: _kv_value(v)} for k, v in flatten_kv(obj)]
    return pd.DataFrame(rows, columns=[key, value])


def _kv_value(v: Any) -> Any:
    v = _native(v)
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v) if not isinstance(v, str) else v


_PAIR_KEYS = ("ci", "bounds")


def flat_record(d: Mapping[str, Any], prefix: str = "", *, json_keys: Iterable[str] = ("choices",)) -> dict[str, Any]:
    """A nested stats dict as one flat row: sub-dicts joined with ``_``; 2-number interval lists -> ``_lo``/``_hi``;
    other lists and the ``json_keys`` dicts -> JSON strings."""
    json_keys = set(json_keys)
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k not in json_keys:
            out.update(flat_record(v, key + "_", json_keys=json_keys))
        elif isinstance(v, dict):
            out[key] = _native(v)
        elif (isinstance(v, (list, tuple)) and len(v) == 2 and any(t in k for t in _PAIR_KEYS)
              and all(isinstance(x, (int, float)) and not isinstance(x, bool) or x is None for x in v)):
            out[key + "_lo"], out[key + "_hi"] = _native(v[0]), _native(v[1])
        elif isinstance(v, (list, tuple)):
            out[key] = _native(list(v))
        else:
            out[key] = _native(v)
    return out


# =====================================================================================================================
# Column documentation
# =====================================================================================================================

B_DOC = "benchmark family: aime, hmmt, omni_math_hard or gpqa_diamond"
COMMON: dict[str, str] = {
    "item_id": "problem id in the legacy form <benchmark>/<source id> (e.g. aime/2025/7, gpqa_diamond/rec...)",
    "benchmark": B_DOC,
    "panel": "item panel: L96 (legacy main 96), C48 (48-item E2 fitting subset of L96), V48 (the same 48 items, "
             "fresh draws), T200 (200 unused test items), D32 (legacy dev 32)",
    "config_id": "allocation n x a: n independent calls, each with a forced reasoning allowance of a*A tokens "
                 "(A = 1,024); diag5 = the five A-level diagnostic draws; c48fit = the E2 class-building scope over "
                 "one C48 item's fitting draws",
    "config": "allocation n x a (n independent calls at a*A reasoning tokens each; A = 1,024)",
    "n": "number of independent calls in the allocation",
    "a": "per-call reasoning allowance in units of A = 1,024 tokens",
    "B": "total nominal reasoning budget n*a (units of A)",
    "R": "rounds per call (always 1: independent calls)",
    "pool_id": "sha256 of (item, config, view, ordered member ids): one evaluated pool",
    "candidate_id": "generation node id (sha256 of every GenerationSpec field); one forced THINK+ANSWER generation",
    "node_id": "generation node id (sha256 of the GenerationSpec of this call)",
    "shard": "provider shard the item is pinned to: mancer (endpoint mancer/fp8) or parasail (parasail/fp8)",
    "predictor": "E5 predictor: P0 (benchmark + budget + configuration), P1 (P0 + five-draw features + provider), "
                 "P2 (mechanism glm: simulated coverage x eta), P3 (ridge on P1 inputs + pairwise products/squares), "
                 "P1-conc / P2-conc (same without the answer-concentration features)",
    "p_delivered": "predicted probability that the allocation's delivered (plurality) answer is correct",
    "p_coverage": "predicted probability that at least one call of the allocation is correct (coverage)",
    "imputed": "true when the item had no diagnostic draws and its features were imputed with development means "
               "(never happened on T200)",
    "fold": "cross-fitting fold whose model produced the row (0-4 = L96 outer fold held out; all_L96 = fit on all "
            "96 L96 items, predicting T200)",
    "model": "E2 model: beta_full_{diffuse,concentrated} (Beta(1/2,1/2) per item-depth cell + Dirichlet answer-class "
             "model), glm_full_{diffuse,concentrated} (pooled glm response + class model), shallow_{diffuse,"
             "concentrated} (A-level draws only + glm depth shift), beta_nocomp (no class competition), ridge "
             "(unstructured ridge); diffuse/concentrated = treatment of the unseen-answer residual",
    "k": "crossover multiplier: compares one call at k*A with k calls at A (k = 2, 4, 8, 16)",
    "arm": "LLM selector arm: A0/B0 = problem only; A1/B1 = problem + the five diagnostic answers",
    "sample": "E8b decision index per (arm, item), 0-4 (K = 5 independent decisions)",
    "selector": "selector name: fixed:<config> (always that option), stat:<P>/<menu> (LEX rule on cross-fitted "
                "predictions), sealed:<P>/<menu> (LEX rule on the sealed T200 predictions), heuristic/<menu> "
                "(diagnostic plurality-share rule), LLM:<arm> (E8a allocator), oracle/<menu> (clairvoyant "
                "ceiling), splithalf/M17 (LEX split-half reference), LLM:B0/B1 (E8b objective-aware LLM "
                "selectors: told the LEX objective, M15 menu, 5 decisions per item)",
    "kind": "row kind (see file notes)",
    "menu": "option menu the selector chooses from: M15 (all n x a <= 16A), M5 (1x16, 2x8, 4x4, 8x2, 16x1), M9sealed "
            "(the 9 sealed E5 configurations), M17 (M15 + 1x25 + 25x1), or a single fixed option",
    "overhead_kind": "latency overhead added to the selector: none; diag (the five diagnostic calls' parallel "
                     "wall-clock); llm (the E8a decision call latency, summed over retries); diag+llm (both)",
    "choice": "the option the selector chose for this item (E8b: the modal option of its 5 decisions, ties -> "
              "menu order)",
    "stability": "E8b: share of the item's decisions that chose the modal option (1.0 = all 5 identical)",
    "n_decisions": "E8b: number of LLM decisions scored (5 per item; summed over items in summary tables)",
    "n_fallback": "E8b: decisions still invalid after one retry, scored as the pre-registered fallback 1x16",
    "n_invalid_fallback": "E8b: decisions still invalid after one retry, scored as the fallback 1x16 (summed)",
    "n_unknown_decisions": "E8b: decisions whose chosen option has an unknown outcome (excluded from the mean)",
    "choices_k": "E8b: the item's decisions in sample order, '|'-separated option labels",
    "choice_stability": "E8b: mean over items of the modal-choice share across the 5 decisions",
    "share_items_all_identical": "E8b: share of items on which all 5 decisions chose the same option",
    "deep_C": "always-1x16 consistency p^2 + (1-p)^2 on the scoring half",
    "deep_latency": "always-1x16 latency (s) on the scoring half",
    "deep25_C": "always-1x25 consistency on the scoring half",
    "deep25_latency": "always-1x25 latency (s) on the scoring half",
    "acc": "delivered accuracy of the chosen option (C48: bank-estimated tie-averaged p; T200: realized S_known)",
    "C": "consistency p^2 + (1-p)^2 of the chosen option (C48 bank estimate; missing on T200: one realization)",
    "lat": "latency (s) of the chosen option: parallel wall-clock of its calls plus the selector overhead",
    "overhead": "selector latency overhead added to this item (s)",
    "solver_A": "solver reasoning compute of the chosen option, n*a (units of A)",
    "latency": "parallel wall-clock latency (s): expected slowest call of the option's n calls",
    "m": "bank size: number of usable graded draws of the item at this allowance",
    "key": "flattened JSON path (dots join nested keys, [i] indexes lists of objects)",
    "value": "value at that path (lists as JSON)",
    "source": "where the value comes from",
    "cost_usd": "OpenRouter cost in US dollars (actual_cost_usd reported with the response)",
    "latency_s": "end-to-end latency of the call in seconds (THINK + ANSWER for a forced generation)",
    "observed_provider": "provider name reported by OpenRouter for the response",
    "observed_model": "model slug reported by OpenRouter for the response",
    "prompt_tokens": "prompt tokens billed for the (THINK) call",
    "completion_tokens": "completion tokens billed (reasoning + visible answer)",
    "reasoning_tokens": "reasoning tokens generated (forced: exactly the THINK allowance)",
    "count": "number of items (or decisions) with this value",
    "n_items": "number of items entering the statistic",
    "benchmark_weight": "estimand weight of the benchmark",
}

#: leaf-name docs for flattened statistics dicts (``passk.stats`` / ``_vendor.ard_stats``)
STAT_LEAF: dict[str, str] = {
    "estimate": "point estimate (estimand-weighted mean of item-level differences)",
    "mean": "estimand-weighted mean over benchmarks of per-benchmark item means",
    "macro": "equal-weight (macro) version of the estimate",
    "se": "bootstrap standard error",
    "t": "studentized statistic estimate/se",
    "p": "two-sided bootstrap p-value",
    "p_holm": "Holm-adjusted p-value within the stated family",
    "reject": "Holm decision at family-wise alpha 0.05",
    "rank": "Holm step (1 = smallest p)",
    "alternative": "alternative hypothesis",
    "null_value": "null value of the contrast",
    "ci95_studentized_lo": "lower end of the 95% studentized bootstrap interval",
    "ci95_studentized_hi": "upper end of the 95% studentized bootstrap interval",
    "ci95_percentile_lo": "lower end of the 95% percentile bootstrap interval",
    "ci95_percentile_hi": "upper end of the 95% percentile bootstrap interval",
    "ci95_lo": "lower end of the 95% percentile bootstrap interval",
    "ci95_hi": "upper end of the 95% percentile bootstrap interval",
    "ci_primary_bonferroni_lo": "lower end of the Bonferroni-simultaneous interval over the family (1-alpha/n_primary)",
    "ci_primary_bonferroni_hi": "upper end of the Bonferroni-simultaneous interval over the family",
    "ci_one_sided_upper_percentile": "one-sided upper 95% percentile bound",
    "ci_one_sided_upper_studentized": "one-sided upper 95% studentized bound",
    "primary_alpha": "per-test alpha used for the simultaneous interval (alpha / n_primary)",
    "components": "number of contrast components (1 = a single paired difference)",
    "n_primary": "size of the test family used for the simultaneous interval",
    "n_resamples": "bootstrap replicates (benchmark-stratified, item-clustered; seed 20260925)",
    "domains": "benchmarks (bootstrap strata) in order",
    "alpha": "family-wise alpha",
    "n_items": "items entering the statistic (both compared cells known)",
    "n_dropped_unknown": "items dropped because a compared cell was unknown",
    "locked": "the locked mixed allocation M*",
    "bounds_best_worst_lo": "best-case bound of the estimate (unknown grades resolved in favour of the mixed arm)",
    "bounds_best_worst_hi": "worst-case bound of the estimate (unknown grades resolved against the mixed arm)",
    "dropped_items_lt3_known": "items dropped because fewer than 3 of the five 16A configurations were known",
    "value": "value",
}
for _b in BENCH:
    STAT_LEAF[f"by_domain_{_b}"] = f"per-benchmark value for {_b} (item mean or mean difference)"
    STAT_LEAF[f"n_{_b}"] = f"number of {_b} items"
    STAT_LEAF[f"weights_{_b}"] = f"estimand weight of {_b}"

PREFIX_WORDS: dict[str, str] = {
    "acc": "delivered accuracy", "C": "consistency p^2+(1-p)^2", "lat": "latency in s (incl. selector overhead)",
    "p": "delivered accuracy p", "latency": "latency (s)", "mixture": "T200 mixture estimand (.10/.10/.50/.30)",
    "macro": "equal-weight macro estimand", "minus_1x16": "selector minus fixed:1x16",
    "acc_mixture": "delivered accuracy, mixture estimand", "acc_macro": "delivered accuracy, macro estimand",
    "lat_macro": "latency (s), macro", "lat_mixture": "latency (s), mixture", "deep_acc": "always-1x16 accuracy",
    "deep_latency": "always-1x16 latency (s)", "acc_minus_deep": "accuracy minus always-1x16",
    "latency_minus_deep": "latency minus always-1x16 (s)",
}


def _prefix_words(prefix: str) -> str:
    """Describe a flattened-key prefix by its longest known leading components (``minus_1x16_acc_macro`` ->
    "selector minus fixed:1x16, delivered accuracy, macro estimand")."""
    if not prefix:
        return ""
    for key in sorted(PREFIX_WORDS, key=len, reverse=True):
        if prefix == key:
            return PREFIX_WORDS[key]
        if prefix.startswith(key + "_"):
            return PREFIX_WORDS[key] + ", " + _prefix_words(prefix[len(key) + 1:])
    head, _, rest = prefix.partition("_")
    return head + (", " + _prefix_words(rest) if rest else "")


def _leaf_doc(col: str) -> str | None:
    """Split ``<prefix>_<stat leaf>`` and describe both halves."""
    for leaf in sorted(STAT_LEAF, key=len, reverse=True):
        if col == leaf:
            return STAT_LEAF[leaf]
        if col.endswith("_" + leaf):
            return f"{_prefix_words(col[: -len(leaf) - 1])}: {STAT_LEAF[leaf]}"
    return None


_METRIC_WORDS = {
    "H_known": "coverage H (some member correct; missing when unknown grades make it undetermined)",
    "S_known": "delivered plurality accuracy S (seeded tie rule; missing when undetermined)",
    "S_tie": "analytic tie-averaged delivered accuracy",
    "S_lo": "delivered accuracy lower bound (unknown = wrong)",
    "S_hi": "delivered accuracy upper bound (unknown = right)",
    "H_lo": "coverage lower bound", "H_hi": "coverage upper bound",
    "entropy": "answer-class entropy over votes (nats)",
    "top_wrong_mass": "share of votes in the largest known-wrong class",
}
_WEIGHT_WORDS = {"macro": "equal benchmark weights", "mixture": "weights .10/.10/.50/.30",
                 "micro": "pooled over items"}
PATTERNS: list[tuple[str, Callable[[re.Match], str]]] = [
    (r"^(H_known|S_known|S_tie|S_lo|S_hi|H_lo|H_hi|entropy|top_wrong_mass)_(macro|mixture|micro)$",
     lambda m: f"{_METRIC_WORDS[m.group(1)]}; {m.group(2)} mean over items ({_WEIGHT_WORDS[m.group(2)]})"),
    (r"^reliability_macro$", lambda m: "reliability S/H: S_known_macro / H_known_macro"),
    (r"^choice_sample(\d)$", lambda m: f"option chosen in decision sample {m.group(1)} (fallback 1x16 when invalid "
                                       "after one retry)"),
    (r"^(aime|hmmt|omni_math_hard|gpqa_diamond)$", lambda m: f"per-benchmark value for {m.group(1)}"),
]


def describe(col: str, explicit: Mapping[str, str] | None) -> str:
    if explicit and col in explicit:
        return explicit[col]
    for pattern, fn in PATTERNS:
        m = re.match(pattern, col)
        if m:
            return fn(m)
    if col in COMMON:
        return COMMON[col]
    doc = _leaf_doc(col)
    if doc:
        return doc
    return "UNDOCUMENTED (meaning not verified from the code)"


# =====================================================================================================================
# the package
# =====================================================================================================================

@dataclass
class TableEntry:
    rel: str
    purpose: str
    unit: str
    rows: int
    columns: list[tuple[str, str, str]]
    source: str
    notes: str = ""


@dataclass
class FileEntry:
    rel: str
    purpose: str
    source: str


@dataclass
class Package:
    study_root: Path
    out: Path
    code_root: Path = CODE_ROOT
    plan_path: Path | None = PLAN_PATH
    legacy_root: Path | None = None
    load_strata: bool = True
    tables: list[TableEntry] = field(default_factory=list)
    files: list[FileEntry] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    checks: list[tuple[str, str]] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- helpers
    def path(self, *parts: str) -> Path:
        return self.study_root.joinpath(*parts)

    def need(self, rel: str) -> Path | None:
        p = self.study_root / rel
        if not p.exists():
            self.gap(f"source missing: {rel}")
            return None
        return p

    def gap(self, message: str) -> None:
        if message not in self.gaps:
            self.gaps.append(message)

    def check(self, name: str, result: str) -> None:
        self.checks.append((name, result))

    def table(self, rel: str, df: pd.DataFrame, *, purpose: str, unit: str, source: str,
              cols: Mapping[str, str] | None = None, notes: str = "") -> None:
        df = intify_counts(df.reset_index(drop=True))
        rows = write_csv(df, self.out / rel)
        columns = [(c, infer_type(df[c]), describe(c, cols)) for c in df.columns]
        for c, _t, meaning in columns:
            if meaning.startswith("UNDOCUMENTED"):
                self.gap(f"{rel}: column {c!r} has no verified meaning")
        self.tables.append(TableEntry(rel, purpose, unit, rows, columns, source, notes))

    def copy(self, src: Path, rel: str, *, purpose: str) -> bool:
        if not Path(src).exists():
            self.gap(f"file to copy missing: {src}")
            return False
        dst = self.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        self.files.append(FileEntry(rel, purpose, str(src)))
        return True

    def text(self, rel: str, content: str, *, purpose: str, source: str) -> None:
        dst = self.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
        self.files.append(FileEntry(rel, purpose, source))

    # ---------------------------------------------------------------- shared inputs
    def splits(self) -> dict:
        if "splits" not in self._cache:
            p = self.need("artifacts/manifests/splits.json")
            self._cache["splits"] = _read_json(p) if p else {}
        return self._cache["splits"]

    def assignment(self) -> dict[str, str]:
        if "assignment" not in self._cache:
            p = self.need("artifacts/shards/assignment.json")
            self._cache["assignment"] = dict(_read_json(p)["assignment"]) if p else {}
        return self._cache["assignment"]

    def draws(self) -> pd.DataFrame | None:
        if "draws" not in self._cache:
            p = self.need("analysis_inputs/draws.parquet")
            self._cache["draws"] = pd.read_parquet(p) if p else None
        return self._cache["draws"]

    def panel_sets(self) -> dict[str, set[str]]:
        return {k: set(v) for k, v in (self.splits().get("panels") or {}).items()}

    def roles(self, item_id: str) -> list[str]:
        sets = self.panel_sets()
        return [p for p in PANELS if item_id in sets.get(p, set())]

    def config(self) -> dict:
        if "config" not in self._cache:
            p = self.code_root / "configs" / "passk_cont_v1.json"
            self._cache["config"] = _read_json(p) if p.exists() else {}
            if not p.exists():
                self.gap(f"config missing: {p}")
        return self._cache["config"]

    def git(self, *args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(self.code_root), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    # =================================================================================================================
    # 00_study
    # =================================================================================================================
    def section_study(self) -> None:
        d = "csv/00_study"
        sp = self.splits()
        panels = sp.get("panels") or {}
        assignment = self.assignment()
        c48 = sp.get("c48") or {}
        strata_of: dict[str, str] = {}
        if self.load_strata and panels:
            try:
                from agent_allocation.benchmarks import text_manifest as tm
                from agent_allocation.passk.splits import load_inventories
                inv = load_inventories(self.config())
                for items in inv.values():
                    for it in items:
                        strata_of[it.public.item_id] = tm._stratum_name(it)
            except Exception as exc:                                    # noqa: BLE001 - offline cache missing
                self.gap(f"benchmark inventories not loadable (stratum columns empty): {exc}")
        items = sorted(set().union(*[set(v) for v in panels.values()])) if panels else []
        order = {p: {i: k for k, i in enumerate(panels.get(p, []))} for p in PANELS}
        cells = c48.get("cells") or {}
        strata_tab = c48.get("strata") or {}
        rows = []
        for item in items:
            bench = item.split("/")[0]
            cell = cells.get(item)
            size_sel = (strata_tab.get(bench) or {}).get(cell) if cell is not None else None
            row = {"item_id": item, "benchmark": bench}
            for p in PANELS:
                row[f"in_{p}"] = item in order[p]
            row["provider_shard"] = assignment.get(item) or (sp.get("assignment_t200") or {}).get(item)
            for p in ("L96", "D32", "C48", "T200", "X80"):
                row[f"{p.lower()}_order"] = order[p].get(item)
            row["stratum"] = strata_of.get(item)
            row["t200_stratum"] = strata_of.get(item) if item in order["T200"] else None
            bank = (c48.get("bank_correct") or {}).get(item, None) if item in order["L96"] else None
            row["legacy_bank_correct_of_25"] = bank
            row["c48_eligible"] = item in set(c48.get("eligible") or []) if item in order["L96"] else None
            row["c48_bank_unknown_excluded"] = (item in set(c48.get("excluded_unknown") or [])
                                                if item in order["L96"] else None)
            row["c48_stratum"] = cell
            row["c48_cell_size"] = None if size_sel is None else size_sel.get("size")
            row["c48_cell_selected"] = None if size_sel is None else size_sel.get("selected")
            row["c48_inclusion_prob"] = (c48.get("inclusion_prob") or {}).get(item)
            rows.append(row)
        frame = pd.DataFrame(rows)
        for c in ("l96_order", "d32_order", "c48_order", "t200_order", "x80_order", "legacy_bank_correct_of_25",
                  "c48_cell_size", "c48_cell_selected"):
            if c in frame:
                frame[c] = frame[c].astype("Int64")
        t200_assign = sp.get("assignment_t200") or {}
        mism = [i for i, s in t200_assign.items() if assignment and assignment.get(i) != s]
        self.check("T200 shard assignment: splits.json == shards/assignment.json",
                   "ok" if not mism else f"{len(mism)} mismatches")
        if strata_of and sp.get("t200_strata"):
            bad = []
            for b, info in sp["t200_strata"].items():
                got = frame[frame.in_T200 & (frame.benchmark == b)].t200_stratum.value_counts().to_dict()
                if got != info.get("t200_strata"):
                    bad.append(b)
            self.check("per-item T200 strata reproduce splits.json t200_strata counts",
                       "ok" if not bad else f"mismatch for {bad}")
        cols = {
            **{f"in_{p}": f"item is in panel {p}" for p in PANELS},
            "provider_shard": "provider shard the item is pinned to (artifacts/shards/assignment.json; L96/D32 keep "
                              "the legacy forced_v2 assignment, T200 the seeded 1:1 assignment)",
            "l96_order": "position in the L96 list (legacy main order)",
            "d32_order": "position in the D32 list (legacy dev order)",
            "c48_order": "position in the C48 list (legacy main order)",
            "t200_order": "position in the T200 list (frozen seeded rank order)",
            "x80_order": "position in the X80 list (frozen seeded order; X80 is frozen but unused)",
            "stratum": "the benchmark's stratum label (text_manifest._STRATUM_RULE: aime year; hmmt year x "
                       "problem_type; omni_math_hard domain; gpqa_diamond domain) from the pinned inventory",
            "t200_stratum": "stratum used for the T200 stratified draw (T200 items only)",
            "legacy_bank_correct_of_25": "correct count among the item's 25 legacy forced-A roots (C48 strata "
                                         "input); empty for the 2 items with unknown bank grades",
            "c48_eligible": "L96 item eligible for C48 (bank-known)",
            "c48_bank_unknown_excluded": "L96 item excluded from C48 because a bank grade is unknown",
            "c48_stratum": "C48 shallow-bank bin: 0, 1-5, 6-15 or 16-25 correct of 25",
            "c48_cell_size": "eligible items in the item's (benchmark, bin) cell",
            "c48_cell_selected": "items selected into C48 from that cell",
            "c48_inclusion_prob": "recorded inclusion probability selected/size (C48 items only)",
        }
        self.table(f"{d}/splits_items.csv", frame, purpose="Every item used anywhere (L96 u D32 u T200) with its "
                   "panel memberships, provider shard, panel positions, strata and C48 sampling design.",
                   unit="one item", source="artifacts/manifests/splits.json, artifacts/shards/assignment.json, "
                   "pinned benchmark inventories (strata)", cols=cols)
        # C48 strata
        srows = [{"benchmark": b, "bin": bin_, "size": v.get("size"), "selected": v.get("selected"),
                  "target_per_cell": c48.get("target_per_cell")}
                 for b in BENCH for bin_, v in sorted((strata_tab.get(b) or {}).items(),
                                                      key=lambda kv: ["0", "1-5", "6-15", "16-25"].index(kv[0])
                                                      if kv[0] in ["0", "1-5", "6-15", "16-25"] else 9)]
        self.table(f"{d}/splits_c48_strata.csv", pd.DataFrame(srows), purpose="C48 stratified design: eligible "
                   "and selected items per (benchmark, shallow-bank bin).", unit="one (benchmark, bin) cell",
                   source="splits.json c48.strata",
                   cols={"bin": "shallow-bank bin (correct of 25 legacy forced-A roots)", "size": "eligible items",
                         "selected": "items selected into C48", "target_per_cell": "target items per cell (3)"})
        trows = []
        for b in BENCH:
            info = (sp.get("t200_strata") or {}).get(b) or {}
            for name in sorted(set(info.get("pool_strata") or {}) | set(info.get("t200_strata") or {})):
                trows.append({"benchmark": b, "rule": info.get("rule"), "stratum": name,
                              "pool_count": (info.get("pool_strata") or {}).get(name),
                              "t200_count": (info.get("t200_strata") or {}).get(name),
                              "x80_count": (info.get("x80_strata") or {}).get(name),
                              "benchmark_eligible": info.get("eligible"), "benchmark_pool": info.get("pool"),
                              "legacy_reserve_in_t200": info.get("legacy_reserve_in_t200")})
        tf = pd.DataFrame(trows)
        for c in ("pool_count", "t200_count", "x80_count"):
            if c in tf:
                tf[c] = tf[c].astype("Int64")
        self.table(f"{d}/splits_t200_strata.csv", tf, purpose="T200/X80 stratified draw: pool and drawn counts per "
                   "stratum.", unit="one (benchmark, stratum)", source="splits.json t200_strata",
                   cols={"rule": "stratum rule of the benchmark", "stratum": "stratum label",
                         "pool_count": "unused eligible items in the stratum (after dedup/exclusions)",
                         "t200_count": "items drawn into T200", "x80_count": "items drawn into X80 (subset of T200)",
                         "benchmark_eligible": "eligible items of the benchmark",
                         "benchmark_pool": "unused pool size of the benchmark",
                         "legacy_reserve_in_t200": "legacy forced_v2 reserve items that landed in T200 (JSON list)"})
        drows = []
        for item, reason in sorted((sp.get("duplicate_exclusions") or {}).items()):
            kind, _, partner = str(reason).partition(":")
            drows.append({"item_id": item, "reason": kind, "partner_item_id": partner or None})
        self.table(f"{d}/splits_duplicate_exclusions.csv", pd.DataFrame(drows, columns=["item_id", "reason",
                   "partner_item_id"]), purpose="Items excluded from the T200 pool by the frozen duplicate registry.",
                   unit="one excluded item", source="splits.json duplicate_exclusions",
                   cols={"reason": "partner_of_used (duplicate of a legacy main/dev item), duplicate_of (another "
                                   "member kept as representative) or conflicting_gold (duplicates with different "
                                   "golds; all dropped)", "partner_item_id": "the duplicate partner"})
        self._study_facts(d)
        self._gates(d)
        num = self.code_root / "docs" / "passk_cont_report" / "numbers.json"
        self.copy(num, f"{d}/report_numbers.source.json", purpose="the report's numbers.json (every number in "
                  "report.pdf), copied verbatim")

    def _study_facts(self, d: str) -> None:
        cfg = self.config()
        facts: list[tuple[str, Any, str]] = []

        def add(key: str, value: Any, source: str) -> None:
            facts.append((key, value, source))
        cf = "configs/passk_cont_v1.json"
        for key in ("study_id", "preset", "primary_model", "seed"):
            add(f"config.{key}", cfg.get(key), cf)
        for sec in ("reasoning", "sampling", "provider", "forced", "budget", "budget_by_shard", "scheduler_by_shard",
                    "passk", "statistics"):
            for k, v in flatten_kv(cfg.get(sec) or {}, sec):
                if k.endswith(".note") or k.startswith("passk.plan"):
                    continue
                add(f"config.{k}", v, cf)
        add("config.shards", cfg.get("shards"), cf)
        add("A (tokens per unit of reasoning allowance)", A_TOKENS, cf + " reasoning.main_a")
        add("allowances used (units of A)", [1, 2, 4, 8, 16, 25], "00_contract.md; 25A only in legacy deep/25 and "
            "the LEX c48_25a partition")
        try:
            from agent_allocation.api.forced import GENERATION_PROMPT, TEMPLATE_SHA256
            add("forced.generation_prompt_suffix", GENERATION_PROMPT, "api/forced.py GENERATION_PROMPT")
            add("forced.template_sha256", TEMPLATE_SHA256.get("qwen3.8-27b"), "api/forced.py TEMPLATE_SHA256")
        except Exception as exc:                                        # noqa: BLE001
            self.gap(f"api.forced constants not importable: {exc}")
        g0 = self.path("gates", "gate0.json")
        if g0.exists():
            gate0 = _read_json(g0)
            crit = gate0.get("criteria") or {}
            add("gate0.code_head", (gate0.get("code") or {}).get("head"), "gates/gate0.json")
            add("gate0.forecast_total_usd", (crit.get("forecast") or {}).get("total_usd"), "gates/gate0.json")
            add("gate0.forecast_by_shard_usd", (crit.get("forecast") or {}).get("by_shard"), "gates/gate0.json")
            add("gate0.key_limit_remaining_usd", (crit.get("key") or {}).get("limit_remaining"), "gates/gate0.json")
            add("gate0.planned_new_nodes", (crit.get("node_ids") or {}).get("planned"), "gates/gate0.json")
            add("gate0.planned_counts_by_partition", (crit.get("planned_counts") or {}).get("counts"),
                "gates/gate0.json")
            pd_ = crit.get("prompt_and_decoding") or {}
            add("prompt_version", pd_.get("prompt_version"), "gates/gate0.json")
            add("decoding_config_hash", pd_.get("decoding_config_hash"), "gates/gate0.json (== forced_v2)")
            add("splits_sha256", (crit.get("gate_s") or {}).get("splits_sha256"), "gates/gate0.json")
        rm = self.path("records", "run_manifest.json")
        if rm.exists():
            man = _read_json(rm)
            for k in ("model", "permaslug", "providers", "prompt_version", "passk_seed", "cfg_seed", "A",
                      "n_candidates", "code_commit", "created_at"):
                add(f"records.run_manifest.{k}", man.get(k), "records/run_manifest.json")
        seal = self.path("sealed", "SEAL_predictions.json")
        if seal.exists():
            s = _read_json(seal)
            for k in ("code_commit", "reviewed_commit", "created_at", "splits"):
                if k in s:
                    add(f"seal.{k}", s.get(k), "sealed/SEAL_predictions.json")
            add("seal.n_artifacts_hashed", len(s.get("artifacts") or {}), "sealed/SEAL_predictions.json")
        ai = self.path("analysis_inputs", "manifest.json")
        if ai.exists():
            m = _read_json(ai)
            for k in ("created_at", "code_commit", "releases", "labelled_partitions", "n_rows", "n_rows_by_partition"):
                add(f"analysis_inputs.manifest.{k}", m.get(k), "analysis_inputs/manifest.json")
        draws = self.draws()
        if draws is not None:
            for col in ("decoding_config_hash", "prompt_version", "grader_version", "observed_model", "think_model",
                        "answer_model", "completion_headroom", "provider_only", "observed_provider"):
                if col in draws:
                    vc = draws[col].astype(object).where(draws[col].notna(), None).value_counts(dropna=False)
                    add(f"draws.distinct.{col}", {str(k): int(v) for k, v in vc.items()},
                        "analysis_inputs/draws.parquet (value -> rows)")
            if "endpoint_lock_hash" in draws:
                lk = draws.groupby("shard")["endpoint_lock_hash"].agg(lambda s: sorted(set(s.dropna())))
                add("draws.endpoint_lock_hash_by_shard", {k: v for k, v in lk.items()},
                    "analysis_inputs/draws.parquet (adopted forced_v2 endpoint locks)")
            exact = draws[~draws.absent.astype(bool)]
            ok = bool((exact.think_out == exact.allowance_a * A_TOKENS).all())
            add("draws.every_present_think_exactly_its_allowance", ok, "analysis_inputs/draws.parquet think_out")
        sp = self.splits()
        for k in ("study_id", "version", "passk_seed", "legacy_seed"):
            add(f"splits.{k}", sp.get(k), "artifacts/manifests/splits.json")
        for k, v in flatten_kv(sp.get("salts") or {}, "splits.salts"):
            add(k, v, "artifacts/manifests/splits.json")
        for k in ("L96", "C48", "V48", "T200", "D32", "X80"):
            add(f"panel_size.{k}", len((sp.get("panels") or {}).get(k, [])), "artifacts/manifests/splits.json")
        # ledger generation window
        for shard in ("mancer", "parasail"):
            lj = self.path("artifacts", "shards", shard, "state", "ledger_journal.jsonl")
            if lj.exists():
                times = [e["recorded_at"] for e in _jsonl(lj) if e.get("kind") == "settle"]
                if times:
                    add(f"ledger.{shard}.first_settle_utc", min(times), "shard ledger_journal.jsonl")
                    add(f"ledger.{shard}.last_settle_utc", max(times), "shard ledger_journal.jsonl")
        for label, when in MEMORY_NOTE_EVENTS:
            add(f"timeline.{label}", when, "memory note passk-cont-run.md (approximate, local EDT)")
        for label, short in KEY_COMMITS:
            full = self.git("rev-parse", "--verify", "--quiet", f"{short}^{{commit}}")
            add(f"commit.{label}", full or f"{short} (not found in this repository)", "git log (memory note)")
        add("commit.worktree HEAD at export", self.git("rev-parse", "HEAD"), "git rev-parse HEAD")
        add("branch", self.git("rev-parse", "--abbrev-ref", "HEAD"), "git")
        add("legacy run root (read-only)", str(self.legacy_root or "/orcd/data/tpoggio/001/mabdel03/"
                                               "agents_scaling_results/forced_budget_v2/artifacts"), "plan")
        frame = pd.DataFrame([{"key": k, "value": _kv_value(v) if not isinstance(v, (dict, list)) else
                               json.dumps(_native(v), sort_keys=True), "source": s} for k, v, s in facts])
        self.table(f"{d}/study_facts.csv", frame, purpose="Frozen protocol parameters, seeds, hashes, commits and "
                   "timeline of the study (key/value).", unit="one fact", source="config, gates, records, seal, "
                   "analysis_inputs manifest, draws, ledgers, git, memory note",
                   cols={"key": "fact name (config.* = configs/passk_cont_v1.json; draws.distinct.* = value -> row "
                                "counts in draws.parquet; commit.* = full sha)", "value": "value (JSON for lists/"
                         "objects)", "source": "file or record the fact was read from"})

    def _gates(self, d: str) -> None:
        gdir = self.path("gates")
        if not gdir.is_dir():
            self.gap("source missing: gates/")
            return
        rows, detail = [], []
        for p in sorted(gdir.glob("*.json")):
            g = _read_json(p)
            crit = g.get("criteria") or g.get("checks") or {}
            passed = {}
            if isinstance(crit, dict):
                for name, v in crit.items():
                    if isinstance(v, dict) and "passed" in v:
                        passed[name] = bool(v["passed"])
            elif isinstance(crit, list):
                for v in crit:
                    if isinstance(v, dict) and "name" in v:
                        passed[str(v["name"])] = bool(v.get("passed"))
            stem = p.stem
            shard = next((s for s in ("mancer", "parasail") if s in stem), None)
            split = next((s for s in ("v48", "t200") if stem.endswith(s)), None)
            rows.append({"gate_file": p.name, "gate": g.get("gate", stem), "shard": shard, "split": split,
                         "passed": g.get("passed"), "complete": g.get("complete"),
                         "evaluated_at": g.get("evaluated_at") or g.get("reviewed_at") or g.get("created_at"),
                         "n_criteria": len(passed), "n_criteria_failed": sum(not v for v in passed.values()),
                         "criteria_failed": [k for k, v in passed.items() if not v], "criteria": passed})
            for k, v in flatten_kv(g):
                detail.append({"gate_file": p.name, "key": k, "value": _kv_value(v)})
        self.table(f"{d}/gates.csv", pd.DataFrame(rows), purpose="Every recorded gate (Gate S, 0, 1 smoke, 2, 3, 3 "
                   "permaslug sample, 4 per split, graded, E0 gate, analysis review): pass/fail and per-criterion "
                   "results.", unit="one gate record file", source="gates/*.json",
                   cols={"gate_file": "file under gates/", "gate": "gate name", "shard": "provider shard (per-shard "
                         "gates)", "split": "held-out split (Gate 4 per split)", "passed": "gate passed (empty = "
                         "the record has no passed field, e.g. analysis_review)", "complete": "all criteria evaluated",
                         "evaluated_at": "UTC time recorded in the gate file", "n_criteria": "criteria with a "
                         "passed flag", "n_criteria_failed": "criteria that failed", "criteria_failed": "names of "
                         "failed criteria (JSON)", "criteria": "criterion -> passed (JSON)"})
        self.table(f"{d}/gates_detail.csv", pd.DataFrame(detail), purpose="Every scalar field of every gate record "
                   "(thresholds, counts, hashes, forecasts).", unit="one (gate file, JSON path)",
                   source="gates/*.json", cols={"gate_file": "file under gates/"})

    # =================================================================================================================
    # 01_e0_audit
    # =================================================================================================================
    def section_e0(self) -> None:
        d = "csv/01_e0_audit"
        e0 = self.path("e0")
        if not e0.is_dir():
            self.gap("source missing: e0/")
            return
        lm = e0 / "legacy_metrics.csv"
        if lm.exists():
            self.table(f"{d}/legacy_metrics.csv", pd.read_csv(lm), purpose="Every legacy forced_v2 headline "
                       "recomputed from one library under the 96/94/93-item cohorts (legacy tie mode).",
                       unit="one (cohort, panel, configuration)", source="e0/legacy_metrics.csv",
                       cols={"cohort": "item cohort: 96 = all L96; 94 = drop the 2 items with unknown bank grades; "
                                       "93 = also drop items with any unknown main-partition grade",
                             "n_items": "items in the cohort with this configuration"},
                       notes="config_id 25x1 = legacy vote@25 and 1x25 = legacy deep/25 (legacy-only readouts).")
        ex = e0 / "exclusions.csv"
        if ex.exists():
            exf = pd.read_csv(ex, keep_default_na=False, na_values=[""])
            self.table(f"{d}/exclusions.csv", exf, purpose="Cohort membership and exclusion reasons of every L96 "
                       "item.", unit="one L96 item", source="e0/exclusions.csv",
                       cols={"in_96": "in the 96 cohort (all)", "in_94": "no unknown grade in the item's root/deep "
                             "bank", "in_93": "no unknown grade anywhere in the legacy main partition",
                             "bank_unknown_grades": "unknown grades among the item's bank draws",
                             "main_unknown_grades": "unknown grades among all legacy main-partition candidates",
                             "unknown_codes": "grader detail codes of the unknowns (5 s Math-Verify timeouts)",
                             "reason": "why the item leaves a cohort"})
            rows = []
            for cohort, col in (("96", "in_96"), ("94", "in_94"), ("93", "in_93")):
                sub = exf[exf[col].astype(bool)]
                for b in BENCH + ("all",):
                    rows.append({"cohort": cohort, "benchmark": b,
                                 "n_items": int(len(sub) if b == "all" else (sub.benchmark == b).sum())})
            self.table(f"{d}/cohort_summary.csv", pd.DataFrame(rows), purpose="Item counts of the 96/94/93 "
                       "cohorts by benchmark.", unit="one (cohort, benchmark)", source="derived from e0/exclusions.csv",
                       cols={"cohort": "96, 94 or 93 (see exclusions.csv)", "benchmark": B_DOC + " or all"})
        rp = e0 / "legacy_replay.parquet"
        replay_cols = {"policy_id": "legacy forced_v2 policy checkpoint id (main/<item>/<name>/current)",
                       "name": "legacy policy name (sample/k or deep/m)", "family": "sample or deep",
                       "selected_candidate_id": "candidate the legacy sealed run delivered",
                       "y": "legacy sealed delivered correctness (1/0)",
                       "delivered_state": "legacy sealed delivered state",
                       "delivered_id": "candidate delivered by this study's replay (legacy tie key)",
                       "delivered_correct": "correctness of the replayed delivered candidate",
                       "equivalence_unknown": "Math-Verify pair verdicts that were unknown in the replay's class build",
                       "n_tied": "modal classes tied in the replay", "status": "single | selected | abstained",
                       "match": "replayed delivered id == legacy delivered id",
                       "drift_eligible": "the replay met an unknown equivalence, so the legacy memoized verdict could "
                                         "differ (exempt from the 100% match gate)"}
        if rp.exists():
            self.table(f"{d}/legacy_replay.csv", pd.read_parquet(rp), purpose="Legacy tie replay: every legacy "
                       "sample/deep checkpoint re-read from the pair cache under the legacy tie key vs the sealed "
                       "forced_v2 selection.", unit="one legacy policy checkpoint", source="e0/legacy_replay.parquet",
                       cols=replay_cols)
        dr = e0 / "legacy_replay_drift.csv"
        if dr.exists():
            self.table(f"{d}/legacy_replay_drift.csv", pd.read_csv(dr), purpose="The drift-eligible replay "
                       "checkpoints (unknown equivalence during the class build).", unit="one checkpoint",
                       source="e0/legacy_replay_drift.csv", cols=replay_cols)
        ap = e0 / "audit.json"
        if ap.exists():
            audit = _read_json(ap)
            self.table(f"{d}/tie_replay_summary.csv", kv_frame(audit.get("replay") or {}), purpose="Tie replay "
                       "summary (checkpoints, match rates, drift).", unit="one statistic",
                       source="e0/audit.json replay",
                       cols={"key": "statistic: checkpoints, match_rate, clean_checkpoints (no unknown equivalence), "
                             "clean_match_rate, drift_eligible, drift_mismatches, gate_100pct_clean"})
            self.table(f"{d}/audit.csv", kv_frame(audit), purpose="E0 audit.json flattened: identity, model/provider "
                       "pinning, exact budgets, cohort sizes, headline reproductions, replay.", unit="one JSON leaf",
                       source="e0/audit.json")
            self.copy(ap, f"{d}/audit.source.json", purpose="e0/audit.json verbatim")
        tr = e0 / "correct_count_trajectories.csv"
        if tr.exists():
            self.table(f"{d}/correct_count_trajectories.csv", pd.read_csv(tr), purpose="Per L96 item: correct/"
                       "unknown counts among the first k legacy A roots and the legacy-tie sample/k plurality outcome "
                       "(k = 1..25), plus each deep/m draw.", unit="one (item, family, k)",
                       source="e0/correct_count_trajectories.csv",
                       cols={"family": "sample (k roots at A) or deep (one call at k*A)",
                             "k": "sample: number of roots; deep: allowance multiple",
                             "c_lo": "correct draws among them", "c_unknown": "draws with unknown grade",
                             "plurality_correct": "sample/k plurality (legacy tie) delivered correct; deep: the draw's "
                                                  "correctness"})
        for name, purpose, unit, cols in (
                ("legacy_physical_ledger.csv", "Legacy bank usage: each legacy request counted once.",
                 "one (source, partition, shard, allowance)",
                 {"source": "legacy", "partition": "legacy_main or legacy_pilot",
                  "allowance_a": "reasoning allowance (units of A)", "think_in": "THINK prompt tokens (sum)",
                  "think_out": "THINK completion tokens (sum)", "answer_in": "ANSWER prompt tokens (sum)",
                  "answer_out": "ANSWER completion tokens (sum)", "cost_usd": "cost (sum, USD)",
                  "latency_s": "latency (sum of per-call seconds)", "requests": "number of generations",
                  "nominal_think_tokens": "sum of allowance_a * 1,024"}),
                ("legacy_logical_ledger_by_config.csv", "Legacy logical ledger: mean per-pool usage by "
                 "configuration (every member charged to every pool it serves).", "one configuration",
                 {"nominal_C_tokens": "nominal compute n*R*a*1,024 tokens", "cost_usd": "mean pool cost (USD)",
                  "think_out": "mean pool THINK tokens", "answer_out": "mean pool ANSWER tokens"})):
            p = e0 / name
            if p.exists():
                self.table(f"{d}/{name}", pd.read_csv(p), purpose=purpose, unit=unit, source=f"e0/{name}", cols=cols)
        pv = e0 / "provenance.parquet"
        if pv.exists():
            prov = pd.read_parquet(pv).drop(columns=["raw_path"], errors="ignore")
            self.table(f"{d}/provenance.csv", prov, purpose="One row per legacy HTTP call (THINK and ANSWER of every "
                       "legacy bank node): request hash, generation id, provider, served model, tokens, cost.",
                       unit="one HTTP call (candidate x purpose)", source="e0/provenance.parquet (raw_path dropped)",
                       cols={"purpose": "think or answer", "request_sha": "sha256 of the request body (THINK without "
                             "max_tokens)", "generation_id": "OpenRouter generation id", "provider": "provider name "
                             "reported", "provider_only": "provider.only pin sent", "served_model": "model served",
                             "status": "completed or absent", "tokens_in": "prompt tokens", "tokens_out": "completion "
                             "tokens", "cost_usd_node": "node cost in USD, attached to the THINK row only (the ANSWER "
                             "row is empty)", "latency_s_node": "node end-to-end latency (s), repeated on both rows",
                             "raw_sha256": "sha256 of the stored raw response", "allowance_a": "reasoning allowance "
                             "(units of A)", "replicate": "root slot (A) or 0 (deep)", "partition": "legacy partition",
                             "kind": "root (A) or deep (>= 2A)"})
        cc = e0 / "candidate_catalog.parquet"
        if cc.exists():
            cat = pd.read_parquet(cc, columns=["candidate_id", "item_id", "partition", "kind", "allowance_a",
                                               "replicate", "n_policy_memberships", "policy_memberships"])
            self.table(f"{d}/candidate_policy_memberships.csv", cat, purpose="Legacy candidates and the forced_v2 "
                       "policy checkpoints (logical pools) each one serves.", unit="one legacy candidate",
                       source="e0/candidate_catalog.parquet (other columns duplicate 02_draws/draws_all.csv)",
                       cols={"n_policy_memberships": "number of legacy policy checkpoints using the candidate",
                             "policy_memberships": "legacy policy ids (JSON list) from manifests/policies/*.jsonl",
                             "kind": "root or deep", "allowance_a": "reasoning allowance (units of A)",
                             "replicate": "root slot (A) or 0 (deep)", "partition": "legacy partition"})
        rm = e0 / "reuse_manifest.json"
        if rm.exists():
            man = _read_json(rm)
            ent = pd.DataFrame(man.get("entries") or [])
            self.table(f"{d}/reuse_manifest_entries.csv", ent, purpose="Reuse manifest: the identity key under which "
                       "each legacy draw is reused in this study (model + permaslug, provider tag, item, THINK body "
                       "sha without max_tokens, templates, decoding, allowance, headroom, stopping, grader, split "
                       "role).", unit="one reused legacy candidate", source="e0/reuse_manifest.json entries",
                       cols={"model": "served model", "permaslug": "model permaslug (verified on sampled /generation "
                             "records)", "provider_tag": "provider.only pin", "think_body_sha": "sha256 of the THINK "
                             "request body without max_tokens", "prompt_version": "template pack version",
                             "decoding_config_hash": "decoding configuration hash", "allowance_a": "reasoning "
                             "allowance (units of A)", "headroom": "ANSWER max_tokens", "stopping": "stopping mode "
                             "(forced_think_exact)", "parser_grader": "grader/parser version", "split_role": "panels "
                             "the item belongs to", "partition": "legacy partition", "reusable_as": "role in this "
                             "study (L96 A slot rK, or deep allowance)"},
                       notes="Keyed on: " + ", ".join(man.get("key_fields") or []) + "; not keyed on: "
                             + ", ".join(man.get("not_keyed_on") or []) + ".")
        self._e0_plot_data(d)

    def _e0_plot_data(self, d: str) -> None:
        """Recompute the three E0 decomposition plots' bar values (legacy tie mode, known-outcome pools, macro)."""
        draws = self.draws()
        if draws is None:
            return
        try:
            from agent_allocation.passk.e0.run import LEGACY_CONFIGS
            from agent_allocation.passk.metrics import MACRO_WEIGHTS, pool_metrics, weighted_mean
            from agent_allocation.passk.pairs import PairCache
            from agent_allocation.passk.pools import build_pools
            from agent_allocation.passk.readout import readout_pools
        except Exception as exc:                                        # noqa: BLE001
            self.gap(f"E0 plot data not recomputed (imports failed: {exc})")
            return
        audit_p = self.path("e0", "audit.json")
        cache_path = None
        if audit_p.exists():
            cp = Path(_read_json(audit_p).get("pair_cache_path") or "")
            cache_path = cp if cp.exists() else None
        cache_path = cache_path or self.path("catalog", "pair_verdicts.parquet")
        if not Path(cache_path).exists():
            self.gap("E0 plot data not recomputed: no pair cache")
            return
        legacy = draws[draws.source == "legacy"].copy()
        try:
            pools = build_pools(legacy, panels=("L96",), configs=LEGACY_CONFIGS)
            metrics = pool_metrics(readout_pools(pools, legacy, cache=PairCache(cache_path), mode="legacy"), legacy)
        except Exception as exc:                                        # noqa: BLE001
            self.gap(f"E0 plot data not recomputed: {exc}")
            return
        known = metrics.dropna(subset=["H_known", "S_known"])
        plots = {"decomposition_sample_k.png": ("1x1", "2x1", "4x1", "8x1", "16x1", "25x1"),
                 "decomposition_deep_m.png": ("1x1", "1x2", "1x4", "1x8", "1x16", "1x25"),
                 "decomposition_b16_by_benchmark.png": ("16x1", "1x16")}
        rows = []
        for config_id in LEGACY_CONFIGS:
            sub = known[known.config_id == config_id]
            g = sub.groupby("benchmark")[["gen_fail", "sel_fail", "S_known"]].mean()
            in_plot = [p for p, cs in plots.items() if config_id in cs]
            for scope in ("macro",) + BENCH:
                if scope == "macro":
                    vals = {k: weighted_mean(g[k].to_dict(), MACRO_WEIGHTS) for k in ("gen_fail", "sel_fail",
                                                                                      "S_known")}
                    n = len(sub)
                    plot_scope = [p for p in in_plot if p != "decomposition_b16_by_benchmark.png"]
                else:
                    if scope not in g.index:
                        continue
                    vals = g.loc[scope].to_dict()
                    n = int((sub.benchmark == scope).sum())
                    plot_scope = [p for p in in_plot if p == "decomposition_b16_by_benchmark.png"]
                rows.append({"config_id": config_id, "scope": scope, "S_known": vals["S_known"],
                             "sel_fail": vals["sel_fail"], "gen_fail": vals["gen_fail"], "n_pools": n,
                             "in_plots": plot_scope})
        self.table(f"{d}/decomposition_plot_data.csv", pd.DataFrame(rows), purpose="Bar values of the three E0 "
                   "decomposition plots: delivered accuracy S, selection failure H-S and generation failure 1-H per "
                   "legacy configuration (known-outcome pools, legacy tie mode).", unit="one (configuration, scope)",
                   source=f"recomputed from analysis_inputs/draws.parquet (legacy rows) with pair cache {cache_path}",
                   cols={"scope": "macro (equal benchmark weights over per-benchmark means) or one benchmark",
                         "S_known": "delivered plurality accuracy (legacy tie key)", "sel_fail": "selection failure "
                         "H - S (covered but not delivered)", "gen_fail": "generation failure 1 - H (no member "
                         "correct)", "n_pools": "known-outcome pools (items) averaged", "in_plots": "figures that "
                         "draw this row (JSON list; figures/e0/*.png)"})
        lm = self.path("e0", "legacy_metrics.csv")
        if lm.exists():
            ref = pd.read_csv(lm)
            ref = ref[(ref.cohort.astype(str) == "96")].set_index("config_id")["S_known_macro"]
            got = {c: weighted_mean(metrics[metrics.config_id == c].groupby("benchmark").S_known.mean().to_dict(),
                                    MACRO_WEIGHTS) for c in ref.index if c in set(metrics.config_id)}
            diff = max((abs(got[c] - ref[c]) for c in got), default=float("nan"))
            self.check("E0 plot recomputation reproduces e0/legacy_metrics.csv S_known_macro (cohort 96)",
                       f"max |diff| = {diff:.3g}")

    # =================================================================================================================
    # 02_draws
    # =================================================================================================================
    DRAW_COLS = ["candidate_id", "source_table", "source", "partition", "draw_role", "panel_roles", "item_id",
                 "benchmark", "shard", "provider_only", "observed_provider", "observed_model", "kind", "allowance_a",
                 "replicate", "completion_headroom", "absent", "pending", "graded", "parse_status", "final_present",
                 "vote_exact_key", "vote_text", "option", "answer_len_chars", "answer_finish_reason",
                 "answer_failures", "correct", "grade_state", "think_latency_s", "latency_s", "think_in", "think_out",
                 "answer_in", "answer_out", "cost_usd", "generation_id_think", "generation_id_answer", "think_model",
                 "think_provider", "answer_model"]

    def c48_25a_rows(self) -> pd.DataFrame | None:
        """The LEX 25A draws in the draws.parquet schema (vote keys re-derived with the harness's own extractor)."""
        if "c48_25a" in self._cache:
            return self._cache["c48_25a"]
        dp, cp = self.path("lex", "c48_25a_draws.parquet"), self.path("lex", "c48_25a_calls.jsonl")
        if not dp.exists() or not cp.exists():
            self.gap("source missing: lex/c48_25a_draws.parquet or lex/c48_25a_calls.jsonl")
            self._cache["c48_25a"] = None
            return None
        graded = pd.read_parquet(dp)
        calls = {(r["item_id"], r["replicate"]): r for r in _jsonl(cp) if r.get("transport_status") == "ok"}
        from agent_allocation.contracts import GenerationResult, GenerationSpec, OutputKind, load_record
        from agent_allocation.extraction.final import extract_candidate
        from agent_allocation.passk.extract import _raw_facts, _usage, load_public_tasks
        from agent_allocation.selection.keys import option_letters, vote_key
        tasks = load_public_tasks(self.path("artifacts"))
        rows = []
        for g in graded.itertuples(index=False):
            call = calls.get((g.item_id, int(g.replicate)))
            row = {c: None for c in self.DRAW_COLS}
            row.update(candidate_id=g.candidate_id, source_table="lex/c48_25a_draws", source="passk",
                       partition="c48_25a", draw_role="c48_25a", panel_roles=self.roles(g.item_id), item_id=g.item_id,
                       benchmark=g.benchmark, shard=g.shard, observed_provider=g.observed_provider, kind="deep",
                       allowance_a=int(g.allowance_a), replicate=int(g.replicate), absent=False, pending=False,
                       graded=True, parse_status=g.parse_status, latency_s=float(g.latency_s),
                       cost_usd=float(g.cost_usd))
            corr = None if _is_missing(g.correct) else bool(g.correct)
            row["correct"] = corr
            row["grade_state"] = ("invalid" if g.parse_status != "valid" or g.grade_status == "invalid_output"
                                  else "unknown" if corr is None else "correct" if corr else "wrong")
            if call is not None:
                spec = load_record(GenerationSpec, call["spec"])
                res = load_record(GenerationResult, call["result"])
                row["completion_headroom"] = spec.completion_headroom
                row["observed_model"] = res.observed_model
                row["generation_id_answer"] = res.generation_id
                row.update(_usage(call["result"]))
                row["cost_usd"] = float(g.cost_usd)
                task = tasks.get(g.item_id)
                if task is not None:
                    cand = extract_candidate(res, spec, task)
                    row["final_present"] = cand.final_answer is not None
                    row["answer_finish_reason"] = cand.finish_reason
                    row["answer_len_chars"] = None if cand.visible_content is None else len(cand.visible_content)
                    letters = option_letters(task) if task.output_kind is OutputKind.OPTION else None
                    key = vote_key(cand, task, letters=letters)
                    if key is not None:
                        row["vote_text"], row["vote_exact_key"] = key.text, key.exact
                        if task.output_kind is OutputKind.OPTION:
                            row["option"] = key.exact
                facts = _raw_facts(self.study_root, call.get("raw"))
                for k in ("generation_id_think", "think_latency_s", "think_model", "think_provider", "answer_model",
                          "answer_failures", "provider_only"):
                    if k in facts:
                        row[k] = facts[k]
            rows.append(row)
        frame = pd.DataFrame(rows, columns=self.DRAW_COLS)
        self._cache["c48_25a"] = frame
        return frame

    def section_draws(self) -> None:
        d = "csv/02_draws"
        draws = self.draws()
        if draws is None:
            return
        main = draws.copy()
        main["source_table"] = "analysis_inputs/draws"
        for c in self.DRAW_COLS:
            if c not in main:
                main[c] = None
        frames = [main[self.DRAW_COLS]]
        extra = self.c48_25a_rows()
        if extra is not None:
            frames.append(extra)
        allx = pd.concat(frames, ignore_index=True)
        for c in ("allowance_a", "replicate", "completion_headroom", "answer_len_chars", "answer_failures",
                  "think_in", "think_out", "answer_in", "answer_out"):
            allx[c] = pd.to_numeric(allx[c], errors="coerce").astype("Int64")
        for c in ("absent", "pending", "graded", "final_present"):
            allx[c] = allx[c].astype(object).where(allx[c].notna(), None)
        allx["reasoning_budget_tokens"] = (allx["allowance_a"] * A_TOKENS).astype("Int64")
        cols = self.DRAW_COLS[:14] + ["reasoning_budget_tokens"] + self.DRAW_COLS[14:]
        allx = allx[cols]
        dup = int(allx.candidate_id.duplicated().sum())
        self.check("draws_all candidate_id unique", "ok" if not dup else f"{dup} duplicates")
        self.table(f"{d}/draws_all.csv", allx, purpose="Every generation used anywhere in the study: the legacy "
                   "forced_v2 L96/D32 banks, every new passk partition and the LEX c48_25a 25A draws.",
                   unit="one forced generation (THINK exactly its allowance + ANSWER)",
                   source="analysis_inputs/draws.parquet + lex/c48_25a_draws.parquet (+ c48_25a_calls.jsonl, "
                          "lex/raw for c48_25a vote keys and THINK facts)",
                   cols=DRAW_DOCS,
                   notes="Dropped as internal hashing/constant: raw_response_path, raw_response_sha256, "
                         "think_body_sha, "
                         "answer_body_sha, think_text_sha, decoding_config_hash, prompt_version, endpoint_lock_hash, "
                         "grader_version (their distinct values are in 00_study/study_facts.csv). c48_25a rows: "
                         "correct/grade_state as used by the LEX analysis, where a grade of status invalid_output "
                         "was left unknown (correct empty, grade_state invalid) and excluded from the 25A bank, "
                         "whereas the core pipeline scores invalid output as false.")

    # =================================================================================================================
    # 03_pools
    # =================================================================================================================
    def section_pools(self) -> None:
        d = "csv/03_pools"
        rp, pp, cp = (self.need("analysis_inputs/readouts.parquet"), self.need("analysis_inputs/pools.parquet"),
                      self.need("analysis_inputs/classes.parquet"))
        draws = self.draws()
        if rp is not None:
            ro = pd.read_parquet(rp)
            ro["B"] = ro["n"].astype(int) * ro["a"].astype(int)
            dev_p = self.path("analysis_inputs", "dev_training_readouts.parquet")
            if dev_p.exists():
                dev = pd.read_parquet(dev_p, columns=["pool_id", "panel"])
                keys = set(zip(dev.pool_id, dev.panel))
                ro["in_e5_dev_training"] = [(p, q) in keys for p, q in zip(ro.pool_id, ro.panel)]
            if pp is not None and draws is not None:
                pools = pd.read_parquet(pp, columns=["pool_id", "member_ids"]).drop_duplicates("pool_id")
                lat = dict(zip(draws.candidate_id, pd.to_numeric(draws.latency_s, errors="coerce")))
                cost = dict(zip(draws.candidate_id, pd.to_numeric(draws.cost_usd, errors="coerce")))
                tok = dict(zip(draws.candidate_id, pd.to_numeric(draws.think_out, errors="coerce")))
                mem = dict(zip(pools.pool_id, pools.member_ids))

                def agg(pid: str, table: dict, fn) -> float | None:
                    vals = [table.get(m) for m in mem.get(pid, [])]
                    vals = [v for v in vals if v is not None and not _is_missing(v)]
                    return float(fn(vals)) if vals else None
                ro["wallclock_s"] = [agg(p, lat, max) for p in ro.pool_id]
                ro["cost_usd_sum"] = [agg(p, cost, sum) for p in ro.pool_id]
                ro["think_tokens_sum"] = pd.array([agg(p, tok, sum) for p in ro.pool_id], dtype="Float64")
            front = ["item_id", "benchmark", "panel", "config_id", "n", "a", "B", "pool_id"]
            ro = ro[front + [c for c in ro.columns if c not in front and c != "class_first_ids"]]
            self.table(f"{d}/pool_outcomes.csv", ro, purpose="Readout of every evaluated pool (L96, C48, V48, T200 x "
                       "every configuration): delivered answer, coverage H, plurality accuracy S under the seeded tie "
                       "rule, tie-averaged S, bounds, entropy, top-wrong mass and the decomposition.",
                       unit="one pool (item x panel x configuration)", source="analysis_inputs/readouts.parquet "
                       "(+ pools/draws for wall-clock and cost)", cols=POOL_DOCS,
                       notes="class_first_ids dropped (recoverable from pool_classes.csv). mode is always passk "
                             "(seeded uniform tie rule); the legacy tie mode appears only in 01_e0_audit.")
        if pp is not None:
            pools = pd.read_parquet(pp)
            self.table(f"{d}/pool_members.csv", pools, purpose="Members of every pool and class-building scope, in "
                       "frozen replicate order.", unit="one pool or scope", source="analysis_inputs/pools.parquet",
                       cols={"member_ids": "candidate ids in frozen member order (JSON list): the prefix replicates "
                             "r0..r(n-1) at allowance a (A-level L96/C48 pools use legacy slots 0..n-1)",
                             "view": "pool view (current)", "candidate_order_seed": "seed recorded with the pool "
                             "order (20260925)", "source_partitions": "draw partitions the members come from (JSON)",
                             "a": "per-call allowance (units of A); 0 for the c48fit scope (mixed allowances)",
                             "n": "number of members", "B": "total nominal budget (units of A)"})
        if cp is not None:
            cl = pd.read_parquet(cp)
            self.table(f"{d}/pool_classes.csv", cl, purpose="Gold-free answer classes of every pool and scope (exact "
                       "vote key, then mutual complete-link Math-Verify from the pair cache; GPQA by option).",
                       unit="one answer class of one pool", source="analysis_inputs/classes.parquet",
                       cols={"class_index": "class position in build order", "class_id": "class label (math:<key> "
                             "or opt:<letter>)", "member_ids": "voting members in the class (JSON list, frozen "
                             "order)", "support": "votes in the class", "n_votes": "voting members of the pool"})

    # =================================================================================================================
    # 04_e1
    # =================================================================================================================
    def section_e1(self) -> None:
        d = "csv/04_e1"
        ep = self.need("sealed/e1_selection.json")
        if ep is not None:
            e1 = _read_json(ep)
            rows = []
            for cid, t in (e1.get("table") or {}).items():
                rows.append({"config_id": cid, "role": "reference" if cid == "1x16" else "candidate",
                             "selected": cid == e1.get("selected"), "mixture": t.get("mixture"),
                             "macro": t.get("macro"), **{b: (t.get("by_family") or {}).get(b) for b in BENCH},
                             "n_items": t.get("n_items"), "n_unknown_tie": t.get("n_unknown_tie"),
                             "cohort_n_common": (e1.get("cohort") or {}).get("n_common"),
                             "cohort_excluded_unknown": (e1.get("cohort") or {}).get("excluded_unknown"),
                             "rule": e1.get("rule")})
            self.table(f"{d}/e1_lock.csv", pd.DataFrame(rows), purpose="The sealed E1 lock: L96 tie-averaged plurality "
                       "accuracy of the candidate mixed allocations and the 1x16 reference on the common known cohort; "
                       "M* = argmax mixture.", unit="one configuration", source="sealed/e1_selection.json",
                       cols={"role": "candidate (8x2, 4x4, 2x8) or reference (1x16)", "selected": "locked as M*",
                             "mixture": "mixture-weighted (.10/.10/.50/.30) mean of per-benchmark tie-averaged "
                             "accuracy", "macro": "equal-weight mean", "n_unknown_tie": "items with an undetermined "
                             "tie-averaged value", "cohort_n_common": "common known cohort size",
                             "cohort_excluded_unknown": "items excluded from the cohort (JSON)", "rule": "lock rule"})
            self.copy(ep, f"{d}/e1_selection.source.json", purpose="sealed/e1_selection.json verbatim")
        num = self.code_root / "docs" / "passk_cont_report" / "numbers.json"
        if num.exists():
            menus = _read_json(num).get("menus") or {}
            rows = [{"panel": p, "config_id": c, **v} for p in ("L96", "V48", "T200") for c, v in
                    (menus.get(p) or {}).items()]
            self.table(f"{d}/menus_by_panel.csv", pd.DataFrame(rows), purpose="Allocation menus per panel as "
                       "reported: plurality accuracy (mixture and macro) and coverage (macro) of every configuration.",
                       unit="one (panel, configuration)", source="docs/passk_cont_report/numbers.json menus",
                       cols={"S_mixture": "delivered plurality accuracy (S_known, seeded ties), mixture weights",
                             "S_macro": "delivered plurality accuracy, macro", "H_macro": "coverage, macro",
                             "n": "items"})
        else:
            self.gap("docs/passk_cont_report/numbers.json missing (menus_by_panel.csv)")
        pj = self.path("scored", "primary.json")
        prim = _read_json(pj) if pj.exists() else ((_read_json(num).get("primary") if num.exists() else None))
        if prim:
            rows = []
            holm = prim.get("holm") or {}
            for test, fam in (("mixed_vs_deep", "primary (Holm-2)"), ("p2_vs_p1", "primary (Holm-2)"),
                              ("mixed_vs_deep_tie_averaged", "secondary")):
                r = prim.get(test)
                if not r:
                    continue
                row = {"test": test, "family": fam,
                       "description": {"mixed_vs_deep": f"locked M* ({prim.get('locked')}) minus deep 1x16 delivered "
                                                        "accuracy on T200 (seeded ties)",
                                       "p2_vs_p1": "item-level mean Brier over the five 16A configurations, P2 minus "
                                                   "P1, T200",
                                       "mixed_vs_deep_tie_averaged": "M* minus 1x16 with the analytic tie-averaged "
                                                                     "readout"}[test]}
                row.update(flat_record(r))
                for k in ("p_holm", "reject", "rank"):
                    if k in (holm.get(test) or {}):
                        row[k] = holm[test][k]
                rows.append(row)
            self.table(f"{d}/primary_tests.csv", pd.DataFrame(rows), purpose="The pre-registered primary family "
                       "(Holm over two tests, T200 mixture estimand) and the tie-averaged secondary readout.",
                       unit="one test", source="scored/primary.json",
                       cols={"test": "test id", "family": "test family", "description": "what is contrasted"})
            if pj.exists():
                self.copy(pj, f"{d}/primary.source.json", purpose="scored/primary.json verbatim")
        else:
            self.gap("scored/primary.json missing (primary_tests.csv)")
        for name, purpose, unit, cols in (
                ("menus.csv", "Full menu table: every panel x configuration with H, S, S_tie and bounds, entropy and "
                 "top-wrong mass under macro, mixture and micro weights.", "one (panel, configuration)",
                 {"n_items": "items with the configuration"}),
                ("menu_contrasts.csv", "Secondary menu contrasts: every B=16A configuration vs 1x16 and every B=8A "
                 "configuration vs 1x8 per panel (macro), with Bonferroni-simultaneous intervals.",
                 "one (panel, configuration)", {"reference": "reference configuration", "ci95_lo": "95% studentized "
                 "lower", "ci95_hi": "95% studentized upper", "simul_lo": "Bonferroni-simultaneous lower (over the "
                 "menu's comparisons)", "simul_hi": "Bonferroni-simultaneous upper", "p": "two-sided bootstrap p",
                 "estimate": "macro mean difference configuration minus reference (S_known)",
                 "error": "why the contrast could not be computed (empty when it was)"}),
                ("finite_bank_coverage.csv", "Finite-bank coverage G = 1 - C(m-c,k)/C(m,k) per item for the 16A and 8A "
                 "menus (never k > m).", "one (panel, item, configuration)",
                 {"G_lo": "coverage with unknown grades counted wrong", "G_hi": "coverage with unknown grades counted "
                  "right"})):
            p = self.path("scored", name)
            if p.exists():
                self.table(f"{d}/{name}", pd.read_csv(p), purpose=purpose, unit=unit, source=f"scored/{name}",
                           cols=cols)
            else:
                self.gap(f"source missing: scored/{name}")

    # =================================================================================================================
    # 05_e2
    # =================================================================================================================
    def section_e2(self) -> None:
        d = "csv/05_e2"
        v48 = self.path("sealed", "v48")
        e2_pred_cols = {"pred_coverage": "predicted coverage (posterior mean)", "pred_plurality": "predicted "
                        "plurality (delivered) accuracy (posterior mean)", "pi_lo": "5% posterior quantile of the "
                        "plurality prediction", "pi_hi": "95% posterior quantile of the plurality prediction",
                        "cov_pi_lo": "5% posterior quantile of coverage", "cov_pi_hi": "95% posterior quantile of "
                        "coverage", "in_menu": "configuration is one of the 9 scored menu configurations (16A, 8A)",
                        "budget": "budget group 16A or 8A (empty for crossover-only configurations)",
                        "pred_rank": "rank of pred_plurality within (item, model, budget); 1 = best"}
        cross_cols = {"pred_deep": "predicted P(1 x kA correct)", "pred_wide_cov": "predicted P(k x A covered)",
                      "pred_diff": "predicted crossover pred_deep - pred_wide_cov", "pred_sign": "sign of pred_diff",
                      "p_sign_pos": "posterior probability the difference is positive", "diff_lo": "5% posterior "
                      "quantile of the difference", "diff_hi": "95% posterior quantile", "obs_deep": "observed V48 "
                      "1 x kA coverage (correctness)", "obs_wide_cov": "observed V48 k x A coverage", "obs_diff":
                      "observed difference", "residual": "obs_diff - pred_diff", "sign_agree": "sign(obs_diff) == "
                      "pred_sign (a tie obs_diff = 0 counts as disagreement)"}
        for name, rel, purpose, unit, cols in (
                ("predictions.parquet", "v48_predictions.csv", "Sealed E2 predictions for every V48 item x "
                 "configuration x model (made from C48 fitting draws before any V48 label).",
                 "one (item, configuration, model)", e2_pred_cols),
                ("crossover.parquet", "v48_crossover_predictions.csv", "Sealed crossover predictions sign(h(kA) - "
                 "k-call coverage) for k = 2, 4, 8, 16.", "one (item, model, k)", cross_cols),
                ("population.parquet", "v48_population_predictions.csv", "Sealed population-level (macro and "
                 "per-benchmark) predictions with 95% predictive intervals including Bernoulli noise.",
                 "one (model, configuration, quantity, scope)",
                 {"quantity": "plurality or coverage", "scope": "macro or a benchmark", "pred_mean": "predicted "
                  "population mean", "pi_lo": "2.5% predictive quantile", "pi_hi": "97.5% predictive quantile"}),
                ("treatment_gap.parquet", "v48_treatment_gap.csv", "Diffuse vs concentrated unseen-residual "
                 "treatments: predicted plurality gap per item x configuration.", "one (base model, item, "
                 "configuration)", {"diffuse": "prediction under the diffuse treatment", "concentrated": "prediction "
                 "under the concentrated-wrong treatment", "base": "base model family", "gap": "|diffuse - "
                 "concentrated|", "material": "gap > 0.10 (reported as uncertainty)"}),
                ("illustrative.parquet", "v48_illustrative_items.csv", "Frozen illustrative items: largest predicted "
                 "|crossover| per benchmark (beta_full_diffuse).", "one benchmark", {"pred_diff": "predicted "
                 "crossover difference"}),
                ("illustrative_curves.parquet", "v48_illustrative_curves.csv", "Depth-response curves of the "
                 "illustrative items.", "one (item, response model, allowance)",
                 {"response": "response model (beta or glm)", "p_mean": "posterior mean per-call correctness at "
                  "allowance a", "p_lo": "5% quantile", "p_hi": "95% quantile", "h_over_a_median": "median of "
                  "h(aA)/a (per-unit efficiency)", "h_over_a_lo": "5% quantile", "h_over_a_hi": "95% quantile",
                  "margin_diffuse_mean": "mean modal margin of the correct class, diffuse treatment",
                  "margin_diffuse_lo": "5% quantile", "margin_diffuse_hi": "95% quantile",
                  "p_correct_modal_diffuse": "probability the correct class is modal, diffuse",
                  "margin_concentrated_mean": "mean modal margin, concentrated treatment", "margin_concentrated_lo":
                  "5% quantile", "margin_concentrated_hi": "95% quantile", "p_correct_modal_concentrated":
                  "probability the correct class is modal, concentrated"})):
            p = v48 / name
            if p.exists():
                self.table(f"{d}/{rel}", pd.read_parquet(p), purpose=purpose, unit=unit, source=f"sealed/v48/{name}",
                           cols=cols)
            else:
                self.gap(f"source missing: sealed/v48/{name}")
        sc = self.path("scores", "e2")
        for name, rel, purpose, unit, cols in (
                ("overall.parquet", "e2_scores_overall.csv", "E2 scores per model over the 9 menu configurations "
                 "on V48.", "one model", {"plurality_brier": "Brier of pred_plurality vs observed delivered "
                 "correctness", "plurality_logscore": "mean log score (higher is better)", "coverage_brier": "Brier "
                 "of pred_coverage vs observed coverage", "ranking_accuracy": "paired ranking accuracy within item x "
                 "budget (ties 1/2)", "pop_pi_coverage_rate": "share of the 9 configurations whose observed macro "
                 "plurality lies inside the 95% population interval"}),
                ("by_config.parquet", "e2_scores_by_config.csv", "E2 scores per model x configuration on V48.",
                 "one (model, configuration)", {
                     "plurality_n": "items with known plurality", "plurality_brier": "Brier (plurality)",
                     "plurality_logscore": "mean log score (plurality)", "plurality_obs_macro": "observed macro "
                     "plurality accuracy", "plurality_pred_macro": "predicted macro", "plurality_pop_error":
                     "pred - obs (macro)", "plurality_pop_pi_covered": "observed macro inside the 95% population "
                     "interval", "coverage_n": "items with known coverage", "coverage_brier": "Brier (coverage)",
                     "coverage_logscore": "mean log score (coverage)", "coverage_obs_macro": "observed macro coverage",
                     "coverage_pred_macro": "predicted macro coverage", "coverage_pop_error": "pred - obs (macro)",
                     "coverage_pop_pi_covered": "observed macro coverage inside the 95% population interval"}),
                ("crossover.parquet", "e2_crossover_scored.csv", "Crossover predictions joined to V48 observations "
                 "with residuals and sign agreement.", "one (item, model, k)", cross_cols)):
            p = sc / name
            if p.exists():
                self.table(f"{d}/{rel}", pd.read_parquet(p), purpose=purpose, unit=unit, source=f"scores/e2/{name}",
                           cols=cols)
            else:
                self.gap(f"source missing: scores/e2/{name}")
        models = v48 / "models"
        fits, coefs, cvs = [], [], []
        if (models / "pooled_glm.json").exists():
            g = _read_json(models / "pooled_glm.json")
            self._model_rows("pooled_glm", "response", g.get("fit") or {}, fits, coefs, None)
            for variant, t in (g.get("cv") or {}).items():
                if isinstance(t, dict):
                    for r in t.get("table") or []:
                        cvs.append({"model_file": "pooled_glm.json", "fit": variant, **r,
                                    "best_ridge": t.get("best_ridge"), "chosen_variant": g["cv"].get("chosen")})
        if (models / "ridge.json").exists():
            g = _read_json(models / "ridge.json")
            names = ([f"bench:{b}" for b in BENCH] + ["log2_n", "log2_a", "rate_a", "rate1", "top_share",
                                                      "n_classes", "novote", "log2_n*rate_a", "log2_n*top_share",
                                                      "log2_a*rate_a"])
            for target in ("coverage", "plurality"):
                t = g.get(target) or {}
                self._model_rows("ridge", target, t.get("fit") or {}, fits, coefs, names)
                for r in t.get("cv") or []:
                    cvs.append({"model_file": "ridge.json", "fit": target, **r, "best_ridge": t.get("ridge")})
        self._write_model_tables(d, "e2", fits, coefs, cvs, "sealed/v48/models/*.json",
                                 "E2 pooled glm (benchmark x categorical depth + Gauss-Hermite item intercept, fit "
                                 "on C48 draws) and the unstructured ridge ablation")
        kv = []
        for f in ("seeds.json", "manifest.json"):
            p = v48 / f
            if p.exists():
                kv += [{"file": f, "key": k, "value": _kv_value(v)} for k, v in flatten_kv(_read_json(p))]
        if kv:
            self.table(f"{d}/e2_fit_settings.csv", pd.DataFrame(kv), purpose="E2 fit settings and the sealed "
                       "manifest (seeds, posterior draws S, priors, interval levels, class source, hashes).",
                       unit="one JSON leaf", source="sealed/v48/{seeds,manifest}.json",
                       cols={"file": "source file under sealed/v48/"})

    def _model_rows(self, model: str, fit_name: str, fit: Mapping, fits: list, coefs: list,
                    names: Sequence[str] | None, model_file: str | None = None) -> None:
        if not fit:
            return
        cols = fit.get("columns")
        source = "stored"
        if not cols:
            cols = list(names) if names is not None else None
            source = "derived from the design code" if names is not None else "unknown"
        coef = list(fit.get("coef") or [])
        if cols is not None and len(cols) != len(coef):
            self.gap(f"{model}/{fit_name}: {len(coef)} coefficients but {len(cols)} derived column names")
            cols, source = None, "unknown"
        fits.append({"model_file": model_file or f"{model}.json", "model": model, "fit": fit_name,
                     "intercept": fit.get("intercept"), "sigma_item": fit.get("sigma"),
                     "item_intercept": fit.get("item_intercept"), "ridge": fit.get("ridge"),
                     "gh_nodes": fit.get("gh_nodes"), "converged": fit.get("converged"), "n_iter": fit.get("n_iter"),
                     "objective": fit.get("objective"), "n_coef": len(coef)})
        xm, xs = fit.get("x_mean") or [None] * len(coef), fit.get("x_scale") or [None] * len(coef)
        for j, c in enumerate(coef):
            coefs.append({"model_file": model_file or f"{model}.json", "model": model, "fit": fit_name, "index": j,
                          "column": None if cols is None else cols[j], "column_name_source": source, "coef": c,
                          "x_mean": xm[j] if j < len(xm) else None, "x_scale": xs[j] if j < len(xs) else None})

    def _write_model_tables(self, d: str, tag: str, fits: list, coefs: list, cvs: list, source: str,
                            what: str) -> None:
        if fits:
            self.table(f"{d}/{tag}_model_fits.csv", pd.DataFrame(fits), purpose=f"Fitted {what}: one row per fit "
                       "(intercept, item random-effect sd, ridge, convergence).", unit="one fitted glm", source=source,
                       cols=MODEL_FIT_DOCS)
        if coefs:
            self.table(f"{d}/{tag}_model_coefficients.csv", pd.DataFrame(coefs), purpose=f"Coefficients of the "
                       f"{what}, on standardized inputs ((x - x_mean) / x_scale).", unit="one coefficient",
                       source=source, cols=MODEL_COEF_DOCS)
        if cvs:
            self.table(f"{d}/{tag}_model_cv.csv", pd.DataFrame(cvs), purpose=f"Problem-grouped 5-fold CV tables "
                       f"used to choose ridge strengths ({what}).", unit="one (fit, ridge value)", source=source,
                       cols={"model_file": "model file", "fit": "which fit / target", "ridge": "ridge strength "
                             "tried", "log_loss": "grouped-CV log loss", "brier": "grouped-CV Brier",
                             "best_ridge": "ridge chosen", "chosen_variant": "response variant chosen (categorical vs "
                             "monotone depth)", "n_rows": "rows in the eta training frame",
                             "model": "E5 predictor"})

    # =================================================================================================================
    # 06_e5
    # =================================================================================================================
    def section_e5(self) -> None:
        d = "csv/06_e5"
        t200 = self.path("sealed", "t200")
        p = t200 / "predictions.parquet"
        if p.exists():
            self.table(f"{d}/t200_predictions.csv", pd.read_parquet(p), purpose="Sealed E5 predictions for all 200 "
                       "T200 items x 9 configurations x 6 predictors (made before any T200 label).",
                       unit="one (item, configuration, predictor)", source="sealed/t200/predictions.parquet")
        else:
            self.gap("source missing: sealed/t200/predictions.parquet")
        p = t200 / "features.parquet"
        if p.exists():
            self.table(f"{d}/t200_features.csv", pd.read_parquet(p), purpose="Label-free five-draw diagnostic "
                       "features of every T200 item (t200_diag r0-4), as sealed.", unit="one T200 item",
                       source="sealed/t200/features.parquet", cols=FEATURE_DOCS)
        else:
            self.gap("source missing: sealed/t200/features.parquet")
        for name, rel, purpose, unit in (
                ("dev_pool_table.parquet", "e5_dev_pool_table.csv", "The E5 development training frame: L96 (+ C48 "
                 "B=32A) pool outcomes with the item's features; the withheld configurations are physically absent.",
                 "one development pool"),
                ("dev_draw_frame.parquet", "e5_dev_draw_frame.csv", "The E5 draw-level training frame (P2 response "
                 "model): every development draw's correctness.", "one development draw")):
            p = t200 / "models" / name
            if p.exists():
                self.table(f"{d}/{rel}", pd.read_parquet(p), purpose=purpose, unit=unit,
                           source=f"sealed/t200/models/{name}", cols={**FEATURE_DOCS, **DEV_FRAME_DOCS})
        sc = self.path("scores", "e5")
        for name, purpose, unit, cols in (
                ("summary", "E5 scores per predictor on T200: the primary item-level Brier (mixture and macro) and "
                 "secondary metrics over the 16A configurations and the withheld cells.", "one predictor",
                 E5_SUMMARY_DOCS),
                ("contrasts", "Mixture-weighted item-level Brier differences between predictors (point estimates; the "
                 "bootstrap test of P2-P1 is in 04_e1/primary_tests.csv).", "one contrast",
                 {"contrast": "predictor a - predictor b", "mixture_diff": "mixture-weighted mean of item-level "
                  "Brier(a) - Brier(b)"}),
                ("calibration", "Calibration bins of p_delivered over the known 16A cells.", "one (predictor, bin)",
                 {"bin": "decile bin of the prediction (0 = [0, .1))", "n": "cells in the bin", "mean_pred":
                  "mean prediction", "mean_obs": "observed delivered accuracy"}),
                ("item_brier", "Item-level mean Brier over each item's known 16A configurations (the primary unit).",
                 "one (predictor, item)", {"brier": "mean squared error of p_delivered vs delivered correctness over "
                                                   "the item's known 16A configurations", "n_known": "known 16A "
                                                   "configurations of the item"}),
                ("policy", "21A select-per-item policy (5 diagnostics + the configuration with the highest predicted "
                 "delivered probability) and every fixed 16A configuration, realized T200 accuracy.",
                 "one policy", {"policy": "select_per_item or fixed_<config>", "budget_A": "nominal budget (units of "
                                "A; 21 = 16 + 5 diagnostics)", "accuracy_mixture": "realized delivered accuracy, "
                                "mixture", "accuracy_macro": "realized delivered accuracy, macro",
                                "predictor": "predictor (empty for fixed configurations)"}),
                ("residuals", "Mean residual (observed - predicted delivered) by predictor and benchmark.",
                 "one (predictor, benchmark)", {"mean": "mean of delivered - p_delivered", "size": "known cells"}),
                ("dropped_items", "Items dropped from the primary Brier (fewer than 3 known 16A configurations).",
                 "one (predictor, item)", {"brier": "item Brier", "n_known": "known 16A configurations"})):
            p = sc / f"{name}.parquet"
            if p.exists():
                self.table(f"{d}/e5_{name}.csv", pd.read_parquet(p), purpose=purpose, unit=unit,
                           source=f"scores/e5/{name}.parquet", cols=cols)
            else:
                self.gap(f"source missing: scores/e5/{name}.parquet")
        fits, coefs, cvs = [], [], []
        models = t200 / "models"
        for name in ("P0", "P1", "P2", "P3", "P1-conc", "P2-conc"):
            p = models / f"{name}.json"
            if not p.exists():
                self.gap(f"source missing: sealed/t200/models/{name}.json")
                continue
            m = _read_json(p)
            if name.startswith("P2"):
                conc = [] if name.endswith("-conc") else ["f1", "f2", "f3", "f4", "f5", "top_gap", "n_classes",
                                                          "entropy"]
                eta_names = ["logit_p_call_oof", "log_n", "log_a"] + [f"bench:{b}" for b in BENCH[1:]] + conc
                self._model_rows(name, "response", m.get("response") or {}, fits, coefs, None, f"{name}.json")
                self._model_rows(name, "eta", m.get("eta") or {}, fits, coefs, eta_names, f"{name}.json")
            else:
                for target, fit in (m.get("fits") or {}).items():
                    self._model_rows(name, target, fit, fits, coefs, None, f"{name}.json")
            for target, t in (m.get("cv") or {}).items():
                for r in t.get("table") or []:
                    cvs.append({"model_file": f"{name}.json", "model": name, "fit": target, **r,
                                "best_ridge": t.get("best_ridge"), "n_rows": t.get("n_rows")})
        self._write_model_tables(d, "e5", fits, coefs, cvs, "sealed/t200/models/*.json",
                                 "E5 predictors P0-P3 (+conc ablations) fit on L96")
        kv = []
        for f in ("seeds.json", "manifest.json", "models/latency.json"):
            p = t200 / f
            if p.exists():
                kv += [{"file": f, "key": k, "value": _kv_value(v)} for k, v in flatten_kv(_read_json(p))]
        if kv:
            self.table(f"{d}/e5_fit_settings.csv", pd.DataFrame(kv), purpose="E5 fit settings and the sealed "
                       "manifest (seed, simulations, CV folds, training/withheld configurations, latency-feature "
                       "decision, class source, hashes).", unit="one JSON leaf",
                       source="sealed/t200/{seeds,manifest,models/latency}.json",
                       cols={"file": "source file under sealed/t200/"})

    # =================================================================================================================
    # 07_e8a
    # =================================================================================================================
    def section_e8a(self) -> None:
        d = "csv/07_e8a_llm_allocator"
        cp = self.need("e8a/calls.jsonl")
        if cp is not None:
            rows = []
            for r in _jsonl(cp):
                lat = None
                raw = r.get("raw")
                if raw and (self.study_root / raw).exists():
                    try:
                        payload = json.loads(gzip.decompress((self.study_root / raw).read_bytes()))
                        lat = (payload.get("result") or {}).get("latency_seconds")
                    except (OSError, ValueError):
                        lat = None
                row = {k: v for k, v in r.items() if k != "raw"}
                row["cost_usd"] = _num(row.get("cost_usd"))
                row["latency_s"] = lat
                rows.append(row)
            dec = pd.DataFrame(rows)
            if len(dec):
                dec = dec.sort_values(["panel", "arm", "item_id", "parse_try", "transport_try"], kind="mergesort")
                if dec.latency_s.isna().any():
                    self.gap(f"e8a: {int(dec.latency_s.isna().sum())} calls without a readable raw latency")
            self.table(f"{d}/decisions.csv", dec, purpose="Every E8a allocator call (arms A0/A1, panels L96/T200), "
                       "including the outcome-blind retry of an invalid choice.", unit="one allocator call attempt",
                       source="e8a/calls.jsonl (raw path dropped; latency_s from e8a/raw/*.json.gz)", cols=E8_DOCS)
        for panel in ("L96", "T200"):
            p = self.path("e8a", f"items_{panel}.parquet")
            if p.exists():
                self.table(f"{d}/items_{panel}.csv", pd.read_parquet(p),
                           purpose=f"E8a per-item scoring on {panel}: the realized delivered correctness of every "
                                   "M5 option and of each arm's choice and reference policy.", unit=f"one {panel} item",
                           source=f"e8a/items_{panel}.parquet", cols=E8A_ITEM_DOCS)
            else:
                self.gap(f"source missing: e8a/items_{panel}.parquet")
        rp = self.need("e8a/results.json")
        if rp is not None:
            res = _read_json(rp)
            lv, tests, choices, spend = [], [], [], []
            for panel, pr in (res.get("panels") or {}).items():
                for policy, est in (pr.get("levels") or {}).items():
                    for estimand in ("mixture", "macro"):
                        r = est.get(estimand)
                        if isinstance(r, dict):
                            lv.append({"panel": panel, "policy": policy, "estimand": estimand, **flat_record(r)})
                for fam, tt in (pr.get("tests") or {}).items():
                    for name, r in tt.items():
                        tests.append({"panel": panel, "family": fam, "test": name, **flat_record(r)})
                for arm, by_b in (pr.get("choices") or {}).items():
                    for b, cnt in by_b.items():
                        for cfg, n in cnt.items():
                            choices.append({"panel": panel, "arm": arm, "benchmark": b, "config": cfg, "count": n})
                for arm, s in (pr.get("spend") or {}).items():
                    spend.append({"panel": panel, "arm": arm, **s,
                                  "invalid_after_retry": (pr.get("invalid_after_retry") or {}).get(arm),
                                  "n_items": pr.get("n_items")})
            self.table(f"{d}/results_summary.csv", pd.DataFrame(lv), purpose="E8a accuracy levels with 95% "
                       "bootstrap intervals for each arm and reference policy (mixture and macro estimands).",
                       unit="one (panel, policy, estimand)", source="e8a/results.json levels",
                       cols={"policy": "A0, A1 (allocator arms; invalid after retry -> fallback 1x16), "
                             "A0_invalid0/A1_invalid0 (sensitivity: invalid -> delivered 0), always_deep (1x16), "
                             "random (per-item mean of the five options), oracle (per-item best realized option; "
                             "clairvoyant ceiling), each fixed M5 option, P1_select (T200: sealed P1 highest predicted "
                             "option)", "estimand": "mixture (.10/.10/.50/.30) or macro",
                             "ci95_percentile_lo": "lower 95% percentile bootstrap bound",
                             "ci95_percentile_hi": "upper 95% percentile bootstrap bound"})
            self.table(f"{d}/results_tests.csv", pd.DataFrame(tests), purpose="E8a tests: primary Holm-2 family "
                       "(A0 and A1 vs always-1x16 on T200, mixture) and secondary descriptive contrasts.",
                       unit="one (panel, test)", source="e8a/results.json tests",
                       cols={"family": "primary (Holm over two, alpha .05) or secondary (descriptive)",
                             "test": "contrast a_vs_b = mean(a - b)"})
            self.table(f"{d}/choices.csv", pd.DataFrame(choices), purpose="E8a choice distribution by arm and "
                       "benchmark.", unit="one (panel, arm, benchmark, option)", source="e8a/results.json choices")
            self.table(f"{d}/spend.csv", pd.DataFrame(spend), purpose="E8a calls and spend per panel and arm.",
                       unit="one (panel, arm)", source="e8a/results.json spend",
                       cols={"calls": "allocator calls incl. retries", "usd": "spend (USD)", "mean_prompt_tokens":
                             "mean prompt tokens per call", "mean_completion_tokens": "mean completion tokens per "
                             "call", "invalid_after_retry": "items still invalid after one retry (fallback 1x16)",
                             "n_items": "items in the panel"})
            self.copy(rp, f"{d}/results.source.json", purpose="e8a/results.json verbatim")

    # =================================================================================================================
    # 08_lex_front
    # =================================================================================================================
    def section_lex(self) -> None:
        d = "csv/08_lex_front"
        lex = self.path("lex")
        for name, purpose, unit, cols in (
                ("front_estimates", "Per C48 item and each of the 17 options: bank-estimated delivered accuracy p, "
                 "consistency C = p^2+(1-p)^2 and expected parallel wall-clock latency.", "one (item, option)",
                 {"m": "bank size at allowance a (C48 u V48 draws, + c48_25a at 25A)", "p": "expected tie-averaged "
                  "plurality correctness of n draws sampled without replacement from the bank (exact enumeration "
                  "when C(m,n) <= 5,000, else 5,000 seeded subsets)", "C": "consistency p^2 + (1-p)^2", "latency":
                  "expected max of n call latencies sampled without replacement (s)"}),
                ("front_winners", "Per-problem lexicographic winner (accuracy within 0.02 -> consistency within 0.02 "
                 "-> fastest -> fewer tokens), in-sample.", "one C48 item",
                 {"cost": "tokens n*a (units of A; final tie-break)", "p": "the winner's bank-estimated delivered "
                  "accuracy", "C": "the winner's consistency p^2 + (1-p)^2", "latency": "the winner's expected "
                  "parallel wall-clock (s)", "m": "bank size at the winner's allowance"}),
                ("front_splithalf", "Split-half check: winners chosen on one half of each item's draws, scored on the "
                 "other half (averaged over both directions), with always-1x16 and always-1x25 on the same half.",
                 "one C48 item", {"p": "accuracy of the half-chosen option on the other half", "C": "its consistency",
                                  "latency": "its latency (s)", "deep_p": "1x16 accuracy on the scoring half",
                                  "deep_C": "1x16 consistency", "deep_latency": "1x16 latency (s)", "deep25_p": "1x25 "
                                  "accuracy on the scoring half", "deep25_C": "1x25 consistency", "deep25_latency":
                                  "1x25 latency (s)"})):
            p = lex / f"{name}.parquet"
            if p.exists():
                self.table(f"{d}/{name}.csv", pd.read_parquet(p), purpose=purpose, unit=unit,
                           source=f"lex/{name}.parquet", cols=cols)
            else:
                self.gap(f"source missing: lex/{name}.parquet")
        rp = lex / "front_report.json"
        if rp.exists():
            rep = _read_json(rp)
            self.table(f"{d}/fixed_options.csv", pd.DataFrame(rep.get("fixed_options") or []), purpose="Population "
                       "level of the 17 fixed options (macro over benchmarks of per-benchmark means).",
                       unit="one option", source="lex/front_report.json fixed_options",
                       cols={"p": "macro mean accuracy", "C": "macro mean consistency", "latency": "macro mean "
                             "latency (s)", "n_items": "items with an estimate"})
            rows = []
            for section in ("in_sample", "split_half"):
                for subject, metrics in (rep.get(section) or {}).items():
                    if not isinstance(metrics, dict):
                        continue
                    for metric, r in metrics.items():
                        if isinstance(r, dict):
                            rows.append(self._lex_row(section, subject, metric, r))
            for pred, r in (rep.get("t200_policy") or {}).items():
                for metric in ("acc", "latency", "deep_acc", "deep_latency", "acc_minus_deep", "latency_minus_deep"):
                    if isinstance(r.get(metric), dict):
                        rows.append(self._lex_row("t200_policy", pred, metric, r[metric]))
            self.table(f"{d}/lex_summary.csv", pd.DataFrame(rows), purpose="LEX front report flattened: in-sample "
                       "per-problem lex policy vs 1x16/1x25/population choice, the split-half check, and the T200 "
                       "policy readout of each sealed predictor.", unit="one (section, subject, metric)",
                       source="lex/front_report.json", cols={
                           "section": "in_sample (C48, optimistic), split_half (C48, honest), t200_policy "
                                      "(exploratory, realized T200)",
                           "subject": "lex (per-problem lex choice), 1x16 / 1x25 / deep_1x16 / deep_1x25 / "
                                      "<population choice> (fixed options), lex_minus_1x16 / lex_minus_1x25 "
                                      "(differences), or the predictor (t200_policy)",
                           "metric": "p / acc (accuracy), C (consistency), latency (s), deep_* (always-1x16), "
                                     "*_minus_deep (difference vs always-1x16)",
                           "stat": "level (mean with 95% interval) or difference (estimate with 95% interval and "
                                   "bootstrap p)",
                           "estimate": "mean (level) or mean difference (difference); macro over benchmarks",
                           "ci95_lo": "lower 95% percentile bootstrap bound", "ci95_hi": "upper 95% bound",
                           "p": "two-sided bootstrap p (differences only)"})
            wc = [{"benchmark": "all", "config": c, "count": n}
                  for c, n in (rep.get("winner_distribution") or {}).items()]
            for b, cnt in (rep.get("winner_by_benchmark") or {}).items():
                wc += [{"benchmark": b, "config": c, "count": n} for c, n in cnt.items()]
            self.table(f"{d}/lex_winner_counts.csv", pd.DataFrame(wc), purpose="Distribution of the per-problem "
                       "lexicographic winners (C48, in-sample).", unit="one (benchmark or all, option)",
                       source="lex/front_report.json winner_distribution / winner_by_benchmark",
                       cols={"benchmark": B_DOC + " or all"})
            bc = [{"allowance_a": int(a), **v} for a, v in (rep.get("bank_census_by_allowance") or {}).items()]
            self.table(f"{d}/lex_bank_census.csv", pd.DataFrame(bc), purpose="C48 bank sizes per allowance and "
                       "draws excluded as unknown.", unit="one allowance", source="lex/front_report.json "
                       "bank_census_by_allowance", cols={"allowance_a": "allowance (units of A)", "draws": "draws in "
                       "the pooled C48 banks at this allowance", "unknown": "draws with unknown correctness "
                       "(excluded from the banks)"})
            pc = [{"predictor": p, "config": c, "count": n} for p, r in (rep.get("t200_policy") or {}).items()
                  for c, n in (r.get("choices") or {}).items()]
            self.table(f"{d}/lex_t200_policy_choices.csv", pd.DataFrame(pc), purpose="Options chosen by the LEX rule "
                       "on each sealed predictor's T200 predictions.", unit="one (predictor, option)",
                       source="lex/front_report.json t200_policy.*.choices")
            self.table(f"{d}/lex_report_kv.csv", kv_frame(rep), purpose="lex/front_report.json flattened (every "
                       "scalar; includes delta, menu, population_lex_choice, feasible split-half menu).",
                       unit="one JSON leaf", source="lex/front_report.json")
            self.copy(rp, f"{d}/front_report.source.json", purpose="lex/front_report.json verbatim")
        else:
            self.gap("source missing: lex/front_report.json")
        p = lex / "c48_25a_draws.parquet"
        if p.exists():
            self.table(f"{d}/c48_25a_draws.csv", pd.read_parquet(p), purpose="The 384 forced 25A draws (8 per C48 "
                       "item) generated for the LEX front, as graded by passk.lex.grade25.", unit="one 25A draw",
                       source="lex/c48_25a_draws.parquet",
                       cols={"replicate": "replicate r1-8 (legacy deep/25 is r0)", "allowance_a": "25",
                             "parse_status": "parse status of the ANSWER", "final_answer": "extracted final answer "
                             "text", "correct": "semantic correctness; empty = unknown, incl. grade status "
                             "invalid_output (left unknown here, unlike the core pipeline)", "operational": "grader "
                             "operational success", "grade_status": "grader status (graded / invalid_output)",
                             "reasoning_tokens": "THINK tokens (exactly 25,600)"})
        cp = lex / "c48_25a_calls.jsonl"
        if cp.exists():
            rows = []
            for r in _jsonl(cp):
                spec, res = r.get("spec") or {}, r.get("result") or {}
                usage = res.get("usage") or {}
                raw_u = usage.get("raw_usage") or {}
                rows.append({"item_id": r.get("item_id"), "benchmark": r.get("benchmark"), "shard": r.get("shard"),
                             "replicate": r.get("replicate"), "node_id": r.get("node_id"), "attempt": r.get("attempt"),
                             "transport_status": r.get("transport_status"), "error": r.get("error"),
                             "partition": spec.get("partition_id"), "kind": spec.get("kind"),
                             "reasoning_budget": spec.get("reasoning_budget"),
                             "completion_headroom": spec.get("completion_headroom"),
                             "prompt_version": spec.get("prompt_version"),
                             "decoding_config_hash": spec.get("decoding_config_hash"),
                             "endpoint_lock_hash": spec.get("endpoint_lock_hash"),
                             "completion_status": res.get("completion_status"),
                             "generation_id": res.get("generation_id"), "finish_reason": res.get("finish_reason"),
                             "http_status": res.get("http_status"), "reasoning_tokens": r.get("reasoning_tokens"),
                             "think_in": (raw_u.get("think") or {}).get("prompt_tokens"),
                             "think_out": (raw_u.get("think") or {}).get("completion_tokens"),
                             "answer_in": (raw_u.get("answer") or {}).get("prompt_tokens"),
                             "answer_out": (raw_u.get("answer") or {}).get("completion_tokens"),
                             "latency_s": r.get("latency_s"), "cost_usd": _num(r.get("cost_usd")),
                             "observed_model": r.get("observed_model"), "observed_provider": r.get("observed_provider"),
                             "visible_content": res.get("visible_content")})
            self.table(f"{d}/c48_25a_calls.csv", pd.DataFrame(rows), purpose="Every c48_25a transport attempt "
                       "(3 transient 502s were retried).", unit="one transport attempt",
                       source="lex/c48_25a_calls.jsonl "
                       "(raw path dropped; spec/result fields flattened)",
                       cols={"attempt": "transport attempt (1-3)", "transport_status": "ok or error_envelope",
                             "error": "error detail of a failed attempt", "partition": "c48_25a", "kind": "deep",
                             "reasoning_budget": "THINK max_tokens (25,600)", "completion_headroom": "ANSWER "
                             "max_tokens", "prompt_version": "template pack version", "decoding_config_hash":
                             "decoding configuration hash", "endpoint_lock_hash": "adopted forced_v2 endpoint lock",
                             "completion_status": "completion status", "generation_id": "OpenRouter generation id of "
                             "the ANSWER call", "finish_reason": "ANSWER finish reason", "http_status": "HTTP status",
                             "think_in": "THINK prompt tokens", "think_out": "THINK completion tokens", "answer_in":
                             "ANSWER prompt tokens", "answer_out": "ANSWER completion tokens", "visible_content":
                             "the visible ANSWER text (rationale + final answer)", "replicate": "replicate r1-8"})

    @staticmethod
    def _lex_row(section: str, subject: str, metric: str, r: Mapping) -> dict:
        ci = r.get("ci95") or [None, None]
        is_diff = "estimate" in r
        return {"section": section, "subject": subject, "metric": metric, "stat": "difference" if is_diff else "level",
                "estimate": r.get("estimate") if is_diff else r.get("mean"), "ci95_lo": ci[0], "ci95_hi": ci[1],
                "p": r.get("p")}

    # =================================================================================================================
    # 09_selector_comparison
    # =================================================================================================================
    def section_selcmp(self) -> None:
        d = "csv/09_selector_comparison"
        sd = self.path("lex", "selcmp")
        if not sd.is_dir():
            self.gap("source missing: lex/selcmp/")
            return
        known = {"c48_selector_items": ("Per C48 item and selector: the option chosen and its bank-scored accuracy, "
                                        "consistency and latency incl. the selector's overhead.",
                                        "one (selector, C48 item)"),
                 "t200_selector_items": ("Per T200 item and selector: the option chosen and its realized delivered "
                                         "correctness and wall-clock incl. overhead.", "one (selector, T200 item)"),
                 "oof_predictions": ("Cross-fitted (5 problem-grouped L96 folds) E5 predictions for every L96 item x "
                                     "M15 option x predictor.", "one (item, option, predictor)"),
                 "t200_m15_predictions": ("Post-hoc refit on all of L96 predicting T200 for the M15 menu (reproduces "
                                          "the sealed 9-option predictions exactly).",
                                          "one (item, option, predictor)")}
        for p in sorted(sd.iterdir()):
            if p.suffix == ".parquet":
                purpose, unit = known.get(p.stem, (f"lex/selcmp/{p.name} converted to CSV.", "one row of the "
                                                   "source parquet"))
                self.table(f"{d}/{p.stem}.csv", pd.read_parquet(p), purpose=purpose, unit=unit,
                           source=f"lex/selcmp/{p.name}", cols=SELCMP_ITEM_DOCS)
            elif p.suffix == ".csv":
                self.table(f"{d}/{p.name}", pd.read_csv(p), purpose=f"lex/selcmp/{p.name} (normalized).",
                           unit="one row of the source CSV", source=f"lex/selcmp/{p.name}")
            elif p.suffix == ".json" and p.name == "report.json":
                self._selcmp_report(d, _read_json(p))
                self.copy(p, f"{d}/report.source.json", purpose="lex/selcmp/report.json verbatim")
            elif p.is_file():
                self.copy(p, f"{d}/{p.stem}.source{p.suffix}", purpose=f"lex/selcmp/{p.name} verbatim")

    def _selcmp_report(self, d: str, rep: dict) -> None:
        for panel, fname in (("c48", "c48_summary.csv"), ("t200", "t200_summary.csv")):
            sel = (rep.get(panel) or {}).get("selectors") or []
            if sel:
                rows = [flat_record(r) for r in sel]
                what = ("bank-estimated accuracy, consistency and latency (macro)" if panel == "c48" else
                        "realized accuracy (mixture and macro) and latency; consistency is not measurable per item")
                self.table(f"{d}/{fname}", pd.DataFrame(rows), purpose=f"Selector comparison on {panel.upper()}: "
                           f"every selector's {what}, with 95% bootstrap intervals, differences vs fixed:1x16, "
                           "lexicographic ranks and bootstrap win shares.",
                           unit="one selector", source=f"lex/selcmp/report.json {panel}.selectors",
                           cols=SELCMP_SUMMARY_DOCS)
        rows = []
        for panel in ("c48", "t200"):
            sec = rep.get(panel) or {}
            for key, val in sec.items():
                if key in ("selectors",) or not isinstance(val, dict):
                    continue
                for selector, vs_map in val.items():
                    if not isinstance(vs_map, dict):
                        continue
                    for vs, metrics in vs_map.items():
                        if not isinstance(metrics, dict):
                            continue
                        for metric, r in metrics.items():
                            if isinstance(r, dict) and "estimate" in r:
                                ci = r.get("ci95") or [None, None]
                                rows.append({"panel": panel, "table": key, "selector": selector,
                                             "vs": vs if ":" in str(vs) else f"LLM:{vs}", "metric": metric,
                                             "estimate": r.get("estimate"), "ci95_lo": ci[0], "ci95_hi": ci[1],
                                             "p": r.get("p"), "n_items": r.get("n_items")})
        if rows:
            self.table(f"{d}/pairwise.csv", pd.DataFrame(rows), purpose="Pairwise contrasts between selectors "
                       "(statistical/heuristic selectors minus the LLM selectors), macro or mixture, with 95% "
                       "bootstrap intervals.", unit="one (panel, selector, comparator, metric)",
                       source="lex/selcmp/report.json <panel>.stat_minus_llm (and any other pairwise table)",
                       cols={"table": "source table in report.json", "vs": "comparator selector",
                             "metric": "acc / C / lat (C48, macro) or acc_macro / acc_mixture / lat_macro (T200)",
                             "estimate": "mean of selector minus comparator (accuracy in proportion, latency in s)",
                             "ci95_lo": "lower 95% percentile bootstrap bound", "ci95_hi": "upper bound",
                             "p": "two-sided bootstrap p", "n_items": "items entering the contrast"})
        orders = []
        for panel in ("c48", "t200"):
            for key, val in (rep.get(panel) or {}).items():
                if key.startswith("lex_order") and isinstance(val, list):
                    orders += [{"panel": panel, "ordering": key, "rank": i + 1, "selector": s}
                               for i, s in enumerate(val)]
        if orders:
            self.table(f"{d}/lex_order.csv", pd.DataFrame(orders), purpose="Population lexicographic rankings "
                       "(repeated LEX selection: accuracy within 0.02 -> consistency within 0.02 (C48) -> lowest "
                       "latency).", unit="one (panel, ordering, rank)", source="lex/selcmp/report.json lex_order_*",
                       cols={"ordering": "lex_order_deployable / lex_order_all (C48, macro) or lex_order_macro / "
                             "lex_order_mixture (T200)", "rank": "1 = lexicographic winner"})
        cf = rep.get("crossfit") or {}
        folds = []
        for f, info in list((cf.get("folds") or {}).items()) + ([("full_fit", cf["full_fit"])] if cf.get("full_fit")
                                                                 else []):
            row = {"fold": str(f)}
            row.update(flat_record({k: v for k, v in info.items()}, json_keys=("best_ridges",)))
            folds.append(row)
        if folds:
            self.table(f"{d}/crossfit_folds.csv", pd.DataFrame(folds), purpose="Cross-fitting folds (5 problem-"
                       "grouped L96 folds = the E5 CV fold assignment) and the full L96 fit: sizes, latency-feature "
                       "decision, chosen ridges, held-out items.", unit="one fold (or the full fit)",
                       source="lex/selcmp/report.json crossfit", cols=CROSSFIT_DOCS)
        q = [{"predictor": k, **v} for k, v in (cf.get("oof_quality_vs_c48_bank") or {}).items()]
        if q:
            self.table(f"{d}/crossfit_oof_quality.csv", pd.DataFrame(q), purpose="Out-of-fold prediction quality "
                       "against the C48 bank estimates (all 15 M15 cells per item).", unit="one predictor",
                       source="lex/selcmp/report.json crossfit.oof_quality_vs_c48_bank",
                       cols={"mae_vs_bank_p": "mean |p_delivered - bank p|", "brier_vs_bank_p": "mean squared "
                             "difference to the bank p", "mean_within_item_spearman": "mean within-item Spearman "
                             "correlation of predicted vs bank p across options"})
        hr, ht = [], []
        for fit_name, menus in (rep.get("heuristic_rules") or {}).items():
            for menu, val in menus.items():
                if fit_name == "c48_crossfit":
                    for fold, r in val.items():
                        hr.append({"fit": fit_name, "fold": fold, "menu": menu,
                                   **{k: v for k, v in r.items() if k != "table"}})
                else:
                    hr.append({"fit": fit_name, "fold": "all_L96", "menu": menu,
                               **{k: v for k, v in val.items() if k != "table"}})
                    ht += [{"fit": fit_name, "menu": menu, **t} for t in val.get("table") or []]
        if hr:
            self.table(f"{d}/heuristic_rules.csv", pd.DataFrame(hr), purpose="Diagnostic-agreement heuristic "
                       "('plurality share f1 of the five diagnostic answers >= tau -> X, else 1x16') chosen per "
                       "fold and on all of L96.", unit="one (fit, fold, menu)",
                       source="lex/selcmp/report.json heuristic_rules", cols=HEUR_DOCS)
        if ht:
            self.table(f"{d}/heuristic_rule_table.csv", pd.DataFrame(ht), purpose="Every (tau, X) candidate rule "
                       "evaluated on all of L96 for the T200 heuristic.", unit="one candidate rule",
                       source="lex/selcmp/report.json heuristic_rules.t200_fit_on_L96.*.table", cols=HEUR_DOCS)
        ov = rep.get("overheads") or {}
        orows = [{"arm": r.get("arm"), "panel": r.get("panel"), "quantity": "llm_decision_latency_s", **r}
                 for r in ov.get("llm_decision") or []]
        for k, v in ov.items():
            if not isinstance(v, (list, dict)):
                orows.append({"quantity": k, "mean_s": v})
        if orows:
            of = pd.DataFrame(orows)
            of = of[["quantity"] + [c for c in of.columns if c != "quantity"]]
            self.table(f"{d}/overheads.csv", of, purpose="Selector overheads added to latency: the diagnostics' "
                       "parallel wall-clock and the LLM decision latency.", unit="one overhead quantity",
                       source="lex/selcmp/report.json overheads",
                       cols={"quantity": "llm_decision_latency_s (per arm and panel) or diag_wallclock_macro_<panel> "
                             "(macro mean of the five diagnostics' max latency)", "mean_s": "mean seconds",
                             "max_s": "max seconds", "attempts": "decision calls incl. retries", "missing": "attempts "
                             "without a recorded latency", "arm": "E8a arm", "panel": "panel"})
        self.table(f"{d}/report_kv.csv", kv_frame(rep), purpose="lex/selcmp/report.json flattened (every scalar: "
                   "definitions, menus, cross-fit details, reproduction check, runtime).", unit="one JSON leaf",
                   source="lex/selcmp/report.json")

    # =================================================================================================================
    # 10_e8b
    # =================================================================================================================
    def section_e8b(self) -> None:
        d = "csv/10_e8b_lex_llm"
        cp = self.need("e8b/calls.jsonl")
        if cp is None:
            return
        rows = []
        for r in _jsonl(cp):
            row = {k: v for k, v in r.items() if k != "raw"}
            row["cost_usd"] = _num(row.get("cost_usd"))
            row["chosen"] = (row.get("choice") if row.get("valid") else E8B_FALLBACK) if row.get("final") else None
            rows.append(row)
        dec = pd.DataFrame(rows)
        if not len(dec):
            self.gap("e8b/calls.jsonl is empty")
            return
        dec = dec.sort_values(["panel", "arm", "item_id", "sample", "parse_try", "transport_try"], kind="mergesort")
        self.table(f"{d}/decisions.csv", dec, purpose="Every E8b LLM-selector call (arms B0/B1, panels C48/T200, "
                   "K = 5 decisions per item and arm), including the outcome-blind retry of an invalid choice.",
                   unit="one selector call attempt", source="e8b/calls.jsonl (raw path dropped)", cols=E8_DOCS)
        final = dec[dec.final.astype(bool)]
        summ = []
        for (panel, arm, item), g in final.groupby(["panel", "arm", "item_id"], sort=True):
            by_s = dict(zip(g["sample"].astype(int), g["chosen"]))
            att = dec[(dec.panel == panel) & (dec.arm == arm) & (dec.item_id == item)]
            choices = [by_s.get(s) for s in range(5)]
            present = [c for c in choices if c is not None]
            counts = pd.Series(present).value_counts() if present else pd.Series(dtype=int)
            top = counts.max() if len(counts) else 0
            modal = sorted(c for c, n in counts.items() if n == top) if len(counts) else []
            per_sample_lat = att.groupby("sample").latency_s.sum(min_count=1)
            summ.append({"panel": panel, "arm": arm, "item_id": item, "benchmark": g.benchmark.iloc[0],
                         "shard": g.shard.iloc[0], **{f"choice_sample{s}": choices[s] for s in range(5)},
                         "n_decisions": len(present), "n_fallback_1x16": int((~g.valid.astype(bool)).sum()),
                         "n_distinct_choices": int(len(counts)), "modal_choice": modal[0] if modal else None,
                         "modal_tied": len(modal) > 1, "modal_share": (top / len(present)) if present else None,
                         "n_calls": int(len(att)), "cost_usd": float(att.cost_usd.sum()),
                         "decision_latency_s_mean": float(per_sample_lat.mean()) if len(per_sample_lat) else None})
        self.table(f"{d}/choice_summary.csv", pd.DataFrame(summ), purpose="Per item and arm: the five E8b choices, "
                   "the modal choice and its share (choice stability), fallbacks, cost and decision latency.",
                   unit="one (panel, arm, item)", source="derived from e8b/calls.jsonl",
                   cols={"n_decisions": "final decisions (5 expected)", "n_fallback_1x16": "decisions invalid after "
                         "one retry (scored as 1x16)", "n_distinct_choices": "distinct options among the decisions",
                         "modal_choice": "most frequent option (alphabetically first when tied)", "modal_tied": "more "
                         "than one option shares the top count", "modal_share": "share of decisions equal to the "
                         "modal choice (choice stability)", "n_calls": "calls incl. retries", "cost_usd": "total "
                         "decision cost (USD)", "decision_latency_s_mean": "mean over decisions of the summed call "
                         "latency of that decision (retries included)"})
        cb = (final.groupby(["panel", "arm", "benchmark", "chosen"]).size().rename("count").reset_index()
              .rename(columns={"chosen": "config"}))
        self.table(f"{d}/choices_by_benchmark.csv", cb, purpose="E8b choice distribution by arm and benchmark (all "
                   "five decisions per item).", unit="one (panel, arm, benchmark, option)",
                   source="derived from e8b/calls.jsonl", cols={"count": "decisions choosing the option"})

    # =================================================================================================================
    # 11_costs
    # =================================================================================================================
    def section_costs(self) -> None:
        d = "csv/11_costs"
        stages: dict[str, list[dict]] = {}
        order = ["core: Gate 0 endpoint probes", "core: Gate 1 canary", "core: Gate 1 smoke", "core: main generation",
                 "E8a LLM allocator", "LEX c48_25a 25A draws", "E8b LLM selectors (LEX objective)",
                 "legacy forced_v2 draws reused (not spent in this study)"]
        core_parts = {"gate0": order[0], "canary": order[1], "smoke": order[2]}
        for shard in ("mancer", "parasail"):
            lj = self.path("artifacts", "shards", shard, "state", "ledger_journal.jsonl")
            if not lj.exists():
                self.gap(f"source missing: artifacts/shards/{shard}/state/ledger_journal.jsonl")
                continue
            agg: dict[str, list[Decimal]] = {}
            for e in _jsonl(lj):
                if e.get("kind") == "settle":
                    agg.setdefault(e.get("partition_id"), []).append(Decimal(str(e.get("amount_usd") or "0")))
            for part in sorted(agg):
                stage = core_parts.get(part, order[3])
                stages.setdefault(stage, []).append({
                    "component": part, "shard": shard, "calls": len(agg[part]), "usd": sum(agg[part], Decimal(0)),
                    "source": f"artifacts/shards/{shard}/state/ledger_journal.jsonl (settle events)",
                    "counted_in_study_total": True})
        for path, stage, keys in (("e8a/calls.jsonl", order[4], ("panel", "arm")),
                                  ("lex/c48_25a_calls.jsonl", order[5], ("shard",)),
                                  ("e8b/calls.jsonl", order[6], ("panel", "arm"))):
            p = self.path(path)
            if not p.exists():
                self.gap(f"source missing: {path}")
                continue
            groups: dict[tuple, list[Decimal]] = {}
            for r in _jsonl(p):
                groups.setdefault(tuple(r.get(k) for k in keys), []).append(Decimal(str(r.get("cost_usd") or "0")))
            for key in sorted(groups):
                stages.setdefault(stage, []).append({
                    "component": "/".join(str(k) for k in key), "shard": key[0] if keys == ("shard",) else "both",
                    "calls": len(groups[key]), "usd": sum(groups[key], Decimal(0)), "source": path,
                    "counted_in_study_total": True})
        draws = self.draws()
        if draws is not None:
            leg = draws[draws.source == "legacy"]
            for (part, shard), g in leg.groupby(["partition", "shard"]):
                stages.setdefault(order[7], []).append({
                    "component": part, "shard": shard, "calls": int(len(g)),
                    "usd": sum((Decimal(repr(float(v))) for v in pd.to_numeric(g.cost_usd).dropna()), Decimal(0)),
                    "source": "analysis_inputs/draws.parquet cost_usd", "counted_in_study_total": False})
        rows: list[dict] = []
        core_total, core_calls, total, total_calls = Decimal(0), 0, Decimal(0), 0
        for stage in order:
            det = stages.get(stage) or []
            if not det:
                continue
            det = sorted(det, key=lambda r: (str(r["shard"]), str(r["component"])))
            rows += [{"row_type": "detail", "stage": stage, **r} for r in det]
            sub = sum((r["usd"] for r in det), Decimal(0))
            calls = sum(r["calls"] for r in det)
            counted = det[0]["counted_in_study_total"]
            rows.append({"row_type": "subtotal", "stage": stage, "component": "subtotal", "shard": "all",
                         "calls": calls, "usd": sub, "source": "sum of this stage's detail rows",
                         "counted_in_study_total": counted})
            if stage.startswith("core"):
                core_total, core_calls = core_total + sub, core_calls + calls
            if counted:
                total, total_calls = total + sub, total_calls + calls
        if core_calls:
            rows.append({"row_type": "subtotal", "stage": "core run (Gate 0 + Gate 1 + main)", "component": "subtotal",
                         "shard": "all", "calls": core_calls, "usd": core_total, "source": "sum of the core stages",
                         "counted_in_study_total": True})
            self.check("core-run ledger total vs key-usage delta 96.26 (memory note)", f"ledger {core_total} USD")
        if rows:
            rows.append({"row_type": "total", "stage": "study total (new spend)", "component": "total", "shard": "all",
                         "calls": total_calls, "usd": total, "source": "sum of the counted stage subtotals",
                         "counted_in_study_total": True})
        frame = pd.DataFrame(rows, columns=["row_type", "stage", "component", "shard", "calls", "usd",
                                            "counted_in_study_total", "source"])
        frame["usd"] = frame["usd"].map(lambda v: float(v) if v is not None else None)
        self.table(f"{d}/spend_by_stage.csv", frame, purpose="Spend by stage: the core run from the shard ledgers "
                   "(Gate 0 probes, canary, smoke, every main partition), E8a, LEX c48_25a and E8b from their call "
                   "logs; legacy forced_v2 draws listed but not counted.", unit="one (stage, component, shard) or "
                   "a subtotal/total", source="shard ledger journals, e8a/e8b/lex call logs, draws.parquet",
                   cols={"row_type": "detail, subtotal (per stage, and the whole core run) or total",
                         "stage": "study stage",
                         "component": "partition (core), panel/arm (E8a, E8b) or shard (LEX)",
                         "shard": "provider shard (both/all when aggregated)", "calls": "billed calls (settled ledger "
                         "events or logged call attempts incl. retries)", "usd": "spend in USD (exact decimal sums)",
                         "counted_in_study_total": "counted in this study's new spend", "source": "where the amounts "
                         "come from"},
                   notes="Core-run amounts are the settled ledger events; the OpenRouter key usage moved 181.32 -> "
                         "277.58 over the core run (96.26 all-in, memory note).")
        if draws is not None:
            allx = pd.concat([draws.assign(source_table="analysis_inputs/draws"),
                              (self.c48_25a_rows() if self.c48_25a_rows() is not None else pd.DataFrame())],
                             ignore_index=True)
            allx = allx[~allx.absent.astype(bool)]
            for c in ("think_in", "think_out", "answer_in", "answer_out", "cost_usd", "latency_s"):
                allx[c] = pd.to_numeric(allx[c], errors="coerce")
            g = allx.groupby(["source", "partition", "shard", "allowance_a"], dropna=False)
            summ = g[["think_in", "think_out", "answer_in", "answer_out", "cost_usd", "latency_s"]].sum(min_count=1)
            summ["requests"] = g.size()
            summ["mean_cost_usd"] = summ.cost_usd / summ.requests
            summ["mean_latency_s"] = summ.latency_s / summ.requests
            summ = summ.reset_index().rename(columns={"latency_s": "latency_s_sum"})
            summ["allowance_a"] = summ.allowance_a.astype(int)
            self.table(f"{d}/draw_costs_by_partition.csv", summ, purpose="Physical usage per generation partition, "
                       "shard and allowance (each generation counted once).", unit="one (source, partition, shard, "
                       "allowance)", source="02_draws/draws_all.csv (present draws)",
                       cols={"source": "legacy (forced_v2, reused) or passk (this study)", "partition": "draw "
                             "partition", "allowance_a": "reasoning allowance (units of A)", "think_in": "THINK prompt "
                             "tokens (sum)", "think_out": "THINK completion tokens (sum)", "answer_in": "ANSWER prompt "
                             "tokens (sum)", "answer_out": "ANSWER completion tokens (sum)", "cost_usd": "cost (sum, "
                             "USD)", "latency_s_sum": "sum of per-call latencies (s)", "requests": "generations",
                             "mean_cost_usd": "mean cost per generation", "mean_latency_s": "mean latency per "
                             "generation (s)"})
        krows = [{"time": t, "label": lab, "key_usage_usd": u, "key_limit_usd": lim, "limit_remaining_usd": rem,
                  "shard": None, "stage": None, "source": "memory note passk-cont-run.md"}
                 for t, lab, u, lim, rem in MEMORY_NOTE_KEY_USAGE]
        for shard in ("mancer", "parasail"):
            pu = self.path("artifacts", "shards", shard, "state", "provider_usage.jsonl")
            if not pu.exists():
                continue
            for r in _jsonl(pu):
                if r.get("endpoint") != "key":
                    continue
                data = r.get("data") or {}
                krows.append({"time": r.get("fetched_at"), "label": "OpenRouter /key snapshot",
                              "key_usage_usd": data.get("usage"), "key_limit_usd": data.get("limit"),
                              "limit_remaining_usd": data.get("limit_remaining"), "shard": shard,
                              "stage": r.get("stage"), "source": f"artifacts/shards/{shard}/state/provider_usage.jsonl "
                              "(usage fields only)"})
        self.table(f"{d}/key_usage_checkpoints.csv", pd.DataFrame(krows), purpose="OpenRouter key-usage "
                   "checkpoints: the memory-note values and every /key snapshot the dispatcher recorded.",
                   unit="one checkpoint", source="memory note + shard provider_usage.jsonl",
                   cols={"time": "time (memory note: local EDT; snapshots: UTC fetched_at)", "label": "what the "
                         "checkpoint is", "key_usage_usd": "cumulative key usage (USD; includes spend outside this "
                         "study before 181.32)", "key_limit_usd": "key limit (USD)", "limit_remaining_usd": "limit "
                         "remaining (USD)", "shard": "shard whose dispatcher recorded the snapshot",
                         "stage": "dispatcher stage recorded with the snapshot", "source": "source"})

    # =================================================================================================================
    # prompts / prereg / figures / code manifest
    # =================================================================================================================
    def section_prompts(self) -> None:
        tdir = self.code_root / "prompts"
        for name in ("system.txt", "root.txt", "private.txt", "peer.txt", "pooled.txt", "output_requirements.json"):
            used = name in ("system.txt", "root.txt", "output_requirements.json")
            self.copy(tdir / name, f"prompts/solver_templates/{name}",
                      purpose=("solver template used by every forced generation in this study (ROOT and DEEP nodes)"
                               if used else "legacy forced_v2 protocol template (PRIVATE/PEER/POOLED nodes); part of "
                               "the frozen template pack (prompt_version) but not used by this study's generations"))
        cfg = self.config()
        try:
            from agent_allocation.api import forced
            self.text("prompts/xhigh_system_preamble.txt", forced.XHIGH_INSTRUCTION + "\n",
                      purpose="the reasoning-effort preamble the Qwen3.8 chat template places at the start of every "
                              "system turn", source="src/agent_allocation/api/forced.py XHIGH_INSTRUCTION")
            proto = {"xhigh_instruction": forced.XHIGH_INSTRUCTION,
                     "generation_prompt_suffix": forced.GENERATION_PROMPT,
                     "chat_template": "qwen3.8-27b", "chat_template_sha256": forced.TEMPLATE_SHA256.get("qwen3.8-27b"),
                     "forced": {k: v for k, v in (cfg.get("forced") or {}).items() if k != "template_file"},
                     "sampling": cfg.get("sampling"), "completion_headroom_answer": (cfg.get("reasoning") or {}).get(
                         "text_completion_headroom"), "A_tokens": A_TOKENS,
                     "think_call": "completions endpoint; prompt = rendered chat prompt ending '<think>\\n'; "
                                   "max_tokens "
                                   "= allowance; logit_bias -100 on </think>, <|im_end|>, <|endoftext|> so the call "
                                   "ends exactly at the budget (finish_reason length)",
                     "answer_call": "prompt = THINK prompt + the realized reasoning + think_close_text; max_tokens = "
                                    "headroom (4,096); no logit_bias; up to answer_attempts attempts",
                     "provider": {"only": ["mancer/fp8 or parasail/fp8 (item's shard)"], "allow_fallbacks": False,
                                  "require_parameters": True}, "seed_sent": False, "response_cache": False}
            self.text("prompts/forced_protocol.json", json.dumps(proto, indent=1, sort_keys=True) + "\n",
                      purpose="forced THINK + ANSWER protocol constants", source="api/forced.py + configs/"
                      "passk_cont_v1.json")
        except Exception as exc:                                        # noqa: BLE001
            self.gap(f"api.forced not importable for the protocol constants: {exc}")
        self._prompt_examples()

    def _raw_prompt(self, path: Path) -> str | None:
        try:
            payload = json.loads(gzip.decompress(path.read_bytes()))
            return ((payload.get("request_body") or {}).get("think") or {}).get("prompt")
        except (OSError, ValueError):
            return None

    def _prompt_examples(self) -> None:
        ex_rows = []
        try:
            from agent_allocation.api.forced import render_chat_prompt
            from agent_allocation.contracts import GenerationSpec, load_record
            from agent_allocation.passk.extract import load_public_tasks, read_jobs
            from agent_allocation.prompts.render import render_messages
        except Exception as exc:                                        # noqa: BLE001
            self.gap(f"prompt rendering unavailable: {exc}")
            return
        tasks = load_public_tasks(self.path("artifacts"))
        sp = self.splits()
        by_b = (sp.get("panels_by_benchmark") or {}).get("T200") or {}
        assignment = self.assignment()
        draws = self.draws()
        seed = int(self.config().get("seed", 20260917))
        for b in BENCH:
            items = by_b.get(b) or []
            if not items or not assignment:
                self.gap(f"no T200 item for the {b} solver prompt example")
                continue
            item = items[0]
            shard = assignment.get(item)
            shard_dir = self.path("artifacts", "shards", shard)
            spec = None
            for node_id, rec in read_jobs(shard_dir, "t200_bank"):
                if (rec.get("item_id") == item and rec.get("kind") == "root" and int(rec.get("draw_id", -1)) == 0
                        and int(rec.get("reasoning_budget", 0)) == A_TOKENS):
                    spec = load_record(GenerationSpec, rec)
                    break
            task = tasks.get(item)
            if spec is None or task is None:
                self.gap(f"{b}: no t200_bank r0 spec or public task for the prompt example ({item})")
                continue
            prompt = render_chat_prompt(render_messages(spec, task, [], seed=seed))
            sent, verified = None, "not verified (raw request not found)"
            if draws is not None:
                hit = draws[draws.candidate_id == spec.node_id]
                if len(hit) and isinstance(hit.raw_response_path.iloc[0], str):
                    sent = self._raw_prompt(shard_dir / hit.raw_response_path.iloc[0])
            if sent is not None:
                verified = "true" if sent == prompt else "false"
            rel = f"prompts/examples/solver_think_prompt_{b}.txt"
            self.text(rel, prompt, purpose=f"exact THINK prompt of one {b} generation (T200 {item}, r0 at A)",
                      source="rendered with prompts/render.py + api/forced.render_chat_prompt")
            ex_rows.append({"file": rel, "kind": "solver THINK (ROOT at A; DEEP nodes render identically)",
                            "benchmark": b, "item_id": item, "node_id": spec.node_id, "arm": None,
                            "matches_sent_request": verified})
        diag = {}
        try:
            from agent_allocation.passk import e8a, e8b
            diag = e8a.diagnostic_answers(self.study_root) if self.path("analysis_inputs", "draws.parquet").exists() \
                else {}
            self.text("prompts/llm_selectors/e8a_constants.json", json.dumps({
                "SYSTEM": e8a.SYSTEM, "OPTION_TEXT": e8a.OPTION_TEXT, "USER_HEAD": e8a.USER_HEAD,
                "USER_DIAG": e8a.USER_DIAG, "USER_TAIL": e8a.USER_TAIL, "NO_ANSWER": e8a.NO_ANSWER,
                "MENU": list(e8a.MENU), "FALLBACK": e8a.FALLBACK, "THINK_TOKENS": e8a.THINK_TOKENS,
                "MAX_PARSE_ATTEMPTS": e8a.MAX_PARSE_ATTEMPTS, "prompt_sha256": e8a.prompt_sha256(),
                "option_order_salt": "passk_cont_v1/e8a/order|seed=20260925|<arm>|<item_id>; options sorted by "
                                     "sha256(salt|label)",
                "parse": "exactly one <final>...</final>; lowercased, whitespace removed, '\u00d7' -> 'x'; must be a "
                         "menu label"}, indent=1, ensure_ascii=False) + "\n",
                purpose="frozen E8a allocator prompt constants (PREREG_E8a)", source="src/agent_allocation/passk/"
                "e8a.py")
            self.text("prompts/llm_selectors/e8b_constants.json", json.dumps({
                "SYSTEM": e8a.SYSTEM, "OPTION_TEXT": e8b.OPTION_TEXT, "TYPICAL_SECONDS": e8b.TYPICAL_SECONDS,
                "USER_HEAD": e8b.USER_HEAD, "USER_DIAG": e8b.USER_DIAG, "USER_TAIL": e8b.USER_TAIL,
                "NO_ANSWER": e8a.NO_ANSWER, "MENU": list(e8b.MENU), "FALLBACK": E8B_FALLBACK, "SAMPLES": e8b.SAMPLES,
                "THINK_TOKENS": e8b.THINK_TOKENS, "prompt_sha256": e8b.prompt_sha256(),
                "option_order_salt": "passk_cont_v1/e8b/order|seed=20260925|<arm>|<item_id>|<sample>; options sorted "
                                     "by sha256(salt|label)"}, indent=1, ensure_ascii=False) + "\n",
                purpose="frozen E8b selector prompt constants (PREREG_E8b)", source="src/agent_allocation/passk/"
                "e8b.py")
            t200 = (sp.get("panels") or {}).get("T200") or []
            item = t200[0] if t200 else None
            task = tasks.get(item) if item else None
            for mod, arms, calls_rel, extra in ((e8a, ("A0", "A1"), "e8a/calls.jsonl", None),
                                                (e8b, ("B0", "B1"), "e8b/calls.jsonl", 0)):
                if task is None:
                    self.gap("no T200 task for the LLM-selector prompt examples")
                    break
                calls = _jsonl(self.path(calls_rel)) if self.path(calls_rel).exists() else []
                for arm in arms:
                    if extra is None:
                        msgs = mod.build_messages(arm, item, task.prompt, diag.get(item))
                    else:
                        msgs = mod.build_messages(arm, item, extra, task.prompt, diag.get(item))
                    prompt = render_chat_prompt(msgs)
                    hit = [r for r in calls if r.get("arm") == arm and r.get("item_id") == item
                           and r.get("parse_try") == 0 and r.get("transport_try") == 1
                           and (extra is None or r.get("sample") == extra)]
                    verified = "not verified (call not found)"
                    if hit and (self.study_root / hit[0]["raw"]).exists():
                        sent = self._raw_prompt(self.study_root / hit[0]["raw"])
                        verified = "not verified (raw unreadable)" if sent is None else \
                            ("true" if sent == prompt else "false")
                    tag = mod.__name__.rsplit(".", 1)[-1]
                    rel = f"prompts/llm_selectors/{tag}_example_{arm}.txt"
                    self.text(rel, prompt, purpose=f"exact THINK prompt of one {tag.upper()} {arm} decision (T200 "
                              f"{item}{'' if extra is None else f', sample {extra}'})",
                              source=f"rendered from passk/{tag}.py build_messages + api/forced.render_chat_prompt")
                    ex_rows.append({"file": rel, "kind": f"{tag.upper()} selector THINK", "benchmark": task.benchmark,
                                    "item_id": item, "node_id": hit[0]["node_id"] if hit else None, "arm": arm,
                                    "matches_sent_request": verified})
        except Exception as exc:                                        # noqa: BLE001
            self.gap(f"LLM-selector prompt examples not rendered: {exc}")
        if ex_rows:
            self.table("prompts/prompt_examples.csv", pd.DataFrame(ex_rows), purpose="Index of the rendered example "
                       "prompts and whether each equals the THINK prompt actually sent (raw request body).",
                       unit="one example prompt file", source="rendered + raw request bodies",
                       cols={"file": "package path of the example", "kind": "what kind of call",
                             "node_id": "node id of the call whose sent prompt was compared",
                             "arm": "selector arm (LLM examples)", "matches_sent_request": "true when the rendered "
                             "text is byte-identical to request_body.think.prompt of that call"})

    def section_prereg(self) -> None:
        docs = self.code_root / "docs" / "passk_cont"
        for name, purpose in (("PREREG.md", "core pre-registration (E0/E1/E2/E5), committed before any generation"),
                              ("PREREG_E8a_llm_allocator.md", "E8a addendum (LLM allocator), before any allocator "
                                                              "call"),
                              ("PREREG_LEX_front.md", "LEX addendum (accuracy -> consistency -> latency front)"),
                              ("PREREG_E8b_lex_llm.md", "E8b addendum (LLM selectors told the LEX objective), before "
                                                        "any E8b call"),
                              ("00_contract.md", "frozen build contract (parameters, partitions, schemas, fence)"),
                              ("RUNBOOK.md", "operations runbook")):
            self.copy(docs / name, f"prereg/{name}", purpose=purpose)
        if self.plan_path is not None and Path(self.plan_path).exists():
            self.copy(Path(self.plan_path), "prereg/plan.md", purpose="the approved study plan (adversarially "
                      "reviewed before approval)")
        else:
            self.gap(f"approved plan not found: {self.plan_path}")
        spec = MAIN_CHECKOUT / SPEC_REL
        if spec.exists():
            self.copy(spec, "prereg/spec_remaining_experiments.md", purpose="the experiment spec the plan implements "
                      "(fast path E0+E1+E2+E5; E8/LEX were owner-requested extensions)")

    def section_figures(self) -> None:
        rep = self.code_root / "docs" / "passk_cont_report"
        self.copy(rep / "report.pdf", "figures/report.pdf", purpose="the compiled core-study report")
        fdir = rep / "figures"
        if fdir.is_dir():
            for p in sorted(fdir.iterdir()):
                if p.is_file():
                    self.copy(p, f"figures/report/{p.name}", purpose="report figure")
        else:
            self.gap("docs/passk_cont_report/figures/ missing")
        e0 = self.path("e0")
        if e0.is_dir():
            for p in sorted(e0.glob("*.png")):
                self.copy(p, f"figures/e0/{p.name}", purpose="E0 decomposition plot (data: csv/01_e0_audit/"
                          "decomposition_plot_data.csv)")

    def section_code_manifest(self) -> None:
        lines = ["passk_cont_v1 results package: code manifest", "=" * 44, ""]
        head = self.git("rev-parse", "HEAD")
        lines += [f"worktree: {self.code_root.parent}", f"branch: {self.git('rev-parse', '--abbrev-ref', 'HEAD')}",
                  f"HEAD: {head}  ({self.git('log', '-1', '--format=%s') or ''})"]
        status = self.git("status", "--porcelain", "--untracked-files=no")
        lines.append("tracked-file changes at export: " + ("none" if not status else
                                                            "; ".join(status.splitlines())))
        lines += ["", "Key commits", "-----------"]
        for label, short in KEY_COMMITS:
            full = self.git("rev-parse", "--verify", "--quiet", f"{short}^{{commit}}")
            subj = self.git("log", "-1", "--format=%s", short) if full else None
            lines.append(f"{(full or short + ' (not found)')}  {label}" + (f"\n    {subj}" if subj else ""))
        lines += ["", "Modules (src/agent_allocation/passk and the report)", "----------------------------------------"]
        src = self.code_root / "src" / "agent_allocation" / "passk"
        mods = sorted([p for p in src.rglob("*.py") if "__pycache__" not in p.parts and p.name != "__init__.py"])
        rep = self.code_root / "docs" / "passk_cont_report"
        mods += sorted(p for p in rep.glob("*.py"))
        for p in mods:
            try:
                doc = ast.get_docstring(ast.parse(p.read_text(encoding="utf-8"))) or ""
            except SyntaxError:
                doc = ""
            first = doc.strip().splitlines()[0] if doc.strip() else "(no module docstring)"
            lines.append(f"{p.relative_to(self.code_root)}: {first}")
        for p in sorted((self.code_root / "slurm").glob("passk_*")):
            lines.append(f"{p.relative_to(self.code_root)}: Slurm wrapper (see RUNBOOK.md)")
        lines += ["", "Commands (per stage)", "--------------------"]
        for stage, cmd in STAGE_COMMANDS:
            lines.append(f"[{stage}]\n    {cmd}")
        lines += ["", "Interpreter: agent_allocation/.venv/bin/python with HF_HOME=/orcd/data/tpoggio/001/mabdel03/"
                  ".cache/huggingface HF_HUB_OFFLINE=1 MPLCONFIGDIR=/orcd/data/tpoggio/001/mabdel03/.cache/matplotlib "
                  "PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1", ""]
        self.text("code_manifest.txt", "\n".join(lines), purpose="git commit, branch, key commits, module list with "
                  "one-line purposes, and the commands of every stage", source="git + module docstrings + RUNBOOK")

    # =================================================================================================================
    # dictionary + manifest
    # =================================================================================================================
    def write_dictionary(self) -> None:
        L = ["# DATA_DICTIONARY: passk_cont_v1 results package", "",
             "Generated by `python -m agent_allocation.passk.export_package` (deterministic; re-run to refresh). "
             "Every CSV is UTF-8 with one header row and no index column; lists and objects are JSON strings; "
             "booleans are `true`/`false`; an empty cell is a missing value (for correctness: unknown).", "",
             "Conventions used throughout: A = 1,024 reasoning tokens; an allocation `n x a` (config_id `NxA`, e.g. "
             "`4x4`) is n independent calls each forced to exactly a*A THINK tokens, answered with a 4,096-token "
             "ANSWER call, and read out by plurality over the calls' final answers (answer classes: exact vote key, "
             "then mutual complete-link Math-Verify; GPQA by option letter; ties by a seeded uniform draw). The "
             "mixture estimand weights benchmarks aime .10, hmmt .10, omni_math_hard .50, gpqa_diamond .30; macro "
             "weights them equally. Bootstrap = benchmark-stratified, item-clustered, 10,000 replicates, seed "
             "20260925. Correctness is tri-state: true, false, unknown (empty; a 5 s Math-Verify timeout); invalid "
             "output counts as false.", ""]
        L += ["## Contents", ""]
        for t in self.tables:
            L.append(f"- [`{t.rel}`](#{_anchor(t.rel)}) ({t.rows:,} rows): {t.purpose}")
        L += ["", "## CSV files", ""]
        for t in self.tables:
            L += [f"### {t.rel}", "", f"- Purpose: {t.purpose}", f"- Row unit: {t.unit}", f"- Rows: {t.rows:,}",
                  f"- Source: {t.source}"]
            if t.notes:
                L.append(f"- Notes: {t.notes}")
            L += ["", "| column | type | meaning |", "|---|---|---|"]
            for c, typ, meaning in t.columns:
                L.append(f"| `{c}` | {typ} | {_md(meaning)} |")
            L.append("")
        L += ["## Other files", "", "| path | what it is | source |", "|---|---|---|"]
        for f in sorted(self.files, key=lambda f: f.rel):
            L.append(f"| `{f.rel}` | {_md(f.purpose)} | {_md(f.source)} |")
        L += ["| `MANIFEST.csv` | every file with bytes, sha256 and CSV row count | this exporter |",
              "| `DATA_DICTIONARY.md` | this file | this exporter |",
              "| `README.md` | the study write-up (written separately; never modified by the exporter) | write-up stage |"]
        for p in sorted(self.out.iterdir()) if self.out.exists() else []:
            if p.is_file() and p.name not in OWNED and p.name != README:
                L.append(f"| `{p.name}` | written by another stage at the package root (never modified by the "
                         "exporter) | not this exporter |")
        L.append("")
        undocumented = [(t.rel, c) for t in self.tables for c, _, m in t.columns if m.startswith("UNDOCUMENTED")]
        L += ["## Checks run by the exporter", ""]
        L += [f"- {name}: {result}" for name, result in self.checks] or ["- none"]
        L += ["", "## Gaps and caveats", ""]
        L += [f"- {g}" for g in self.gaps] or ["- none"]
        if undocumented:
            L += ["", f"{len(undocumented)} column(s) above are marked UNDOCUMENTED: their meaning was not verified "
                      "from the code and should not be guessed."]
        L.append("")
        (self.out / "DATA_DICTIONARY.md").write_text("\n".join(L), encoding="utf-8")

    def write_manifest(self) -> None:
        rows_by_rel = {t.rel: t.rows for t in self.tables}
        rows = []
        for p in sorted(self.out.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self.out).as_posix()
            if rel == "MANIFEST.csv":
                continue
            n = rows_by_rel.get(rel)
            if n is None and p.suffix == ".csv":
                with p.open(encoding="utf-8", newline="") as fh:
                    n = max(sum(1 for _ in csv.reader(fh)) - 1, 0)
            rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": _sha256(p), "rows": n})
        frame = pd.DataFrame(rows, columns=["path", "bytes", "sha256", "rows"])
        frame["rows"] = frame["rows"].astype("Int64")
        write_csv(frame, self.out / "MANIFEST.csv")

    # =================================================================================================================
    def prepare_out(self) -> None:
        """Clear only what this exporter owns (``OWNED``); README.md and any other file a write-up stage put at the
        package root (e.g. a methods draft) are left exactly as they are. An owned name present without the
        MANIFEST.csv of an earlier export means ``--out`` is not a package: refuse rather than delete it."""
        out = self.out
        if out.exists():
            entries = [p for p in out.iterdir() if p.name in OWNED]
            names = {p.name for p in entries}
            if names and "MANIFEST.csv" not in names:
                raise RuntimeError(f"{out} holds {sorted(names)} but no MANIFEST.csv from an earlier export; "
                                   "refusing to clear it")
            for p in entries:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        out.mkdir(parents=True, exist_ok=True)
        for sub in TOP_DIRS:
            (out / sub).mkdir(exist_ok=True)
        for sub in CSV_DIRS:
            (out / "csv" / sub).mkdir(parents=True, exist_ok=True)

    def _foreign(self) -> dict[str, str]:
        """sha256 of every file under the package that the exporter does not own (README.md, drafts)."""
        if not self.out.exists():
            return {}
        out = {}
        for p in sorted(self.out.rglob("*")):
            rel = p.relative_to(self.out)
            if p.is_file() and rel.parts[0] not in OWNED:
                out[rel.as_posix()] = _sha256(p)
        return out

    def run(self) -> dict[str, Any]:
        before = self._foreign()
        self.prepare_out()
        for section in (self.section_study, self.section_e0, self.section_draws, self.section_pools, self.section_e1,
                        self.section_e2, self.section_e5, self.section_e8a, self.section_lex, self.section_selcmp,
                        self.section_e8b, self.section_costs, self.section_prompts, self.section_prereg,
                        self.section_figures, self.section_code_manifest):
            section()
        after = self._foreign()
        changed = sorted({k for k in set(before) | set(after) if before.get(k) != after.get(k)})
        self.check("files not owned by the exporter (README.md, drafts) untouched",
                   "ok" if not changed else f"changed during the export by another writer: {changed}")
        self.write_dictionary()
        self.write_manifest()
        return {"out": str(self.out), "csv_files": len(self.tables), "other_files": len(self.files),
                "rows": sum(t.rows for t in self.tables), "gaps": list(self.gaps),
                "checks": [f"{n}: {r}" for n, r in self.checks]}


def _anchor(text: str) -> str:
    return re.sub(r"[^a-z0-9 _-]", "", text.lower()).replace(" ", "-")


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


# =====================================================================================================================
# per-table column docs
# =====================================================================================================================

DRAW_DOCS = {
    "source_table": "analysis_inputs/draws (legacy + every passk partition) or lex/c48_25a_draws (LEX 25A draws)",
    "source": "legacy (reused forced_v2 generation) or passk (generated by this study)",
    "partition": "generation partition: legacy_main (L96 A slots 0-24 + deep r0 at 2/4/8/16/25A), legacy_pilot (D32 "
                 "roots at A), smoke (Gate 1, D32 x A, r100), e1_fill (L96 2A r1-7, 4A r1-3, 8A r1), c48_extra (C48 4A "
                 "r4-7, 8A r2-7, 16A r1-7), v48_bank (V48 A r0-15, 2A r0-7, 4A r0-3, 8A r0-1, 16A r0), t200_bank (same "
                 "31-call bank on T200), t200_diag (T200 A r0-4 diagnostics), c48_25a (C48 25A r1-8, LEX)",
    "draw_role": "legacy: pool (A slots 0-15: 16xA pool + E2 fitting), bank (slots 16-19: finite-bank only), diag "
                 "(slots 20-24: five-draw diagnostics, features only), deep_r0 (the legacy deep draw); passk: the "
                 "partition name",
    "panel_roles": "panels the item belongs to (JSON list)",
    "provider_only": "provider.only pin sent with the THINK request",
    "kind": "root (one call at A) or deep (one call at >= 2A; same template as root)",
    "allowance_a": "reasoning allowance of the call (units of A = 1,024 tokens)",
    "reasoning_budget_tokens": "THINK max_tokens = allowance_a * 1,024 (forced exactly)",
    "replicate": "replicate index at this (item, allowance) within its partition family (root_slot = draw_id); "
                 "legacy A slots 0-24, legacy deep r0, smoke r100",
    "completion_headroom": "ANSWER call max_tokens",
    "absent": "the node ended without a candidate (failed after its attempts); never replaced",
    "pending": "node never settled at extract time",
    "graded": "a grade record exists",
    "parse_status": "extractor status of the ANSWER text: valid, missing_final, malformed_final, empty_content",
    "final_present": "an extracted final answer exists",
    "vote_exact_key": "exact vote key of the final answer (normalized); GPQA: the option letter",
    "vote_text": "normalized final-answer text used for Math-Verify equivalence",
    "option": "GPQA option letter (gpqa_diamond only)",
    "answer_len_chars": "characters of visible ANSWER text",
    "answer_finish_reason": "finish reason of the ANSWER call (stop or length)",
    "answer_failures": "failed ANSWER attempts before success",
    "correct": "tri-state correctness: true, false (wrong or invalid output), empty (unknown: grader timeout, or not "
               "graded)",
    "grade_state": "correct, wrong, invalid (invalid output), unknown (grader could not decide), missing (no "
                   "candidate)",
    "think_latency_s": "latency of the THINK call (s)",
    "latency_s": "end-to-end latency of the generation, THINK + ANSWER (s)",
    "think_in": "THINK prompt tokens", "think_out": "THINK completion tokens (= the allowance, forced)",
    "answer_in": "ANSWER prompt tokens", "answer_out": "ANSWER completion tokens",
    "cost_usd": "cost of the generation (THINK + ANSWER, USD)",
    "generation_id_think": "OpenRouter generation id of the THINK call",
    "generation_id_answer": "OpenRouter generation id of the ANSWER call",
    "think_model": "model served for THINK", "think_provider": "provider served THINK",
    "answer_model": "model served for ANSWER",
}

POOL_DOCS = {
    "mode": "tie rule of this readout: passk (seeded uniform tie among modal classes)",
    "status": "single (one-call pool), selected (plurality over several calls), abstained (no vote)",
    "delivered_id": "candidate delivered (first member of the winning class)",
    "winner_class": "winning answer class id",
    "support": "votes in the winning class",
    "n_votes": "voting members (present, with a vote key)",
    "n_classes": "answer classes among the votes",
    "n_tied": "modal classes tied (1 = no tie)",
    "class_sizes": "votes per class in build order (JSON list)",
    "equivalence_unknown": "Math-Verify pair verdicts that were unknown (timeout) during the class build",
    "delivered_correct": "correctness of the delivered answer (tri-state; abstained = false)",
    "tie_avg_lo": "analytic tie-averaged delivered correctness, unknown grades counted wrong",
    "tie_avg_hi": "analytic tie-averaged delivered correctness, unknown grades counted right",
    "n_member_correct": "members graded correct", "n_member_unknown": "members with unknown grade",
    "n_absent": "members that ended without a candidate",
    "H_lo": "coverage lower bound (some member correct; unknown = wrong)", "H_hi": "coverage upper bound",
    "S_lo": "delivered correctness lower bound", "S_hi": "delivered correctness upper bound",
    "S_known": "delivered correctness when determined (empty when unknown or the pool is all-absent)",
    "H_known": "coverage when determined", "S_tie": "tie-averaged delivered correctness when determined",
    "entropy": "answer-class entropy over votes (nats)",
    "top_wrong_mass": "votes in the largest class whose first member is known wrong / n_votes",
    "S_operational": "operational delivered correctness (failed delivery = failure) = S_lo",
    "all_absent": "every member absent (missing in the primary analysis)",
    "gen_fail": "decomposition: generation failure 1 - H_known",
    "avail_loss": "decomposition: availability loss H_arc - H_cur (0 for independent pools)",
    "sel_fail": "decomposition: selection failure H_known - S_known",
    "n_calls": "calls in the pool (= n)",
    "in_e5_dev_training": "the pool is a row of the E5 development training table (L96/C48 training configurations, "
                          "withheld configurations excluded, one row per shared pool)",
    "wallclock_s": "derived: parallel wall-clock of the pool = max member latency_s (the realized latency used by the "
                   "selector comparison)",
    "cost_usd_sum": "derived: sum of member costs (USD; logical cost, members shared with other pools counted again)",
    "think_tokens_sum": "derived: sum of member THINK tokens",
}

FEATURE_DOCS = {
    "provider": "shard of the diagnostics (mancer or parasail)",
    "collection": "where the five diagnostics come from: legacy (L96 A slots 20-24), t200_diag (T200 r0-4), imputed",
    "n_diag": "diagnostic draws used (5)",
    "f1": "CONC: plurality share of the five diagnostic answers (largest class / 5)",
    "f2": "CONC: second-largest class / 5", "f3": "CONC: third-largest class / 5", "f4": "CONC: fourth class / 5",
    "f5": "CONC: fifth class / 5",
    "top_gap": "CONC: f1 - f2", "n_classes": "CONC: number of answer classes / 5",
    "entropy": "CONC: entropy of the voter class distribution, normalized by log 5",
    "invalid_frac": "fraction of the five diagnostics casting no vote",
    "log_answer_len": "mean log(1 + ANSWER characters)",
    "finish_length_frac": "fraction of diagnostics whose ANSWER finished by length",
    "latency_z": "mean within-group latency z-score (group = provider x allowance x collection); dropped by the "
                 "grouped-CV latency check (use_latency = false)",
}

DEV_FRAME_DOCS = {
    "coverage": "observed coverage of the development pool (some member correct)",
    "delivered": "E5 training target: observed plurality delivered correctness (plurality_source tells which readout)",
    "plurality_tieavg": "tie-averaged plurality correctness",
    "all_absent": "every member absent",
    "plurality_source": "tie_averaged (analytic tie average), readouts_seeded (seeded readout) or all_absent",
    "partition": "draw partition", "a": "allowance (units of A)", "y": "draw correctness (1/0)",
}

MODEL_FIT_DOCS = {
    "model_file": "model file under sealed/", "model": "model / predictor", "fit": "which fit: response (per-call "
    "glm), eta (P(delivered | covered)), delivered / coverage / plurality (ridge targets)",
    "intercept": "fitted intercept (standardized design)", "sigma_item": "sd of the Gauss-Hermite item random "
    "intercept (0 = no item intercept)", "item_intercept": "fit has an item random intercept", "ridge": "ridge "
    "strength used", "gh_nodes": "Gauss-Hermite nodes", "converged": "optimizer converged", "n_iter": "optimizer "
    "iterations", "objective": "final penalized negative log-likelihood", "n_coef": "number of coefficients",
}
MODEL_COEF_DOCS = {
    "model_file": "model file under sealed/", "model": "model / predictor", "fit": "which fit (see *_model_fits.csv)",
    "index": "coefficient position", "column": "design column name (bench:<b> dummies, log_n, log_a = log a/A, R, "
    "B = n*a/16, provider:parasail, the five-draw features, <benchmark>:d<depth> cells, <feature>*log2a, products "
    "a*b in P3; eta: logit_p_call_oof = logit of the out-of-fold per-call prediction)",
    "column_name_source": "stored (in the model file) or derived from the design code (e2._ridge_row, "
                          "predictors.eta_design)",
    "coef": "coefficient on the standardized column", "x_mean": "column mean used for standardization",
    "x_scale": "column scale used for standardization",
}

E5_SUMMARY_DOCS = {
    "primary_brier_mixture": "primary metric: mixture-weighted mean over items of the item-level mean Brier over the "
                             "known 16A configurations",
    "primary_brier_macro": "the same, equal benchmark weights",
    "n_items_primary": "items entering the primary metric",
    "coverage_brier_16A": "Brier of p_coverage vs observed coverage, 16A cells",
    "log_loss_16A": "log loss of p_delivered, 16A cells", "mae_16A": "mean absolute error, 16A cells",
    "ece_16A": "expected calibration error (10 bins), 16A cells",
    "ranking_accuracy_16A": "paired ranking accuracy of the 16A configurations within items (ties 1/2)",
    "brier_withheld_4x4": "Brier on the withheld 4x4 cell", "brier_withheld_8A": "Brier on the withheld "
    "compositional 8A cells (8x1, 4x2, 2x4)", "brier_all_8A": "Brier on all four 8A cells",
}

E8_DOCS = {
    "panel": "panel: E8a L96/T200; E8b C48/T200",
    "parse_try": "0 = first decision call; 1 = the outcome-blind retry after an invalid choice",
    "transport_try": "transport attempt within a parse try (1-3)",
    "transport_status": "transport outcome (ok)",
    "choice": "parsed option label (empty when invalid)",
    "valid": "the reply contained exactly one <final>LABEL</final> with a menu label",
    "chosen": "option the decision is scored with: the choice, or the 1x16 fallback when still invalid after the "
              "retry (final rows only)",
    "final": "this row ends the decision (valid choice, or the retry)",
    "visible": "visible ANSWER text, truncated (E8a 400 characters, E8b 300)",
    "prompt_sha256": "sha256 of the frozen prompt constants",
    "option_order": "order in which the menu options were shown (JSON list; seeded per item/arm(/sample))",
    "latency_s": "decision call latency (s); E8a: result.latency_seconds from the raw response",
}

E8A_ITEM_DOCS = {
    "16x1": "realized delivered correctness (S_known, seeded ties) of 16x1", "1x16": "realized delivered correctness "
    "of 1x16", "2x8": "realized delivered correctness of 2x8", "4x4": "realized delivered correctness of 4x4",
    "8x2": "realized delivered correctness of 8x2",
    "A0_choice": "A0 chosen option (fallback 1x16 when invalid after retry)", "A0_valid": "A0 choice was valid",
    "A0": "A0 realized delivered correctness", "A0_invalid0": "A0 with invalid scored 0 (sensitivity)",
    "A1_choice": "A1 chosen option", "A1_valid": "A1 choice was valid", "A1": "A1 realized delivered correctness",
    "A1_invalid0": "A1 with invalid scored 0", "always_deep": "1x16 realized correctness",
    "random": "mean of the five options (uniform random choice)", "oracle": "best realized option (clairvoyant)",
    "P1_choice": "sealed P1's highest-predicted M5 option", "P1_select": "realized correctness of P1_choice",
}

SELCMP_ITEM_DOCS = {
    "kind": "fixed, reference (1x25, split-half), stat, stat-sealed, heuristic, llm, ceiling (oracles); later "
            "versions may add further kinds (e.g. E8b)",
    "predictor": COMMON["predictor"],
    "imputed": COMMON["imputed"],
    "n": COMMON["n"], "a": COMMON["a"],
}

SELCMP_SUMMARY_DOCS = {
    "kind": SELCMP_ITEM_DOCS["kind"],
    "choices": "option -> items (JSON object)",
    "mean_overhead_s": "macro mean selector overhead (s)",
    "mean_solver_A": "macro mean solver compute n*a of the chosen options (units of A)",
    "lex_rank_deployable": "rank in the C48 population LEX ordering of deployable selectors (empty = reference)",
    "lex_rank_all": "rank in the C48 LEX ordering including references and oracles",
    "winner_share_deployable": "share of bootstrap replicates in which the selector is the C48 LEX winner",
    "lex_rank_macro": "rank in the T200 LEX ordering (accuracy macro -> latency)",
    "lex_rank_mixture": "rank in the T200 LEX ordering (accuracy mixture -> latency)",
    "winner_share_macro": "share of bootstrap replicates in which the selector is the T200 LEX winner (macro)",
    "winner_share_mixture": "the same, mixture estimand",
    "n_unknown_acc": "T200 items whose chosen option's outcome is unknown",
    "acc_n_items": "items in the accuracy mean", "C_n_items": "items in the consistency mean",
    "lat_n_items": "items in the latency mean",
    "acc_mixture_n_items": "items in the mixture accuracy mean",
    "acc_macro_n_items": "items in the macro accuracy mean",
    "lat_macro_n_items": "items in the macro latency mean", "lat_mixture_n_items": "items in the mixture latency mean",
}

CROSSFIT_DOCS = {
    "fold": "outer fold 0-4 (its L96 items held out) or full_fit (all L96 -> T200)",
    "n_train_items": "training items", "n_test_items": "predicted items", "n_train_pools": "training pools",
    "n_train_draws": "training draws (P2 response model)",
    "train_configs_present": "training configurations present (JSON; withheld 4x4, 8x1, 4x2, 2x4 absent)",
    "latency_choice_use_latency": "the latency feature was kept (grouped CV improved) - never",
    "latency_choice_cv_log_loss_without": "grouped-CV log loss of P1 without latency",
    "latency_choice_cv_log_loss_with": "grouped-CV log loss of P1 with latency",
    "best_ridges": "chosen ridge per predictor and target (JSON)", "test_items": "held-out items (JSON)",
    "seconds": "fit time (s)",
}

HEUR_DOCS = {
    "fit": "c48_crossfit (per outer fold, applied to the fold's C48 items) or t200_fit_on_L96",
    "fold": "fold (fold0-4) or all_L96", "menu": "menu X is chosen from (M15 or M5)",
    "tau": "threshold on f1 (plurality share of the five diagnostic answers)", "X": "option used when f1 >= tau "
    "(otherwise 1x16)", "train_acc": "macro training accuracy of the chosen rule", "train_C": "macro training "
    "consistency (bank items only)", "train_lat": "macro training latency incl. diagnostics (s)", "n_train_items":
    "training items", "rule": "rule label", "acc": "macro training accuracy", "lat": "macro training latency (s)",
    "tokens": "n*a of X", "share_X": "share of training items routed to X",
    "C": "macro training consistency (bank items only)",
}


# =====================================================================================================================
# entry points
# =====================================================================================================================

def export(study_root: Path | str = STUDY_ROOT, out: Path | str = DEFAULT_OUT, *, code_root: Path | str = CODE_ROOT,
           plan_path: Path | str | None = PLAN_PATH, load_strata: bool = True) -> dict[str, Any]:
    pkg = Package(Path(study_root), Path(out), code_root=Path(code_root),
                  plan_path=Path(plan_path) if plan_path else None, load_strata=load_strata)
    return pkg.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passk.export_package")
    parser.add_argument("--study-root", default=str(STUDY_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--code-root", default=str(CODE_ROOT))
    parser.add_argument("--plan", default=str(PLAN_PATH))
    parser.add_argument("--no-strata", action="store_true", help="skip loading the benchmark inventories (strata)")
    args = parser.parse_args(argv)
    summary = export(args.study_root, args.out, code_root=args.code_root, plan_path=args.plan,
                     load_strata=not args.no_strata)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
