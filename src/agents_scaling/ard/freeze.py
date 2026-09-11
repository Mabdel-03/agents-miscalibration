"""ARD36 freeze: write ``<run_root>/FROZEN.yaml`` once, before any generate cell runs.

The study package's :func:`agents_scaling.study.config.freeze` cannot be reused directly: it
copies study_v4's own amendment and prior-exposure registers out of ``study/freeze/`` and
stamps ``study/prompts``' hashes.  ARD36 carries its own registers in ``ard/registers/`` and
its own contract, so this module mirrors the *discipline* (write-once, atomic, registers
copied and hashed, refuse a second freeze) with ARD36's content.

What the manifest must pin, so that nothing downstream can silently re-base:

* the config bytes, the code version and the frozen contract constants;
* the FLOP oracle convention and its per-checkpoint hash, which is the cost axis;
* the item table's content hashes, its split salt and the grader hashes, which are the
  measurement axis;
* the tier actually being run and its decision counts, so the ledger, the coverage gates
  and the cost tables all reconcile against one number.

A second freeze is a new run root, never a silent edit (spec §9.1).

    python -m agents_scaling.ard.freeze --run-id ard36 [--tier reduced|full]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from agents_scaling.ard import contract as C
from agents_scaling.study.types import ProtocolError

REGISTERS_DIR = Path(__file__).resolve().parent / "registers"
AMENDMENTS_FILE = "amendments.json"
PRIOR_EXPOSURE_FILE = "prior_exposure_manifest.json"
FROZEN_FILENAME = "FROZEN.yaml"

DEFAULT_RESULTS_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results")


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _code_version() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[3], timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def _contract_snapshot() -> dict[str, Any]:
    """The frozen constants a reader would otherwise have to trust the code for."""
    return {
        "contract_version": C.CONTRACT_VERSION,
        "study_id": C.STUDY_ID,
        "seed_namespace": C.SEED_NAMESPACE,
        "n_agents": C.N_AGENTS,
        "update_rounds": C.UPDATE_ROUNDS,
        "early_stop_on_unanimity": C.EARLY_STOP_ON_UNANIMITY,
        "sampler": {
            "temperature": C.TEMPERATURE, "top_p": C.TOP_P, "top_k": C.TOP_K,
            "min_p": C.MIN_P, "presence_penalty": C.PRESENCE_PENALTY,
            "repetition_penalty": C.REPETITION_PENALTY,
            "enable_thinking": C.ENABLE_THINKING, "dtype": C.DTYPE,
        },
        "budgets": {
            "thinking_low": C.THINKING_TOKENS_LOW, "thinking_high": C.THINKING_TOKENS_HIGH,
            "final": C.FINAL_TOKENS, "root_thinking": C.ROOT_THINKING_TOKENS,
            "max_tokens_low": C.max_tokens_for(C.THINKING_TOKENS_LOW),
            "max_tokens_high": C.max_tokens_for(C.THINKING_TOKENS_HIGH),
        },
        "prompt_budget": {
            "task": C.TASK_TOKENS_CAP, "report": C.REPORT_TOKENS_CAP,
            "template_overhead": C.TEMPLATE_OVERHEAD_CAP,
            "prompt_cap": C.PROMPT_TOKENS_CAP, "reserve": C.CONTEXT_RESERVE_TOKENS,
        },
        "bank_layout": {
            "root_draws": C.ROOT_DRAWS, "team": list(C.TEAM_DRAWS),
            "auxiliary": list(C.AUXILIARY_DRAWS), "fresh_base": C.FRESH_DRAW_BASE,
            "fresh_prefix_k": list(C.FRESH_PREFIX_K),
        },
        "splits_per_family": dict(C.SPLIT_COUNTS_PER_FAMILY),
        "families": [f.value for f in C.FAMILIES],
        "protocols": [p.value for p in C.Protocol],
        "profiles": {"small": C.PROFILE_SMALL, "large": C.PROFILE_LARGE,
                     "max_model_len": C.MAX_MODEL_LEN},
        "revisions": dict(C.REVISIONS),
        "capture": {"anchors": list(C.CAPTURE_ANCHORS),
                    "relative_layers": list(C.CAPTURE_RELATIVE_LAYERS),
                    "entropy_anchors": sorted(C.ENTROPY_ANCHORS)},
        "statistics": {"primaries": list(C.PRIMARY_CONTRASTS), "n_primary": C.N_PRIMARY,
                       "n_resamples": C.N_RESAMPLES, "alpha": C.ALPHA,
                       "noninferiority_margin": C.NONINFERIORITY_MARGIN},
        "tables": list(C.ALL_TABLES),
        "forbidden_feature_sources": sorted(C.FORBIDDEN_FEATURE_SOURCES),
    }


#: Decision counts per tier, recomputed from the bundle ledger formulas (plan §3).
TIERS: dict[str, dict[str, Any]] = {
    "reduced": {
        "note": "spec §16.4 steps 1-3 applied prospectively at T0 because the clock starts "
                "at implementation: no HUB5, E2d 16 items, challenge <=48, receiver 32 "
                "programs, selector audit on two pools, extensions off",
        "small": {"roots": 6144, "natural": 23040, "high_budget": 480, "challenge": 7200,
                  "forks": 1440, "closed_loop": 7200, "suffix": 1280, "receiver": 640},
        "large": {"roots": 1536, "natural": 4320, "high_budget": 480, "suffix": 1280,
                  "receiver": 640},
    },
    "full": {
        "note": "the spec's default 71,600-decision ledger",
        "small": {"roots": 6144, "natural": 28800, "high_budget": 960, "challenge": 14400,
                  "forks": 1440, "closed_loop": 7200, "selector_audit": 720, "suffix": 1280,
                  "receiver": 1280},
        "large": {"roots": 1536, "natural": 4320, "high_budget": 960, "suffix": 1280,
                  "receiver": 1280},
    },
}


def tier_totals(tier: str) -> dict[str, int]:
    spec = TIERS[tier]
    return {"small_logical_decisions": sum(spec["small"].values()),
            "large_logical_decisions": sum(spec["large"].values())}


def freeze(run_root: str | os.PathLike, *, tier: str = "reduced",
           config_path: str | os.PathLike | None = None,
           extra: Mapping[str, Any] | None = None) -> Path:
    """Write ``FROZEN.yaml`` once.  Raises if the run root is already frozen."""
    root = Path(run_root)
    target = root / FROZEN_FILENAME
    if target.exists():
        raise ProtocolError(
            f"{target} already exists; a study is frozen once per run root "
            f"(a second freeze is a new run root, never a silent edit)"
        )
    if tier not in TIERS:
        raise ProtocolError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")

    root.mkdir(parents=True, exist_ok=True)
    freeze_dir = root / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)

    registers: dict[str, str] = {}
    for name in (AMENDMENTS_FILE, PRIOR_EXPOSURE_FILE):
        src = REGISTERS_DIR / name
        json.loads(src.read_text(encoding="utf-8"))          # must be valid JSON
        shutil.copyfile(src, freeze_dir / name)
        registers[name] = _sha256_file(src)

    cfg_path = Path(config_path) if config_path else (
        Path(__file__).resolve().parents[3] / "configs" / "ard36.yaml")
    config_block: dict[str, Any] = {"path": str(cfg_path)}
    if cfg_path.exists():
        config_block["sha256"] = _sha256_file(cfg_path)
        try:
            config_block["raw"] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:                              # pragma: no cover
            config_block["raw_error"] = str(exc)

    # The item table is the measurement axis; pin its content hashes and salts.
    items_block: dict[str, Any] = {}
    summary = root / "data" / "public" / "items_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text(encoding="utf-8"))
        items_block = {k: s.get(k) for k in
                       ("counts", "totals", "dataset_revisions", "gold_files", "grader_hashes",
                        "split_salt_sha256", "study_seed_sha256", "split_rule", "rank_rule",
                        "option_permutation_namespace", "dropped", "eligible", "reserve",
                        "public_columns")}
        items_block["items_summary_sha256"] = _sha256_file(summary)
        for rel in ("data/public/items.parquet", "data/public/items_reserve.json"):
            p = root / rel
            if p.exists():
                items_block.setdefault("files", {})[rel] = _sha256_file(p)
    else:
        items_block["missing"] = "items_build has not been run for this run root"

    # The FLOP oracle convention is the cost axis; pin it or nothing downstream reconciles.
    oracle_block: dict[str, Any]
    try:
        from agents_scaling.ard import client as _client
        from agents_scaling.study.config import load_config

        cfg = load_config(cfg_path)
        oracle_block = _client.oracle_tables(cfg.checkpoints)
    except Exception as exc:
        oracle_block = {"error": f"{type(exc).__name__}: {exc}",
                        "note": "cost tables cannot reconcile without this; fix before dispatch"}

    manifest: dict[str, Any] = {
        "study_id": C.STUDY_ID,
        # Top level, not only inside ``config``: ``study.config.StudyConfig.frozen`` -- the
        # freeze gate every generate cell calls before its first request -- reads
        # ``config_sha256`` from the top level and refuses the cell when it does not match.
        # Writing the hash only as ``config.sha256`` left that read returning None and
        # suspended every cell in the run (defect D-05).
        "config_sha256": config_block.get("sha256"),
        "run_root": str(root),
        "frozen_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "config": config_block,
        "contract": _contract_snapshot(),
        "tier": {"name": tier, **TIERS[tier], "totals": tier_totals(tier)},
        "items": items_block,
        "oracle_tables": oracle_block,
        "freeze_registers": registers,
        "spec": {
            "path": "expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md",
            "md5": "9c261a4968e1b947ce88e428adc7b03b",
        },
        "extra": dict(extra or {}),
    }
    if not manifest.get("config_sha256"):
        raise ProtocolError(
            f"cannot freeze: the study config at {cfg_path} is missing or unreadable, so "
            "FROZEN.yaml would carry no top-level config_sha256 -- and the freeze gate every "
            "generate cell calls would then refuse the whole run (defect D-05)"
        )
    payload = yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True,
                             default_flow_style=False)
    _atomic_write_bytes(target, payload.encode("utf-8"))
    return target


def load_frozen(run_root: str | os.PathLike) -> Mapping[str, Any]:
    path = Path(run_root) / FROZEN_FILENAME
    if not path.exists():
        raise ProtocolError(f"{path} is missing: freeze the study before running generate cells")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ProtocolError(f"{path} is not a mapping")
    if data.get("study_id") != C.STUDY_ID:
        raise ProtocolError(f"{path} study_id mismatch: {data.get('study_id')!r}")
    if data.get("contract", {}).get("contract_version") != C.CONTRACT_VERSION:
        raise ProtocolError(
            f"{path} was frozen under contract version "
            f"{data.get('contract', {}).get('contract_version')!r}; the loaded contract is "
            f"{C.CONTRACT_VERSION} (stale or edited manifest)"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=C.STUDY_ID)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--tier", choices=sorted(TIERS), default="reduced")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--show", action="store_true", help="print the existing manifest and exit")
    args = ap.parse_args(argv)

    root = args.run_root or (args.results_root / args.run_id)
    if args.show:
        print(yaml.safe_dump(dict(load_frozen(root)), sort_keys=True, default_flow_style=False))
        return 0

    path = freeze(root, tier=args.tier, config_path=args.config)
    data = load_frozen(root)
    totals = data["tier"]["totals"]
    print(f"froze {path}")
    print(f"  tier                {data['tier']['name']}")
    print(f"  decisions           {totals['small_logical_decisions']:,} 8B + "
          f"{totals['large_logical_decisions']:,} 32B")
    print(f"  code_version        {data['code_version']}")
    print(f"  amendments          {len(json.loads((REGISTERS_DIR / AMENDMENTS_FILE).read_text()))}")
    print(f"  items               {data['items'].get('totals') or data['items'].get('missing')}")
    oracle = data["oracle_tables"]
    print(f"  oracle              {'ERROR: ' + oracle['error'] if 'error' in oracle else 'pinned'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
