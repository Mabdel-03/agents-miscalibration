"""Study manifest loader, freeze writer and freeze gate (WP0).

Spec: §6.4/§10.1 (pins), §6.5 (B0 frozen before tier-1 dispatch), §9.1 (freeze before
confirmation), §11.2 ("reject stale hashes"), §11.5 (amendments in the freeze manifest).
Architecture: docs/study_v4/01_architecture.md §1.3.  Corrections: 04_critic_corrections.md
P1-1 (B0 analytic, filled by ``resources/profile.py`` into the frozen copy), P1-9
(``freeze/amendments.json`` + ``freeze/prior_exposure_manifest.json`` are written by
``freeze()`` together with ``FROZEN.yaml``).

``configs/study_v4.yaml`` is the committed manifest.  ``freeze(run_root, extra)`` writes
``<run_root>/FROZEN.yaml`` (config + sha256 + prompt hashes + ``extra``) exactly once and
copies the two freeze registers next to it.  ``StudyConfig.frozen(run_root)`` is the gate
every generate cell passes through: it refuses when ``FROZEN.yaml`` is absent or was
frozen from a config with a different sha256.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from agents_scaling.study import prompts as _prompts
from agents_scaling.study.identity import sha256_hex
from agents_scaling.study.types import (
    FORECAST_DECODING,
    JUDGE_DECODING,
    SOLVER_DECODING,
    Checkpoint,
    Decoding,
    ProtocolError,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "study_v4.yaml"
FREEZE_DIR = Path(__file__).resolve().parent / "freeze"
AMENDMENTS_FILE = "amendments.json"
PRIOR_EXPOSURE_FILE = "prior_exposure_manifest.json"
FROZEN_FILENAME = "FROZEN.yaml"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Caps:
    """Frozen envelopes (§4.3, §6.4, amendment B2 / Table E, P1-1)."""

    task_tokens: int
    prompt_tokens: int
    solver_out: int
    selector_out: int
    forecast_out: int
    own_candidate_tokens: int
    packet_tokens: int
    packet_final_tokens: int
    subtask_result_tokens: int
    hub_prior_plan_tokens: int
    forwarded_results_tokens: int
    solver_calls: int
    selector_calls: int
    dec_max_rounds: int
    cen_max_cycles: int
    #: Amendment B2b (Table E; review P0-A): total recipient tokens of the returned-results
    #: block of one CEN_FLAT hub prompt.  A k-assignment cycle clips every returned result to
    #: ``min(subtask_result_tokens, hub_returned_results_tokens // k)``, so no hub prompt can
    #: exceed ``prompt_tokens`` at any N ≤ 9 (critic P1-1).  Defaulted for backward compatibility.
    hub_returned_results_tokens: int = 16384
    #: Amendment B2b (audit A-6): allowance for the rendered ``last_action_error`` object the
    #: hub is re-prompted with after a typed action error (the policy bounds its tokens).
    hub_action_error_tokens: int = 512


@dataclass(frozen=True)
class ItemCounts:
    hle: int
    bcb: int

    @property
    def total(self) -> int:
        return self.hle + self.bcb


@dataclass(frozen=True)
class Items:
    """Item counts per split and per nested panel (amendments N1, N2, E4b, M1b, D1, AB1)."""

    dev: ItemCounts
    main: ItemCounts
    panels: Mapping[str, int]


@dataclass(frozen=True)
class Budget:
    """§6.5 budget grid.  ``B0_flops`` is ``None`` in the committed yaml and is supplied by
    ``resources/profile.py`` through ``freeze(extra={"B0_flops": ...})``."""

    B0_flops: float | None
    budget_multipliers: tuple[int, ...]
    primary: int


@dataclass(frozen=True)
class StudyConfig:
    study_id: str
    study_seed_hex: str
    split_salt_hex: str
    checkpoints: Mapping[str, Checkpoint]
    flagship: str
    judge_checkpoint: str
    caps: Caps
    items: Items
    budget: Budget
    engine: Mapping[str, Any]
    config_sha256: str
    source_path: Path
    raw: Mapping[str, Any] = field(repr=False, compare=False)

    # ---- derived ---------------------------------------------------------
    @property
    def study_seed(self) -> bytes:
        return bytes.fromhex(self.study_seed_hex)

    @property
    def split_salt(self) -> bytes:
        return bytes.fromhex(self.split_salt_hex)

    def checkpoint(self, size: str) -> Checkpoint:
        return self.checkpoints[size]

    @property
    def flagship_checkpoint(self) -> Checkpoint:
        return self.checkpoints[self.flagship]

    @property
    def judge(self) -> Checkpoint:
        return self.checkpoints[self.judge_checkpoint]

    # ---- freeze gate -----------------------------------------------------
    def frozen(self, run_root: str | os.PathLike) -> "FrozenStudy":
        """Return the frozen manifest for ``run_root`` or raise ``ProtocolError``.

        Generate cells call this before any request (§9.1); F bank cells may pass
        ``FrozenStudy.b0_flops is None`` since B is not applicable to fixed banks.
        """
        path = Path(run_root) / FROZEN_FILENAME
        if not path.exists():
            raise ProtocolError(f"{path} is missing: freeze the study before running generate cells")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ProtocolError(f"{path} is not a mapping")
        if data.get("config_sha256") != self.config_sha256:
            raise ProtocolError(
                f"{path} was frozen from config sha256 {data.get('config_sha256')!r}; "
                f"the loaded config has {self.config_sha256!r} (stale or edited manifest)"
            )
        if data.get("study_id") != self.study_id:
            raise ProtocolError(f"{path} study_id mismatch")
        return FrozenStudy(config=self, path=path, manifest=data)


@dataclass(frozen=True)
class FrozenStudy:
    config: StudyConfig
    path: Path
    manifest: Mapping[str, Any]

    @property
    def b0_flops(self) -> float | None:
        value = self.manifest.get("budget", {}).get("B0_flops")
        return None if value is None else float(value)

    @property
    def frozen_sha256(self) -> str:
        return sha256_hex(self.path.read_bytes())

    @property
    def prompt_hashes(self) -> Mapping[str, str]:
        return dict(self.manifest.get("prompt_hashes", {}))


# --------------------------------------------------------------------------- loading


def _require_hex(value: Any, pattern: re.Pattern[str], what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{what} must match {pattern.pattern}, got {value!r}")
    return value


def _decoding_from_yaml(block: Mapping[str, Any]) -> Decoding:
    return Decoding(
        temperature=float(block["temperature"]),
        top_p=float(block["top_p"]),
        top_k=int(block["top_k"]),
        min_p=float(block["min_p"]),
        presence_penalty=float(block["presence_penalty"]),
        repetition_penalty=float(block["repetition_penalty"]),
        max_tokens=int(block["max_tokens"]),
        enable_thinking=bool(block["enable_thinking"]),
        guided_json=bool(block.get("guided_json", False)),
    )


def load_config(path: str | os.PathLike | None = None) -> StudyConfig:
    """Load and validate ``configs/study_v4.yaml`` (or ``path``)."""
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw_bytes = source.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top level must be a mapping")

    study_id = str(raw["study_id"])
    study_seed_hex = _require_hex(raw["study_seed_hex"], _HEX64, "study_seed_hex")
    split_salt_hex = _require_hex(raw["split_salt_hex"], _HEX64, "split_salt_hex")
    if study_seed_hex == split_salt_hex:
        raise ValueError("study_seed_hex and split_salt_hex must be independent secrets")

    checkpoints: dict[str, Checkpoint] = {}
    for size, block in raw["checkpoints"].items():
        size = str(size)
        model_revision = _require_hex(block["model_revision"], _HEX40, f"checkpoints.{size}.model_revision")
        tokenizer_revision = _require_hex(
            block.get("tokenizer_revision", model_revision), _HEX40, f"checkpoints.{size}.tokenizer_revision"
        )
        checkpoints[size] = Checkpoint(
            size=size,
            hf_id=str(block["hf_id"]),
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            profile=str(block["profile"]),
            tp_size=int(block["tp_size"]),
            served_model_name=str(block.get("served_model_name", size)),
        )
    flagship = str(raw["flagship"])
    judge_checkpoint = str(raw["judge_checkpoint"])
    for name, value in (("flagship", flagship), ("judge_checkpoint", judge_checkpoint)):
        if value not in checkpoints:
            raise ValueError(f"{name}={value!r} is not a declared checkpoint")

    caps = Caps(**{k: int(v) for k, v in raw["caps"].items()})
    if caps.hub_returned_results_tokens < caps.subtask_result_tokens or caps.hub_action_error_tokens < 1:
        raise ValueError(
            "caps.hub_returned_results_tokens must be >= caps.subtask_result_tokens and "
            "caps.hub_action_error_tokens >= 1 (amendment B2b)"
        )
    items_raw = raw["items"]
    items = Items(
        dev=ItemCounts(**{k: int(v) for k, v in items_raw["dev"].items()}),
        main=ItemCounts(**{k: int(v) for k, v in items_raw["main"].items()}),
        panels={str(k): int(v) for k, v in items_raw["panels"].items()},
    )
    budget_raw = raw["budget"]
    budget = Budget(
        B0_flops=None if budget_raw.get("B0_flops") is None else float(budget_raw["B0_flops"]),
        budget_multipliers=tuple(int(m) for m in budget_raw["budget_multipliers"]),
        primary=int(budget_raw["primary"]),
    )
    if budget.primary not in budget.budget_multipliers:
        raise ValueError("budget.primary must be one of budget.budget_multipliers")

    # The yaml records the decoding recipes for the freeze manifest; types.py is the
    # single source of truth, so a divergence is a configuration error.
    decoding_raw = raw.get("decoding", {})
    expected = {"solver": SOLVER_DECODING, "judge": JUDGE_DECODING, "forecast": FORECAST_DECODING}
    for role, constant in expected.items():
        if role in decoding_raw and _decoding_from_yaml(decoding_raw[role]) != constant:
            raise ValueError(f"decoding.{role} in {source} differs from types.{role.upper()}_DECODING")
    if caps.solver_out != SOLVER_DECODING.max_tokens or caps.selector_out != JUDGE_DECODING.max_tokens:
        raise ValueError("caps.solver_out/selector_out must equal the frozen decoding max_tokens")

    return StudyConfig(
        study_id=study_id,
        study_seed_hex=study_seed_hex,
        split_salt_hex=split_salt_hex,
        checkpoints=checkpoints,
        flagship=flagship,
        judge_checkpoint=judge_checkpoint,
        caps=caps,
        items=items,
        budget=budget,
        engine=dict(raw.get("engine", {})),
        config_sha256=sha256_hex(raw_bytes),
        source_path=source,
        raw=raw,
    )


# --------------------------------------------------------------------------- freezing


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def freeze(
    run_root: str | os.PathLike,
    extra: Mapping[str, Any] | None = None,
    *,
    config: StudyConfig | None = None,
    code_version: str | None = None,
) -> Path:
    """Write ``<run_root>/FROZEN.yaml`` once and copy the freeze registers (P1-9, §11.5).

    ``extra`` carries what only the pilot can supply (``B0_flops``, Table E envelopes,
    the alias table, realized admitted-call schedules).  A ``B0_flops`` entry is written
    into the frozen ``budget`` block.  Refuses to overwrite an existing manifest: a second
    freeze is a new run root, never a silent edit (§9.1).
    """
    cfg = config or load_config()
    root = Path(run_root)
    target = root / FROZEN_FILENAME
    if target.exists():
        raise ProtocolError(f"{target} already exists; the study is frozen once per run root")
    extra = dict(extra or {})

    freeze_dir = root / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    register_hashes: dict[str, str] = {}
    for name in (AMENDMENTS_FILE, PRIOR_EXPOSURE_FILE):
        src = FREEZE_DIR / name
        json.loads(src.read_text(encoding="utf-8"))  # must be valid JSON
        shutil.copyfile(src, freeze_dir / name)
        register_hashes[name] = sha256_hex(src.read_bytes())

    manifest: dict[str, Any] = json.loads(json.dumps(cfg.raw))  # yaml-native copy
    manifest["budget"] = dict(manifest.get("budget", {}))
    if "B0_flops" in extra:
        manifest["budget"]["B0_flops"] = extra.pop("B0_flops")
    manifest.update(
        {
            "config_sha256": cfg.config_sha256,
            "config_path": str(cfg.source_path),
            "frozen_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "code_version": code_version,
            "prompt_hashes": dict(_prompts.prompt_hashes()),
            "freeze_registers": register_hashes,
            "extra": extra,
        }
    )
    payload = yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True, default_flow_style=False)
    _atomic_write_bytes(target, payload.encode("utf-8"))
    return target


def load_amendments() -> list[dict[str, Any]]:
    return json.loads((FREEZE_DIR / AMENDMENTS_FILE).read_text(encoding="utf-8"))


def load_prior_exposure_manifest() -> dict[str, Any]:
    return json.loads((FREEZE_DIR / PRIOR_EXPOSURE_FILE).read_text(encoding="utf-8"))
