"""config.py: manifest loading, pins, freeze writer and the freeze gate (§6.4, §9.1, §11.2)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from agents_scaling.study import config as C
from agents_scaling.study.types import FORECAST_DECODING, JUDGE_DECODING, SOLVER_DECODING, ProtocolError

VERIFIED_SNAPSHOTS = {
    "32B": "9216db5781bf21249d130ec9da846c4624c16137",
    "14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "4B": "1cfa9a7208912126459214e8b04321603b3df60c",
}


def test_yaml_loads_with_pins(study_config):
    cfg = study_config
    assert cfg.study_id == "agent_design_v4_orcd"
    assert re.fullmatch(r"[0-9a-f]{64}", cfg.study_seed_hex)
    assert re.fullmatch(r"[0-9a-f]{64}", cfg.split_salt_hex)
    assert cfg.study_seed_hex != cfg.split_salt_hex
    assert len(cfg.study_seed) == 32 and len(cfg.split_salt) == 32
    assert set(cfg.checkpoints) == {"32B", "14B", "8B", "4B"}
    for size, ckpt in cfg.checkpoints.items():
        assert re.fullmatch(r"[0-9a-f]{40}", ckpt.model_revision)
        assert ckpt.model_revision == ckpt.tokenizer_revision == VERIFIED_SNAPSHOTS[size]
        assert ckpt.hf_id == f"Qwen/Qwen3-{size}"
        assert ckpt.profile == f"{size}-long"
        assert ckpt.served_model_name == size
        assert ckpt.tp_size == (2 if size == "32B" else 1)
    assert cfg.flagship == "32B" and cfg.judge_checkpoint == "32B"
    assert cfg.flagship_checkpoint.model_cell == "Qwen3-32B@9216db57"


def test_caps_items_budget(study_config):
    caps = study_config.caps
    assert (caps.task_tokens, caps.prompt_tokens, caps.solver_out, caps.selector_out, caps.forecast_out) == (
        4096, 32768, 8192, 1024, 256)
    assert (caps.own_candidate_tokens, caps.packet_tokens, caps.packet_final_tokens) == (8192, 2048, 1024)
    assert (caps.subtask_result_tokens, caps.hub_prior_plan_tokens, caps.forwarded_results_tokens) == (4096, 8192, 8192)
    assert (caps.solver_calls, caps.selector_calls, caps.dec_max_rounds, caps.cen_max_cycles) == (64, 64, 8, 8)
    items = study_config.items
    assert (items.dev.hle, items.dev.bcb, items.main.hle, items.main.bcb) == (30, 30, 200, 200)
    assert items.panels == {"N": 100, "D": 100, "E": 30, "M": 100, "M_extension": 150, "B": 100}
    budget = study_config.budget
    assert budget.B0_flops is None
    assert budget.budget_multipliers == (1, 2, 4, 8) and budget.primary == 4
    assert study_config.engine["vllm_version"] == "0.21.0"


def test_yaml_decoding_blocks_agree_with_types(study_config):
    raw = study_config.raw["decoding"]
    assert C._decoding_from_yaml(raw["solver"]) == SOLVER_DECODING
    assert C._decoding_from_yaml(raw["judge"]) == JUDGE_DECODING
    assert C._decoding_from_yaml(raw["forecast"]) == FORECAST_DECODING


def test_loader_rejects_divergent_decoding(tmp_path: Path, study_config):
    raw = yaml.safe_load(study_config.source_path.read_text())
    raw["decoding"]["solver"]["temperature"] = 1.0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="decoding.solver"):
        C.load_config(bad)


def test_frozen_refuses_without_manifest_then_accepts_after_freeze(tmp_run_root: Path, study_config):
    with pytest.raises(ProtocolError, match="missing"):
        study_config.frozen(tmp_run_root)

    target = C.freeze(tmp_run_root, {"B0_flops": 5.66e15, "table_e": {"L_root_max": 6144}}, config=study_config,
                      code_version="test")
    assert target == tmp_run_root / "FROZEN.yaml"
    frozen = study_config.frozen(tmp_run_root)
    assert frozen.b0_flops == 5.66e15
    assert frozen.manifest["config_sha256"] == study_config.config_sha256
    assert frozen.manifest["extra"] == {"table_e": {"L_root_max": 6144}}
    assert frozen.manifest["study_seed_hex"] == study_config.study_seed_hex
    assert frozen.prompt_hashes and all(len(h) == 64 for h in frozen.prompt_hashes.values())
    assert len(frozen.frozen_sha256) == 64

    # the two freeze registers were copied verbatim
    for name in ("amendments.json", "prior_exposure_manifest.json"):
        copied = tmp_run_root / "freeze" / name
        assert copied.read_bytes() == (C.FREEZE_DIR / name).read_bytes()
        assert frozen.manifest["freeze_registers"][name] == C.sha256_hex(copied.read_bytes())

    # frozen once per run root
    with pytest.raises(ProtocolError, match="already exists"):
        C.freeze(tmp_run_root, config=study_config)


def test_frozen_rejects_stale_config_hash(tmp_run_root: Path, tmp_path: Path, study_config):
    C.freeze(tmp_run_root, config=study_config)
    edited = tmp_path / "edited.yaml"
    edited.write_bytes(study_config.source_path.read_bytes() + b"\n# edited after freeze\n")
    stale = C.load_config(edited)
    assert stale.config_sha256 != study_config.config_sha256
    with pytest.raises(ProtocolError, match="stale"):
        stale.frozen(tmp_run_root)


def test_amendment_register_complete():
    amendments = C.load_amendments()
    ids = [a["id"] for a in amendments]
    audit_rows = ["F1", "F2", "S1", "S1b", "N1", "N1b", "R1", "A1", "A2", "A3", "B1", "B2", "E1", "E2", "E3",
                  "J1", "C1", "C2", "D1", "N2", "E4", "M1", "M2", "AB1", "P1", "P2", "P3", "P4", "L1", "X1"]
    additions = ["E3'", "N9B1", "N1-SRS", "E4b", "M1b", "C1b"]
    assert ids == audit_rows + additions
    for entry in amendments:
        assert {"id", "deviation", "spec_clause", "unsupported_claim", "manifest_wording"} <= set(entry)
        assert all(isinstance(entry[k], str) and entry[k] for k in ("deviation", "spec_clause", "manifest_wording"))
    by_id = {a["id"]: a for a in amendments}
    assert "AST canonicalization" in by_id["S1b"]["manifest_wording"]
    assert "strip_one_enclosing_fence" in by_id["E3'"]["manifest_wording"]
    assert "N=9 cells exist only at B4" in by_id["N9B1"]["manifest_wording"]


def test_prior_exposure_manifest():
    manifest = C.load_prior_exposure_manifest()
    assert manifest["datasets_used_by_prior_runs"] == ["gpqa", "mmlu_pro", "math (MATH-500/hendrycks)", "truthfulqa"]
    assert manifest["hle_verified_first_downloaded"] == "2026-09-05"
    assert manifest["bigcodebench_first_downloaded"] == "2026-09-05"
    assert manifest["viewed_beyond_field_inspection"] is False
    assert isinstance(manifest["notes"], str) and manifest["notes"]
