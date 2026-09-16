"""The strict compiler: the spec-17.4 rejections, unknown-key paths, scratch refusal, lock-hash
pins, the E2b grid cross-check, and the parametric contract every value flows into (plan
§1.2, R1, R2).

``load_study`` runs ``compile_checks`` by default; tests that build a violating config on
purpose pass ``check=False`` and inspect the violation list directly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn.errors import ConfigError, ContractViolation

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "nextdyn"


def _study_yaml(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load((CONFIG_DIR / "study.yaml").read_text())
    mutate(data)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locks_beside(tmp_path: Path, *, source_inventory_hash="a" * 64, prior_use_inventory_hash="b" * 64) -> dict:
    """Lock files beside a temporary study.yaml, with the datasets lock carrying the inventory
    hashes; returns the ``locks`` block that pins them."""
    for name in ("models.lock.json", "prompts.lock.json"):
        shutil.copy(CONFIG_DIR / name, tmp_path / name)
    datasets = json.loads((CONFIG_DIR / "datasets.lock.json").read_text())
    datasets["source_inventory_hash"] = source_inventory_hash
    datasets["prior_use_inventory_hash"] = prior_use_inventory_hash
    (tmp_path / "datasets.lock.json").write_text(json.dumps(datasets, sort_keys=True, indent=2) + "\n")
    return {
        "models_lock_hash": _sha(tmp_path / "models.lock.json"),
        "datasets_lock_hash": _sha(tmp_path / "datasets.lock.json"),
        "prompts_lock_hash": _sha(tmp_path / "prompts.lock.json"),
        "source_inventory_hash": source_inventory_hash,
        "prior_use_inventory_hash": prior_use_inventory_hash,
        "git_commit": "0" * 40,
        "container_digests": {"asys_env": "sha256:" + "0" * 64},
    }


def _confirm(d):
    d["study"]["mode"] = "confirmation"
    d["study"]["confirmation_authorized"] = True


def _fill_ceilings(d):
    d["budgets"]["campaign_gpu_hours"] = 670
    d["budgets"]["campaign_cpu_hours"] = 120
    d["budgets"]["storage_limit_bytes"] = 200 * 10**9
    d["analysis"]["confirmatory_sample_size_lock"] = {"MATH": 300, "MMLU_PRO": 300}
    d["analysis"]["analysis_lock_hash"] = "1" * 64


# ---------------------------------------------------------------------------- loading


def test_committed_configs_compile_with_no_violations(study_cfg, compute_cfg, analysis_cfg):
    full = C.load_study(CONFIG_DIR / "study.yaml")          # check=True: compute/analysis/protocols beside it
    assert full.study.mode == K.MODE_PILOT_ONLY and not full.study.confirmation_authorized
    assert C.compile_checks(full, compute=compute_cfg, analysis=analysis_cfg,
                            protocols=CONFIG_DIR / "protocols.yaml") == []
    assert study_cfg.study.mode == K.MODE_TEST
    assert study_cfg.protocol.revision_rounds == 2 and study_cfg.protocol.score_rounds == (0, 1, 2)
    assert full.protocol.revision_rounds == 4 and full.protocol.score_rounds == (0, 1, 2, 3, 4)


def test_compiler_rejects_unknown_fields_naming_the_path(tmp_path):
    path = _study_yaml(tmp_path, lambda d: d["budgets"].__setitem__("thinking_tokens", 2048))
    with pytest.raises(ConfigError, match=r"study\.budgets\.thinking_tokens: unknown key"):
        C.load_study(path)
    path = _study_yaml(tmp_path, lambda d: d["experiments"]["E2b"].__setitem__("depth", [4]))
    with pytest.raises(ConfigError, match=r"study\.experiments\.E2b\.depth: unknown key"):
        C.load_study(path)
    path = _study_yaml(tmp_path, lambda d: d["models"]["primary"].pop("profile"))
    with pytest.raises(ConfigError, match=r"study\.models\.primary\.profile: missing key"):
        C.load_study(path)
    path = _study_yaml(tmp_path, lambda d: d["protocol"].__setitem__("n_agents", "five"))
    with pytest.raises(ConfigError, match=r"study\.protocol\.n_agents"):
        C.load_study(path)
    # the facts that moved out of their duplicate homes are unknown keys now (R1)
    path = _study_yaml(tmp_path, lambda d: d["cohorts"]["sub_panels"].__setitem__("e3a", 150))
    with pytest.raises(ConfigError, match=r"study\.cohorts\.sub_panels\.e3a: unknown key"):
        C.load_study(path)
    path = _study_yaml(tmp_path, lambda d: d["experiments"]["E2b"].__setitem__("pilot_problems", 30))
    with pytest.raises(ConfigError, match=r"pilot_problems: unknown key"):
        C.load_study(path)


def test_native_thinking_budget_on_envelope_adapter_is_an_error(tmp_path):
    path = _study_yaml(tmp_path, lambda d: d["models"]["family_transfer"].__setitem__("thinking_tokens", 2048))
    violations = C.compile_checks(C.load_study(path, check=False))
    assert any("family_transfer" in str(v) and "thinking budget" in str(v) for v in violations)
    with pytest.raises(ContractViolation, match="thinking budget"):
        C.load_study(path)                                               # the default path refuses too
    path = _study_yaml(tmp_path, lambda d: d["models"]["family_transfer"].__setitem__("total_output_tokens", None))
    assert any("total_output_tokens" in str(v) for v in C.compile_checks(C.load_study(path, check=False)))
    path = _study_yaml(tmp_path, lambda d: d["models"]["primary"].__setitem__("total_output_tokens", 3072))
    assert any("primary" in str(v) for v in C.compile_checks(C.load_study(path, check=False)))


def test_a_thinking_override_on_a_native_cell_is_refused_not_ignored(tmp_path):
    """The D-03 shape: a knob that compiles and is then dropped.  ``models.*.thinking_tokens``
    has no implementation in v1, so a value is refused; the envelope cap rides on the contract
    and changes its sha."""
    path = _study_yaml(tmp_path, lambda d: d["models"]["primary"].__setitem__("thinking_tokens", 512))
    violations = C.compile_checks(C.load_study(path, check=False))
    assert any("primary" in str(v) and "silent no-op" in str(v) for v in violations)
    assert C.load_study(path, check=False).contract.tokens.thinking_cap == 2048   # the override never took
    base = C.load_study(CONFIG_DIR / "study.yaml")
    assert base.contract.model("qwen_nonthinking").total_output_tokens == 3072
    assert base.contract.model("primary").total_output_tokens is None
    path = _study_yaml(tmp_path, lambda d: d["models"]["qwen_nonthinking"].__setitem__("total_output_tokens", 2048))
    other = C.load_study(path, check=False)
    assert other.contract.model("qwen_nonthinking").total_output_tokens == 2048
    assert other.contract.sha256() != base.contract.sha256()


def test_oracle_selector_in_deployed_list_is_an_error(tmp_path):
    path = _study_yaml(tmp_path, lambda d: d["selection"].__setitem__(
        "secondary_deployed", ["random_valid", "oracle_analysis"]))
    assert any("oracle_analysis" in str(v) for v in C.compile_checks(C.load_study(path, check=False)))
    with pytest.raises(ContractViolation, match="oracle_analysis"):
        C.load_study(path)
    path = _study_yaml(tmp_path, lambda d: d["selection"].__setitem__("primary", "oracle"))
    assert any("'oracle'" in str(v) for v in C.compile_checks(C.load_study(path, check=False)))
    assert C.compile_checks(C.load_study(CONFIG_DIR / "study.yaml")) == []


def test_confirmation_with_null_locks_is_an_error(tmp_path):
    violations = C.compile_checks(C.load_study(_study_yaml(tmp_path, _confirm), check=False))
    nulls = {str(v).split(" ")[0] for v in violations}
    assert {"study." + p for p in C.CONFIRMATION_REQUIRED} <= nulls
    assert len(violations) == len(C.CONFIRMATION_REQUIRED)
    with pytest.raises(ContractViolation, match=r"12 violation"):
        C.load_study(_study_yaml(tmp_path, _confirm))                   # load_study applies the checks itself

    def unauthorized(d):
        _confirm(d)
        d["study"]["confirmation_authorized"] = False

    assert any("confirmation_authorized" in str(v)
               for v in C.compile_checks(C.load_study(_study_yaml(tmp_path, unauthorized), check=False)))

    locks = _locks_beside(tmp_path)

    def filled(d):
        _confirm(d)
        d["locks"] = locks
        _fill_ceilings(d)

    assert C.compile_checks(C.load_study(_study_yaml(tmp_path, filled), check=False)) == []
    assert C.load_study(_study_yaml(tmp_path, filled)).study.mode == K.MODE_CONFIRMATION


def test_a_stale_lock_hash_in_study_yaml_blocks_confirmation(tmp_path):
    """The study.yaml hash is the spec's pin: it must equal sha256 of the lock file beside it,
    and a non-null inventory hash must equal the datasets lock's field.  The ``.sha256``
    sidecars are never consulted here (a regenerated sidecar cannot launder a tampered lock)."""
    locks = _locks_beside(tmp_path)

    def pinned(d):
        _confirm(d)
        d["locks"] = locks
        _fill_ceilings(d)

    path = _study_yaml(tmp_path, pinned)
    assert C.compile_checks(C.load_study(path, check=False)) == []
    # the lock file changes after the pin (and the sidecar is regenerated to match)
    models = tmp_path / "models.lock.json"
    models.write_text(models.read_text().replace('"blocked"', '"locked"', 1))
    (tmp_path / "models.lock.json.sha256").write_text(_sha(models) + "  models.lock.json\n")
    violations = C.compile_checks(C.load_study(path, check=False))
    assert [str(v) for v in violations if "models_lock_hash" in str(v) and "stale pin" in str(v)]
    with pytest.raises(ContractViolation, match="models_lock_hash"):
        C.load_study(path)
    # a missing lock file behind a non-null hash, and an inventory hash the datasets lock does not carry
    (tmp_path / "prompts.lock.json").unlink()
    text = "\n".join(str(v) for v in C.compile_checks(C.load_study(path, check=False)))
    assert "prompts_lock_hash is set but" in text
    path = _study_yaml(tmp_path, lambda d: (pinned(d), d["locks"].__setitem__("source_inventory_hash", "c" * 64)))
    text = "\n".join(str(v) for v in C.compile_checks(C.load_study(path, check=False)))
    assert "source_inventory_hash" in text and "datasets.lock.json" in text
    # a null hash pins nothing and is only a confirmation blocker
    assert C.lock_hash_checks(C.load_study(CONFIG_DIR / "study.yaml")) == []


def test_e2b_grid_must_match_the_receiver_registry(tmp_path):
    """R1: the E2b grid is declared once (``experiments.E2b``, spec 17.4 names) and
    ``protocols.yaml`` instantiates exactly it through ``config.E2B_ENDORSEMENT_SUPPORT``."""
    cfg = C.load_study(CONFIG_DIR / "study.yaml")
    assert C.e2b_grid_checks(cfg, CONFIG_DIR / "protocols.yaml") == []
    assert C.E2B_ENDORSEMENT_SUPPORT == {"correct_minority": "1:4", "correct_majority": "4:1", "hidden": "hidden"}
    path = _study_yaml(tmp_path, lambda d: d["experiments"]["E2b"].__setitem__("checker_access", [True]))
    shutil.copy(CONFIG_DIR / "protocols.yaml", tmp_path / "protocols.yaml")
    with pytest.raises(ContractViolation, match="registry-only"):
        C.load_study(path)
    text = "\n".join(str(v) for v in C.e2b_grid_checks(C.load_study(path, check=False), tmp_path / "protocols.yaml"))
    assert "('1:4', False, 'correct')" in text
    # disabling one registry entry is the mirror image
    data = yaml.safe_load((CONFIG_DIR / "protocols.yaml").read_text())
    data["receivers"]["E2B_1_4_CHECK_WRONG"]["enabled"] = False
    (tmp_path / "protocols.yaml").write_text(yaml.safe_dump(data))
    text = "\n".join(str(v) for v in C.e2b_grid_checks(cfg, tmp_path / "protocols.yaml"))
    assert "declared-only [('1:4', True, 'wrong')]" in text
    path = _study_yaml(tmp_path, lambda d: d["experiments"]["E2b"].__setitem__("endorsement", ["majority"]))
    assert any("alias map" in str(v) for v in C.e2b_grid_checks(C.load_study(path, check=False), CONFIG_DIR / "protocols.yaml"))


def test_scratch_paths_are_refused(tmp_path):
    path = _study_yaml(tmp_path, lambda d: d["models"]["primary"].__setitem__(
        "model_id", "/orcd/scratch/orcd/012/mabdel03/models/Qwen3-8B"))
    violations = C.compile_checks(C.load_study(path, check=False))
    assert any("/orcd/scratch" in str(v) and "study.yaml:models.primary.model_id" in str(v) for v in violations)
    with pytest.raises(ContractViolation, match="scratch"):
        C.load_study(path)
    assert C.scratch_paths(*sorted(CONFIG_DIR.glob("*.yaml"))) == []


def test_mode_override_must_be_a_declared_mode():
    with pytest.raises(ConfigError):
        C.load_study(CONFIG_DIR / "study.yaml", mode="dry")


def test_pilot_and_gate_numbers_live_in_study_yaml(study_cfg):
    """Spec 19.1 / 18.6: nothing in plan or gates may hard-code these."""
    p = study_cfg.pilot
    assert (p.natural_items_per_family, p.constructed_paired_items_per_family, p.directional_items_per_family) == (20, 30, 10)
    assert (p.natural_receiver_items, p.natural_receiver_tasks) == (10, ("MATH", "MMLU_PRO"))
    assert (p.synthetic_receiver_problems, p.adapter_check_problems, p.capture_items_per_family) == (30, 12, 10)
    assert (p.smoke_solver_calls_per_model, p.smoke_receiver_episodes) == (20, 5)
    g = study_cfg.gates
    assert (g.max_unresolved_infra_rate, g.max_format_failure_rate, g.max_arm_differential_pp) == (0.01, 0.02, 1.0)
    assert study_cfg.cohorts.sub_panels == C.SubPanelsBlock(second_root=60, bridge=60, fragmented=100,
                                                            e3a_fill=60, capture_natural=30)
    assert study_cfg.protocol.protocols == ("PRIVATE", "PEER", "FRESH")
    assert study_cfg.experiments.E1.bridge_protocol == "PEER_ANSWER_ONLY"


# ---------------------------------------------------------------------------- the contract


def test_contract_is_parametric_and_every_value_comes_from_yaml(study_cfg):
    contract = study_cfg.contract
    assert isinstance(contract, K.StudyContract)
    assert contract.study_id == study_cfg.study.id == "task_conditioned_dynamics_v1"
    assert contract.tokens == K.TokenContract(thinking_cap=2048, final_cap=1024, served_window=32768, reserve=256,
                                              packet_cap=512, rationale_cap=256, claims_max=4, claim_cap=64)
    assert contract.tokens.max_new_tokens == 3074
    assert contract.sampling == K.Sampling(temperature="0.6", top_p="0.95", top_k=20, min_p="0.0",
                                           repetition_penalty="1.0", presence_penalty="0.0", frequency_penalty="0.0")
    assert contract.sampling.as_floats()["temperature"] == 0.6
    assert contract.banks == K.BankSpec(difficulty_draws=16, donor_initial=32, donor_max=64, natural_root_draws=5)
    assert contract.team.n_agents == 5 and contract.team.revision_rounds == 2      # test mode
    assert set(contract.receiver_tokens) == {"1024/plan", "1024/decision", "1024/one_call",
                                             "4096/plan", "4096/decision", "4096/one_call"}
    assert contract.receiver_tokens["4096/plan"].thinking_cap == 2048
    assert contract.receiver_tokens["4096/plan"].final_cap == 256
    assert contract.receiver_tokens["4096/decision"].final_cap == 512
    assert contract.receiver_tokens["1024/one_call"].thinking_cap == 1024
    assert contract.audit_tokens.thinking_cap == 1024 and contract.audit_tokens.final_cap == 256
    assert contract.selection.deployed == ("current_round_plurality", "random_valid", "confidence_weighted", "pointwise_audit")
    assert contract.selection.analysis_only == ("oracle_analysis",)
    assert contract.selection.pool_scope == "terminal_only" and contract.selection.tie_break == "item_class_hash"
    assert contract.selection.audit_model == "scale" and contract.selection.audit_failure_fallback == "current_round_plurality"
    assert contract.selection.archive_audit_enabled is False
    assert contract.failure_policy.transient_retries == 2
    assert contract.models["primary"].profile == "nd1-8B" and contract.models["scale"].tp == 2
    assert contract.models["scale"].served_model_name == "32B"
    assert (contract.models["primary"].lane, contract.models["scale"].lane, contract.models["family_transfer"].lane) == ("8B", "32B", "8B")
    assert contract.models["family_transfer"].total_output_tokens == 3072
    assert contract.models["family_transfer"].pinned is False and contract.models["primary"].pinned
    assert contract.families == ("MATH", "MMLU_PRO", "TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE")
    assert len(contract.study_seed) == 32 and len(contract.split_salt) == 32
    assert contract.contract_version == 1
    with pytest.raises(ContractViolation):
        contract.model("nope")


def test_contract_sha256_is_stable_and_sensitive(study_cfg):
    contract = study_cfg.contract
    assert contract.sha256() == study_cfg.contract.sha256()
    changed = dataclasses.replace(contract, tokens=dataclasses.replace(contract.tokens, final_cap=1023))
    assert changed.sha256() != contract.sha256()
    assert contract.tokens.budget_hash() != contract.audit_tokens.budget_hash()
    assert contract.tokens.budget_hash() == K.TokenContract(**dataclasses.asdict(contract.tokens)).budget_hash()


def test_contract_shapes_refuse_inconsistent_values():
    with pytest.raises(ContractViolation):
        K.TokenContract(thinking_cap=40000, final_cap=1024, served_window=32768, reserve=256, packet_cap=512,
                        rationale_cap=256, claims_max=4, claim_cap=64)
    with pytest.raises(ContractViolation):
        K.TeamShape(n_agents=5, revision_rounds=4, score_rounds=(1, 2))
    with pytest.raises(ContractViolation):
        K.BankSpec(difficulty_draws=16, donor_initial=64, donor_max=32, natural_root_draws=5)
    cell = dict(name="x", hf_id="a/b", revision=None, tokenizer_revision=None, profile="p", tp=1,
                served_model_name="x", lane="8B")
    with pytest.raises(ContractViolation):
        K.ModelCell(adapter="thinking", total_output_tokens=None, **cell)
    with pytest.raises(ContractViolation, match="needs total_output_tokens"):
        K.ModelCell(adapter=K.ADAPTER_TOTAL_OUTPUT_ENVELOPE, total_output_tokens=None, **cell)
    with pytest.raises(ContractViolation, match="belongs to"):
        K.ModelCell(adapter=K.ADAPTER_NATIVE_THINKING, total_output_tokens=3072, **cell)
    with pytest.raises(ContractViolation, match="lane"):
        K.ModelCell(adapter=K.ADAPTER_NATIVE_THINKING, total_output_tokens=None, **{**cell, "lane": ""})
    with pytest.raises(ContractViolation):
        K.FailurePolicy(transient_retries=2, retry_same_request_and_seed=True, model_output_repair_calls=1,
                        wrong_answer_retry=False, context_overflow="typed_failure", unresolved_semantics="unknown",
                        null_item_ids="fatal", duplicate_logical_keys="fatal", hidden_label_leakage="fatal")


def test_study_validation_refuses_a_non_synchronous_protocol(tmp_path):
    path = _study_yaml(tmp_path, lambda d: d["protocol"].__setitem__("early_stop_on_consensus", True))
    with pytest.raises(ConfigError, match="synchronous"):
        C.load_study(path)
    path = _study_yaml(tmp_path, lambda d: d["selection"].__setitem__("audit_model", "judge"))
    with pytest.raises(ConfigError, match="audit_model"):
        C.load_study(path)
    path = _study_yaml(tmp_path, lambda d: d["experiments"]["E1"].__setitem__("bridge_protocol", "PEER"))
    with pytest.raises(ConfigError, match="bridge_protocol"):
        C.load_study(path)
    path = _study_yaml(tmp_path, lambda d: d["gates"].__setitem__("max_format_failure_rate", 2))
    with pytest.raises(ConfigError, match="fraction"):
        C.load_study(path)


# ---------------------------------------------------------------------------- compute / analysis


def test_compute_config_pins_the_plan_fleet(compute_cfg):
    assert compute_cfg.pools.deadline.port_offset == 0 and compute_cfg.pools.low.port_offset == 200
    assert compute_cfg.pools.audit.port_offset == 100
    assert compute_cfg.pools.low.preemptable_allowed and not compute_cfg.pools.deadline.preemptable_allowed
    assert "nd1-32B:2:pi_manoli:3-00:00:00" in compute_cfg.pools.deadline.spec
    assert compute_cfg.serving.max_num_seqs_by_gpu["h200"]["8B"] == 128
    assert compute_cfg.serving.exclude_nodes == ("node3804",)
    assert compute_cfg.capture.partition == "ou_bcs_low" and compute_cfg.capture.fallback.partition == "mit_preemptable"
    assert compute_cfg.partitions["mit_normal"].cpu == 96 and compute_cfg.partitions["mit_normal"].mem_gb == 386
    assert compute_cfg.partitions["ou_bcs_normal"].gpu == {"a100": 8, "h100": 8}
    assert compute_cfg.partitions["mit_preemptable"].preemptable
    assert compute_cfg.dispatch.global_submit_cap == 380 and compute_cfg.dispatch.throttle_per_endpoint["episode"] == 8
    assert compute_cfg.keepalive.tick_s == 180 and compute_cfg.keepalive.max_launch == 3
    assert compute_cfg.workers.deadline.partition == "mit_normal"
    assert compute_cfg.workers.deadline.overflow_partition == "mit_preemptable"
    assert compute_cfg.cells.reference.thinking_cap == 2048
    assert compute_cfg.cells.kinds["episode"].max_inflight == "n_agents"
    assert compute_cfg.serving.engine.vllm == "0.21.0"


def test_compute_config_refuses_an_unknown_partition_or_pool(tmp_path):
    data = yaml.safe_load((CONFIG_DIR / "compute.yaml").read_text())
    data["pools"]["assignment"]["episode"] = "spot"
    path = tmp_path / "compute.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="unknown pool"):
        C.load_compute(path)
    data = yaml.safe_load((CONFIG_DIR / "compute.yaml").read_text())
    data["workers"]["deadline"]["partition"] = "pi_nobody"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="pi_nobody"):
        C.load_compute(path)
    data = yaml.safe_load((CONFIG_DIR / "compute.yaml").read_text())
    data["cells"]["kinds"]["capture"]["partition"] = "ou_bcs_low"       # a second declaration (R1)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="capture.partition must be null"):
        C.load_compute(path)


def test_analysis_profiles_confirmatory_and_test(analysis_cfg):
    conf = C.load_analysis(CONFIG_DIR / "analysis.yaml", profile="confirmatory").profile
    assert conf.n_resamples == 10000
    assert not hasattr(conf, "holm") and not hasattr(conf.power, "alpha")      # derived from study.yaml (R1)
    assert conf.difficulty_cutpoints == (0.2, 0.5, 0.8) and conf.reliability_bins is None
    assert conf.efficiency.reference_cost is None and conf.efficiency.budget_levels == (0.5, 1.0, 2.0)
    assert conf.efficiency.truncation_rule == "last_complete_round"
    assert conf.power.n_values == (150, 200, 300, 450, 600) and conf.power.n_sims == 1000
    assert conf.power.target == 0.8
    assert conf.equivalence_margin_pp == 5 and set(conf.families) == {"MATH", "MMLU_PRO", "TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE"}
    assert conf.confidence_weighted.fallback == "global_dev_base_rate"
    test = analysis_cfg.profile
    assert analysis_cfg.profile_name == "test"
    assert test.n_resamples == 200 and test.power.n_sims == 10 and test.posterior_draws == 5
    assert test.power.n_values == (4,) and test.power.target == conf.power.target
    with pytest.raises(ConfigError):
        C.load_analysis(CONFIG_DIR / "analysis.yaml", profile="prod")


def test_analysis_test_profile_cannot_introduce_new_knobs(tmp_path):
    data = yaml.safe_load((CONFIG_DIR / "analysis.yaml").read_text())
    data["profiles"]["test"]["surprise"] = 1
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="surprise"):
        C.load_analysis(path, profile="test")


def test_consumer_lookup_prefers_the_most_specific_pattern():
    assert C.consumer_of("study.budgets.thinking_tokens_main") == "agents_scaling.nextdyn.config"
    assert C.consumer_of("compute.partitions.pi_manoli.gpu.a100") == "agents_scaling.nextdyn.fleet"
    assert C.consumer_of("analysis.profiles.test.power.n_sims") == "agents_scaling.nextdyn.power"
    assert C.consumer_of("legacy_sources.full_sweep_v1.cache.items") == "agents_scaling.nextdyn.e0.registry"
    assert C.consumer_of("study.locks.models_lock_hash") == "agents_scaling.nextdyn.freeze"
    assert C.consumer_of("study.budgets.campaign_gpu_hours") == "agents_scaling.nextdyn.plan"
    assert C.consumer_of("study.no.such.key") is None


def test_lock_paths_sit_beside_the_study_config():
    paths = C.lock_paths(CONFIG_DIR / "study.yaml")
    assert paths.models == CONFIG_DIR / "models.lock.json"
    assert paths.datasets.exists() and paths.prompts.exists() and paths.models.exists()
    assert C.sibling_config(CONFIG_DIR / "study.yaml", "protocols.yaml") == CONFIG_DIR / "protocols.yaml"
    assert C.sibling_config(CONFIG_DIR / "study.yaml", "nope.yaml") is None
