"""WP-D: dynorch-doctor checks with a monkeypatched HOME / worktree / command runner (plan §9 Gate 0)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_allocation.config import load_config
from agent_allocation.dynorch import doctor

CFG_PATH = Path(__file__).resolve().parents[3] / "configs" / "dynorch_v1.json"
QUOTA_OK = """Disk quotas for user mabdel03 (uid 225593):
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
nfs001.lb:/home    109G    195G    200G            860k   1000k   1000k
10.1.223.2:/home
                   109G    195G    200G            860k   1000k   1000k
"""
QUOTA_FULL = QUOTA_OK.replace("860k", "995k")


@pytest.fixture
def cfg():
    return load_config(CFG_PATH)


def fake_runner(*, quota: str = QUOTA_OK, codex: str = "codex-cli 0.156.1", openclaude: str = "0.13.0 (OpenClaude)"):
    def run(argv):
        if argv[0] == "quota":
            return 0, quota
        if argv[0].endswith("codex"):
            return 0, codex
        if argv[0].endswith("openclaude"):
            return 0, openclaude
        return 127, "unknown"
    return run


REAL_CONTEXT = doctor.DoctorContext


def make_ctx(cfg, tmp_path, *, quota=QUOTA_OK, codex="codex-cli 0.156.1", openclaude="0.13.0 (OpenClaude)",
             scratch=None, key_mode=0o600, env_extra=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    worktree = tmp_path / "wt"
    for rel in doctor.PROMPT_FILES:
        (worktree / rel).parent.mkdir(parents=True, exist_ok=True)
        (worktree / rel).write_text("x")
    key = tmp_path / "keys" / "openrouter.key"
    key.parent.mkdir(exist_ok=True)
    key.write_text("not-a-key\n")
    key.chmod(key_mode)
    scratch = scratch or (tmp_path / "scratch")
    scratch.mkdir(exist_ok=True)
    prefix = scratch / "npm-prefix"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    (prefix / "bin" / "codex").write_text("#!/bin/sh\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    oc = bin_dir / "openclaude"
    oc.write_text("#!/bin/sh\n")
    oc.chmod(0o755)
    cfg = json.loads(json.dumps(cfg))
    cfg["dynorch"]["scratch_root"] = str(scratch)
    cfg["dynorch"]["codex_prefix"] = str(prefix)
    env = {"PATH": str(bin_dir), "TMPDIR": str(scratch / "tmp"), "npm_config_cache": str(scratch / "npm-cache"),
           "UV_CACHE_DIR": str(scratch / "uv"), "HF_HOME": str(scratch / "hf"), "MPLCONFIGDIR": str(scratch / "mpl")}
    env.update(env_extra or {})
    return REAL_CONTEXT(cfg=cfg, home=home, worktree=worktree, env=env,
                                run_command=fake_runner(quota=quota, codex=codex, openclaude=openclaude),
                                key_file=key, run_root=scratch / "run")


def by_name(checks):
    return {c["check"]: c for c in checks}


def test_parse_quota_files_reads_the_home_line():
    assert doctor.parse_quota_files(QUOTA_OK) == 860_000
    assert doctor.parse_quota_files(QUOTA_FULL) == 995_000
    assert doctor.parse_quota_files("nothing here") is None
    assert doctor.parse_quota_files("nfs:/home 1G 2G 3G 4days 12345 20000 20000") == 12345


def test_clean_environment_passes(cfg, tmp_path):
    checks = by_name(doctor.run_checks(make_ctx(cfg, tmp_path)))
    for name in ("home_file_count_below_990k", "key_file_mode_0600", "no_key_string_in_worktree",
                 "state_paths_outside_home", "codex_binary_version", "openclaude_on_path",
                 "template_sha_matches_forced", "jinja2_importable", "prompt_files_present"):
        assert checks[name]["ok"], checks[name]
    assert checks["home_file_count_below_990k"]["detail"]["source"] == "quota"


def test_home_over_990k_blocks(cfg, tmp_path):
    checks = by_name(doctor.run_checks(make_ctx(cfg, tmp_path, quota=QUOTA_FULL)))
    assert not checks["home_file_count_below_990k"]["ok"]
    assert "home_file_count_below_990k" in doctor.blocking_failures(checks.values())


def test_quota_unparseable_falls_back_to_a_walk_with_a_warning(cfg, tmp_path):
    ctx = make_ctx(cfg, tmp_path, quota="quota: no such command")
    (ctx.home / "a").write_text("1")
    (ctx.home / "b").write_text("2")
    checks = by_name(doctor.run_checks(ctx))
    assert checks["home_file_count_below_990k"]["ok"]
    assert checks["home_file_count_below_990k"]["detail"] == {"files": 2, "limit": 990_000, "source": "walk"}
    assert not checks["home_file_count_from_quota"]["ok"] and not checks["home_file_count_from_quota"]["blocking"]


def test_key_mode_other_than_0600_blocks(cfg, tmp_path):
    checks = by_name(doctor.run_checks(make_ctx(cfg, tmp_path, key_mode=0o640)))
    assert not checks["key_file_mode_0600"]["ok"]


def test_key_shaped_string_in_worktree_is_found_but_never_printed(cfg, tmp_path):
    ctx = make_ctx(cfg, tmp_path)
    leaked = "sk-or-v1-" + "a1b2c3d4e5f6" * 4
    (ctx.worktree / "notes.md").write_text(f"token: {leaked}\n")
    (ctx.worktree / ".git").mkdir()
    (ctx.worktree / ".git" / "x.txt").write_text(leaked)          # skipped directories never count
    checks = by_name(doctor.run_checks(ctx))
    assert not checks["no_key_string_in_worktree"]["ok"]
    assert checks["no_key_string_in_worktree"]["detail"]["files"] == ["notes.md"]
    assert leaked not in json.dumps(checks)
    # the source itself (regexes, startswith checks) is not a key-shaped string
    assert doctor.key_shaped_strings(Path(doctor.__file__).parent) == []


def test_state_path_under_home_blocks(cfg, tmp_path):
    ctx = make_ctx(cfg, tmp_path)
    ctx.cfg["dynorch"]["scratch_root"] = str(ctx.home / "scratch")
    checks = by_name(doctor.run_checks(ctx))
    assert not checks["state_paths_outside_home"]["ok"]
    assert "dynorch.scratch_root" in checks["state_paths_outside_home"]["detail"]["bad"]
    ctx = make_ctx(cfg, tmp_path / "second", env_extra={"TMPDIR": "/home/someone/tmp"})
    checks = by_name(doctor.run_checks(ctx))
    assert checks["state_paths_outside_home"]["detail"]["bad"] == {"env.TMPDIR": "/home/someone/tmp"}


def test_missing_env_path_blocks(cfg, tmp_path):
    ctx = make_ctx(cfg, tmp_path)
    ctx.env = {k: v for k, v in ctx.env.items() if k != "npm_config_cache"}
    checks = by_name(doctor.run_checks(ctx))
    assert checks["state_paths_outside_home"]["detail"]["bad"] == {"env.npm_config_cache": None}


def test_codex_and_openclaude_versions_are_pinned(cfg, tmp_path):
    checks = by_name(doctor.run_checks(make_ctx(cfg, tmp_path, codex="codex-cli 0.155.0")))
    assert not checks["codex_binary_version"]["ok"] and checks["codex_binary_version"]["detail"]["version"] == "0.155.0"
    checks = by_name(doctor.run_checks(make_ctx(cfg, tmp_path / "b", openclaude="0.12.9 (OpenClaude)")))
    assert not checks["openclaude_on_path"]["ok"]
    ctx = make_ctx(cfg, tmp_path / "c")
    (Path(ctx.cfg["dynorch"]["codex_prefix"]) / "bin" / "codex").unlink()
    checks = by_name(doctor.run_checks(ctx))
    assert not checks["codex_binary_version"]["ok"] and checks["codex_binary_version"]["detail"]["version"] is None


def test_base_run_root_and_prompts(cfg, tmp_path):
    ctx = make_ctx(cfg, tmp_path)
    ctx.cfg["base_run_root"] = str(tmp_path / "nope")
    (ctx.worktree / "prompts" / "root.txt").unlink()
    checks = by_name(doctor.run_checks(ctx))
    assert not checks["base_run_root_files_present"]["ok"]
    assert len(checks["base_run_root_files_present"]["detail"]["missing"]) == len(doctor.BASE_RUN_ROOT_FILES)
    assert checks["prompt_files_present"]["detail"]["missing"] == ["prompts/root.txt"]
    real = Path(cfg["base_run_root"])
    if real.exists():
        ctx = make_ctx(cfg, tmp_path / "real")
        assert by_name(doctor.run_checks(ctx))["base_run_root_files_present"]["ok"]


def test_cmd_doctor_exit_codes(cfg, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "DoctorContext", lambda **kw: make_ctx(cfg, tmp_path, quota=QUOTA_FULL))
    assert doctor.cmd_dynorch_doctor(["--config", str(CFG_PATH)]) == 1
    out = capsys.readouterr().out
    assert "home_file_count_below_990k" in out.splitlines()[-1]
    monkeypatch.setattr(doctor, "DoctorContext", lambda **kw: make_ctx(cfg, tmp_path / "ok"))
    assert doctor.cmd_dynorch_doctor(["--config", str(CFG_PATH)]) == 0
