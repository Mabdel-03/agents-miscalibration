"""WP-B: the environment scrub allowlist -- the OpenRouter key never enters a harness environment (plan §6)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.dynorch import stubs  # noqa: F401
from agent_allocation.dynorch.harness.base import (
    ALLOWED_OPENAI_KEYS, INHERITED_KEYS, check_env, harness_env, redacted_env,
)
from agent_allocation.dynorch.runner import scrubbed_env

RUN_KEY = "abcdefabcdefabcdefabcdef"
PROXY = f"http://127.0.0.1:1234/r/{RUN_KEY}/v1"
FAKE_KEY = "sk-or-v1-" + "0" * 64
JOB_ENV = {
    "PATH": "/opt/bin:/usr/bin", "LANG": "en_US.UTF-8", "LC_ALL": "C", "TZ": "UTC", "USER": "u", "LOGNAME": "u",
    "TERM": "xterm", "HOME": "/home/u", "TMPDIR": "/tmp", "SHELL": "/bin/bash", "SLURM_JOB_ID": "1",
    "OPENROUTER_API_KEY": FAKE_KEY, "OPENROUTER_API_KEY_FILE": "/home/u/.config/agents_scaling/x.key",
    "ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
    "OPENAI_API_KEY": "sk-openai-x", "OPENAI_BASE_URL": "https://api.openai.com/v1", "OPENAI_ORG_ID": "org",
    "GEMINI_API_KEY": "g", "CLAUDE_CONFIG_DIR": "/home/u/.claude", "CODEX_HOME": "/home/u/.codex",
    "OPENCLAW_STATE_DIR": "/home/u/.openclaw", "XDG_CONFIG_HOME": "/home/u/.config", "XDG_CACHE_HOME": "/home/u/.cache",
    "HTTP_PROXY": "http://proxy", "HTTPS_PROXY": "http://proxy", "UV_CACHE_DIR": "/x", "HF_HOME": "/y",
    "NODE_OPTIONS": "--inspect", "PYTHONPATH": "/z",
}


def test_scrub_keeps_only_the_allowlist_and_sets_scratch_paths(tmp_path):
    work = tmp_path / "work" / "a1"
    env = harness_env(JOB_ENV, work=work, proxy_base_url=PROXY, run_key=RUN_KEY)
    assert scrubbed_env is harness_env
    inherited = {k for k in env if k in JOB_ENV and env[k] == JOB_ENV[k]}
    assert inherited == set(INHERITED_KEYS) & set(JOB_ENV)
    for forbidden in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_FILE", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
                      "OPENAI_ORG_ID", "GEMINI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "SHELL", "SLURM_JOB_ID",
                      "UV_CACHE_DIR", "HF_HOME", "PYTHONPATH", "OPENCLAW_STATE_DIR"):
        assert forbidden not in env, forbidden
    assert env["OPENAI_API_KEY"] == RUN_KEY and env["OPENAI_BASE_URL"] == PROXY
    assert {k for k in env if k.startswith("OPENAI_")} == {"OPENAI_API_KEY", "OPENAI_BASE_URL"} <= ALLOWED_OPENAI_KEYS
    for key in ("HOME", "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        assert Path(env[key]).is_relative_to(work), key
        assert not str(env[key]).startswith("/home/"), key
    assert env["HOME"] == str(work / "home") and env["TMPDIR"] == str(work / "tmp")
    assert "CLAUDE_CONFIG_DIR" not in env and "CODEX_HOME" not in env       # adapters add their own, under work
    assert env["NODE_OPTIONS"] == "--max-old-space-size=1024"
    serialized = json.dumps(env)
    assert FAKE_KEY not in serialized and "sk-or-" not in serialized and "sk-ant" not in serialized


def test_extra_variables_are_checked_too(tmp_path):
    work = tmp_path / "w"
    with pytest.raises(ValueError, match="OPENROUTER"):
        harness_env(JOB_ENV, work=work, proxy_base_url=PROXY, run_key=RUN_KEY, extra={"OPENROUTER_API_KEY": "x"})
    with pytest.raises(ValueError, match="OpenRouter-shaped key"):
        harness_env(JOB_ENV, work=work, proxy_base_url=PROXY, run_key=RUN_KEY, extra={"SOMETHING": FAKE_KEY})
    with pytest.raises(ValueError, match="OPENAI_\\*"):
        harness_env(JOB_ENV, work=work, proxy_base_url=PROXY, run_key=RUN_KEY, extra={"OPENAI_ORG": "o"})
    with pytest.raises(ValueError, match="not under the work dir"):
        harness_env(JOB_ENV, work=work, proxy_base_url=PROXY, run_key=RUN_KEY, extra={"CODEX_HOME": "/home/u/.codex"})
    with pytest.raises(ValueError, match="run_key"):
        harness_env(JOB_ENV, work=work, proxy_base_url="http://127.0.0.1:1/r/other/v1", run_key=RUN_KEY)
    ok = harness_env(JOB_ENV, work=work, proxy_base_url=PROXY, run_key=RUN_KEY,
                     extra={"CODEX_HOME": str(work / "state" / "codex"), "OPENAI_MODEL": "m"})
    assert ok["CODEX_HOME"].startswith(str(work))


def test_check_env_rejects_tampered_environments(tmp_path):
    work = tmp_path / "w"
    env = harness_env(JOB_ENV, work=work, proxy_base_url=PROXY, run_key=RUN_KEY)
    tampered = {**env, "OPENAI_API_KEY": FAKE_KEY}
    with pytest.raises(ValueError):
        check_env(tampered, work=work, run_key=RUN_KEY)
    with pytest.raises(ValueError, match="HOME"):
        check_env({**env, "HOME": "/home/u"}, work=work, run_key=RUN_KEY)
    with pytest.raises(ValueError, match="localhost"):
        check_env({**env, "OPENAI_BASE_URL": f"http://localhost:1/r/{RUN_KEY}/v1"}, work=work, run_key=RUN_KEY)
    with pytest.raises(ValueError, match="ANTHROPIC"):
        check_env({**env, "ANTHROPIC_API_KEY": "x"}, work=work, run_key=RUN_KEY)


def test_redacted_env_masks_key_shaped_values():
    assert redacted_env({"A": FAKE_KEY, "B": "plain"}) == {"A": "[REDACTED_KEY]", "B": "plain"}
