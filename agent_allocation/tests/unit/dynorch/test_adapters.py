"""WP-B: adapters build argv/env per plan §6 with no key, every state path under scratch, forbidden flags absent."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent_allocation.dynorch.harness import adapters_from_config
from agent_allocation.dynorch.harness.base import (
    FORBIDDEN_PREFIXES, HarnessAdapter, InProcessAdapter, WorkDir, check_env, harness_env, openclaude_result,
)
from agent_allocation.dynorch.harness.codex import CodexAdapter, config_toml
from agent_allocation.dynorch.harness.fake import FakeAdapter
from agent_allocation.dynorch.harness.null import NullAdapter
from agent_allocation.dynorch.harness.openclaude import DISALLOWED_TOOLS, FORBIDDEN_FLAGS, OpenclaudeAdapter
from agent_allocation.dynorch.harness.openclaw import OpenclawAdapter, openclaw_config
from agent_allocation.dynorch.records import RunSpec

PROJECT = Path(__file__).resolve().parents[3]
CFG = json.loads((PROJECT / "configs" / "dynorch_v1.json").read_text("utf-8"))
MODEL = CFG["primary_model"]
RUN_KEY = "0123456789abcdef01234567"
PROXY = f"http://127.0.0.1:18700/r/{RUN_KEY}/v1"
SYSTEM = (PROJECT / "prompts" / "system_dynorch.txt").read_text("utf-8")
SOLVER = (PROJECT / "prompts" / "subagent_dynorch.txt").read_text("utf-8")
TASK = "TASK\nWhat is 6*7?\n\nCOMPUTE BUDGET: 4096 tokens.\n"
POISON_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/mabdel03", "OPENROUTER_API_KEY": "sk-or-v1-" + "a" * 40,
              "ANTHROPIC_API_KEY": "sk-ant-secret", "OPENAI_API_KEY": "sk-openai-secret", "OPENAI_ORG": "org",
              "OPENROUTER_API_KEY_FILE": "/home/x/key", "CLAUDE_CONFIG_DIR": "/home/mabdel03/.claude",
              "CODEX_HOME": "/home/mabdel03/.codex", "TMPDIR": "/tmp", "LANG": "C.UTF-8", "HTTP_PROXY": "http://p",
              "XDG_CONFIG_HOME": "/home/mabdel03/.config"}


def _spec(harness: str, arm: str = "H1", allowed=("Agent",), spawn=("Agent",), rung: int = 4) -> RunSpec:
    return RunSpec(run_key=RUN_KEY, arm=arm, harness=harness, item_id="aime/2025/1", benchmark="aime", stratum="aime",
                   shard_id="parasail", endpoint_tag="parasail/fp8", rung=rung, cap_tokens=rung * 1024, repeat=1,
                   attempt=1, generation=1, partition="main", allowed_tools=tuple(allowed), spawn_tools=tuple(spawn),
                   root_call_ceiling=None, sampling_hash="s", template_sha256="t", prompt_version="dynorch-v1",
                   proxy_version="p", harness_version="h")


def _assert_scrubbed(env: dict, work: Path):
    check_env(env, work=work, run_key=RUN_KEY)
    assert not any(k.startswith(FORBIDDEN_PREFIXES) for k in env)
    assert {k for k in env if k.startswith("OPENAI_")} <= {"OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"}
    assert env["OPENAI_API_KEY"] == RUN_KEY and RUN_KEY in env["OPENAI_BASE_URL"]
    assert "sk-or-" not in json.dumps(env) and "sk-ant" not in json.dumps(env) and "sk-openai" not in json.dumps(env)
    for key in ("HOME", "TMPDIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        assert Path(env[key]).resolve().is_relative_to(work.resolve()), key
    assert "HTTP_PROXY" not in env and env["NO_PROXY"].startswith("127.0.0.1")


@pytest.fixture
def work(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr("agent_allocation.dynorch.harness.base.default_base_env", lambda: POISON_ENV)
    for module in ("null", "fake", "openclaude", "codex", "openclaw"):
        monkeypatch.setattr(f"agent_allocation.dynorch.harness.{module}.default_base_env", lambda: POISON_ENV)
    return tmp_path / "work" / "H1" / "4" / "1" / "aime__2025__1" / "a1"


# -- openclaude ---------------------------------------------------------------------------------------------

def test_openclaude_argv_env_and_files(work: Path):
    adapter = OpenclaudeAdapter(model=MODEL)
    spec = _spec("openclaude")
    adapter.prepare(work, spec, PROXY, SYSTEM, SOLVER, TASK)
    plan = adapter.build_command(work, spec, PROXY)
    argv = list(plan.argv)
    assert argv[0] == "openclaude" and argv[1:4] == ["-p", "--output-format", "json"]
    for flag in ("--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--provider") + 1] == "openai" and argv[argv.index("--model") + 1] == MODEL
    assert argv[argv.index("--system-prompt") + 1] == SYSTEM.rstrip("\n")      # $(cat SYSTEM.md) semantics, §10.2
    assert argv[argv.index("--tools") + 1] == "Agent" and argv[argv.index("--allowedTools") + 1] == "Agent"
    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
    assert argv[argv.index("--setting-sources") + 1] == ""
    i = argv.index("--disallowedTools")
    assert tuple(argv[i + 1:i + 1 + len(DISALLOWED_TOOLS)]) == DISALLOWED_TOOLS
    for flag in FORBIDDEN_FLAGS:
        assert flag not in argv and not any(a.startswith(flag) for a in argv)
    assert "--bare" not in argv and "--max-budget-usd" not in argv and "budget_usd" not in " ".join(argv)
    agents = json.loads(argv[argv.index("--agents") + 1])
    assert agents == json.loads((work / "agents.json").read_text())
    assert agents["solver"]["prompt"] == SOLVER and agents["solver"]["tools"] == []
    assert plan.stdin == TASK.encode() and plan.cwd == work / "ws" and (work / "ws").is_dir()
    env = plan.env
    _assert_scrubbed(env, work)
    assert env["CLAUDE_CONFIG_DIR"] == str(work / "home" / "cfg") and env["HOME"] == str(work / "home")
    assert env["TMPDIR"] == str(work / "tmp")
    for key, value in {"CLAUDE_CODE_USE_OPENAI": "1", "OPENAI_MODEL": MODEL, "CLAUDE_CODE_SUBAGENT_MODEL": MODEL,
                       "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1", "CLAUDE_CODE_AGENT_LIST_IN_MESSAGES": "0",
                       "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1",
                       "DISABLE_AUTOUPDATER": "1", "DISABLE_ERROR_REPORTING": "1", "DISABLE_COMPACT": "1",
                       "CLAUDE_CODE_MAX_RETRIES": "1", "API_TIMEOUT_MS": "3600000",
                       "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
                       "CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED": "1"}.items():
        assert env[key] == value, key
    assert "CLAUDE_CODE_SIMPLE" not in env
    cfg = json.loads((work / "home" / "cfg" / ".openclaude.json").read_text())
    assert cfg["hasCompletedOnboarding"] is True and cfg["bypassPermissionsModeAccepted"] is True
    assert cfg["agentModels"][MODEL] == {"base_url": PROXY, "api_key": RUN_KEY}
    assert set(cfg["agentRouting"].values()) == {MODEL} and "solver" in cfg["agentRouting"]
    assert "localhost" not in PROXY


def test_openclaude_spawn_disabled_arm_offers_no_tools(work: Path):
    adapter = OpenclaudeAdapter(model=MODEL)
    spec = _spec("openclaude", arm="H1n", allowed=(), spawn=())
    adapter.prepare(work, spec, PROXY, SYSTEM, SOLVER, TASK)
    argv = list(adapter.build_command(work, spec, PROXY).argv)
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--allowedTools") + 1] == ""


def test_openclaude_result_parsing():
    ok = openclaude_result(b'{"type":"system"}\n{"type":"result","subtype":"success","is_error":false,"result":"<final>3</final>","num_turns":2,"total_cost_usd":0.1}\n',
                           0, where="t")
    assert ok.cli_result == "<final>3</final>" and ok.reported["num_turns"] == 2
    err = openclaude_result(b'{"type":"result","subtype":"error_during_execution","is_error":true,"result":"API error 402"}',
                            1, where="t")
    assert err.cli_result is None and err.reported["is_error"] is True and "402" in err.reported["error_text"]
    # contract §10.5: OpenClaude keeps subtype "success" on a 402; is_error is the signal
    after_402 = openclaude_result(b'{"type":"result","subtype":"success","is_error":true,"result":"API Error: 402 {\\"error\\":{\\"code\\":\\"budget_exhausted\\"}}","stop_reason":"stop_sequence"}',
                                  1, where="t")
    assert after_402.cli_result is None and after_402.reported["subtype"] == "success"
    assert openclaude_result(b"garbage", 1, where="t").cli_result is None
    no_flag = openclaude_result(b'{"type":"result","subtype":"success","result":"x"}', 0, where="t")
    assert no_flag.cli_result is None                     # a missing is_error never counts as success


def test_openclaude_version_is_recorded_or_absent():
    version = OpenclaudeAdapter(model=MODEL, binary="definitely-not-installed-xyz").version()
    assert version == "absent"
    if shutil.which("openclaude"):
        assert "0.13.0" in OpenclaudeAdapter(model=MODEL).version()


# -- codex --------------------------------------------------------------------------------------------------

def test_codex_config_command_and_env(work: Path):
    adapter = CodexAdapter(model=MODEL, prefix=CFG["dynorch"]["codex_prefix"], wall_base=600, wall_per_rung=120)
    spec = _spec("codex", arm="H3", allowed=("spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent",
                                             "update_plan"), spawn=("spawn_agent",), rung=16)
    adapter.prepare(work, spec, PROXY, SYSTEM, SOLVER, TASK)
    home = work / "state" / "codex"
    assert home.is_dir()
    toml = (home / "config.toml").read_text()
    for line in (f'model = "{MODEL}"', 'model_provider = "meter"', "model_context_window = 131072",
                 'model_reasoning_summary = "none"', "show_raw_agent_reasoning = false", 'approval_policy = "never"',
                 'sandbox_mode = "read-only"', 'web_search = "disabled"', "check_for_update_on_startup = false",
                 "[model_providers.meter]", 'name = "Local metering proxy"', f'base_url = "{PROXY}"',
                 'env_key = "OPENAI_API_KEY"', 'wire_api = "responses"', "request_max_retries = 2",
                 "stream_max_retries = 2", "stream_idle_timeout_ms = 1800000", "[features]", "shell_tool = false",
                 "unified_exec = false", "view_image = false", "code_mode = false", "apps = false", "plugins = false",
                 "hooks = false", "image_generation = false", "skill_search = false", "goals = false",
                 "multi_agent = true", "multi_agent_v2 = false", "[tools.experimental_request_user_input]",
                 "[tools.update_plan]", "[agents]", "enabled = true", "max_concurrent_threads_per_session = 4",
                 "max_depth = 1", "[shell_environment_policy]", 'inherit = "none"'):
        assert line in toml.splitlines(), line
    lines = toml.splitlines()
    assert lines[lines.index("[tools.experimental_request_user_input]") + 1] == "enabled = false"
    assert lines[lines.index("[tools.update_plan]") + 1] == "enabled = true"
    assert "tools.update_plan = " not in toml and "experimental_request_user_input = " not in toml   # structs, not bools
    assert f'model_instructions_file = "{work / "SYSTEM.md"}"' in toml
    assert (work / "SYSTEM.md").read_text() == SYSTEM
    assert 'wire_api = "chat"' not in toml
    plan = adapter.build_command(work, spec, PROXY)
    argv = list(plan.argv)
    assert argv[:3] == ["timeout", "--kill-after=30s", f"{600 + 120 * 16}s"]
    assert argv[3] == str(Path(CFG["dynorch"]["codex_prefix"]) / "bin" / "codex")
    assert argv[4:] == ["exec", "--json", "--ephemeral", "--strict-config", "--skip-git-repo-check", "-C",
                        str(work / "ws"), "-s", "read-only", "-o", str(work / "final.txt"), "-"]
    assert "--ignore-user-config" not in argv and "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert plan.stdin == TASK.encode()
    _assert_scrubbed(plan.env, work)
    assert plan.env["CODEX_HOME"] == str(home)


def test_codex_spawn_disabled_arm_disables_agents(work: Path):
    adapter = CodexAdapter(model=MODEL, prefix=CFG["dynorch"]["codex_prefix"])
    spec = _spec("codex", arm="H3n", allowed=("update_plan",), spawn=())
    adapter.prepare(work, spec, PROXY, SYSTEM, SOLVER, TASK)
    toml = (work / "state" / "codex" / "config.toml").read_text()
    assert "enabled = false" in toml.splitlines()


def test_codex_result_parsing(tmp_path):
    adapter = CodexAdapter(model=MODEL, prefix=str(tmp_path))
    work = tmp_path / "w"
    WorkDir(work).create()
    (work / "final.txt").write_text("<final>7</final>")
    events = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "1", "type": "collab_tool_call", "tool": "spawn_agent", "status": "completed",
                                            "receiver_thread_ids": ["c1"]}},
        {"type": "item.completed", "item": {"id": "2", "type": "agent_message", "text": "<final>7</final>"}},
        {"type": "turn.completed", "usage": {"input_tokens": 900, "output_tokens": 300, "reasoning_output_tokens": 200}},
    ]
    stdout = "\n".join(json.dumps(e) for e in events).encode() + b"\nnot json\n"
    result = adapter.parse_result(stdout, b"", work)
    assert result.cli_result == "<final>7</final>"
    assert result.reported["usage"]["output_tokens"] == 300 and result.reported["spawns"] == 1
    assert result.reported["final_file_matches_last_message"] is True and result.reported["malformed_lines"] == 1
    failed = adapter.parse_result(stdout + b'\n{"type":"turn.failed","error":{"message":"boom"}}', b"", work)
    assert failed.cli_result is None and failed.reported["failures"]


def test_codex_version_is_recorded_or_absent():
    assert CodexAdapter(model=MODEL, prefix="/nonexistent/prefix").version() == "absent"
    real = Path(CFG["dynorch"]["codex_prefix"]) / "bin" / "codex"
    if real.is_file() and shutil.which("node"):
        assert "0.156.1" in CodexAdapter(model=MODEL, prefix=CFG["dynorch"]["codex_prefix"]).version()


# -- null / fake --------------------------------------------------------------------------------------------

def test_null_adapter_is_in_process_and_builds_the_single_request(work: Path):
    adapter = NullAdapter(model=MODEL)
    spec = _spec("null", arm="H0", allowed=(), spawn=())
    assert isinstance(adapter, InProcessAdapter) and adapter.in_process
    adapter.prepare(work, spec, PROXY, SYSTEM, SOLVER, TASK)
    plan = adapter.build_command(work, spec, PROXY)
    assert plan.argv == () and plan.stdin is None
    _assert_scrubbed(plan.env, work)
    body = adapter.request_body(work, spec)
    assert body["stream"] is False and body["max_tokens"] == 4096 and "tools" not in body
    assert body["messages"] == [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]


def test_fake_adapter_launches_the_cli(work: Path):
    adapter = FakeAdapter(model=MODEL)
    spec = _spec("fake", arm="HF", allowed=("spawn_subagent",), spawn=("spawn_subagent",))
    adapter.prepare(work, spec, PROXY, SYSTEM, SOLVER, TASK)
    plan = adapter.build_command(work, spec, PROXY)
    assert plan.argv[1].endswith("tests/dynorch/fake_harness_cli.py") and Path(plan.argv[1]).is_file()
    assert "--tools" in plan.argv and plan.argv[plan.argv.index("--tools") + 1] == "spawn_subagent"
    assert plan.stdin == TASK.encode()
    _assert_scrubbed(plan.env, work)


# -- openclaw (fallback; skip-if-absent for anything touching the CLI) -------------------------------------

def test_openclaw_config_and_gateway_plan(work: Path):
    adapter = OpenclawAdapter(model=MODEL)
    spec = _spec("openclaw", arm="H2", allowed=("sessions_spawn", "subagents"), spawn=("sessions_spawn",))
    adapter.prepare(work, spec, PROXY, SYSTEM, SOLVER, TASK)
    cfg = json.loads((work / "openclaw.json").read_text())
    provider = cfg["models"]["providers"]["meter"]
    assert provider["baseUrl"] == PROXY and provider["api"] == "openai-completions" and provider["apiKey"] == RUN_KEY
    assert provider["models"][0]["compat"]["maxTokensField"] == "max_tokens"
    assert cfg["tools"]["profile"] == "minimal" and cfg["tools"]["allow"] == ["sessions_spawn", "subagents"]
    assert cfg["agents"]["defaults"]["contextPruning"] == {"mode": "off"}
    assert cfg["agents"]["defaults"]["compaction"] == {"mode": "default", "memoryFlush": {"enabled": False}}
    assert cfg["plugins"]["enabled"] is False and cfg["skills"]["allowBundled"] == [] and cfg["channels"] == {}
    assert cfg["gateway"]["bind"] == "loopback" and cfg["gateway"]["auth"]["mode"] == "token"
    assert (work / "workspace" / "AGENTS.md").read_text().startswith(SYSTEM)
    plan = adapter.build_command(work, spec, PROXY)
    assert plan.argv[:3] == ("openclaw", "gateway", "run") and "--allow-unconfigured" in plan.argv
    assert "--bind" in plan.argv and plan.argv[plan.argv.index("--bind") + 1] == "loopback"
    _assert_scrubbed(plan.env, work)
    assert plan.env["OPENCLAW_STATE_DIR"] == str(work / "state" / "openclaw")
    assert plan.env["OPENCLAW_CONFIG_PATH"] == str(work / "openclaw.json")
    assert isinstance(adapter, InProcessAdapter)


@pytest.mark.skipif(shutil.which("openclaw") is None, reason="openclaw CLI absent")
def test_openclaw_version_when_present():
    assert OpenclawAdapter(model=MODEL).version() not in ("absent", "unknown")


# -- registry -----------------------------------------------------------------------------------------------

def test_adapters_from_config_covers_every_configured_harness():
    adapters = adapters_from_config(CFG)
    harnesses = {a["harness"] for a in CFG["dynorch"]["arms"].values()} | {a["harness"] for a in CFG["dynorch"]["extension_arms"].values()}
    assert harnesses <= set(adapters)
    assert all(isinstance(a, HarnessAdapter) for a in adapters.values())
    assert adapters["codex"].binary == str(Path(CFG["dynorch"]["codex_prefix"]) / "bin" / "codex")


def test_openclaw_config_generator_is_deterministic(tmp_path):
    a = openclaw_config(model=MODEL, proxy_base_url=PROXY, run_key=RUN_KEY, workspace=tmp_path, port=19001, token="t")
    b = openclaw_config(model=MODEL, proxy_base_url=PROXY, run_key=RUN_KEY, workspace=tmp_path, port=19001, token="t")
    assert a == b and a["agents"]["defaults"]["subagents"]["maxSpawnDepth"] == 1
