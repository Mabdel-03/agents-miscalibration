"""WP-D: ``allocation dynorch-doctor`` — the Gate 0 preconditions (plan §9 Gate 0; contract §8). Spends nothing.

Checks (each ``{"check", "ok", "blocking", "detail"}``): ``/home`` file count < 990k (``quota -s``, falling back to
a walk of ``$HOME`` with a warning); OpenRouter key file mode 0600; no key-shaped ``sk-or-`` string anywhere in the
worktree; every configured state/scratch/temp path outside ``/home``; the Codex binary in the scratch prefix reports
``codex-cli 0.156.1``; ``openclaude`` on PATH (0.13.0); the vendored chat template matches ``api/forced.py``'s pinned
sha; the sealed base run root's inputs are present; ``jinja2`` 3.1.6 importable; the frozen prompt files exist. The
key's bytes are never read here (only its mode); nothing prints a key.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HOME_FILE_LIMIT = 990_000
CODEX_VERSION = "0.156.1"
OPENCLAUDE_VERSION = "0.13.0"
JINJA2_VERSION = "3.1.6"
_KEY_SHAPED = re.compile(r"sk-or-(?:v\d+-)?[A-Za-z0-9_\-]{8,}")
# The base study's test tree carries four obviously fake keys (fixtures of tests/unit/api and tests/unit/code).
# They are allowlisted by the sha256 of the exact matched string, never by file: any other key-shaped string in
# those files, or these strings anywhere else, still blocks.
KNOWN_FAKE_KEY_DIGESTS: Mapping[str, frozenset[str]] = {
    "tests/unit/api/conftest.py": frozenset({"d470bd61240c8e058b2ce54498c9c7954822471127d08ba53e37e2eb84d675e2"}),
    "tests/unit/api/test_client.py": frozenset({"8c3f57df825ed1df04c74d42902c8c38cc579ffbfbb9f922d105cab152918560"}),
    "tests/unit/code/test_attacks.py": frozenset({"939004f775c115cc484b44f260d1e3361a1b89dad7a6deb0f0de23ecad75f810",
                                                  "04dd6ddea9bc7d2710d5432f90a1cd8161a3f5aaf3140039eb407d28b4178081"}),
}
_SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_TEXT_SUFFIXES = frozenset({".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".sh", ".sbatch", ".cfg", ".ini", ".yaml",
                            ".yml", ".tex", ".csv", ".jinja", ".env", ""})
BASE_RUN_ROOT_FILES: tuple[str, ...] = (
    "manifests/panels.json",
    "manifests/public/aime.jsonl", "manifests/public/hmmt.jsonl", "manifests/public/omni_math_hard.jsonl",
    "manifests/public/gpqa_diamond.jsonl",
    "private/aime.jsonl", "private/hmmt.jsonl", "private/omni_math_hard.jsonl", "private/gpqa_diamond.jsonl",
    "shards/assignment.json",
    "shards/mancer/locks/openrouter_endpoints.json", "shards/parasail/locks/openrouter_endpoints.json",
    "merged/tables/policy_outcomes.parquet", "merged/tables/policy_costs.parquet",
    "merged/tables/sample_bank_items.parquet", "merged/tables/candidates.parquet",
    "calibration.json",
)
PROMPT_FILES: tuple[str, ...] = ("prompts/system_dynorch.txt", "prompts/subagent_dynorch.txt", "prompts/system.txt",
                                 "prompts/root.txt", "prompts/output_requirements.json")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class DoctorContext:
    """Everything the checks touch, injectable for tests: the home directory, the worktree, the environment, and the
    command runner (``argv -> (returncode, stdout)``)."""
    cfg: Mapping[str, Any]
    home: Path = field(default_factory=lambda: Path.home())
    worktree: Path = field(default_factory=project_root)
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    run_command: Callable[[Sequence[str], Mapping[str, str] | None], tuple[int, str]] = None  # type: ignore[assignment]
    key_file: Path | None = None
    run_root: Path | None = None
    walk_limit: int = 2_000_000

    def __post_init__(self) -> None:
        if self.run_command is None:
            self.run_command = _run
        if self.key_file is None:
            from agent_allocation.apikey import key_file_path
            self.key_file = key_file_path()


def _run(argv: Sequence[str], env: Mapping[str, str] | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=60,
                              env=None if env is None else dict(env))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _parse_count(token: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgG]?)\*?", token.strip())
    if not match:
        raise ValueError(f"not a quota count: {token!r}")
    value = float(match.group(1))
    scale = {"": 1, "k": 1_000, "m": 1_000_000, "g": 1_000_000_000}[match.group(2).lower()]
    return int(round(value * scale))


def parse_quota_files(output: str) -> int | None:
    """The ``files`` column of the ``/home`` line of ``quota -s`` (None when no such line parses)."""
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "/home" not in line:
            continue
        parts = line.split()
        if len(parts) == 1 and index + 1 < len(lines):      # long filesystem names wrap onto the next line
            parts = parts + lines[index + 1].split()
        if len(parts) < 6:
            continue
        # <fs> <space> <quota> <limit> [<grace>] <files> <quota> <limit> [<grace>]: the files column is the first
        # count after the three space columns (a grace column, when present, is a duration and never parses).
        for token in parts[4:]:
            try:
                return _parse_count(token)
            except ValueError:
                continue
    return None


def count_home_files(home: Path, limit: int) -> int:
    count = 0
    for _root, dirs, files in os.walk(home, onerror=lambda _e: None):
        count += len(dirs) + len(files)
        if count > limit:
            break
    return count


def home_file_count(ctx: DoctorContext) -> tuple[int, str]:
    """(count, source): ``quota -s`` when it parses, else a walk of ``$HOME`` (source ``walk`` = a warning)."""
    rc, out = ctx.run_command(["quota", "-s"], None)
    if rc in (0, 1):                       # quota exits 1 when a filesystem is over quota; the table still prints
        parsed = parse_quota_files(out)
        if parsed is not None:
            return parsed, "quota"
    return count_home_files(ctx.home, ctx.walk_limit), "walk"


def key_file_mode_ok(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    mode = stat.S_IMODE(path.stat().st_mode)
    return mode == 0o600, f"{path} mode {oct(mode)}"


def key_shaped_strings(root: Path, *, max_bytes: int = 4_000_000,
                       allow: Mapping[str, frozenset[str]] = KNOWN_FAKE_KEY_DIGESTS) -> list[str]:
    """Worktree-relative paths of files containing a key-shaped ``sk-or-...`` string (never the string itself).
    A match whose sha256 is allowlisted for that exact relative path (the base study's fake fixtures) is skipped."""
    from hashlib import sha256
    hits: list[str] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            path = Path(dirpath) / name
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > max_bytes or path.is_symlink():
                    continue
                text = path.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
            relative = str(path.relative_to(root))
            allowed = allow.get(relative, frozenset())
            if any(sha256(m.encode()).hexdigest() not in allowed for m in _KEY_SHAPED.findall(text)):
                hits.append(relative)
    return sorted(hits)


def configured_paths(ctx: DoctorContext) -> dict[str, str | None]:
    """Every path that must resolve outside ``/home``: scratch root, codex prefix, run root, and the cache/temp
    environment variables the sbatch header exports."""
    dyn = ctx.cfg.get("dynorch", {})
    paths: dict[str, str | None] = {
        "dynorch.scratch_root": dyn.get("scratch_root"),
        "dynorch.codex_prefix": dyn.get("codex_prefix"),
        "run_root": str(ctx.run_root) if ctx.run_root else None,
    }
    for var in ("TMPDIR", "npm_config_cache", "UV_CACHE_DIR", "HF_HOME", "MPLCONFIGDIR"):
        paths[f"env.{var}"] = ctx.env.get(var)
    return paths


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def under_home(path: str, home: Path) -> bool:
    """True when ``path`` is on the inode-limited home filesystem: literally under ``/home``, under the directory
    ``/home`` resolves to (``/orcd/home/002`` on this host), or under ``home`` itself (before or after resolving)."""
    literal = Path(path).expanduser()
    resolved = _resolve(literal)
    roots = [Path("/home"), _resolve(Path("/home")), home, _resolve(home)]
    for candidate in (literal, resolved):
        for root in roots:
            if candidate == root or root in candidate.parents:
                return True
    return False


def codex_binary(ctx: DoctorContext) -> Path:
    prefix = ctx.cfg.get("dynorch", {}).get("codex_prefix")
    if not prefix:
        raise KeyError("config.dynorch.codex_prefix is missing")
    return Path(prefix) / "bin" / "codex"


def probe_env(ctx: DoctorContext) -> dict[str, str]:
    """The environment for CLI version probes: ``CODEX_HOME`` (and HOME-derived state) under the scratch root, so
    a bare ``codex --version`` never writes ``~/.codex`` (it does when CODEX_HOME is unset: an observed /home write)."""
    scratch = ctx.cfg.get("dynorch", {}).get("scratch_root")
    if not scratch:
        raise KeyError("config.dynorch.scratch_root is missing")
    probe_home = Path(scratch) / "doctor-probe-home"
    return {**dict(ctx.env), "CODEX_HOME": str(probe_home / "codex"), "TMPDIR": str(probe_home / "tmp"),
            "XDG_CACHE_HOME": str(probe_home / "cache"), "XDG_STATE_HOME": str(probe_home / "state")}


def version_of(ctx: DoctorContext, argv: Sequence[str]) -> str | None:
    env = probe_env(ctx)
    for key in ("CODEX_HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    rc, out = ctx.run_command(argv, env)
    if rc != 0:
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", out)
    return match.group(1) if match else None


def template_sha_ok() -> tuple[bool, str]:
    from agent_allocation.api.forced import TEMPLATE_SHA256, TemplateError, template_path, template_sha256
    try:
        template_path("qwen3.8-27b")
        digest = template_sha256("qwen3.8-27b")
    except (TemplateError, OSError, KeyError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return digest == TEMPLATE_SHA256["qwen3.8-27b"], digest


def run_checks(ctx: DoctorContext) -> list[dict[str, Any]]:
    """Every doctor check; ``blocking=False`` rows are warnings."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any, *, blocking: bool = True) -> None:
        checks.append({"check": name, "ok": bool(ok), "blocking": blocking, "detail": detail})

    count, source = home_file_count(ctx)
    add("home_file_count_below_990k", count < HOME_FILE_LIMIT, {"files": count, "limit": HOME_FILE_LIMIT, "source": source})
    if source == "walk":
        add("home_file_count_from_quota", False, "quota -s did not parse; the count above comes from a walk of $HOME",
            blocking=False)

    ok, detail = key_file_mode_ok(ctx.key_file)
    add("key_file_mode_0600", ok, detail)

    hits = key_shaped_strings(ctx.worktree)
    add("no_key_string_in_worktree", not hits, {"files": hits, "worktree": str(ctx.worktree)})

    bad: dict[str, str | None] = {}
    for name, value in configured_paths(ctx).items():
        if value is None or not Path(value).is_absolute() or under_home(value, ctx.home):
            bad[name] = value
    add("state_paths_outside_home", not bad, {"bad": bad, "home": str(ctx.home)})

    try:
        codex = codex_binary(ctx)
        version = version_of(ctx, [str(codex), "--version"]) if codex.exists() else None
        add("codex_binary_version", version == CODEX_VERSION, {"binary": str(codex), "version": version,
                                                                 "expected": CODEX_VERSION})
    except KeyError as exc:
        add("codex_binary_version", False, str(exc))

    openclaude = shutil.which("openclaude", path=ctx.env.get("PATH"))
    version = version_of(ctx, [openclaude, "--version"]) if openclaude else None
    add("openclaude_on_path", version == OPENCLAUDE_VERSION, {"binary": openclaude, "version": version,
                                                              "expected": OPENCLAUDE_VERSION})

    ok, detail = template_sha_ok()
    add("template_sha_matches_forced", ok, detail)

    base = ctx.cfg.get("base_run_root")
    missing = [rel for rel in BASE_RUN_ROOT_FILES if not base or not (Path(base) / rel).exists()]
    add("base_run_root_files_present", not missing and bool(base), {"base_run_root": base, "missing": missing})

    try:
        import jinja2
        add("jinja2_importable", jinja2.__version__ == JINJA2_VERSION, {"version": jinja2.__version__, "expected": JINJA2_VERSION})
    except ImportError as exc:
        add("jinja2_importable", False, str(exc))

    missing_prompts = [rel for rel in PROMPT_FILES if not (ctx.worktree / rel).exists()]
    add("prompt_files_present", not missing_prompts, {"missing": missing_prompts})
    return checks


def blocking_failures(checks: Sequence[Mapping[str, Any]]) -> list[str]:
    return [c["check"] for c in checks if c["blocking"] and not c["ok"]]


def cmd_dynorch_doctor(argv: list[str]) -> int:
    """``allocation dynorch-doctor --config CFG [--run-dir RUN] [--json]``: exit 1 on any blocking failure."""
    parser = argparse.ArgumentParser(prog="allocation dynorch-doctor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    cfg = load_config(args.config)
    ctx = DoctorContext(cfg=cfg, run_root=Path(args.run_dir) if args.run_dir else None)
    checks = run_checks(ctx)
    print(json.dumps(checks, indent=1, sort_keys=True, default=str))
    failed = blocking_failures(checks)
    warnings = [c["check"] for c in checks if not c["blocking"] and not c["ok"]]
    print(json.dumps({"blocking_failures": failed, "warnings": warnings}), file=sys.stdout)
    return 1 if failed else 0
