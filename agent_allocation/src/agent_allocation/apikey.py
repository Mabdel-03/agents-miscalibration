"""API key loading and redaction. Only the lead-authorized dispatcher process calls ``load_api_key``.

Order: ``OPENROUTER_API_KEY`` (the pack's contract), else the file named by ``OPENROUTER_API_KEY_FILE``, else
``~/.config/agents_scaling/openrouter_allocation_v1.key``. A key file readable by group or others is refused.
Tests and mocks never need a key.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

DEFAULT_KEY_FILE = Path.home() / ".config" / "agents_scaling" / "openrouter_allocation_v1.key"
_KEY_PATTERN = re.compile(r"sk-or-(?:v\d+-)?[A-Za-z0-9_\-]{8,}")
_BEARER_PATTERN = re.compile(r"(?i)(authorization\s*[:=]\s*)?bearer\s+[A-Za-z0-9._\-]{8,}")


class KeyUnavailable(RuntimeError):
    pass


def key_file_path() -> Path:
    return Path(os.environ.get("OPENROUTER_API_KEY_FILE") or DEFAULT_KEY_FILE).expanduser()


def load_api_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not value:
        path = key_file_path()
        if not path.exists():
            raise KeyUnavailable(f"No OpenRouter key: set OPENROUTER_API_KEY or create {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise KeyUnavailable(f"{path} is readable by group/others (mode {oct(mode)}); chmod 600 it")
        value = path.read_text().strip()
    if not value.startswith("sk-or-"):
        raise KeyUnavailable("OpenRouter key has an unexpected format")
    return value


def redact(text: str | bytes | None) -> str | None:
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = _KEY_PATTERN.sub("[REDACTED_KEY]", text)
    return _BEARER_PATTERN.sub("[REDACTED_AUTH]", text)


REDACTED_HEADERS = frozenset({"authorization", "cookie", "set-cookie", "x-api-key", "proxy-authorization"})


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k.lower(): ("[REDACTED]" if k.lower() in REDACTED_HEADERS else (redact(v) or ""))
            for k, v in headers.items()}
