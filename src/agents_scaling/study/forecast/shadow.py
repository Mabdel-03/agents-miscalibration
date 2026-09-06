"""Strict shadow-forecast parsing and the on-disk forecast / report-render files (§8.7).

``parse_forecast`` applies the §3.5 salvage rule (whitespace strip, at most one enclosing
Markdown fence — ``parse.candidate.strip_one_fence``) and then a strict JSON parse
(``parse.candidate.load_strict_json``: duplicate keys, non-finite literals and trailing
text rejected).  The value must be an object with **exactly** the five forecast keys
(handoff ``forecast.schema.json``), each a finite number in [0, 1] or ``null``; fields the
trusted mask marks ``required`` must be numbers (``REQUIRED_NULL`` otherwise).  A
``length``-truncated completion is ``TRUNCATED`` before any parse.  An invalid forecast is
recorded as such — the development-fitted scope-specific marginal prior and missingness
flag are applied in analysis (§8.7), never here.

Files (``<run_root>/forecast/``):
* ``<source_id>.<method>.json`` — the forecast outcome (``forecast_output``): report id,
  request id, manifest, raw content, parsed values, parse status, cost, plus the §8.13
  ``ConfidenceRow`` projection;
* ``reports/<source_id>.<method>.json`` — the report render for the capture stage
  (``manifest.report_render``);
* ``errors/<source_id>.<method>.json`` — the last failure of the cell for that job.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.experiment import io
from agents_scaling.study.forecast.manifest import MASK, required_fields
from agents_scaling.study.forecast.report import Report
from agents_scaling.study.parse.candidate import StrictJSONError, load_strict_json, strip_one_fence
from agents_scaling.study.types import RequestRecord

FORECAST_SCHEMA_VERSION = 1
FORECAST_DIR = "forecast"
REPORTS_DIR = "reports"
ERRORS_DIR = "errors"
#: Handoff ``forecast.schema.json`` keys in wire order.
FORECAST_FIELDS: tuple[str, ...] = ("q_personal", "q_child_contract", "q_team_now", "q_recover", "q_preserve")
PARSE_OK = "ok"
PARSE_FAILURES: tuple[str, ...] = (
    "TRUNCATED",
    "EMPTY",
    "NOT_JSON",
    "DUPLICATE_KEY",
    "NONFINITE",
    "TRAILING_TEXT",
    "SCHEMA",
    "REQUIRED_NULL",
)
FINISH_LENGTH = "length"


@dataclass(frozen=True)
class Forecast:
    """The five probability fields (``None`` = inapplicable / null)."""

    q_personal: float | None
    q_child_contract: float | None
    q_team_now: float | None
    q_recover: float | None
    q_preserve: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in FORECAST_FIELDS}


@dataclass(frozen=True)
class ParsedForecast:
    """``status == "ok"`` with a :class:`Forecast`, or a failure code with a detail string."""

    status: str
    forecast: Forecast | None
    detail: str = ""

    @property
    def valid(self) -> bool:
        return self.status == PARSE_OK and self.forecast is not None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail, "values": None if self.forecast is None else self.forecast.to_dict()}


def _failure(code: str, detail: str) -> ParsedForecast:
    assert code in PARSE_FAILURES, code
    return ParsedForecast(code, None, detail)


def _probability(name: str, value: Any) -> float | None:
    """A finite number in [0, 1] → float; ``None`` → None; anything else → ``SchemaError``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrictJSONError("SCHEMA", f"{name} must be a number or null, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise StrictJSONError("NONFINITE", f"{name} is not finite")
    if not 0.0 <= number <= 1.0:
        raise StrictJSONError("SCHEMA", f"{name}={number!r} outside [0, 1]")
    return number


def parse_forecast(content: str | None, *, finish_reason: str | None = "stop", mask: Mapping[str, Any] = MASK) -> ParsedForecast:
    """Strict parse of a shadow-forecast completion under the trusted ``mask``."""
    if finish_reason == FINISH_LENGTH:
        return _failure("TRUNCATED", "completion hit the 256-token cap")
    if content is None or not isinstance(content, str):
        return _failure("EMPTY", "no content channel")
    try:
        value = load_strict_json(strip_one_fence(content))
        if not isinstance(value, dict):
            raise StrictJSONError("SCHEMA", f"forecast must be an object, got {type(value).__name__}")
        if tuple(sorted(value)) != tuple(sorted(FORECAST_FIELDS)):
            extra = sorted(set(value) - set(FORECAST_FIELDS))
            missing = sorted(set(FORECAST_FIELDS) - set(value))
            raise StrictJSONError("SCHEMA", f"keys must be exactly {list(FORECAST_FIELDS)}; extra={extra} missing={missing}")
        values = {name: _probability(name, value[name]) for name in FORECAST_FIELDS}
    except StrictJSONError as exc:
        return _failure(exc.code, exc.detail)
    null_required = [name for name in required_fields(mask) if values.get(name) is None]
    if null_required:
        return _failure("REQUIRED_NULL", f"required fields are null: {null_required}")
    return ParsedForecast(PARSE_OK, Forecast(**values), "")


# --------------------------------------------------------------------------- files


def forecast_dir(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / FORECAST_DIR


def _name(source_id: str, method: str) -> str:
    method = getattr(method, "value", method)
    return f"{source_id}.{method}.json"


def forecast_path(run_root: str | os.PathLike, source_id: str, method: Any) -> Path:
    return forecast_dir(run_root) / _name(source_id, method)


def report_path(run_root: str | os.PathLike, source_id: str, method: Any) -> Path:
    return forecast_dir(run_root) / REPORTS_DIR / _name(source_id, method)


def error_path(run_root: str | os.PathLike, source_id: str, method: Any) -> Path:
    return forecast_dir(run_root) / ERRORS_DIR / _name(source_id, method)


def forecast_output(
    report: Report,
    manifest: Mapping[str, Any],
    record: RequestRecord,
    parsed: ParsedForecast,
    *,
    aliased: bool,
    produced_at: float,
    cell_id: str | None = None,
) -> dict[str, Any]:
    """The JSON-safe forecast outcome of one (item, method): raw content, parsed values,
    parse status, cost, and the §8.13 ``ConfidenceRow`` projection."""
    response = record.response
    values = None if parsed.forecast is None else parsed.forecast.to_dict()
    return {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "kind": "SHADOW_FORECAST",
        "report_id": report.report_id,
        "source_id": report.source_id,
        "method": report.method.value,
        "cell_id": report.cell_id,
        "seal": report.seal,
        "selection_id": report.selection_id,
        "pool_id": report.pool_id,
        "request_id": record.request_id,
        "forecast_prompt_hash": record.identity.get("input_hash"),
        "aliased": bool(aliased),
        "manifest": dict(manifest),
        "raw_content": response.get("content"),
        "reasoning_present": response.get("reasoning") is not None,
        "finish_reason": response.get("finish_reason"),
        "parse_status": parsed.status,
        "parse_detail": parsed.detail,
        "parsed": values,
        "selected_personal_confidence": report.selected_personal_confidence,
        "personal_confidence_missing": report.personal_confidence_missing,
        "selected_is_sentinel": report.selected_is_sentinel,
        "cost": {
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": response.get("completion_tokens"),
            "reasoning_tokens": response.get("reasoning_tokens"),
            "flops": dict(record.flops),
        },
        "confidence_row": {
            "StateSnapshot_id": report.report_id,
            "scope": list(manifest.get("scope", [])),
            "observer_role": manifest.get("observer_role"),
            "information_set": manifest.get("information_set"),
            "selected_pool_id": manifest.get("selected_pool_id"),
            "operation_id": manifest.get("operation_id"),
            "remaining_allowance": manifest.get("remaining_allowance"),
            "probability_fields": values,
            "forecast_prompt_hash": record.identity.get("input_hash"),
            "future_label_ids_sealed": report.selection_id,
            "parsing_status": parsed.status,
            "cost": dict(record.flops),
        },
        "producer": {"cell_id": cell_id, "produced_at": produced_at},
    }


def write_forecast(run_root: str | os.PathLike, payload: Mapping[str, Any]) -> Path:
    path = forecast_path(run_root, str(payload["source_id"]), str(payload["method"]))
    io.write_json(path, dict(payload))
    return path


def write_report_render(run_root: str | os.PathLike, payload: Mapping[str, Any]) -> Path:
    path = report_path(run_root, str(payload["source_id"]), str(payload["method"]))
    io.write_json(path, dict(payload))
    return path


__all__ = [
    "ERRORS_DIR",
    "FORECAST_DIR",
    "FORECAST_FIELDS",
    "FORECAST_SCHEMA_VERSION",
    "Forecast",
    "PARSE_FAILURES",
    "PARSE_OK",
    "ParsedForecast",
    "REPORTS_DIR",
    "error_path",
    "forecast_dir",
    "forecast_output",
    "forecast_path",
    "parse_forecast",
    "report_path",
    "write_forecast",
    "write_report_render",
]
