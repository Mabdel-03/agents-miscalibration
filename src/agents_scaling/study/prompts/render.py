"""Byte-exact prompt rendering from the frozen templates (WP1).

Spec: §5.5 (the four framing cells; clauses and their placement are frozen literally; the
task, solving instruction, schema and safety text are identical in every cell), §3.5 (the
one-line candidate schema; the failure sentinel replaces raw malformed text), §4.1 (no
arm names, no numeric team size in stateless roots, truthful role disclosure), §4.2
(DEC/CEN/S_HISTORY state contracts), §4.3 (the ordered canonical state array
``[role_contract, task, task_only_anchor, own_saved_candidate,
allowed_peer_or_hub_packets, allowed_public_observations, state_anchor,
output_contract]`` with fixed neutral delimiters; saved objects are fallible task data),
§4.5/§4.6 (focal revision, common consumer, N=1 truthfulness), §8.7 (shadow forecast),
§3.3 (official HLE judge prompt).  Architecture §1.5; corrections P0-1 (main-tier DEC
roots use ``dec_root_truthful``; N-panel roots use ``F00``), P0-2 (authored artifacts),
P1-3 (N=1 never mentions peers; hub at N=1 has zero workers).

Frozen rendering rules
* One ``user`` message, no system prompt.  ``Rendered.messages`` is what enters
  ``input_hash``; ``Rendered.anchors`` holds UTF-8 byte offsets into that content.
* Placeholder substitution is a single pass over the template, so text inside a task or
  a saved object can never be re-substituted.
* Empty clause rule (§5.5 level 0): the placeholder line is removed together with its
  trailing newline.  The ``00`` cell therefore keeps the blank line(s) around the removed
  lines exactly as the template leaves them.
* Saved objects are rendered as compact JSON (``ensure_ascii=False``); an invalid
  candidate record renders as the exact §3.5 sentinel bytes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from agents_scaling.study import prompts
from agents_scaling.study.types import (
    SENTINEL,
    SENTINEL_JSON,
    Assignment,
    Candidate,
    CandidateRecord,
    DelegateAction,
    Domain,
    FinalAction,
    Framing,
    Packet,
    PublicTask,
    SubtaskResult,
)

# --------------------------------------------------------------------------- frozen literals

#: §3.5 one-line candidate schema (byte-exact from the spec).
CANDIDATE_SCHEMA_LINE = (
    '{"approach":"brief method","evidence":[{"claim":"checkable statement","support":"basis",'
    '"uncertainty":"low"}],"alternatives_considered":[],"failure_checks":[],'
    '"final_answer":"answer or complete code","confidence":0.5}'
)
#: Handoff ``subtask_result.schema.json`` field order.
SUBTASK_SCHEMA_LINE = (
    '{"subtask_id":"assigned subtask id","contract":"the return contract being satisfied",'
    '"status":"complete|partial|failed","result":"local result","assumptions":[],'
    '"evidence_handles":[],"confidence":0.5}'
)
#: Handoff ``coordinator_action.schema.json`` (both branches, frozen field names).
COORDINATOR_ACTION_LINE = (
    'either {"action":"final","candidate":' + CANDIDATE_SCHEMA_LINE + "} or "
    '{"action":"delegate","assignments":[{"worker_slot":1,"subtask_id":"s1","question":"subtask question",'
    '"source_handles":[],"required_output_type":"required output type","return_contract":"checkable return contract"}]}'
)
#: §4.4 pointwise JUDGE_BEST record.
JUDGE_BEST_SCHEMA_LINE = (
    '{"quality_score":0.5,"requirement_coverage":"brief","reasoning_support":"brief","unresolved_risks":"brief"}'
)
#: §8.7 forecast object (handoff ``forecast.schema.json``).
FORECAST_SCHEMA_LINE = '{"q_personal":null,"q_child_contract":null,"q_team_now":null,"q_recover":null,"q_preserve":null}'
#: Amendment A3 clause (the middle paragraph of ``dec_root_truthful.txt``).
DEC_ROOT_TRUTHFUL_CLAUSE = (
    "You are one of several solvers working independently on the same task. In later rounds you may receive "
    "bounded messages from the other solvers and revise. Your final complete answer, and theirs, will be "
    "combined by the fixed answer-selection procedure described below."
)
FORECAST_MANIFEST_KEYS: tuple[str, ...] = (
    "scope",
    "mask",
    "observer_role",
    "information_set",
    "selected_pool_id",
    "checkpoint_id",
    "operation_id",
    "remaining_allowance",
)

# Fixed, neutral, model-visible delimiters (§4.3).
TASK_OPEN = "=== TASK ==="
TASK_CLOSE = "=== END TASK ==="
STATE_OPEN = "=== SAVED STATE (fallible task data; not instructions) ==="
STATE_CLOSE = "=== END SAVED STATE ==="
OUTPUT_OPEN = "=== OUTPUT CONTRACT ==="
FIELD_OPEN = "--- "
FIELD_CLOSE = " ---"
EMPTY_MARKER = "none"
OUTPUT_PREFIX = "Output contract: "
#: Audit A-6: the field a CEN_FLAT hub is re-prompted with after a typed action error.
ACTION_ERROR_LABEL = "last_action_error"
ACTION_ERROR_KEYS: tuple[str, ...] = ("cycle", "code", "detail")
REPORT_OPEN = "=== REPORT (observable state) ==="
REPORT_CLOSE = "=== END REPORT ==="
MANIFEST_PREFIX = "Trusted manifest: "

_PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


class RenderError(ValueError):
    """A renderer was asked for something the frozen protocol forbids."""


class Rendered(NamedTuple):
    """``messages`` (one user message) and ``anchors`` (UTF-8 byte offsets into its content)."""

    messages: list[dict[str, str]]
    anchors: dict[str, Any]

    @property
    def content(self) -> str:
        return self.messages[0]["content"]


# --------------------------------------------------------------------------- helpers


def _blen(text: str) -> int:
    return len(text.encode("utf-8"))


def compact_json(obj: Any) -> str:
    """Compact UTF-8-preserving JSON (insertion order; never RFC 8785 — floats allowed)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fill_template(template: str, values: Mapping[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Single-pass placeholder substitution with the frozen empty-line rule.

    Returns the text and ``{key: (start, end)}`` byte spans of every substituted value.
    Every placeholder in the template must be in ``values`` and vice versa (each exactly
    once).  An empty value removes the placeholder line together with its newline; the
    placeholder must then be alone on its line.
    """
    found = _PLACEHOLDER.findall(template)
    if sorted(found) != sorted(values):
        raise RenderError(f"template placeholders {sorted(found)} != values {sorted(values)}")
    if len(set(found)) != len(found):
        raise RenderError("a placeholder occurs more than once")
    parts = _PLACEHOLDER.split(template)  # literal, key, literal, key, ..., literal
    out: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    strip_leading_newline = False
    for index, part in enumerate(parts):
        if index % 2 == 0:  # literal
            if strip_leading_newline:
                if not part.startswith("\n"):
                    raise RenderError("an empty clause placeholder must end its line")
                part = part[1:]
                strip_leading_newline = False
            out.append(part)
            offset += _blen(part)
            continue
        value = values[part]
        if not isinstance(value, str):
            raise RenderError(f"value for {part} must be str")
        if value == "":
            previous = parts[index - 1]
            at_line_start = (index == 1 and previous == "") or previous.endswith("\n")
            if not at_line_start:
                raise RenderError(f"empty clause {part} must start its line")
            spans[part] = (offset, offset)
            strip_leading_newline = True
            continue
        out.append(value)
        spans[part] = (offset, offset + _blen(value))
        offset += _blen(value)
    return "".join(out), spans


def _template_prefix(name: str, placeholder: str) -> str:
    """The frozen role paragraph(s) before ``{{placeholder}}`` (template must be ``prefix\\n\\n{{...}}\\n``)."""
    template = prompts.load_template(name)
    token = "\n\n{{" + placeholder + "}}\n"
    if template.count(token) != 1 or not template.endswith(token):
        raise RenderError(f"template {name} does not end with the state placeholder line")
    return template[: -len(token)]


def selector_description(domain: Domain | str) -> str:
    """Frozen task-specific selector clause (§5.5; amendment S1b for code)."""
    clauses = prompts.load_framing_clauses()
    domain = Domain(domain)
    return clauses["HLE_CHOICE_SELECTOR"] if domain is Domain.HLE else clauses["CODE_SELECTOR"]


def _check_task(task: PublicTask) -> None:
    if not isinstance(task, PublicTask):
        raise TypeError("task must be a PublicTask")
    if not task.task_text:
        raise RenderError(f"{task.source_id}: empty task text")


def saved_object_text(obj: Any) -> str:
    """Render a saved object as fallible task data (§3.5, §4.3).

    * ``Candidate``/``FinalAction``/``DelegateAction``/``Assignment`` → compact JSON;
    * ``CandidateRecord`` → its candidate JSON when valid, else the exact sentinel bytes;
    * the sentinel (as mapping or as its exact string) → ``SENTINEL_JSON``;
    * ``None`` → ``"none"`` (only where a renderer allows an absent object).
    Raw strings other than the sentinel are refused: malformed model text never re-enters
    a prompt.
    """
    if obj is None:
        return EMPTY_MARKER
    if isinstance(obj, str):
        if obj == SENTINEL_JSON:
            return SENTINEL_JSON
        raise RenderError("raw text is not a saved object; pass a Candidate, record or the sentinel")
    if isinstance(obj, CandidateRecord):
        if obj.valid:
            if obj.candidate is None:
                raise RenderError(f"valid candidate record {obj.candidate_id} has no candidate")
            return compact_json(obj.candidate.to_dict())
        return SENTINEL_JSON
    if isinstance(obj, (Candidate, FinalAction, DelegateAction, Assignment)):
        return compact_json(obj.to_dict())
    if isinstance(obj, Mapping):
        if dict(obj) == SENTINEL:
            return SENTINEL_JSON
        raise RenderError("mappings other than the sentinel are not saved objects; use the typed dataclasses")
    raise RenderError(f"unsupported saved object type {type(obj).__name__}")


def action_error_text(error: Mapping[str, Any]) -> str:
    """Render the typed error object a CEN_FLAT hub is re-prompted with (audit A-6; §4.2
    "invalid action objects ... produce typed errors").

    Exactly the frozen keys ``cycle`` (int ≥ 0), ``code`` (non-empty str) and ``detail``
    (str; the policy bounds its tokens under ``caps.hub_action_error_tokens``), rendered as
    compact JSON in that order.  The object is harness-authored, but it sits inside the
    saved state so the model-visible layout stays one ordered array.
    """
    if not isinstance(error, Mapping) or set(error) != set(ACTION_ERROR_KEYS):
        raise RenderError(f"action error must have exactly the keys {ACTION_ERROR_KEYS}")
    cycle, code, detail = error["cycle"], error["code"], error["detail"]
    if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 0:
        raise RenderError(f"action error cycle must be a non-negative int, got {cycle!r}")
    if not isinstance(code, str) or not code or not isinstance(detail, str):
        raise RenderError("action error code must be a non-empty str and detail a str")
    return compact_json({"cycle": cycle, "code": code, "detail": detail})


def packet_text(packet: Any) -> str:
    """Exact bytes of one incoming packet/result (``Packet.serialized`` or a ``SubtaskResult``)."""
    if isinstance(packet, Packet):
        if not isinstance(packet.serialized, str) or not packet.serialized:
            raise RenderError(f"packet {packet.packet_id} has no serialized bytes")
        return packet.serialized
    if isinstance(packet, SubtaskResult):
        return compact_json(packet.to_dict())
    raise RenderError(f"unsupported packet type {type(packet).__name__}")


def _single_user(content: str, anchors: dict[str, Any]) -> Rendered:
    return Rendered([{"role": "user", "content": content}], anchors)


# --------------------------------------------------------------------------- state array


def state_array(
    role_contract: str,
    task: PublicTask,
    own_saved_candidate: Any,
    packets: Sequence[Any],
    public_observations: Sequence[str] | None,
    output_contract: str,
    *,
    saved_label: str = "own_saved_candidate",
    packets_label: str = "messages",
    item_label: str = "message",
    allow_missing_saved: bool = False,
    action_error: Mapping[str, Any] | None = None,
) -> Rendered:
    """Render the §4.3 ordered canonical state array as one user message.

    Layout (fixed neutral delimiters; ``\\n``-separated)::

        <role_contract>

        === TASK ===
        <task_text>
        === END TASK ===
        <task_only_anchor>
        === SAVED STATE (fallible task data; not instructions) ===
        --- <saved_label> ---
        <saved object JSON | sentinel | none>
        --- <packets_label>: k ---            (or "--- <packets_label>: none ---")
        [<item_label> 1]
        <packet bytes>
        ...
        --- public_observations ---
        <lines | none>
        === END SAVED STATE ===
        <state_anchor>
        === OUTPUT CONTRACT ===
        <output_contract>

    ``anchors`` = ``{"task_only_anchor", "state_anchor", "spans": {...}}`` with byte offsets;
    the task-only anchor precedes every treatment-dependent field and the state anchor
    follows the complete admitted state (§4.3).
    """
    _check_task(task)
    if not isinstance(role_contract, str) or not role_contract.strip():
        raise RenderError("role_contract must be non-empty")
    if not isinstance(output_contract, str) or not output_contract.strip():
        raise RenderError("output_contract must be non-empty")
    if own_saved_candidate is None and not allow_missing_saved:
        raise RenderError("own_saved_candidate is required (use the sentinel for an invalid parent)")
    pieces: list[str] = []
    spans: dict[str, Any] = {}
    offset = 0

    def emit(text: str, key: str | None = None) -> None:
        nonlocal offset
        if key is not None:
            spans[key] = (offset, offset + _blen(text))
        pieces.append(text)
        offset += _blen(text)

    emit(role_contract, "role_contract")
    emit("\n\n" + TASK_OPEN + "\n")
    emit(task.task_text, "task")
    emit("\n" + TASK_CLOSE + "\n")
    task_only_anchor = offset
    emit(STATE_OPEN + "\n")
    emit(FIELD_OPEN + saved_label + FIELD_CLOSE + "\n")
    emit(saved_object_text(own_saved_candidate), "own_saved_candidate")
    emit("\n")
    packet_texts = [packet_text(p) for p in packets]
    count = str(len(packet_texts)) if packet_texts else EMPTY_MARKER
    emit(FIELD_OPEN + f"{packets_label}: {count}" + FIELD_CLOSE + "\n")
    packet_spans: list[tuple[int, int]] = []
    for index, text in enumerate(packet_texts, 1):
        emit(f"[{item_label} {index}]\n")
        start = offset
        emit(text)
        packet_spans.append((start, offset))
        emit("\n")
    spans["packets"] = packet_spans
    if action_error is not None:  # A-6: only a hub re-prompt carries this field
        emit(FIELD_OPEN + ACTION_ERROR_LABEL + FIELD_CLOSE + "\n")
        emit(action_error_text(action_error), ACTION_ERROR_LABEL)
        emit("\n")
    emit(FIELD_OPEN + "public_observations" + FIELD_CLOSE + "\n")
    observations = list(public_observations or [])
    if any(not isinstance(o, str) or not o for o in observations):
        raise RenderError("public observations must be non-empty strings")
    emit("\n".join(observations) if observations else EMPTY_MARKER, "public_observations")
    emit("\n" + STATE_CLOSE + "\n")
    state_anchor = offset
    emit(OUTPUT_OPEN + "\n")
    emit(output_contract, "output_contract")
    emit("\n")
    anchors = {
        "task_only_anchor": task_only_anchor,
        "state_anchor": state_anchor,
        "spans": spans,
        "labels": {"saved": saved_label, "packets": packets_label},
    }
    return _single_user("".join(pieces), anchors)


# --------------------------------------------------------------------------- roots


def render_root(task: PublicTask, framing: Framing | str, native_clause: str | None = None) -> Rendered:
    """§5.5 factorial root prompt from ``independent_root.txt`` (single user message).

    ``framing`` ``00/01/10/11`` fills ``{{TEAM_CLAUSE_OR_EMPTY}}`` with ``TEAM_1`` or nothing
    and ``{{VOTE_AWARE_..._OR_EMPTY}}`` with ``VOTE_1 + "\\n" + <selector>`` or nothing (an
    empty clause removes its line, see the module docstring).  ``Framing.NATIVE`` requires
    ``native_clause`` (a policy's truthful role wording that refers to the selector below);
    it takes the VOTE slot followed by the selector, and the TEAM slot is empty, which makes
    ``render_root(task, NATIVE, DEC_ROOT_TRUTHFUL_CLAUSE)`` byte-identical to
    :func:`render_dec_root`.  Anchors: ``task`` span; both anchors sit at the end of the task.
    """
    _check_task(task)
    framing = Framing(framing)
    clauses = prompts.load_framing_clauses()
    selector = selector_description(task.domain)
    if framing is Framing.NATIVE:
        if not isinstance(native_clause, str) or not native_clause.strip():
            raise RenderError("Framing.NATIVE requires a non-empty native_clause")
        team = ""
        vote = native_clause + "\n" + selector
    else:
        if native_clause is not None:
            raise RenderError("native_clause is only valid with Framing.NATIVE")
        team = clauses["TEAM_1"] if framing.team_frame else ""
        vote = clauses["VOTE_1"] + "\n" + selector if framing.vote_aware else ""
    text, spans = fill_template(
        prompts.load_template("independent_root"),
        {
            "TEAM_CLAUSE_OR_EMPTY": team,
            "VOTE_AWARE_CLAUSE_AND_FROZEN_SELECTOR_DESCRIPTION_OR_EMPTY": vote,
            "EXACT_PUBLIC_TASK": task.task_text,
            "EXACT_CANDIDATE_JSON_SCHEMA": CANDIDATE_SCHEMA_LINE,
        },
    )
    task_span = spans["EXACT_PUBLIC_TASK"]
    anchors = {
        "task_only_anchor": task_span[1],
        "state_anchor": task_span[1],
        "spans": {
            "task": task_span,
            "team_clause": spans["TEAM_CLAUSE_OR_EMPTY"],
            "vote_clause": spans["VOTE_AWARE_CLAUSE_AND_FROZEN_SELECTOR_DESCRIPTION_OR_EMPTY"],
            "output_contract": spans["EXACT_CANDIDATE_JSON_SCHEMA"],
        },
        "framing": framing.value,
    }
    return _single_user(text, anchors)


def render_dec_root(task: PublicTask, N: int) -> Rendered:
    """Main-tier DEC root from ``dec_root_truthful.txt`` (§4.1 truthful disclosure, P0-1).

    ``N`` is validated only (it must be ≥ 2: the clause says "several solvers", and a
    false team claim at N=1 is prohibited by §4.6); no numeric team size enters the prompt
    (§5.5), so every N ≥ 2 shares the same bytes and the N-panel uses ``F00`` instead.
    """
    _check_task(task)
    if not isinstance(N, int) or isinstance(N, bool) or N < 2:
        raise RenderError(f"render_dec_root needs N >= 2 (got {N!r}); N=1 uses the neutral 00 root")
    text, spans = fill_template(
        prompts.load_template("dec_root_truthful"),
        {
            "FROZEN_SELECTOR_DESCRIPTION": selector_description(task.domain),
            "EXACT_PUBLIC_TASK": task.task_text,
            "EXACT_CANDIDATE_JSON_SCHEMA": CANDIDATE_SCHEMA_LINE,
        },
    )
    task_span = spans["EXACT_PUBLIC_TASK"]
    anchors = {
        "task_only_anchor": task_span[1],
        "state_anchor": task_span[1],
        "spans": {
            "task": task_span,
            "selector": spans["FROZEN_SELECTOR_DESCRIPTION"],
            "output_contract": spans["EXACT_CANDIDATE_JSON_SCHEMA"],
        },
        "framing": Framing.NATIVE.value,
    }
    return _single_user(text, anchors)


# --------------------------------------------------------------------------- revisions


def revision_instruction() -> str:
    """The frozen ``focal_revision.txt`` paragraph (before its state placeholder)."""
    return _template_prefix("focal_revision", "CANONICAL_TASK_OWN_STATE_ALLOWED_PACKETS_AND_OUTPUT_CONTRACT")


def dec_revision_contract(task: PublicTask, N: int) -> str:
    """Truthful DEC revision disclosure: ``dec_revision_contract.txt`` (N ≥ 2) or the N=1
    variant that never mentions peers (P1-3, §4.6)."""
    if not isinstance(N, int) or isinstance(N, bool) or N < 1:
        raise RenderError(f"N must be a positive int, got {N!r}")
    selector = selector_description(task.domain)
    if N == 1:
        text, _ = fill_template(prompts.load_template("dec_revision_contract_n1"), {"FROZEN_SELECTOR_DESCRIPTION": selector})
    else:
        text, _ = fill_template(
            prompts.load_template("dec_revision_contract"),
            {"N": str(N), "N_MINUS_1": str(N - 1), "FROZEN_SELECTOR_DESCRIPTION": selector},
        )
    return text.rstrip("\n")


def render_dec_revision(task: PublicTask, own: Any, peer_packets: Sequence[Any], N: int, round: int) -> Rendered:
    """DEC round ``round ≥ 1`` revision for one member (§4.2 DEC row, §4.3).

    ``own`` is the member's immediately preceding candidate (or the sentinel);
    ``peer_packets`` must hold exactly ``N-1`` packets from the previous round (unavailable
    peers arrive as typed unavailable packets, never as omissions).  The round number is
    not model-visible (the templates carry no round text); it is validated and echoed in
    ``anchors``.
    """
    if not isinstance(round, int) or isinstance(round, bool) or round < 1:
        raise RenderError(f"round must be >= 1, got {round!r}")
    if len(peer_packets) != N - 1:
        raise RenderError(f"DEC N={N} needs exactly {N - 1} peer packets, got {len(peer_packets)}")
    role = dec_revision_contract(task, N) + "\n\n" + revision_instruction()
    rendered = state_array(role, task, own, peer_packets, None, OUTPUT_PREFIX + CANDIDATE_SCHEMA_LINE)
    rendered.anchors.update({"N": N, "round": round})
    return rendered


def render_s_history(task: PublicTask, previous_or_sentinel: Any) -> Rendered:
    """S_HISTORY successive call: task + the previous valid candidate or its sentinel (§4.2).

    Uses the ``focal_revision`` instruction with an empty message list; earlier reasoning
    and earlier candidates are not retained (portable state only).
    """
    return state_array(revision_instruction(), task, previous_or_sentinel, [], None, OUTPUT_PREFIX + CANDIDATE_SCHEMA_LINE)


def render_focal_revision(task: PublicTask, own_root: Any, peer_packets: Sequence[Any]) -> Rendered:
    """D-module / transport focal revision: own root + 0..d peer packets (§4.5, §4.6)."""
    return state_array(revision_instruction(), task, own_root, peer_packets, None, OUTPUT_PREFIX + CANDIDATE_SCHEMA_LINE)


def render_common_consumer(task: PublicTask, saved_state: Any) -> Rendered:
    """Fresh common-consumer attempt from task + the exact revised focal state only (§4.5)."""
    role = prompts.load_template("common_consumer").rstrip("\n")
    return state_array(role, task, saved_state, [], None, OUTPUT_PREFIX + CANDIDATE_SCHEMA_LINE)


# --------------------------------------------------------------------------- hub / worker


def render_hub(
    task: PublicTask,
    N_total: int,
    N_workers: int,
    prior_plan: DelegateAction | None,
    returned_results: Sequence[Any],
    cycle: int,
    final: bool = False,
    action_error: Mapping[str, Any] | None = None,
) -> Rendered:
    """CEN_FLAT coordinator call (§4.2 CEN_FLAT row; ``central_hub.txt``).

    ``cycle == 0`` has no prior plan and no results; ``cycle ≥ 1`` requires the previous
    valid ``DelegateAction`` and one returned object per assignment, in planned order.
    ``final=True`` appends ``cen_final_instruction.txt`` to the role contract (the reserved
    no-more-work call).  ``N_workers`` is the exact available roster capacity (``N_total-1``
    at most; 0 at N=1, P1-3).  ``action_error`` (audit A-6) is the typed error object of the
    hub's immediately preceding invalid action, rendered as the ``last_action_error`` field
    after the returned results (see :func:`action_error_text`); it is absent otherwise, so
    prompts without an error keep their frozen bytes.
    """
    for name, value in (("N_total", N_total), ("N_workers", N_workers), ("cycle", cycle)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RenderError(f"{name} must be a non-negative int, got {value!r}")
    if N_total < 1 or N_workers > N_total - 1:
        raise RenderError(f"N_workers={N_workers} must be <= N_total-1={N_total - 1}")
    if cycle == 0:
        if prior_plan is not None or len(returned_results) != 0:
            raise RenderError("cycle 0 has no prior plan and no returned results")
    else:
        if not isinstance(prior_plan, DelegateAction):
            raise RenderError("cycle >= 1 requires the previous valid DelegateAction")
        if len(returned_results) != len(prior_plan.assignments):
            raise RenderError(
                f"{len(prior_plan.assignments)} assignments but {len(returned_results)} returned results"
            )
    prefix = _template_prefix("central_hub", "CANONICAL_TASK_PLAN_RETURNS_ALLOWED_OBSERVATIONS_AND_ACTION_CONTRACT")
    role, _ = fill_template(prefix, {"N_TOTAL": str(N_total), "N_WORKERS": str(N_workers)})
    if final:
        role = role + "\n\n" + prompts.load_template("cen_final_instruction")
    rendered = state_array(
        role,
        task,
        prior_plan,
        returned_results,
        None,
        OUTPUT_PREFIX + COORDINATOR_ACTION_LINE,
        saved_label="prior_plan",
        packets_label="returned_results",
        item_label="returned_result",
        allow_missing_saved=True,
        action_error=action_error,
    )
    rendered.anchors.update(
        {"cycle": cycle, "final": bool(final), "N_total": N_total, "N_workers": N_workers, "action_error": action_error is not None}
    )
    return rendered


def render_worker(task: PublicTask, assignment: Assignment, forwarded: Sequence[Any]) -> Rendered:
    """CEN_FLAT worker call: task + current assignment + hub-forwarded results (§4.2)."""
    if not isinstance(assignment, Assignment):
        raise RenderError("assignment must be an Assignment")
    role = _template_prefix("central_worker", "TASK_CURRENT_ASSIGNMENT_EXPLICIT_SOURCE_VIEW_AND_SUBTASK_SCHEMA")
    rendered = state_array(
        role,
        task,
        assignment,
        forwarded,
        None,
        OUTPUT_PREFIX + SUBTASK_SCHEMA_LINE,
        saved_label="assignment",
        packets_label="forwarded_results",
        item_label="forwarded_result",
    )
    rendered.anchors.update({"worker_slot": assignment.worker_slot, "subtask_id": assignment.subtask_id})
    return rendered


# --------------------------------------------------------------------------- selectors / judges


def render_judge_best(task: PublicTask, candidate: Candidate) -> Rendered:
    """Pointwise JUDGE_BEST scorer prompt (§4.4): task + one anonymous full candidate."""
    if not isinstance(candidate, Candidate):
        raise RenderError("candidate must be a Candidate")
    role = _template_prefix("judge_best", "EXACT_PUBLIC_TASK_ANONYMOUS_FULL_CANDIDATE_AND_ALLOWED_PUBLIC_RESULTS")
    return state_array(
        role,
        task,
        candidate,
        [],
        None,
        OUTPUT_PREFIX + JUDGE_BEST_SCHEMA_LINE,
        saved_label="candidate",
        packets_label="allowed_public_results",
        item_label="public_result",
    )


def render_hle_judge(question: str, response: str, correct_answer: str) -> Rendered:
    """The official cais/hle ``JUDGE_PROMPT`` via ``str.format`` (§3.3; evaluator identity only).

    Callers live in ``study.evaluation``: ``correct_answer`` is protected data.
    """
    for name, value in (("question", question), ("response", response), ("correct_answer", correct_answer)):
        if not isinstance(value, str) or not value:
            raise RenderError(f"{name} must be a non-empty str")
    template = prompts.load_template("hle_judge")
    parts = re.split(r"(\{question\}|\{response\}|\{correct_answer\})", template)
    values = {"{question}": question, "{response}": response, "{correct_answer}": correct_answer}
    out: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    for part in parts:
        text = values.get(part, part)
        if part in values:
            spans[part[1:-1]] = (offset, offset + _blen(text))
        out.append(text)
        offset += _blen(text)
    content = "".join(out)
    if content != template.format(question=question, response=response, correct_answer=correct_answer):
        raise RenderError("judge prompt rendering diverged from str.format")
    return _single_user(content, {"spans": spans})


def render_forecast(report: str, manifest: Mapping[str, Any]) -> Rendered:
    """§8.7 shadow forecast: the frozen instruction + trusted manifest + compiled report bytes.

    ``manifest`` must carry the trusted-request keys of §8.7 (scope, mask, observer_role,
    information_set, selected_pool_id, checkpoint_id, operation_id, remaining_allowance);
    ``report`` is the compiled FINAL_HANDOFF_REPORT text (WP5 compiler), inserted verbatim.
    """
    if not isinstance(report, str) or not report.strip():
        raise RenderError("report must be a non-empty str")
    missing = [k for k in FORECAST_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise RenderError(f"forecast manifest lacks {missing}")
    manifest_json = json.dumps(dict(manifest), ensure_ascii=False, separators=(",", ":"), allow_nan=False, sort_keys=True)
    head = MANIFEST_PREFIX + manifest_json + "\n" + OUTPUT_PREFIX + FORECAST_SCHEMA_LINE
    body = REPORT_OPEN + "\n" + report + "\n" + REPORT_CLOSE
    text, spans = fill_template(
        prompts.load_template("forecast"),
        {
            "TRUSTED_SCOPE_MASK_INFORMATION_SET_AND_FORECAST_SCHEMA": head,
            "BOUNDED_FINAL_HANDOFF_OR_REGISTERED_OPERATION_STATE": body,
        },
    )
    body_span = spans["BOUNDED_FINAL_HANDOFF_OR_REGISTERED_OPERATION_STATE"]
    report_start = body_span[0] + _blen(REPORT_OPEN + "\n")
    anchors = {
        "spans": {
            "manifest": spans["TRUSTED_SCOPE_MASK_INFORMATION_SET_AND_FORECAST_SCHEMA"],
            "report": (report_start, report_start + _blen(report)),
        },
        "state_anchor": body_span[1],
    }
    return _single_user(text, anchors)


__all__ = [name for name in globals() if not name.startswith("_")]
