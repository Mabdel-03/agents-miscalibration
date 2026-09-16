"""Locked prompt templates and their renderers (spec 6.2, 6.3, 9.4; plan §1.6).

The templates are **files**, not strings in code: ``templates/root.txt`` and
``templates/revision.txt`` hold the spec 6.2 / 6.3 blocks byte for byte
(``test_root_and_revision_templates_match_spec_6_2_6_3`` re-extracts them from the spec
markdown and compares bytes), and ``locks.lock_prompts`` hashes the whole directory into
``prompts.lock.json`` so a template edit is a frozen-manifest change, not a silent one.

Three properties the templates must keep, each pinned by a test:

* **No condition is nameable.**  No template mentions a protocol label (PRIVATE, PEER,
  IN0_OUT1, ...) or describes the communication graph; the only thing that varies between
  arms is the *content* of ``OTHER DRAFTS`` (rule R5,
  ``test_no_template_contains_a_protocol_label``).  That is what makes the aliasing byte
  tests possible: the IN0_OUT1 focal prompt is byte-identical to the PRIVATE prompt because
  both render ``NONE`` into the same bytes.
* **No horizon.**  Nothing says how many rounds remain, so a round-4 prompt of an R=8
  episode is byte-identical to the round-4 prompt of an R=4 episode
  (``test_r4_prefix_of_r8_is_identical_when_prompts_are_horizon_free``).
* **No slot, role, persona or social count.**  Spec 6.2: "No role diversity, expertise
  persona, social-count hint, or agent-slot identifier appears in the root prompt."  Peers
  are named only by the per-episode pseudonym inside the packet bytes.

Item and family are *structural* here (WP3 owns the concrete classes): a task item supplies
``rendered_question`` -- the spec's ``{question_with_options_if_any}`` -- and a family
supplies ``answer_instruction`` -- the spec's ``{task_specific_answer_instruction}``.
Nothing else about an item reaches a prompt, which is also why gold cannot (R7).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final, Mapping, Protocol, Sequence, runtime_checkable

from agents_scaling.nextdyn.errors import ContractViolation, ManifestError
from agents_scaling.study import identity

TEMPLATES_DIR: Final[Path] = Path(__file__).resolve().parent / "templates"

ROOT: Final[str] = "root"
REVISION: Final[str] = "revision"
RECEIVER_PLAN: Final[str] = "receiver_plan"
RECEIVER_DECISION: Final[str] = "receiver_decision"
AUDIT: Final[str] = "audit"
TEMPLATE_NAMES: Final[tuple[str, ...]] = (ROOT, REVISION, RECEIVER_PLAN, RECEIVER_DECISION, AUDIT)

FILLER_WORDS_FILE: Final[str] = "filler_words.txt"

#: The literal the spec requires when a recipient has no visible peer packets (spec 6.3).
NONE_LITERAL: Final[str] = "NONE"

SYSTEM_MARKER: Final[str] = "SYSTEM"
USER_MARKER: Final[str] = "USER"

ROLE_SYSTEM: Final[str] = "system"
ROLE_USER: Final[str] = "user"

#: Placeholder names, spelled exactly as the spec spells them.
Q: Final[str] = "question_with_options_if_any"
ANSWER_INSTRUCTION: Final[str] = "task_specific_answer_instruction"
OWN_PACKET: Final[str] = "own_previous_public_packet"
PEER_PACKETS: Final[str] = "visible_peer_packets_or_literal_NONE"
SUPPORT_FIELDS: Final[str] = "support_fields_or_literal_NONE"
CHECK_TOOL_STATUS: Final[str] = "check_tool_status"
CHECK_RESULTS: Final[str] = "check_results_or_literal_NONE"
CANDIDATE_PACKET: Final[str] = "candidate_packet"

Messages = tuple[dict[str, str], ...]


@runtime_checkable
class TaskItem(Protocol):
    """What a prompt needs from an item: the rendered question and nothing else (R7)."""

    @property
    def rendered_question(self) -> str: ...


@runtime_checkable
class AnswerFormat(Protocol):
    """What a prompt needs from a family: the task-specific answer instruction (spec 6.2)."""

    @property
    def answer_instruction(self) -> str: ...


# --------------------------------------------------------------------------- template files


def template_path(name: str) -> Path:
    if name not in TEMPLATE_NAMES:
        raise ContractViolation(f"unknown template {name!r}; declared: {TEMPLATE_NAMES}")
    return TEMPLATES_DIR / f"{name}.txt"


def template_text(name: str) -> str:
    """The raw template bytes, decoded as UTF-8; no normalisation, no stripping."""
    path = template_path(name)
    if not path.is_file():
        raise ManifestError(f"template {name!r} is missing at {path}")
    return path.read_bytes().decode("utf-8")


def template_hashes() -> dict[str, str]:
    """``relative path -> sha256``, exactly what ``locks.lock_prompts`` writes."""
    return {str(p.relative_to(TEMPLATES_DIR)): identity.sha256_hex(p.read_bytes())
            for p in sorted(TEMPLATES_DIR.rglob("*")) if p.is_file()}


def assert_prompts_locked(lock: Any) -> None:
    """Refuse to render against a drifted or skeleton ``prompts.lock.json`` (plan D5)."""
    locked = dict(getattr(lock, "templates", {}) or {})
    if not locked:
        raise ManifestError("prompts.lock.json holds no template hashes; run lock-inputs")
    current = template_hashes()
    if current != locked:
        drift = sorted(set(current) ^ set(locked)) or \
            sorted(k for k in current if current[k] != locked.get(k))
        raise ManifestError(f"templates drifted from prompts.lock.json: {drift}")


def filler_words() -> tuple[str, ...]:
    """The frozen inert vocabulary (comment lines excluded).

    Asserted here rather than at use: no digits and no bare option letter, so a
    length-matched filler packet can carry no answer and no task content.
    """
    path = TEMPLATES_DIR / FILLER_WORDS_FILE
    if not path.is_file():
        raise ManifestError(f"{FILLER_WORDS_FILE} is missing at {path}")
    words = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#"))
    if not words:
        raise ManifestError(f"{FILLER_WORDS_FILE} holds no words")
    for word in words:
        if any(ch.isdigit() for ch in word) or len(word) == 1:
            raise ManifestError(f"filler word {word!r} carries a digit or is a bare letter")
    return words


# --------------------------------------------------------------------------- rendering


def _split_roles(text: str) -> tuple[str, str]:
    """``SYSTEM``/``USER`` marker lines -> the two chat message bodies.

    The markers are the spec's own rendering of the two roles; they never reach the model.
    """
    lines = text.split("\n")
    if not lines or lines[0] != SYSTEM_MARKER:
        raise ManifestError(f"a template must begin with a {SYSTEM_MARKER!r} line")
    try:
        user_at = lines.index(USER_MARKER)
    except ValueError as exc:
        raise ManifestError(f"a template must carry a {USER_MARKER!r} line") from exc
    system = "\n".join(lines[1:user_at]).strip("\n")
    user = "\n".join(lines[user_at + 1:]).strip("\n")
    if not system or not user:
        raise ManifestError("a template must carry a non-empty system and user body")
    return system, user


_PLACEHOLDER = re.compile(r"\{([A-Za-z_]+)\}")


def _substitute(body: str, values: Mapping[str, str]) -> str:
    """Replace exactly the template's declared ``{placeholder}`` tokens, once each.

    The token set is read from the *template*, never from the substituted result: a packet
    body is JSON and carries braces of its own, and a substituted value must never be
    rescanned as if it were part of the template.
    """
    declared = set(_PLACEHOLDER.findall(body))
    if declared != set(values):
        raise ManifestError(
            f"template placeholders {sorted(declared)} != supplied {sorted(values)}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], body)


def render_template(name: str, values: Mapping[str, str]) -> Messages:
    """One template plus its substitutions -> the frozen two-message chat request."""
    system, user = _split_roles(template_text(name))
    return (
        {"role": ROLE_SYSTEM, "content": _substitute(system, {})},
        {"role": ROLE_USER, "content": _substitute(user, values)},
    )


def _packet_block(serialized: Sequence[str]) -> str:
    """Peer packets, one serialized packet per line, or the literal ``NONE`` (spec 6.3)."""
    lines = [s for s in serialized]
    return "\n".join(lines) if lines else NONE_LITERAL


def render_root(item: TaskItem, family: AnswerFormat) -> Messages:
    """Spec 6.2 root prompt: question and answer format, nothing else."""
    return render_template(ROOT, {
        Q: item.rendered_question,
        ANSWER_INSTRUCTION: family.answer_instruction,
    })


def render_revision(item: TaskItem, own: str | None, peers: Sequence[str]) -> Messages:
    """Spec 6.3 revision prompt.

    ``own`` is the agent's own previous transmitted packet bytes (``None`` renders ``NONE``
    -- the FRESH arm uses the root template instead and never reaches here), ``peers`` the
    visible peer packets already in their blind display order.  PRIVATE and the isolated
    focal slot of IN0_OUT1 both render the same literal ``NONE`` here, which is why their
    prompts are byte-identical.
    """
    return render_template(REVISION, {
        Q: item.rendered_question,
        OWN_PACKET: own if own is not None else NONE_LITERAL,
        PEER_PACKETS: _packet_block(peers),
    })


def render_receiver_plan(item: TaskItem, draft: str, packets: Sequence[str],
                         support: Sequence[str], *, checker_available: str) -> Messages:
    """Spec 9.4 phase 1: plan which local steps to check before answering."""
    return render_template(RECEIVER_PLAN, {
        Q: item.rendered_question,
        OWN_PACKET: draft,
        PEER_PACKETS: _packet_block(packets),
        SUPPORT_FIELDS: _packet_block(support),
        CHECK_TOOL_STATUS: checker_available,
    })


def render_receiver_decision(item: TaskItem, draft: str, packets: Sequence[str],
                             support: Sequence[str], *, check_results: Sequence[str]) -> Messages:
    """Spec 9.4 phase 3 (and the E2a one-call assay, which passes no check results)."""
    return render_template(RECEIVER_DECISION, {
        Q: item.rendered_question,
        OWN_PACKET: draft,
        PEER_PACKETS: _packet_block(packets),
        SUPPORT_FIELDS: _packet_block(support),
        CHECK_RESULTS: _packet_block(check_results),
    })


def render_audit(item: TaskItem, candidate: str) -> Messages:
    """Spec 6.5 pointwise audit: one candidate, one ``p_correct``, no new synthesis."""
    return render_template(AUDIT, {Q: item.rendered_question, CANDIDATE_PACKET: candidate})


__all__ = [
    "ANSWER_INSTRUCTION", "AUDIT", "AnswerFormat", "CANDIDATE_PACKET", "CHECK_RESULTS",
    "CHECK_TOOL_STATUS", "FILLER_WORDS_FILE", "Messages", "NONE_LITERAL", "OWN_PACKET",
    "PEER_PACKETS", "Q", "RECEIVER_DECISION", "RECEIVER_PLAN", "REVISION", "ROLE_SYSTEM",
    "ROLE_USER", "ROOT", "SUPPORT_FIELDS", "TEMPLATES_DIR", "TEMPLATE_NAMES", "TaskItem",
    "assert_prompts_locked", "filler_words", "render_audit", "render_receiver_decision",
    "render_receiver_plan", "render_revision", "render_root", "render_template",
    "template_hashes", "template_path", "template_text",
]
