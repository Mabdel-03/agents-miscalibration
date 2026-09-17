"""One explicit ``<final>`` element per response, extracted deterministically (MASTER_SPEC §8, §11, §15.10/14).

Rules, applied in this order:

1. ``candidate_id == node_id == spec.node_id``; ``result.node_id`` and ``task.item_id``/``task.benchmark`` must agree
   with ``spec`` (a mismatch raises: it is a harness error, never a model outcome).
2. ``transport_status`` other than ``OK`` -> ``EMPTY``: no accepted completion exists, even when a body carried text
   (a cache HIT or provider mismatch is never a fresh valid draw). ``visible_content`` is retained for audit and the
   transport status stays on ``GenerationResult``, reported separately by ``metrics``.
3. ``CompletionStatus.BLOCKED`` -> ``REFUSAL`` (a filtered response is not a parse failure).
4. ``visible_content`` ``None`` or whitespace-only -> ``EMPTY``.
5. Exactly one ``<final>`` followed by exactly one ``</final>`` -> ``VALID``, *whatever* ``finish_reason`` says:
   validity is a property of the closed final element, and truncation is reported from ``finish_reason``.
6. Neither ``<final>`` nor ``</final>`` present -> ``MISSING``.
7. Anything else -- two or more opening tags, a stray, reversed or missing closer -- -> ``MALFORMED``. A closed element
   whose inner text is blank (after the fence strip for Python) is also ``MALFORMED``: it holds no answer.
8. ``OutputKind.PYTHON`` only (lead decision W06-2): a final whose text, after stripping at most one outer fence, is
   not a Python program -- ``ast.parse`` raises (``SyntaxError``, a null byte, nesting too deep) -- is ``MALFORMED``:
   no program, not a wrong program. Parsing never executes anything and reads no gold; warnings are suppressed.

Tags are matched byte-exactly (no case folding). There is no text heuristic, no "Answer:" scraping, no LLM repair,
and nothing is read from hidden reasoning (``GenerationResult`` does not carry it). ``final_answer`` is the inner text
with outer whitespace stripped; for ``OutputKind.PYTHON`` at most one outer Markdown fence whose info string is one
of ``ALLOWED_FENCE_INFO`` is removed as well. MATH/OPTION/BOOLEAN finals are kept as written: normalization belongs
to class construction in ``agent_allocation.selection.keys``. ``rationale`` is the inner text of exactly one
``<rationale>`` pair, else ``None``; it never influences validity.

Config keys consumed: none.
"""
from __future__ import annotations

import ast
import warnings

from agent_allocation._vendor.extraction_tags import TAG_EMPTY, TAG_MISSING, TAG_OK, strict_single_pair, strip_one_fence
from agent_allocation.contracts import (
    Candidate, CompletionStatus, GenerationResult, GenerationSpec, OutputKind, ParseStatus, PublicTask, TransportStatus,
)

__all__ = [
    "ALLOWED_FENCE_INFO", "EXTRACTOR_VERSION", "FINAL_CLOSE", "FINAL_OPEN", "empty_candidate", "extract_candidate",
    "is_python_program", "terminal_without_response",
]

EXTRACTOR_VERSION = "06-final-v2"
FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"
RATIONALE_OPEN = "<rationale>"
RATIONALE_CLOSE = "</rationale>"
#: Info strings of the one outer fence that may be stripped from a Python final ("" is a bare fence).
ALLOWED_FENCE_INFO: frozenset[str] = frozenset({"", "python", "py", "python3"})


def extract_candidate(result: GenerationResult, spec: GenerationSpec, task: PublicTask) -> Candidate:
    if not isinstance(result, GenerationResult):
        raise TypeError(f"extract_candidate expects GenerationResult, got {type(result).__name__}")
    if not isinstance(spec, GenerationSpec):
        raise TypeError(f"extract_candidate expects GenerationSpec, got {type(spec).__name__}")
    if not isinstance(task, PublicTask):
        raise TypeError(f"extract_candidate expects PublicTask, got {type(task).__name__}")
    node_id = spec.node_id
    if result.node_id != node_id:
        raise ValueError(f"result.node_id {result.node_id!r} does not match spec.node_id {node_id!r}")
    if task.item_id != spec.item_id or task.benchmark != spec.benchmark:
        raise ValueError(f"task {task.benchmark}/{task.item_id} does not match spec {spec.benchmark}/{spec.item_id}")

    content = result.visible_content
    visible = content if isinstance(content, str) else ""

    def build(status: ParseStatus, final_answer: str | None = None, rationale: str | None = None) -> Candidate:
        return Candidate(candidate_id=node_id, node_id=node_id, item_id=spec.item_id, visible_content=visible,
                         final_answer=final_answer, rationale=rationale, parse_status=status,
                         finish_reason=result.finish_reason, output_kind=task.output_kind)

    if result.transport_status is not TransportStatus.OK:
        return build(ParseStatus.EMPTY)
    if result.completion_status is CompletionStatus.BLOCKED:
        return build(ParseStatus.REFUSAL)
    if not visible.strip():
        return build(ParseStatus.EMPTY)

    rationale_status, rationale = strict_single_pair(visible, RATIONALE_OPEN, RATIONALE_CLOSE)
    rationale = rationale if rationale_status == TAG_OK else None

    status, inner = strict_single_pair(visible, FINAL_OPEN, FINAL_CLOSE)
    if status == TAG_MISSING:
        return build(ParseStatus.MISSING, rationale=rationale)
    if status == TAG_EMPTY or status != TAG_OK or inner is None:
        return build(ParseStatus.MALFORMED, rationale=rationale)
    final_answer = inner
    if task.output_kind is OutputKind.PYTHON:
        final_answer = strip_one_fence(inner, ALLOWED_FENCE_INFO)
    if not final_answer.strip():
        return build(ParseStatus.MALFORMED, rationale=rationale)
    if task.output_kind is OutputKind.PYTHON and not is_python_program(final_answer):
        return build(ParseStatus.MALFORMED, rationale=rationale)
    return build(ParseStatus.VALID, final_answer=final_answer, rationale=rationale)


def is_python_program(source: str) -> bool:
    """Whether ``source`` parses as a Python module (``ast.parse``; nothing is compiled or executed)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return True


def terminal_without_response(spec: GenerationSpec, task: PublicTask) -> Candidate:
    """The candidate of a node made terminal without any API response (contract v2 R03-1 context-limit policy).

    It holds no visible content, so it is ``EMPTY``: it casts no vote, is graded ``INVALID_OUTPUT`` and renders as
    absent to children, and a checkpoint containing it can still be selected and sealed instead of waiting forever.
    """
    if not isinstance(spec, GenerationSpec) or not isinstance(task, PublicTask):
        raise TypeError("terminal_without_response expects (GenerationSpec, PublicTask)")
    if task.item_id != spec.item_id or task.benchmark != spec.benchmark:
        raise ValueError(f"task {task.benchmark}/{task.item_id} does not match spec {spec.benchmark}/{spec.item_id}")
    return empty_candidate(spec.node_id, task)


def empty_candidate(node_id: str, task: PublicTask) -> Candidate:
    """``terminal_without_response`` for readers that hold only the node id (e.g. metrics' state-backup reader)."""
    if not isinstance(node_id, str) or not node_id or not isinstance(task, PublicTask):
        raise TypeError("empty_candidate expects (non-empty node_id, PublicTask)")
    return Candidate(candidate_id=node_id, node_id=node_id, item_id=task.item_id, visible_content="", final_answer=None,
                     rationale=None, parse_status=ParseStatus.EMPTY, finish_reason=None, output_kind=task.output_kind)
