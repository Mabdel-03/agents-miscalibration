"""The E2b local-step checker: a harness-side tool loop (spec 9.3; plan §1.8, D17).

The checker is the treatment of hypothesis H4: it is the difference between "you were told
which draft the panel endorses" and "you can find out for yourself".  For that contrast to
mean anything, the tool has to be *exactly* as strong as the spec says and no stronger, so
every restriction below is enforced in code rather than left to the prompt:

* it reads **only** the named displayed packet's named row plus the problem's fixed
  coefficients -- :func:`mod97.local_step_valid` is the whole implementation;
* it never recomputes the chain, never evaluates the program, never holds the correct final
  answer, never names the injected error and never ranks the two packets.  A locally
  consistent *downstream* row of the wrong trace therefore returns ``valid``, which is the
  honest answer and the reason the tool cannot be turned into a binary search;
* the budget is spent by **requests**, not by information: the first ``max_checks`` requests
  consume slots including duplicates and malformed ones ("no adaptive exception grants extra
  checks"), and anything beyond is not served at all and is logged ``over_budget``;
* when the tool is disabled every response carries the *same schema* with status ``disabled``,
  so the two checker arms differ in what the response says and not in its shape (spec 9.4:
  "This is tool access, not a placebo that secretly provides correctness feedback").  A
  disabled arm is served by a ``CheckStepTool(enabled=False)``, never by the absence of one:
  it spends the same slots, logs the same ``over_budget`` events and writes the same
  ``tool_events`` rows, which is what makes the two prompts the same length and spec 9.6's
  checker-use report defined on both arms.

Why a harness loop and not vLLM tool calling (plan D17, amendment A2): tool-calling would put
the request inside the model's sampled tokens and therefore inside the request identity, and
the number of calls would stop being a controlled quantity.  Here the plan phase emits JSON,
this module serves it, and the decision phase reads a typed transcript.

Producer of the ``tool_events`` table (``tables._PRODUCERS``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import mod97
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.tables import TOOL_STATUSES, ToolEventRow
from agents_scaling.study import identity


class CheckStatus(str, Enum):
    """The four statuses a *served* request can answer with (spec 9.3)."""

    VALID = "valid"
    INVALID = "invalid"
    DISABLED = "disabled"
    BAD_REQUEST = "bad_request"


#: The fifth ``tool_events.status`` value: an *event* that was never served because the budget
#: was already spent.  It is deliberately not a :class:`CheckStatus`, because the receiver
#: receives no response for it at all.
OVER_BUDGET: Final[str] = "over_budget"

#: ``tool_events.access_condition``: which arm the episode was in when the request arrived.
ACCESS_ENABLED: Final[str] = "enabled"
ACCESS_DISABLED: Final[str] = "disabled"

#: The plan phase's JSON contract (spec 9.4 step 1), declared once and rendered by
#: ``templates/receiver_plan.txt``.
PLAN_FIELD: Final[str] = "check_requests"
PACKET_FIELD: Final[str] = "packet_id"
STEP_FIELD: Final[str] = "step_id"

if OVER_BUDGET not in TOOL_STATUSES or any(s.value not in TOOL_STATUSES for s in CheckStatus):
    raise RuntimeError("checker statuses must be exactly the tool_events vocabulary (R1)")


@dataclass(frozen=True)
class CheckRequest:
    """One requested check.  ``step_id`` may be anything the model emitted -- including a
    value that names no displayed operation, which is a ``bad_request`` that still costs a
    slot."""

    packet_id: str
    step_id: int


@dataclass(frozen=True)
class CheckResponse:
    """The typed response.  Identical shape in every arm (spec 9.4 step 2)."""

    packet_id: str
    step_id: int
    status: CheckStatus

    def as_dict(self) -> dict[str, Any]:
        return {PACKET_FIELD: self.packet_id, STEP_FIELD: int(self.step_id),
                "status": self.status.value}

    def as_line(self) -> str:
        """The single line the decision prompt shows for this response."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def parse_plan_output(text: str | None) -> tuple[CheckRequest, ...]:
    """The checks a plan output asked for; invalid plan output yields ``()`` (spec 9.4).

    "Invalid plan output yields an empty check set and continues to decision without a repair
    call" -- so this never raises and never asks the model again.  Entries that are not an
    object with a string ``packet_id`` and an integer ``step_id`` are dropped; entries that
    *are* well formed but name an unknown packet or step are kept, because deciding that is
    the tool's job and costs the receiver a slot (spec 9.3).
    """
    if not isinstance(text, str) or not text.strip():
        return ()
    try:
        obj = json.loads(text)
    except ValueError:
        return ()
    if not isinstance(obj, dict):
        return ()
    entries = obj.get(PLAN_FIELD)
    if not isinstance(entries, list):
        return ()
    out: list[CheckRequest] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        packet = entry.get(PACKET_FIELD)
        step = entry.get(STEP_FIELD)
        if isinstance(step, bool) or not isinstance(step, int) or not isinstance(packet, str):
            continue
        out.append(CheckRequest(packet_id=packet, step_id=int(step)))
    return tuple(out)


class CheckStepTool:
    """``check_step(packet_id, step_id)`` over the two displayed packets of one problem.

    ``packets`` maps the *displayed* packet id (the id the receiver sees in its prompt, never
    a correctness label) to the :class:`mod97.Trace` that packet displays.  The tool holds no
    final answer of its own: :attr:`problem` supplies coefficients and
    :func:`mod97.local_step_valid` compares one stated row against them.
    """

    def __init__(self, packets: Mapping[str, mod97.Trace], problem: mod97.Problem, *,
                 enabled: bool, max_checks: int) -> None:
        self.packets: Mapping[str, mod97.Trace] = dict(packets)
        self.problem = problem
        self.enabled = bool(enabled)
        self.max_checks = int(max_checks)
        if self.max_checks < 0:
            raise ContractViolation("max_checks must be a non-negative request budget")
        for packet_id, trace in self.packets.items():
            if not packet_id:
                raise ContractViolation("a displayed packet needs a non-empty packet id")
            if trace.problem_id != problem.problem_id:
                raise ContractViolation(
                    f"packet {packet_id} displays problem {trace.problem_id}, not "
                    f"{problem.problem_id}: the tool serves one problem")
        self._step_ids = frozenset(mod97.step_ids(problem))

    @property
    def access_condition(self) -> str:
        return ACCESS_ENABLED if self.enabled else ACCESS_DISABLED

    def check_step(self, packet_id: str, step_id: int) -> CheckResponse:
        """One check.  Reads the named packet's named row and nothing else.

        ``disabled`` short-circuits *before* the lookup, so a disabled tool cannot be probed
        for the existence of a packet or a step id
        (``test_disabled_checker_never_returns_validity``).
        """
        request = CheckRequest(packet_id=str(packet_id), step_id=step_id)  # type: ignore[arg-type]
        if not self.enabled:
            return CheckResponse(request.packet_id, _step_or_zero(step_id), CheckStatus.DISABLED)
        if isinstance(step_id, bool) or not isinstance(step_id, int):
            return CheckResponse(request.packet_id, _step_or_zero(step_id), CheckStatus.BAD_REQUEST)
        trace = self.packets.get(request.packet_id)
        if trace is None or int(step_id) not in self._step_ids:
            return CheckResponse(request.packet_id, int(step_id), CheckStatus.BAD_REQUEST)
        row = trace.row_of(int(step_id))
        if row is None:                                     # pragma: no cover - traces are complete
            return CheckResponse(request.packet_id, int(step_id), CheckStatus.BAD_REQUEST)
        valid = mod97.local_step_valid(self.problem, row)
        return CheckResponse(request.packet_id, int(step_id),
                             CheckStatus.VALID if valid else CheckStatus.INVALID)

    def serve(self, requests: Sequence[CheckRequest], *, decision_id: str,
              episode_id: str) -> tuple[tuple[CheckResponse, ...], tuple[ToolEventRow, ...]]:
        """Serve one plan phase's requests under the fixed budget policy (spec 9.3).

        Slots are consumed **in request order** by the first ``max_checks`` requests whatever
        they are -- a repeat of an earlier request is served again with the same typed answer
        and still costs a slot, and so does a request naming nothing.  Beyond the budget the
        request is not served: no response is produced and an ``over_budget`` event records
        that the receiver asked.
        """
        responses: list[CheckResponse] = []
        events: list[ToolEventRow] = []
        seen: set[tuple[str, Any]] = set()
        for index, request in enumerate(requests):
            key = (str(request.packet_id), request.step_id)
            duplicate = key in seen
            seen.add(key)
            request_json = json.dumps({PACKET_FIELD: request.packet_id,
                                       STEP_FIELD: request.step_id},
                                      sort_keys=True, separators=(",", ":"), default=str)
            if index >= self.max_checks:
                events.append(self._event(decision_id, episode_id, index, request, OVER_BUDGET,
                                          duplicate=duplicate, elapsed_s=0.0,
                                          request_json=request_json, response_json=None))
                continue
            started = time.perf_counter()
            response = self.check_step(request.packet_id, request.step_id)
            elapsed = time.perf_counter() - started
            responses.append(response)
            events.append(self._event(decision_id, episode_id, index, request,
                                      response.status.value, duplicate=duplicate,
                                      elapsed_s=elapsed, request_json=request_json,
                                      response_json=response.as_line()))
        return tuple(responses), tuple(events)

    def _event(self, decision_id: str, episode_id: str, index: int, request: CheckRequest,
               status: str, *, duplicate: bool, elapsed_s: float, request_json: str,
               response_json: str | None) -> ToolEventRow:
        step = request.step_id if isinstance(request.step_id, int) and not isinstance(request.step_id, bool) else None
        return ToolEventRow(
            tool_event_id=tool_event_id(decision_id, index),
            decision_id=decision_id,
            episode_id=episode_id,
            request_index=int(index),
            packet_id=str(request.packet_id) or None,
            step_id=step,
            access_condition=self.access_condition,
            status=status,
            elapsed_s=float(elapsed_s),
            duplicate=bool(duplicate),
            request_json=request_json,
            response_json=response_json if response_json is not None else json.dumps(None),
        )


def tool_event_id(decision_id: str, request_index: int) -> str:
    """``tool_events`` primary key: one id per (decision, request index) -- the unique pair."""
    return "te:" + identity.sha256_hex(identity.jcs([decision_id, int(request_index)]))[:32]


def _step_or_zero(step_id: Any) -> int:
    return int(step_id) if isinstance(step_id, int) and not isinstance(step_id, bool) else 0


__all__ = [
    "ACCESS_DISABLED", "ACCESS_ENABLED", "CheckRequest", "CheckResponse", "CheckStatus",
    "CheckStepTool", "OVER_BUDGET", "PACKET_FIELD", "PLAN_FIELD", "STEP_FIELD",
    "parse_plan_output", "tool_event_id",
]
