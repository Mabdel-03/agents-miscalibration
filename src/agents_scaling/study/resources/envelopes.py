"""Table E: frozen per-role prompt envelopes and minimal full-cap paths (WP4).

Spec §4.3 (32,768 rendered input tokens; 4,096 task envelope; ≤8,192 recipient tokens for
a full own candidate; ≤2,048 per peer packet; "Wrappers, schema and role instructions must
also fit"), §6.5 ("Resolve [a minimal feasible path's] cost from maximum permitted rendered
lengths and full decode reservations"; B0 = the largest such reservation over the
mandatory N=5 methods), §10.2 (worst-case reservation envelopes are published).  Audit
amendment B2 (Table E: subtask_result ≤ 4,096 recipient tokens, hub prior plan ≤ 8,192,
forwarded results ≤ 8,192 total) and critic P1-1 (``L_root_max`` = task envelope + the
*measured* wrapper of the longest root cell; hub/worker envelopes frozen so B0 is
computable and no CEN_FLAT prompt can exceed 32,768).

Wrapper tokens are ``rendered prompt tokens − task tokens`` (chat-template control tokens
included), measured with the pinned tokenizer on any task: the templates are fixed bytes,
so the wrapper is task-independent up to tokenizer boundary effects, which the +task_cap
bound absorbs.  Every envelope is *checked* against ``PROMPT_TOKENS_CAP``: a summed envelope
that does not fit raises :class:`EnvelopeError` instead of being clipped (review P0-A), so
``profile.py`` refuses to freeze an infeasible Table E.  Amendment B2b bounds the hub's
returned-results block (``caps.hub_returned_results_tokens``) and reserves an allowance for
the ``last_action_error`` re-prompt object (``caps.hub_action_error_tokens``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agents_scaling.study.config import Caps
from agents_scaling.study.inference.tokens import count_tokens, render_chat_token_ids
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.oracle import FlopOracle
from agents_scaling.study.types import (
    PACKET_UNAVAILABLE_JSON,
    PROMPT_TOKENS_CAP,
    SENTINEL_JSON,
    SOLVER_OUT_CAP,
    Assignment,
    Domain,
    Method,
    Packet,
    PublicTask,
)

#: Team sizes the envelopes are tabulated for (§6.3 membership grid).
ENVELOPE_NS: tuple[int, ...] = (1, 2, 3, 5, 9)
#: Critic P1-1 fallback when no measurement is available (architecture §4 used 6,144 = 4,096 + 2,048).
FALLBACK_WRAPPER_TOKENS = 2048
WRAPPER_FIXED_KEYS: tuple[str, ...] = ("root", "s_history", "focal_revision", "consumer", "worker")


class EnvelopeError(ValueError):
    """A summed Table E envelope exceeds the prompt cap: the caps are infeasible (fail closed)."""


def synthetic_task(domain: Domain = Domain.HLE, text: str = "Synthetic profiling task.") -> PublicTask:
    """A minimal public task for wrapper measurement (never a study item)."""
    return PublicTask(
        source_id=f"synthetic:{domain.value}",
        domain=domain,
        split="dev",
        task_text=text,
        answer_format="exactMatch" if domain is Domain.HLE else "code",
        stratum="synthetic",
        rank=0,
        task_tokens=0,
    )


def unavailable_packet(sender_slot: int) -> Packet:
    """The typed §4.3 unavailable packet (exact bytes) used as a wrapper-measurement filler."""
    return Packet(
        packet_id=f"unavailable-{sender_slot}",
        sender_slot=sender_slot,
        candidate_sha256="",
        fields={},
        spans=(),
        truncated={},
        final_partial=False,
        recipient_tokens=0,
        serialized=PACKET_UNAVAILABLE_JSON,
        unavailable=True,
    )


def _prompt_tokens(tokenizer: Any, rendered: R.Rendered) -> int:
    return len(render_chat_token_ids(tokenizer, rendered.messages, True))


def measure_wrappers(tokenizer: Any, task: PublicTask | None = None, Ns: Sequence[int] = ENVELOPE_NS) -> dict[str, int]:
    """Wrapper tokens per role (see the module docstring), keyed ``root``, ``s_history``,
    ``focal_revision``, ``consumer``, ``worker``, ``dec_revision@N<n>``, ``hub@N<n>``,
    ``hub_final@N<n>`` for every ``n`` in ``Ns``.

    ``root`` is the maximum over the four framing cells and the truthful DEC root.
    """
    task = task if task is not None else synthetic_task()
    base = count_tokens(task.task_text, tokenizer)
    out: dict[str, int] = {}
    roots = [_prompt_tokens(tokenizer, R.render_root(task, f)) for f in ("00", "01", "10", "11")]
    roots.append(_prompt_tokens(tokenizer, R.render_dec_root(task, 5)))
    out["root"] = max(roots) - base
    out["s_history"] = _prompt_tokens(tokenizer, R.render_s_history(task, SENTINEL_JSON)) - base
    out["focal_revision"] = _prompt_tokens(tokenizer, R.render_focal_revision(task, SENTINEL_JSON, [])) - base
    out["consumer"] = _prompt_tokens(tokenizer, R.render_common_consumer(task, SENTINEL_JSON)) - base
    assignment = Assignment(1, "", "", (), "", "")
    out["worker"] = _prompt_tokens(tokenizer, R.render_worker(task, assignment, [])) - base
    for n in Ns:
        if not isinstance(n, int) or n < 1:
            raise ValueError("Ns must be positive ints")
        fillers = [unavailable_packet(s) for s in range(1, n)]
        out[f"dec_revision@N{n}"] = _prompt_tokens(tokenizer, R.render_dec_revision(task, SENTINEL_JSON, fillers, n, 1)) - base
        out[f"hub@N{n}"] = _prompt_tokens(tokenizer, R.render_hub(task, n, n - 1, None, [], 0)) - base
        out[f"hub_final@N{n}"] = _prompt_tokens(tokenizer, R.render_hub(task, n, n - 1, None, [], 0, final=True)) - base
    for key, value in out.items():
        if value <= 0:
            raise ValueError(f"wrapper measurement for {key!r} is not positive ({value})")
    return out


def fallback_wrappers(Ns: Sequence[int] = ENVELOPE_NS, tokens: int = FALLBACK_WRAPPER_TOKENS) -> dict[str, int]:
    """Every wrapper at the P1-1 fallback (recorded as ``wrapper_source="fallback"``)."""
    out = {key: tokens for key in WRAPPER_FIXED_KEYS}
    for n in Ns:
        out[f"dec_revision@N{n}"] = tokens
        out[f"hub@N{n}"] = tokens
        out[f"hub_final@N{n}"] = tokens
    return out


@dataclass(frozen=True)
class TableE:
    """Frozen envelopes for one study (``caps`` block) and one wrapper table.

    ``task_tokens`` defaults to the task envelope (profile-time bound); a policy may pass
    the item's actual task token count for its runtime reservations (still "maximum
    permitted rendered length" for that episode).
    """

    caps: Caps
    wrappers: Mapping[str, int]
    task_tokens: int | None = None
    prompt_cap: int = PROMPT_TOKENS_CAP

    def __post_init__(self) -> None:
        missing = [k for k in WRAPPER_FIXED_KEYS if k not in self.wrappers]
        if missing:
            raise ValueError(f"wrapper table lacks {missing}")

    @property
    def L_task(self) -> int:
        return int(self.caps.task_tokens if self.task_tokens is None else self.task_tokens)

    def wrapper(self, key: str) -> int:
        try:
            return int(self.wrappers[key])
        except KeyError as exc:
            raise KeyError(f"no wrapper measurement for {key!r}") from exc

    def _bounded(self, n: int, what: str) -> int:
        """The envelope itself, or :class:`EnvelopeError` when it exceeds the prompt cap (never clipped)."""
        n = int(n)
        if n > int(self.prompt_cap):
            raise EnvelopeError(f"{what} envelope is {n} tokens > prompt cap {self.prompt_cap}: infeasible caps (Table E / B2b)")
        return n

    @property
    def hub_returned_results_tokens(self) -> int:
        return int(self.caps.hub_returned_results_tokens)

    @property
    def hub_action_error_tokens(self) -> int:
        return int(self.caps.hub_action_error_tokens)

    def hub_returned_max(self, N: int) -> int:
        """Largest returned-results block a hub prompt can carry at team size ``N`` (B2b):
        ``min((N−1)·subtask_result_tokens, hub_returned_results_tokens)``."""
        if not isinstance(N, int) or isinstance(N, bool) or N < 1:
            raise ValueError(f"N must be a positive int, got {N!r}")
        return min((N - 1) * int(self.caps.subtask_result_tokens), self.hub_returned_results_tokens)

    # ---- per-role maxima -----------------------------------------------------------------
    def root_prompt_max(self) -> int:
        return self._bounded(self.L_task + self.wrapper("root"), "root")

    def s_history_prompt_max(self) -> int:
        return self._bounded(self.L_task + self.wrapper("s_history") + self.caps.own_candidate_tokens, "s_history")

    def dec_revision_prompt_max(self, N: int) -> int:
        return self._bounded(
            self.L_task + self.wrapper(f"dec_revision@N{N}") + self.caps.own_candidate_tokens + (N - 1) * self.caps.packet_tokens,
            f"dec_revision@N{N}",
        )

    def hub_prompt_max(self, N: int, *, cycle0: bool) -> int:
        """Cycle-0 hub prompt (task + wrapper) or the maximum later hub prompt: prior plan +
        the bounded returned-results block + the ``last_action_error`` allowance.  The later
        value dominates every hub prompt of an episode (a re-prompt after a cycle-0 error
        carries no plan and no results; the reserved final call carries the same state)."""
        if cycle0:
            return self._bounded(self.L_task + self.wrapper(f"hub@N{N}"), f"hub@N{N} (cycle 0)")
        return self._bounded(
            self.L_task
            + self.wrapper(f"hub_final@N{N}")
            + self.caps.hub_prior_plan_tokens
            + self.hub_returned_max(N)
            + self.hub_action_error_tokens,
            f"hub_final@N{N}",
        )

    def worker_prompt_max(self, *, cycle0: bool) -> int:
        forwarded = 0 if cycle0 else self.caps.forwarded_results_tokens
        return self._bounded(self.L_task + self.wrapper("worker") + self.caps.hub_prior_plan_tokens + forwarded, "worker")

    def focal_prompt_max(self, d: int) -> int:
        return self._bounded(
            self.L_task + self.wrapper("focal_revision") + self.caps.own_candidate_tokens + d * self.caps.packet_tokens, f"focal_revision@d{d}"
        )

    def consumer_prompt_max(self) -> int:
        return self._bounded(self.L_task + self.wrapper("consumer") + self.caps.own_candidate_tokens, "consumer")

    def to_dict(self, Ns: Sequence[int] = ENVELOPE_NS) -> dict[str, Any]:
        return {
            "prompt_cap": self.prompt_cap,
            "task_tokens": self.L_task,
            "own_candidate_tokens": self.caps.own_candidate_tokens,
            "packet_tokens": self.caps.packet_tokens,
            "packet_final_tokens": self.caps.packet_final_tokens,
            "subtask_result_tokens": self.caps.subtask_result_tokens,
            "hub_prior_plan_tokens": self.caps.hub_prior_plan_tokens,
            "forwarded_results_tokens": self.caps.forwarded_results_tokens,
            "hub_returned_results_tokens": self.hub_returned_results_tokens,
            "hub_action_error_tokens": self.hub_action_error_tokens,
            "hub_returned_max": {str(n): self.hub_returned_max(n) for n in Ns},
            "wrappers": dict(self.wrappers),
            "root_prompt_max": self.root_prompt_max(),
            "s_history_prompt_max": self.s_history_prompt_max(),
            "consumer_prompt_max": self.consumer_prompt_max(),
            "worker_prompt_max": {"cycle0": self.worker_prompt_max(cycle0=True), "later": self.worker_prompt_max(cycle0=False)},
            "dec_revision_prompt_max": {str(n): self.dec_revision_prompt_max(n) for n in Ns},
            "hub_prompt_max": {
                str(n): {"cycle0": self.hub_prompt_max(n, cycle0=True), "later": self.hub_prompt_max(n, cycle0=False)} for n in Ns
            },
            "focal_prompt_max": {str(d): self.focal_prompt_max(d) for d in (0, 1, 2, 4, 8)},
        }


# --------------------------------------------------------------------------- minimal paths


#: How CEN_FLAT's minimal path is counted in B0 (recorded in the manifest):
#: ``cycle`` = cycle-0 hub + (N−1) workers + the reserved final hub call (the lead's B0
#: definition; the smallest budget at which the hub can actually delegate once);
#: ``hub_only`` = the cycle-0 hub call + the reserved final hub call (architecture §4's
#: reading: the hub may answer ``final`` immediately).
CEN_MINIMAL_PATHS: tuple[str, ...] = ("cycle", "hub_only")


def minimal_path(
    method: Method, oracle: FlopOracle, table: TableE, N: int, out_cap: int = SOLVER_OUT_CAP, *, cen_path: str = "cycle"
) -> dict[str, Any]:
    """The §6.5 minimal full-cap path of ``method`` at team size ``N`` (all-reservation cost).

    * ``S_FRESH``/``S_HISTORY``: one root; VOTE costs 0.
    * ``IND_VOTE``/``DEC``/``DEC_ONE_ROUND``/``IND_PRIVATE_REVISION``: N roots; VOTE costs 0
      (a DEC round is optional work, listed separately as ``first_round``).
    * ``CEN_FLAT``: per ``cen_path`` (:data:`CEN_MINIMAL_PATHS`): cycle-0 hub call [+ (N−1)
      workers at the cycle-0 worker envelope] + the reserved final hub call at the
      later-cycle hub envelope (architecture §1.11; §6.5 "including coordinator work").
    * ``DEGREE``: 9 roots + focal revision at degree 8 + 4 consumers (fixed protocol).
    """
    if cen_path not in CEN_MINIMAL_PATHS:
        raise ValueError(f"cen_path must be one of {CEN_MINIMAL_PATHS}, got {cen_path!r}")
    method = Method(method)
    R_root = oracle.reservation(table.root_prompt_max(), out_cap)
    calls: list[dict[str, Any]] = []

    def add(role: str, prompt: int, count: int = 1) -> None:
        calls.append({"role": role, "prompt_tokens_max": prompt, "count": count, "each": oracle.reservation(prompt, out_cap)})

    extra: dict[str, Any] = {}
    if method in (Method.S_FRESH, Method.S_HISTORY):
        add("root", table.root_prompt_max())
        if method is Method.S_HISTORY:
            extra["next_revision"] = oracle.reservation(table.s_history_prompt_max(), out_cap)
    elif method in (Method.IND_VOTE, Method.DEC, Method.DEC_ONE_ROUND, Method.IND_PRIVATE_REVISION):
        add("root", table.root_prompt_max(), N)
        if method is not Method.IND_VOTE:
            extra["first_round"] = N * oracle.reservation(table.dec_revision_prompt_max(N), out_cap)
    elif method is Method.CEN_FLAT:
        add("hub", table.hub_prompt_max(N, cycle0=True))
        if N > 1 and cen_path == "cycle":
            add("worker", table.worker_prompt_max(cycle0=True), N - 1)
        add("hub_final", table.hub_prompt_max(N, cycle0=False))
        extra["cen_path"] = cen_path
    elif method is Method.DEGREE:
        add("root", table.root_prompt_max(), 9)
        add("focal_revision", table.focal_prompt_max(8))
        add("consumer", table.consumer_prompt_max(), 4)
    else:
        raise ValueError(f"{method.value} has no minimal path (F banks are fixed opportunities)")
    total = sum(c["each"] * c["count"] for c in calls)
    return {"method": method.value, "N": N, "calls": calls, "total": total, "root_reservation": R_root, **extra}


B0_METHODS: tuple[Method, ...] = (Method.IND_VOTE, Method.DEC, Method.CEN_FLAT, Method.S_FRESH, Method.S_HISTORY)


def b0_candidates(oracle: FlopOracle, table: TableE, N: int = 5, *, cen_path: str = "cycle") -> dict[str, dict[str, Any]]:
    """Minimal paths of the five mandatory methods at ``N`` (B0 = the maximum; amendment B1)."""
    return {m.value: minimal_path(m, oracle, table, N, cen_path=cen_path) for m in B0_METHODS}


__all__ = [
    "B0_METHODS",
    "CEN_MINIMAL_PATHS",
    "ENVELOPE_NS",
    "EnvelopeError",
    "FALLBACK_WRAPPER_TOKENS",
    "TableE",
    "WRAPPER_FIXED_KEYS",
    "b0_candidates",
    "fallback_wrappers",
    "measure_wrappers",
    "minimal_path",
    "synthetic_task",
    "unavailable_packet",
]
