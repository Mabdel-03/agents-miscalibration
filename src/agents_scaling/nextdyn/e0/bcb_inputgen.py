"""The dev-frozen BigCodeBench input generator, policy ``v1`` (spec 7.4; plan §2.6).

The behavioural repair needs inputs to run candidate programs on, and where those inputs come
from decides whether the repair is a measurement or a leak.  Two rules are structural here:

* **The generator never reads a hidden test.**  It parses the *public* ``instruct_prompt``
  only -- the signature and the docstring's declared argument types -- and it imports nothing
  from the legacy evaluation directory.  ``tests/nextdyn/test_e0.py::
  test_hidden_code_tests_absent_from_generator_and_selection_sandbox`` walks this module's
  import closure and the job payload it builds to prove it.
* **The policy is frozen before it is used.**  :data:`POLICY_VERSION` is ``v1``, the type
  vocabulary is closed, and an argument the policy cannot type makes the task **ineligible with
  a typed reason** instead of getting a default value.  A generator that quietly falls back to
  ``0`` for anything it does not understand manufactures agreement between programs that do
  different things.

Values are drawn from ``study.identity.blind_order_key(study_seed, "ND_BCB_INPUT", task_id, k)``,
so the input set for a task is a deterministic function of the study seed and nothing else --
not of the candidate programs, and not of any outcome.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.study.identity import blind_order_key

POLICY_VERSION: Final[str] = "dev_frozen_v1"
INPUT_NAMESPACE: Final[str] = "ND_BCB_INPUT"
AUDIT_NAMESPACE: Final[str] = "ND_BCB_AUDIT"
ENTRY_POINT: Final[str] = "task_func"

#: The closed type vocabulary policy v1 can generate.  Anything else is a typed ineligibility.
SUPPORTED_TYPES: Final[tuple[str, ...]] = ("int", "float", "str", "bool", "list", "dict", "path")

#: Reasons a task is ineligible.  Each is a *statement about the task*, never a silent skip.
REASON_NO_SIGNATURE: Final[str] = "signature_unparseable"
REASON_UNTYPED_ARG: Final[str] = "argument_type_not_in_policy"
REASON_OPAQUE_ARG: Final[str] = "argument_is_an_opaque_object"
REASON_NONDETERMINISTIC: Final[str] = "requires_network_gui_time_or_unseedable_randomness"
REASON_NO_ARGS: Final[str] = "entry_point_takes_no_arguments"

#: Imports whose behaviour cannot be made deterministic inside the sandbox.
NONDETERMINISTIC_MARKERS: Final[tuple[str, ...]] = (
    "requests", "urllib", "socket", "http.client", "boto3", "tkinter", "matplotlib.pyplot",
    "datetime.now", "time.time", "os.urandom", "uuid4", "secrets",
)

_TYPE_ALIASES: Final[Mapping[str, str]] = {
    "int": "int", "integer": "int", "float": "float", "number": "float", "double": "float",
    "str": "str", "string": "str", "bool": "bool", "boolean": "bool",
    "list": "list", "tuple": "list", "sequence": "list", "array": "list",
    "dict": "dict", "mapping": "dict", "path": "path", "pathlib.path": "path", "file": "path",
}


@dataclass(frozen=True)
class Argument:
    name: str
    declared: str | None
    kind: str | None
    has_default: bool


@dataclass(frozen=True)
class TaskInputs:
    """Either a typed input set or a typed ineligibility; never a partly-guessed set."""

    task_id: str
    eligible: bool
    reason: str | None
    arguments: tuple[Argument, ...]
    inputs: tuple[Mapping[str, Any], ...]
    policy_version: str = POLICY_VERSION

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)


def _normalize_type(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"\[.*$", "", str(text).strip().strip("'\"")).lower()
    return _TYPE_ALIASES.get(cleaned)


def parse_signature(instruct_prompt: str) -> tuple[tuple[Argument, ...], str | None]:
    """Arguments of ``task_func`` from the **public** prompt: annotations, then the docstring.

    The prompt is the text the model was shown, so nothing read here was hidden from the
    candidate programs either.
    """
    match = re.search(r"def\s+" + ENTRY_POINT + r"\s*\((.*?)\)\s*(->[^:]*)?:", instruct_prompt, re.S)
    if not match:
        return (), REASON_NO_SIGNATURE
    try:
        tree = ast.parse(f"def {ENTRY_POINT}({match.group(1)}): pass")
    except SyntaxError:
        return (), REASON_NO_SIGNATURE
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    docs = _docstring_types(instruct_prompt)
    defaults = len(function.args.defaults)
    positional = function.args.args
    arguments: list[Argument] = []
    for index, arg in enumerate(positional):
        annotation = ast.unparse(arg.annotation) if arg.annotation is not None else None
        declared = annotation or docs.get(arg.arg)
        arguments.append(Argument(name=arg.arg, declared=declared,
                                  kind=_normalize_type(declared),
                                  has_default=index >= len(positional) - defaults))
    return tuple(arguments), None


def _docstring_types(text: str) -> dict[str, str]:
    """``name (type): description`` lines of the prompt's Parameters/Args block."""
    out: dict[str, str] = {}
    for name, declared in re.findall(r"^\s*(\w+)\s*\(([^)]+)\)\s*:", text, re.M):
        out.setdefault(name, declared)
    return out


def _is_nondeterministic(instruct_prompt: str) -> bool:
    return any(marker in instruct_prompt for marker in NONDETERMINISTIC_MARKERS)


def _value(kind: str, seed_bytes: bytes, index: int) -> Any:
    """One deterministic value of ``kind`` from the blind key bytes."""
    n = int.from_bytes(seed_bytes[: 8], "big")
    if kind == "int":
        return int(n % 1000) - 500
    if kind == "float":
        return round(((n % 1_000_000) / 1_000.0) - 500.0, 6)
    if kind == "bool":
        return bool(n % 2)
    if kind == "str":
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "
        return "".join(alphabet[(n >> (5 * i)) % len(alphabet)] for i in range(1 + n % 12))
    if kind == "list":
        return [int((n >> (7 * i)) % 100) for i in range(1 + n % 5)]
    if kind == "dict":
        return {f"k{i}": int((n >> (11 * i)) % 100) for i in range(1 + n % 3)}
    if kind == "path":
        # a path inside the sandbox's own /work; the harness creates the file before the call
        return f"/work/input_{index}.txt"
    raise ContractViolation(f"policy {POLICY_VERSION} cannot generate type {kind!r}")


def generate(task_id: str, instruct_prompt: str, study_seed: bytes, *, max_inputs: int,
             namespace: str = INPUT_NAMESPACE) -> TaskInputs:
    """At most ``max_inputs`` typed inputs for one task, or a typed ineligibility.

    The seed is ``blind_order_key(study_seed, namespace, task_id, k)``, so the audit suite
    (``namespace=AUDIT_NAMESPACE``) is disjoint from the selection suite by construction and
    neither depends on any candidate program.
    """
    if _is_nondeterministic(instruct_prompt):
        return TaskInputs(task_id, False, REASON_NONDETERMINISTIC, (), ())
    arguments, failure = parse_signature(instruct_prompt)
    if failure:
        return TaskInputs(task_id, False, failure, (), ())
    if not arguments:
        return TaskInputs(task_id, False, REASON_NO_ARGS, (), ())
    for argument in arguments:
        if argument.kind is None:
            reason = REASON_UNTYPED_ARG if argument.declared else REASON_OPAQUE_ARG
            return TaskInputs(task_id, False, reason, arguments, ())
    inputs: list[dict[str, Any]] = []
    for k in range(int(max_inputs)):
        key = blind_order_key(study_seed, namespace, task_id, k)
        inputs.append({argument.name: _value(argument.kind, key[i * 8:(i + 1) * 8] or key, k)
                       for i, argument in enumerate(arguments)})
    return TaskInputs(task_id, True, None, arguments, tuple(inputs))


def census(results: Sequence[TaskInputs]) -> dict[str, Any]:
    """Eligible fraction and the typed reasons for the rest (spec 7.4 reporting)."""
    reasons: dict[str, int] = {}
    for result in results:
        if not result.eligible:
            reasons[str(result.reason)] = reasons.get(str(result.reason), 0) + 1
    eligible = sum(1 for r in results if r.eligible)
    return {"policy_version": POLICY_VERSION, "n_tasks": len(results), "n_eligible": eligible,
            "eligible_fraction": (eligible / len(results)) if results else None,
            "ineligible_reasons": dict(sorted(reasons.items())),
            "supported_types": list(SUPPORTED_TYPES)}


__all__ = [
    "AUDIT_NAMESPACE", "Argument", "ENTRY_POINT", "INPUT_NAMESPACE", "NONDETERMINISTIC_MARKERS",
    "POLICY_VERSION", "REASON_NO_ARGS", "REASON_NO_SIGNATURE", "REASON_NONDETERMINISTIC",
    "REASON_OPAQUE_ARG", "REASON_UNTYPED_ARG", "SUPPORTED_TYPES", "TaskInputs", "census",
    "generate", "parse_signature",
]
