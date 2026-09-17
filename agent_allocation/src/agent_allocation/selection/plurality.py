"""Text plurality over gold-blind answer classes (MASTER_SPEC §11.1, §15.13).

Class construction, in the checkpoint's ``candidate_ids`` order:

1. Candidates without a vote key cast no vote (``selection.keys.vote_key``).
2. Exact keys group first: equal ``VoteKey.exact`` means the same class.
3. MATH only, and only when an equivalence callable is injected: exact-key groups are admitted by **mutual,
   complete-link** equivalence. Processing distinct exact keys in first-occurrence order, a group joins the first
   existing class (creation order) for which ``eq(a, b) is True and eq(b, a) is True`` for *every* pair of a distinct
   answer text of the group and a distinct answer text already in that class; otherwise it founds a new class.
   ``None`` (unknown/timeout) and ``False`` never merge, so a one-direction Math-Verify success cannot merge and a
   nontransitive chain cannot collapse into one class. Each ordered pair is evaluated at most once per call;
   exceptions from the callable propagate (fail loudly). Exact keys only apply value-preserving rewrites
   (``selection.keys.math_exact_key``), so step 2 never merges what step 3 would keep apart.

Winner: greatest support (candidate occurrences). Tied classes are ordered by :func:`class_tie_key`, a hash of
``(seed, item_id, canonical class id)`` only -- independent of completion order, candidate ids and gold. The
canonical class id is ``"<kind>:" + min(exact keys in the class)``, so it does not depend on processing order either.
Within the winning class the first member in ``candidate_ids`` order is delivered: a genuine generated candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from agent_allocation.contracts import OutputKind, canonical_hash
from agent_allocation.selection.keys import VoteKey

__all__ = [
    "AnswerClass", "MathEquivalent", "PluralityOutcome", "build_classes", "class_tie_key", "memoized_equivalence",
    "plurality",
]

MathEquivalent = Callable[[str, str], "bool | None"]

_CLASS_PREFIX = {OutputKind.OPTION: "option", OutputKind.BOOLEAN: "boolean", OutputKind.MATH: "math"}


@dataclass(frozen=True)
class AnswerClass:
    class_id: str
    member_ids: tuple[str, ...]          # candidate_ids order
    exact_keys: tuple[str, ...]          # distinct exact keys, first-occurrence order

    @property
    def support(self) -> int:
        return len(self.member_ids)


@dataclass(frozen=True)
class PluralityOutcome:
    classes: tuple[AnswerClass, ...]     # creation order
    winner: AnswerClass | None
    selected_candidate_id: str | None
    tied_class_ids: tuple[str, ...]      # classes sharing the top support, tie-key order (winner first)
    n_votes: int
    equivalence_calls: int
    equivalence_unknown: int


def memoized_equivalence(fn: MathEquivalent) -> MathEquivalent:
    """Wrap an equivalence callable so each ordered pair is evaluated once for the wrapper's lifetime.

    Use one wrapper per partition run: overlapping checkpoints (``sample/k`` prefixes, archives) then see the same
    verdict for the same pair, including a cached ``None`` from a timeout, so class construction is consistent
    across checkpoints. Results must be ``bool | None``.
    """
    cache: dict[tuple[str, str], bool | None] = {}

    def equivalent(a: str, b: str) -> bool | None:
        if (a, b) not in cache:
            value = fn(a, b)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"math_equivalent must return bool | None, got {type(value).__name__}")
            cache[(a, b)] = value
        return cache[(a, b)]

    return equivalent


def class_tie_key(seed: int, item_id: str, class_id: str) -> str:
    return canonical_hash({"purpose": "agent06/class_tie", "seed": seed, "item_id": item_id, "class_id": class_id})


class _Equivalence:
    def __init__(self, fn: MathEquivalent) -> None:
        self._fn = fn
        self._cache: dict[tuple[str, str], bool | None] = {}
        self.calls = 0
        self.unknown = 0

    def one_way(self, a: str, b: str) -> bool | None:
        if (a, b) not in self._cache:
            value = self._fn(a, b)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"math_equivalent must return bool | None, got {type(value).__name__}")
            self.calls += 1
            self.unknown += value is None
            self._cache[(a, b)] = value
        return self._cache[(a, b)]

    def mutual(self, a: str, b: str) -> bool:
        return self.one_way(a, b) is True and self.one_way(b, a) is True


def build_classes(votes: Sequence[tuple[str, VoteKey]], *, kind: OutputKind,
                  math_equivalent: MathEquivalent | None = None) -> tuple[tuple[AnswerClass, ...], int, int]:
    """Classes in creation order plus (equivalence calls, unknown equivalence results)."""
    if kind not in _CLASS_PREFIX:
        raise ValueError(f"no answer classes for output kind {kind.value!r}")
    members: dict[str, list[str]] = {}
    texts: dict[str, list[str]] = {}                 # distinct answer texts per exact key, first-occurrence order
    for candidate_id, key in votes:
        members.setdefault(key.exact, []).append(candidate_id)
        seen = texts.setdefault(key.exact, [])
        if key.text not in seen:
            seen.append(key.text)
    order = list(members)                            # first-occurrence order of distinct exact keys
    groups: list[list[str]] = []
    calls = unknown = 0
    if kind is OutputKind.MATH and math_equivalent is not None:
        eq = _Equivalence(math_equivalent)
        for key in order:
            for group in groups:
                if all(eq.mutual(mine, theirs) for mine in texts[key] for other in group for theirs in texts[other]):
                    group.append(key)
                    break
            else:
                groups.append([key])
        calls, unknown = eq.calls, eq.unknown
    else:
        groups = [[key] for key in order]
    position = {cid: i for i, (cid, _) in enumerate(votes)}
    classes = []
    for group in groups:
        ids = sorted((cid for key in group for cid in members[key]), key=position.__getitem__)
        classes.append(AnswerClass(class_id=f"{_CLASS_PREFIX[kind]}:{min(group)}", member_ids=tuple(ids),
                                   exact_keys=tuple(group)))
    return tuple(classes), calls, unknown


def plurality(votes: Sequence[tuple[str, VoteKey]], *, kind: OutputKind, item_id: str, seed: int,
              math_equivalent: MathEquivalent | None = None) -> PluralityOutcome:
    ids = [cid for cid, _ in votes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{item_id}: duplicate candidate ids would double-count a sampling opportunity")
    classes, calls, unknown = build_classes(votes, kind=kind, math_equivalent=math_equivalent)
    if not classes:
        return PluralityOutcome((), None, None, (), 0, calls, unknown)
    top = max(c.support for c in classes)
    tied = sorted((c for c in classes if c.support == top), key=lambda c: (class_tie_key(seed, item_id, c.class_id),
                                                                           c.class_id))
    winner = tied[0]
    return PluralityOutcome(classes=classes, winner=winner, selected_candidate_id=winner.member_ids[0],
                            tied_class_ids=tuple(c.class_id for c in tied), n_votes=len(votes),
                            equivalence_calls=calls, equivalence_unknown=unknown)
