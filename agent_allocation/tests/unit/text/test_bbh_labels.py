"""Independent label checks on every pinned BBH row: small solvers recompute the gold for three of the four
formats and must agree with the adapter's normalized gold; the fourth (logical deduction) is checked
structurally. A mismatch means the adapter mislabels items (or the pinned data is wrong) and blocks the benchmark."""
from __future__ import annotations

import re

from agent_allocation.contracts import OutputKind

_BOOL_TOKENS = re.compile(r"True|False|not|and|or|\(|\)")
_OPTION = re.compile(r"^\(([A-Z])\) (.+)$", re.MULTILINE)


def _items(bbh_inventory, subtask):
    items = [item for item in bbh_inventory if item.public.subtask == subtask]
    assert len(items) == 250
    return items


def _options(prompt: str) -> dict[str, str]:
    return dict(_OPTION.findall(prompt.split("\nOptions:\n")[-1]))


def solve_boolean(prompt: str) -> str:
    assert prompt.endswith(" is")
    expression = prompt[: -len(" is")]
    tokens = expression.split()
    assert tokens and all(_BOOL_TOKENS.fullmatch(token) for token in tokens), expression
    value = eval(" ".join(tokens), {"__builtins__": {}}, {})  # noqa: S307 - tokens restricted to boolean syntax
    assert isinstance(value, bool)
    return str(value)


def solve_tracking(prompt: str) -> str:
    story, _, _ = prompt.partition("\nOptions:\n")
    options = _options(prompt)
    names_sentence = story.split(" are ", 1)[0]
    names = [name.strip() for name in re.split(r", and |, ", names_sentence)]
    assert len(names) == 7, names
    setup = story.split(": ", 1)[1].split(".\n", 1)[0]
    parts = [part.strip() for part in re.split(r", and |, ", setup)]
    holder_of: dict[str, str] = {}
    for part in parts:
        owner = next(name for name in names if part.startswith(name + " "))
        texts = [text for text in options.values() if part.endswith(" " + text)]
        assert len(texts) == 1, (part, texts)
        holder_of[owner] = texts[0]
    assert len(holder_of) == 7
    for first, second in re.findall(r"(?:First|Then|Finally), (\w+) and (\w+) (?:swap|trade|switch)", story):
        holder_of[first], holder_of[second] = holder_of[second], holder_of[first]
    asked = re.search(r"At the end of the [\w ]+?, (\w+) ", story).group(1)
    final = holder_of[asked]
    matches = [letter for letter, text in options.items() if text == final]
    assert len(matches) == 1
    return matches[0]


def _hour(token: str) -> int:
    match = re.fullmatch(r"(\d{1,2})(am|pm)", token)
    assert match, token
    hour = int(match.group(1)) % 12
    return hour + 12 if match.group(2) == "pm" else hour


def solve_temporal(prompt: str) -> str:
    story, _, _ = prompt.partition("\nOptions:\n")
    options = _options(prompt)
    wake = _hour(re.search(r"woke up at (\d{1,2}(?:am|pm))", story).group(1))
    close = _hour(re.search(r"closed after (\d{1,2}(?:am|pm))", story).group(1))
    busy = [(_hour(a), _hour(b)) for a, b in re.findall(r"from (\d{1,2}(?:am|pm)) to (\d{1,2}(?:am|pm))", story)]
    free = []
    for letter, text in options.items():
        start, end = (_hour(part) for part in text.split(" to "))
        if start >= wake and end <= close and all(end <= b_start or start >= b_end for b_start, b_end in busy):
            free.append(letter)
    assert len(free) == 1, (free, prompt)
    return free[0]


def test_boolean_expression_labels(bbh_inventory):
    for item in _items(bbh_inventory, "boolean_expressions"):
        assert item.public.output_kind is OutputKind.BOOLEAN
        assert solve_boolean(item.public.prompt) == item.private.gold_answer, item.public.item_id


def test_tracking_shuffled_objects_labels(bbh_inventory):
    for item in _items(bbh_inventory, "tracking_shuffled_objects_seven_objects"):
        assert solve_tracking(item.public.prompt) == item.private.gold_answer, item.public.item_id


def test_temporal_sequence_labels(bbh_inventory):
    for item in _items(bbh_inventory, "temporal_sequences"):
        assert solve_temporal(item.public.prompt) == item.private.gold_answer, item.public.item_id


def test_logical_deduction_structure(bbh_inventory):
    for item in _items(bbh_inventory, "logical_deduction_seven_objects"):
        options = _options(item.public.prompt)
        assert tuple(options) == tuple("ABCDEFG") == item.public.allowed_choices
        assert item.private.gold_answer in options
        assert item.public.prompt.startswith("The following paragraphs each describe a set of seven objects")
        assert len(set(options.values())) == 7                  # distinct statements: letter identity is answer identity


def test_planted_canaries_in_the_source_document_never_reach_public_records():
    import json

    from agent_allocation.benchmarks import bbh_logic4
    from agent_allocation.contracts import BBH_SUBTASKS, dump_record
    canary = "CANARY-BBH-4f1d9e"
    sources = {}
    for subtask in BBH_SUBTASKS:
        if subtask == "boolean_expressions":
            example = {"input": "True and not False is", "target": "True"}
        else:
            example = {"input": "Which?\nOptions:\n(A) one\n(B) two", "target": "(B)"}
        document = {"canary": canary, "rationale": canary, "examples": [dict(example) for _ in range(250)]}
        sources[subtask] = json.dumps(document).encode()
    items = bbh_logic4.items_from_bytes(sources)
    assert len(items) == 1000 and all(item.eligible for item in items)
    assert canary not in json.dumps([dump_record(item.public) for item in items])
