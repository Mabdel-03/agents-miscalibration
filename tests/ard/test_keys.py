"""Primary-key uniqueness assertions: the checker must find duplicates, must let a genuinely
unique table pass, and must degrade to a recorded fallback rather than crash when the v2
``cell_kind`` discriminator is absent.

Every table here is synthetic and tiny.  The frozen v1 tables are the checker's real proving
input — it reports 3 violations against them — but a test that reads them would assert a
property of one run root rather than a property of this module, and could not exercise the
clean-table or missing-table paths at all.
"""

from __future__ import annotations

import json

from agents_scaling.ard import keys as KEYS
from agents_scaling.study.aggregate import _write_parquet


def _decisions(ids):
    return [{"decision_id": d, "episode_id": "ep0", "round": 0} for d in ids]


def _episodes(rows):
    """``rows`` are ``(episode_id, cell_kind)`` pairs; ``cell_kind`` of ``None`` omits it."""
    out = []
    for episode_id, cell_kind in rows:
        row = {"episode_id": episode_id, "item_id": "math500-0000"}
        if cell_kind is not None:
            row["cell_kind"] = cell_kind
        out.append(row)
    return out


def _tables(tmp_path, **tables):
    root = tmp_path / "tables"
    root.mkdir(exist_ok=True)
    for name, rows in tables.items():
        _write_parquet(root / f"{name}.parquet", rows)
    return root


def test_unique_table_passes(tmp_path):
    root = _tables(tmp_path, labels=[{"decision_id": f"d{i}"} for i in range(8)])
    check = KEYS.check_table(root, "labels", ("decision_id",))
    assert check.status == KEYS.STATUS_OK
    assert (check.n_rows, check.n_distinct, check.n_duplicate_groups) == (8, 8, 0)
    assert check.examples == ()
    assert all(v.table != "labels" for v in KEYS.check_keys(root, strict=False))


def test_duplicate_key_is_reported_with_counts_and_examples(tmp_path):
    # "a" three times, "b" twice, "c" once: 6 rows, 3 distinct, 2 duplicate groups.
    root = _tables(tmp_path, decisions=_decisions(["a", "a", "a", "b", "b", "c"]))
    check = KEYS.check_table(root, "decisions", ("decision_id",))
    assert check.status == KEYS.STATUS_DUPLICATE
    assert (check.n_rows, check.n_distinct) == (6, 3)
    assert (check.n_duplicate_groups, check.n_excess_rows) == (2, 3)
    # Worst first, so the most-multiplied key is the one a reader sees.
    assert [(e.values, e.n_rows) for e in check.examples] == [(("a",), 3), (("b",), 2)]


def test_examples_are_capped_and_deterministic(tmp_path):
    ids = [f"d{i}" for i in range(KEYS.EXAMPLE_LIMIT + 4) for _ in range(2)]
    root = _tables(tmp_path, decisions=_decisions(ids))
    check = KEYS.check_table(root, "decisions", ("decision_id",))
    assert check.n_duplicate_groups == KEYS.EXAMPLE_LIMIT + 4
    assert len(check.examples) == KEYS.EXAMPLE_LIMIT
    assert check.examples == KEYS.check_table(root, "decisions", ("decision_id",)).examples


def test_cell_kind_absent_falls_back_and_records_the_note(tmp_path):
    """The v1 shape: no ``cell_kind``, and the EPISODE/ROOTS rows collide without it."""
    root = _tables(tmp_path, episodes=_episodes([("ep0", None), ("ep0", None), ("ep1", None)]))
    check = KEYS.check_table(root, "episodes", KEYS.PRIMARY_KEYS["episodes"])
    assert check.declared_key == (KEYS.CELL_KIND, "episode_id")
    assert check.key == ("episode_id",)
    assert check.status == KEYS.STATUS_DUPLICATE
    assert "cell_kind" in check.note and "episode_id" in check.note


def test_cell_kind_present_is_used_as_declared(tmp_path):
    """The v2 shape: the discriminator separates the rows that collided in v1.

    This asserts only that the checker *uses* the column when it exists.  Whether v2's
    ``cell_kind`` actually makes ``episodes`` unique is a property of v2's writer, and is not
    knowable from v1.
    """
    root = _tables(tmp_path, episodes=_episodes(
        [("ep0", "EPISODE"), ("ep0", "ROOTS"), ("ep1", "EPISODE")]))
    check = KEYS.check_table(root, "episodes", KEYS.PRIMARY_KEYS["episodes"])
    assert check.key == (KEYS.CELL_KIND, "episode_id")
    assert check.status == KEYS.STATUS_OK
    assert check.note == ""


def test_required_column_missing_is_unverifiable_not_a_crash(tmp_path):
    root = _tables(tmp_path, decisions=[{"episode_id": "ep0"}, {"episode_id": "ep1"}])
    check = KEYS.check_table(root, "decisions", ("decision_id",))
    assert check.status == KEYS.STATUS_UNVERIFIABLE
    assert check.key == ()
    assert check.n_rows == 2                       # still read from the footer
    assert "decision_id" in check.note


def test_missing_table_and_empty_sentinel_differ(tmp_path):
    root = _tables(tmp_path, labels=[])            # writes labels.empty, no parquet
    empty = KEYS.check_table(root, "labels", ("decision_id",))
    assert empty.status == KEYS.STATUS_OK and empty.n_rows == 0

    absent = KEYS.check_table(root, "decisions", ("decision_id",))
    assert absent.status == KEYS.STATUS_UNVERIFIABLE
    assert "does not exist" in absent.note


def test_strict_governs_unverifiable_only(tmp_path):
    """A measured duplicate is a violation under either setting; an absent table is not."""
    root = _tables(tmp_path, decisions=_decisions(["a", "a"]))
    strict = {v.table for v in KEYS.check_keys(root, strict=True)}
    lenient = {v.table for v in KEYS.check_keys(root, strict=False)}
    assert "decisions" in strict and "decisions" in lenient
    assert {"labels", "episodes", "round_summary"} <= strict
    assert lenient == {"decisions"}


def test_violations_are_sorted_worst_first(tmp_path):
    root = _tables(
        tmp_path,
        decisions=_decisions(["a", "a"]),
        labels=[{"decision_id": "x"}],
        episodes=_episodes([("ep0", None), ("ep0", None), ("ep1", None), ("ep1", None)]),
        round_summary=[{"episode_id": "ep0", "round": 0}],
    )
    violations = KEYS.check_keys(root, strict=False)
    assert [v.table for v in violations] == ["episodes", "decisions"]
    assert violations[0].n_duplicate_groups == 2


def test_report_carries_clean_checks_too(tmp_path):
    root = _tables(tmp_path, labels=[{"decision_id": f"d{i}"} for i in range(3)])
    payload = KEYS.report(root, strict=False)
    assert payload["n_checks"] == len(KEYS.PRIMARY_KEYS)
    by_table = {c["table"]: c for c in payload["checks"]}
    assert by_table["labels"]["status"] == KEYS.STATUS_OK
    assert payload["ok"] is True and payload["violations"] == []
    json.dumps(payload)                            # the CLI's contract: it must serialise


def test_cli_exit_codes(tmp_path, capsys):
    root = _tables(tmp_path, decisions=_decisions(["a", "a"]),
                   labels=[{"decision_id": "x"}],
                   episodes=_episodes([("ep0", "EPISODE")]),
                   round_summary=[{"episode_id": "ep0", "round": 0}])
    assert KEYS.main(["--tables", str(root)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_violations"] == 1 and payload["violations"][0]["table"] == "decisions"

    assert KEYS.main(["--tables", str(root), "--no-strict"]) == 0
    assert json.loads(capsys.readouterr().out)["strict"] is False


def test_cli_exit_zero_on_a_clean_directory(tmp_path):
    root = _tables(tmp_path, decisions=_decisions(["a", "b"]),
                   labels=[{"decision_id": "x"}],
                   episodes=_episodes([("ep0", "EPISODE"), ("ep0", "ROOTS")]),
                   round_summary=[{"episode_id": "ep0", "round": 0},
                                  {"episode_id": "ep0", "round": 1}])
    assert KEYS.main(["--tables", str(root)]) == 0


def test_missing_tables_dir_is_reported_not_raised(tmp_path):
    violations = KEYS.check_keys(tmp_path / "nope", strict=True)
    assert len(violations) == len(KEYS.PRIMARY_KEYS)
    assert all(v.n_duplicate_groups == 0 and "does not exist" in v.note for v in violations)
