"""The structural guards WP0 can already assert (plan §1.1 R1, R2, D2, §3.3; contract §0).

Every guard is an AST walk over the **package** only: tests are never counted as readers
(a field only a test reads is the D-03 shape).  A declared consumer module that has not
landed yet is reported as *pending* (and counted by ``test_pending_consumers_are_listed``),
a landed one must read the attribute by name.  Guards that need modules other WPs own
(``test_pipeline_dry_run``, the item-file access rule, the gold import closure, the single
``DecisionRow`` site) land with those WPs.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest
import yaml

import agents_scaling.nextdyn as nextdyn
from agents_scaling.nextdyn import ARD_IMPORTS
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import kinds, tables

CONFIG_STEMS = ("study", "compute", "analysis", "protocols", "compositions", "legacy_sources")

PACKAGE_DIR = Path(nextdyn.__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
TESTS_DIR = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs" / "nextdyn"
PACKAGE = "agents_scaling.nextdyn"


def _package_modules() -> dict[str, Path]:
    out = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = path.relative_to(PACKAGE_DIR).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join([PACKAGE, *parts]) if parts else PACKAGE] = path
    return out


def _test_modules() -> dict[str, Path]:
    return {f"tests.nextdyn.{p.stem}": p for p in sorted(TESTS_DIR.glob("*.py"))}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _attribute_reads(tree: ast.AST, *, skip_assign_targets: frozenset[str] = frozenset()) -> set[str]:
    """Attribute names read anywhere, plus dotted-string segments (``"locks.git_commit"``
    read through ``getattr`` chains), excluding the bodies of the named assignments."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in skip_assign_targets for t in node.targets):
            continue
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "." in node.value \
                and re.fullmatch(r"[A-Za-z0-9_*.]+", node.value):
            names.update(node.value.split("."))
    # the skipped assignments hide the CONSUMERS dict itself; re-walk everything else once
    return names


def _attribute_reads_outside_consumers(path: Path) -> set[str]:
    tree = _tree(path)
    names: set[str] = set()
    skip = {"CONSUMERS"}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in skip for t in targets):
                continue
        names |= _attribute_reads(node)
    return names


# ---------------------------------------------------------------------------- R2: dataclass fields


def _field_consumer_state(consumer: str, field_name: str, *, declaring_module: Path) -> str:
    """'ok' | 'pending' | 'unread' | 'self': the consumer must be a package module other
    than the declaring one, and once it exists it must read ``field_name`` by attribute."""
    module_path = _package_modules().get(consumer)
    if module_path is None:
        return "pending"
    if module_path.resolve() == declaring_module.resolve():
        return "self"
    return "ok" if field_name in _attribute_reads_outside_consumers(module_path) else "unread"


def _check_field_consumers(table, dataclasses_, *, declaring_module: Path, key):
    unread, pending, undeclared = [], [], []
    for cls in dataclasses_:
        for f in dataclasses.fields(cls):
            consumer = table.get(key(cls, f.name))
            if consumer is None:
                undeclared.append(f"{cls.__name__}.{f.name}")
                continue
            state = _field_consumer_state(consumer, f.name, declaring_module=declaring_module)
            if state == "pending":
                pending.append((f"{cls.__name__}.{f.name}", consumer))
            elif state != "ok":
                unread.append((f"{cls.__name__}.{f.name}", consumer, state))
    return undeclared, unread, pending


def test_every_kindspec_field_is_read_outside_kinds_py(capsys):
    """Every ``KindSpec`` field names its consuming package module in
    ``kinds.KIND_FIELD_CONSUMERS``; a landed consumer must read it, tests never count."""
    undeclared, unread, pending = _check_field_consumers(
        kinds.KIND_FIELD_CONSUMERS, (kinds.KindSpec,), declaring_module=PACKAGE_DIR / "kinds.py",
        key=lambda cls, name: name)
    stale = sorted(set(kinds.KIND_FIELD_CONSUMERS) - {f.name for f in dataclasses.fields(kinds.KindSpec)})
    print(f"pending KindSpec field consumers: {pending}")
    assert not stale, f"KIND_FIELD_CONSUMERS names fields KindSpec does not have: {stale}"
    assert not undeclared, f"KindSpec fields without a declared consumer: {undeclared}"
    assert not unread, f"KindSpec fields whose landed consumer never reads them (D-03 shape): {unread}"
    assert all(c.startswith(PACKAGE + ".") for _, c in pending)


def test_every_contract_dataclass_field_is_read_outside_contract_py(capsys):
    """Rule R2 one level up: a YAML leaf that flows into a contract field satisfies the
    config-leaf guard, so every contract field names its own package consumer
    (``contract.CONTRACT_FIELD_CONSUMERS``) and a landed consumer must read it by name."""
    undeclared, unread, pending = _check_field_consumers(
        K.CONTRACT_FIELD_CONSUMERS, K.CONTRACT_DATACLASSES, declaring_module=PACKAGE_DIR / "contract.py",
        key=lambda cls, name: (cls.__name__, name))
    declared = {(cls.__name__, f.name) for cls in K.CONTRACT_DATACLASSES for f in dataclasses.fields(cls)}
    stale = sorted(set(K.CONTRACT_FIELD_CONSUMERS) - declared)
    print(f"pending contract field consumers ({len(pending)}): {pending}")
    assert not stale, f"CONTRACT_FIELD_CONSUMERS names fields that do not exist: {stale}"
    assert not undeclared, f"contract fields without a declared consumer: {undeclared}"
    assert not unread, f"contract fields whose landed consumer never reads them (D-03 shape): {unread}"
    # the consumers WP0 itself provides are landed, so those fields are held strictly already
    landed = {f"{cls}.{name}" for (cls, name), consumer in K.CONTRACT_FIELD_CONSUMERS.items()
              if consumer in _package_modules()}
    assert {"ModelCell.name", "ModelCell.lane", "ModelCell.hf_id", "SelectionSpec.audit_thinking_tokens"} <= landed
    assert not any(name in landed for name, _ in pending)


# ---------------------------------------------------------------------------- R2: config leaves


def _leaves() -> list[tuple[str, str]]:
    """Every leaf of all six YAML files (lists are leaves)."""
    out = []
    for stem in CONFIG_STEMS:
        data = yaml.safe_load((CONFIG_DIR / f"{stem}.yaml").read_text())
        out.extend((f"{stem}.{path}", stem) for path, _ in C.leaf_paths(data))
    return out


def _leaf_values() -> dict[str, object]:
    out = {}
    for stem in CONFIG_STEMS:
        data = yaml.safe_load((CONFIG_DIR / f"{stem}.yaml").read_text())
        out.update({f"{stem}.{path}": value for path, value in C.leaf_paths(data)})
    return out


def _consumer_reads(consumer: str, needle: str) -> str:
    """'ok' | 'pending' | 'unread' for one (consumer, key name) pair."""
    if consumer.startswith("slurm/"):
        template = REPO_ROOT / consumer
        if not template.exists():
            return "pending"
        return "ok" if needle in template.read_text() else "unread"
    module_path = _package_modules().get(consumer)
    if module_path is None:
        return "pending"
    reads = _attribute_reads_outside_consumers(module_path)
    return "ok" if needle in reads else "unread"


def _needle(leaf: str) -> str:
    parts = leaf.split(".")
    pattern = next((p for p in C.CONSUMERS if C.consumer_of(leaf) == C.CONSUMERS[p]
                    and len(p.split(".")) == len(parts)
                    and all(a == "*" or a == b for a, b in zip(p.split("."), parts))), None)
    assert pattern is not None
    named = [seg for seg in pattern.split(".") if seg != "*"]
    return named[-1]


def test_every_config_leaf_has_a_declared_consumer_that_reads_it():
    """R2, static: every leaf of every YAML under configs/nextdyn names a consumer, and every
    consumer that exists reads an attribute of that name (D-03 cannot recur silently)."""
    missing, unread = [], []
    for leaf, _stem in _leaves():
        consumer = C.consumer_of(leaf)
        if consumer is None:
            missing.append(leaf)
            continue
        state = _consumer_reads(consumer, _needle(leaf))
        if state == "unread":
            unread.append((leaf, consumer))
    assert not missing, f"config leaves without a declared consumer: {missing}"
    assert not unread, f"declared consumers that never read the key: {unread}"


def test_consumer_patterns_all_match_some_leaf():
    leaves = [leaf for leaf, _ in _leaves()]
    dead = [p for p in C.CONSUMERS if not any(
        len(p.split(".")) == len(leaf.split(".")) and all(a == "*" or a == b for a, b in zip(p.split("."), leaf.split(".")))
        for leaf in leaves)]
    assert not dead, f"CONSUMERS patterns that match no config leaf: {dead}"


def test_pending_consumers_are_listed(capsys):
    """Consumers whose module has not landed yet.  This count goes to zero as WPs land."""
    pending = sorted({(C.consumer_of(leaf)) for leaf, _ in _leaves()
                      if C.consumer_of(leaf) is not None and _consumer_reads(C.consumer_of(leaf), _needle(leaf)) == "pending"})
    n_leaves = sum(1 for leaf, _ in _leaves() if _consumer_reads(C.consumer_of(leaf), _needle(leaf)) == "pending")
    print(f"pending consumers ({len(pending)} modules, {n_leaves} leaves): {pending}")
    assert all(p.startswith(PACKAGE + ".") or p.startswith("slurm/") for p in pending)
    assert {stem for _, stem in _leaves()} == set(CONFIG_STEMS)


LAUNCH_BLOCKERS = (
    "study.locks.models_lock_hash", "study.locks.datasets_lock_hash", "study.locks.prompts_lock_hash",
    "study.locks.source_inventory_hash", "study.locks.prior_use_inventory_hash", "study.locks.git_commit",
    "study.locks.container_digests", "study.budgets.campaign_gpu_hours", "study.budgets.campaign_cpu_hours",
    "study.budgets.storage_limit_bytes", "study.analysis.confirmatory_sample_size_lock",
    "study.analysis.analysis_lock_hash",
)


def test_launch_blockers_are_consumed_by_the_module_that_acts_on_them():
    """``config`` only null-checks and hash-checks these; the declared consumer must be the
    module that acts on the value (plan's ceiling admission, the frozen manifest, release)."""
    assert set(LAUNCH_BLOCKERS) == {"study." + p for p in C.CONFIRMATION_REQUIRED}
    acting = {PACKAGE + "." + m for m in ("plan", "freeze", "release")}
    for leaf in LAUNCH_BLOCKERS:
        assert C.consumer_of(leaf) in acting, (leaf, C.consumer_of(leaf))


# ---------------------------------------------------------------------------- R1: one fact, one declaration

#: (a, b): two leaf paths that would declare one fact twice.  Neither may exist with a non-null
#: value while the other does; the second home of each fact was removed on review.
FORBIDDEN_DUPLICATE_PAIRS = (
    ("study.cohorts.sub_panels.e3a", "study.experiments.E3a.target_items_per_task"),
    ("study.cohorts.sub_panels.capture", "study.instrumentation.target_items_per_task"),
    ("study.cohorts.sub_panels.e2a", "study.experiments.E2a.items_per_task"),
    ("study.cohorts.sub_panels.natprot", "study.cohorts.natural_target_per_family"),
    ("study.cohorts.fragmented_target_per_family", "study.cohorts.sub_panels.fragmented"),
    ("study.experiments.E1.natural_protocols", "study.protocol.protocols"),
    ("study.experiments.E1.constructed_protocols", "study.protocol.protocols"),
    ("study.experiments.E2b.pilot_problems", "study.pilot.synthetic_receiver_problems"),
    ("study.experiments.E2b.adapter_check_problems", "study.pilot.adapter_check_problems"),
    ("analysis.profiles.confirmatory.holm.n_primary", "study.analysis.primary_hypotheses"),
    ("analysis.profiles.confirmatory.holm.alpha", "study.analysis.familywise_alpha"),
    ("analysis.profiles.confirmatory.power.alpha", "study.analysis.familywise_alpha"),
    ("compute.cells.kinds.capture.partition", "compute.capture.partition"),
)


def test_no_fact_is_declared_twice_across_the_configs():
    values = _leaf_values()
    both = [(a, b) for a, b in FORBIDDEN_DUPLICATE_PAIRS
            if values.get(a) is not None and values.get(b) is not None]
    assert not both, f"one fact, two declarations (R1): {both}"
    # and the surviving home of each pair exists (the list cannot rot into vacuity)
    missing = [(a, b) for a, b in FORBIDDEN_DUPLICATE_PAIRS if values.get(a) is None and values.get(b) is None]
    assert not missing, f"neither home of the fact exists any more: {missing}"
    # the E2b grid is declared once and instantiated by the registry (checked by the compiler)
    assert C.e2b_grid_checks(C.load_study(CONFIG_DIR / "study.yaml"), CONFIG_DIR / "protocols.yaml") == []


# ---------------------------------------------------------------------------- D2: ARD imports


def _ard_import_targets(tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names if a.name.startswith("agents_scaling.ard"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("agents_scaling.ard"):
            out.extend(f"{node.module}.{a.name}" for a in node.names)
    return out


def test_ard_imports_are_exactly_the_allowlist():
    """Every ``agents_scaling.ard`` import in the package names an allowlisted attribute or
    module, and every allowlisted name exists in the ard source (so the list cannot rot)."""
    for module, path in _package_modules().items():
        for target in _ard_import_targets(_tree(path)):
            allowed = target in ARD_IMPORTS or any(target.startswith(m + ".") for m in ARD_IMPORTS if _is_module(m))
            assert allowed, f"{module} imports {target}, which is not in nextdyn.ARD_IMPORTS (vendor it with a parity test)"
    for name in ARD_IMPORTS:
        assert _ard_name_exists(name), f"ARD_IMPORTS names {name}, which does not exist in agents_scaling.ard"


def _ard_path(name: str) -> tuple[Path, list[str]]:
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        base = REPO_ROOT / "src" / Path(*parts[:i])
        if base.with_suffix(".py").exists():
            return base.with_suffix(".py"), parts[i:]
        if (base / "__init__.py").exists():
            return base / "__init__.py", parts[i:]
    raise AssertionError(name)


def _is_module(name: str) -> bool:
    return not _ard_path(name)[1]


def _ard_name_exists(name: str) -> bool:
    path, attr = _ard_path(name)
    if not attr:
        return True
    if len(attr) != 1:
        return False
    tree = _tree(path)
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    defined |= {n.target.id for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    return attr[0] in defined


# ---------------------------------------------------------------------------- §3.3: no dispatch numbers in code

DISPATCH_MODULES = ("kinds", "deps", "cli", "dispatch", "fleet", "runner", "run_one", "status", "plan", "cells", "gates")
ALLOWED_NUMBERS = {0, 1}
#: A unit conversion says what a minute, a gigabyte or a percent *is*; no operator would ever
#: tune it in ``compute.yaml``, and it is exempt by its declaration site -- the name of the
#: module-level constant it is assigned to.  An inline ``24.0`` in an expression, or a policy
#: quantity dressed as a constant (``TARGET_WALL_HOURS``), is still an offender.
UNIT_CONSTANT = re.compile(r"^(?:[A-Z0-9]+_PER_[A-Z0-9_]+)$")


def test_no_numeric_dispatch_constant_lives_in_code():
    """Every throttle, cap, poll interval, walltime and cell size comes from compute.yaml (§3.3).

    Ints *and* floats: ``plan.TARGET_WALL_HOURS = 24.0`` -- the wall clock the whole fleet floor
    is sized against -- slipped an int-only guard for the length of a work package.

    Exempt: process exit codes (``EXIT_*``), JSON ``indent=`` widths, slice bounds used to
    shorten hashes in messages, and named unit conversions (:data:`UNIT_CONSTANT`) -- none of
    them is a dispatch quantity.
    """
    offenders = []
    for stem in DISPATCH_MODULES:
        path = PACKAGE_DIR / f"{stem}.py"
        if not path.exists():
            continue
        tree = _tree(path)
        exempt: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if any(name.startswith("EXIT_") or UNIT_CONSTANT.match(name) for name in names):
                    exempt.update(id(c) for c in ast.walk(node))
            elif isinstance(node, ast.keyword) and node.arg == "indent":
                exempt.update(id(c) for c in ast.walk(node))
            elif isinstance(node, ast.Slice):
                exempt.update(id(c) for c in ast.walk(node))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                    and not isinstance(node.value, bool):
                if node.value not in ALLOWED_NUMBERS and id(node) not in exempt:
                    offenders.append(f"{stem}.py:{node.lineno}: {node.value}")
    assert not offenders, f"numeric constants in dispatch modules (move them to compute.yaml): {offenders}"


def test_the_fleet_floors_target_wall_clock_comes_from_compute_yaml(compute_cfg):
    """plan §3.3: the hours the floor is sized against are a dispatch number like any other."""
    from agents_scaling.nextdyn import plan as PLAN

    assert not hasattr(PLAN, "TARGET_WALL_HOURS")
    assert PLAN._target_hours(compute_cfg) == float(compute_cfg.dispatch.target_wall_hours)
    assert C.consumer_of("compute.dispatch.target_wall_hours") == PACKAGE + ".plan"


def test_compute_yaml_carries_every_plan_throttle_and_cap(compute_cfg):
    d = compute_cfg.dispatch
    assert (d.throttle_per_endpoint["episode"], d.throttle_per_endpoint["donor_bank"],
            d.throttle_per_endpoint["receiver"], d.throttle_per_endpoint["selection_audit"]) == (8, 3, 6, 6)
    assert d.lane_submit_caps == {"8B": 200, "32B": 80, "audit": 60, "cpu": 40, "capture": 4}
    assert d.bank_submit_cap == 80 and d.global_submit_cap == 380
    assert (compute_cfg.keepalive.tick_s, compute_cfg.keepalive.max_launch) == (180, 3)
    assert (d.planner_pass_s, d.cpu_driver_pass_s, d.lane_driver_pass_s) == (900, 300, 600)


# ---------------------------------------------------------------------------- R1: single column declaration

_COLUMN_LIST_NAME = re.compile(r"(COLUMNS|SCHEMA|FIELDS|KEYS)", re.IGNORECASE)


def test_every_row_dataclass_is_the_only_declaration_of_its_table_columns():
    """No module but tables.py declares a table (``__table__``), a parquet schema, a
    primary-key mapping, or a module-level list of column names."""
    all_columns = {c for spec in tables.TABLES.values() for c in spec.column_names}
    offenders = []
    for module, path in _package_modules().items():
        if path.resolve() == (PACKAGE_DIR / "tables.py").resolve():
            continue
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                    isinstance(b, (ast.Assign, ast.AnnAssign)) and "__table__" in ast.dump(b) for b in node.body):
                offenders.append(f"{module}: class {node.name} declares __table__")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "schema" \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "pa":
                offenders.append(f"{module}:{node.lineno}: pa.schema(...)")
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                value = node.value
                if not names or not _COLUMN_LIST_NAME.search(names[0]) or value is None:
                    continue
                if isinstance(value, (ast.List, ast.Tuple)):
                    strings = {e.value for e in value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
                    if len(strings & all_columns) >= 2:
                        offenders.append(f"{module}: {names[0]} lists table columns {sorted(strings & all_columns)[:3]}")
    assert not offenders, offenders
    # and inside tables.py each table name is declared exactly once
    names = [t.__table__ for t in tables.ROW_TYPES]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------- D3: scratch


def test_scratch_paths_are_refused_everywhere():
    for path in list(PACKAGE_DIR.rglob("*.py")) + list(CONFIG_DIR.glob("*")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name == "config.py":
            continue  # declares SCRATCH_PREFIX to refuse it
        assert "/orcd/scratch" not in text, f"{path} references /orcd/scratch"
    assert C.scratch_paths(*(CONFIG_DIR.glob("*.yaml"))) == []
    assert sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")) == sorted(CONFIG_STEMS)


# ---------------------------------------------------------------------------- vocabulary consistency


def test_worker_modules_named_by_the_kind_table_are_pending_or_present():
    """A landed WP's worker must resolve; a pending one is listed, never silently absent."""
    pending, present = [], []
    modules = _package_modules()
    for spec in kinds.KIND_TABLE.values():
        module, _, attr = spec.worker.partition(":")
        path = modules.get(module)
        if path is None:
            pending.append(spec.worker)
            continue
        defined = {n.name for n in _tree(path).body if isinstance(n, ast.FunctionDef)}
        assert attr in defined, f"{spec.worker}: module present but callable missing"
        present.append(spec.worker)
    assert set(pending) | set(present) == {s.worker for s in kinds.KIND_TABLE.values()}
    print(f"pending workers: {pending}")


def test_table_producers_and_consumers_are_package_modules():
    modules = _package_modules()
    pending = set()
    for spec in tables.TABLES.values():
        for name in (spec.producer, *spec.consumers):
            assert name.startswith(PACKAGE + ".")
            if name not in modules:
                pending.add(name)
    print(f"pending table producers/consumers: {sorted(pending)}")
    assert PACKAGE + ".tables" not in pending


def test_every_row_field_the_gates_read_is_a_field_the_writer_writes():
    """D-05, caught structurally: a reader may not name a column that does not exist.

    Two of these were live at once and both scored a healthy run as a failure.  ``_valid``
    tested ``parse_status.startswith("OK")`` -- the enum's *name*, where the row carries its
    value ``"ok"`` -- so 117 of 144 draws that parsed cleanly were counted as format failures.
    And ``generated_tokens`` is not a column at all (``DecisionRow`` has ``thinking_tokens``,
    ``final_tokens``, ``discarded_span_tokens``), so ``getattr(row, "generated_tokens", 0)``
    returned 0.0 for every row and the smoke reported zero generated tokens on a run producing
    ~1,800 per draw.

    Both were invisible to the existing guards: the config-consumer walk sees a knob being
    read, not whether the name being read exists on the other side of the writer.
    """
    import ast

    from agents_scaling.nextdyn import gates, tables

    columns = {f.name for f in dataclasses.fields(tables.DecisionRow)}
    tree = ast.parse(Path(gates.__file__).read_text(encoding="utf-8"))
    read: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2):
            continue
        target, name = node.args[0], node.args[1]
        if not (isinstance(target, ast.Name) and target.id == "row"):
            continue
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            read.add(name.value)
    assert read, "the scan found no row reads at all; it has stopped testing anything"
    missing = sorted(read - columns)
    assert not missing, (f"gates.py reads {missing} off a decisions row, and DecisionRow has no "
                         f"such column; a default-valued getattr made that silent")


def test_the_gate_agrees_with_the_parser_about_which_statuses_are_valid():
    """One declaration of "gradeable" (R1), not one per reader."""
    from agents_scaling.nextdyn import gates, schema

    for status in schema.ParseStatus:
        row = type("Row", (), {"parse_status": status.value})()
        assert gates._valid(row) is (status in schema.VALID_STATUSES), status
        # and the enum member itself reads the same as its value, so neither spelling drifts
        assert gates._valid(type("Row", (), {"parse_status": status})()) is gates._valid(row)
