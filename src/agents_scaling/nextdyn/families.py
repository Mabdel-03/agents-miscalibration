"""The task-family registry: one row per dataset panel (plan §2.1, §1.10; spec 5.1, 5.2).

ARD36 hard-coded two families in a closed ``Family`` enum and spread their facts over
``contract.py`` (option count), ``items_build.py`` (renderer), ``graders/__init__.py``
(dispatch) and the loaders, so "add a task family" meant editing four modules and the
answer instruction drifted between the root and revision templates.  Here a family is
**one frozen row**: where its rows come from, how a task is rendered, what the model is
told to emit, which canonicalizer groups answers, which grader resolves them, and what a
GRADE cell of that family costs (plan §1.10: "New task family = a ``FamilySpec`` entry, a
grader module, a ``datasets.lock.json`` entry"; nothing in the episode loop changes).

This module is **runtime-safe by construction** and is on the generation side of R7: it
holds no gold, imports no loader and imports no grader.  ``canonicalizer`` and ``grader``
are dotted ``"module:callable"`` *names* resolved lazily by the writer side
(:func:`resolve`), so importing a family never pulls ``agents_scaling.benchmarks``, the HF
``datasets`` library or :mod:`agents_scaling.nextdyn.graders.families` (the gold store)
into a generation worker.  ``graders.mc`` is the one grader module that is runtime-safe
(pure MC parsing, gold only ever arrives as an argument) and supplies the parametric
alphabet the renderer prints.

The six families (plan §2.1, D19):

===========================================  ==============================================
family                                       source
===========================================  ==============================================
``MATH500``                                  ``HuggingFaceH4/MATH-500`` test (pinned arrow)
``MATH_EXPANDED``                            ``EleutherAI/hendrycks_math`` test, 7 subjects,
                                             MATH-500 overlap removed by text hash (D19)
``MMLU_PRO``                                 ``TIGER-Lab/MMLU-Pro`` test, 10-option rows
``TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE``   ``truthfulqa/truthful_qa`` MC1 (4-13 options)
``MOD97``                                    generated (WP5 ``mod97``); no dataset rows
``BIGCODEBENCH``                             ``study_v4`` raw parquet; **E0 only**
===========================================  ==============================================

``MATH500`` and ``MATH_EXPANDED`` are two *panels* of the one cohort family ``MATH``
(``study.yaml`` ``cohorts.families``); :data:`TASK_FAMILY` is that map and the only place it
is written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn.errors import ConfigError
from agents_scaling.nextdyn.graders.mc import mc_letters

# --------------------------------------------------------------------------- vocabulary

ANSWER_FORMAT_EXACT: Final[str] = "exactMatch"
ANSWER_FORMAT_MC: Final[str] = "multipleChoice"
ANSWER_FORMATS: Final[tuple[str, ...]] = (ANSWER_FORMAT_EXACT, ANSWER_FORMAT_MC)

#: ``GradeResources.container``: a plain CPU grader, or one that must run inside the
#: BigCodeBench evaluator image (spec 7.5 / plan §2.6; the only apptainer family).
CONTAINER_NONE: Final[str] = "none"
CONTAINER_APPTAINER: Final[str] = "apptainer"
CONTAINERS: Final[tuple[str, ...]] = (CONTAINER_NONE, CONTAINER_APPTAINER)

#: Where a loader reads its rows from.  ``hf_arrow`` goes through
#: ``benchmarks.loaders._load_hf`` (revision-pinned, offline, shared cache);
#: ``hub_parquet`` reads snapshot parquet files with pyarrow (the hendrycks script loader is
#: dead, docs/OPERATIONS.md:231); ``local_parquet`` is a path on the data filesystem;
#: ``generated`` has no dataset at all (MOD97 problems come from a seeded generator).
READER_HF_ARROW: Final[str] = "hf_arrow"
READER_HUB_PARQUET: Final[str] = "hub_parquet"
READER_LOCAL_PARQUET: Final[str] = "local_parquet"
READER_GENERATED: Final[str] = "generated"
READERS: Final[tuple[str, ...]] = (READER_HF_ARROW, READER_HUB_PARQUET, READER_LOCAL_PARQUET,
                                   READER_GENERATED)


# --------------------------------------------------------------------------- the row parts


@dataclass(frozen=True)
class LoaderSpec:
    """Where one family's raw rows live, pinned (spec 5.1: no ``main``, no ``latest``).

    ``revision`` is the full commit for a hub source and ``None`` for a generated or local
    one; ``parts`` names the per-subject files of a multi-file snapshot (the seven
    ``hendrycks_math`` subjects) and is empty otherwise.  ``expected_rows`` is the row count
    the lock asserts before any filter.
    """

    reader: str
    repo_id: str | None
    config_name: str | None
    split: str | None
    revision: str | None
    expected_rows: int | None
    parts: tuple[str, ...] = ()
    local_path: str | None = None
    benchmark_key: str | None = None

    def __post_init__(self) -> None:
        if self.reader not in READERS:
            raise ConfigError(f"LoaderSpec.reader {self.reader!r} not in {READERS}")
        if self.reader in (READER_HF_ARROW, READER_HUB_PARQUET):
            if not self.repo_id or not self.revision or not self.split:
                raise ConfigError(f"{self.reader} loader needs repo_id, revision and split")
        object.__setattr__(self, "parts", tuple(self.parts))


@dataclass(frozen=True)
class Renderer:
    """Turns a question (plus displayed options) into the task statement an agent reads.

    Pure text: the renderer never sees gold, never names a protocol and never describes the
    graph (R5).  For a multiple-choice family the options are printed in **displayed**
    order with the parametric alphabet of :func:`agents_scaling.nextdyn.graders.mc.mc_letters`,
    so a 13-option TruthfulQA item prints ``A.``-``M.`` with no second alphabet anywhere.
    """

    answer_format: str
    max_options: int | None

    def __post_init__(self) -> None:
        if self.answer_format not in ANSWER_FORMATS:
            raise ConfigError(f"Renderer.answer_format {self.answer_format!r} not in {ANSWER_FORMATS}")
        if (self.answer_format == ANSWER_FORMAT_MC) != (self.max_options is not None):
            raise ConfigError("Renderer.max_options is set exactly for multipleChoice families")

    def __call__(self, question: str, options: Sequence[str] = ()) -> str:
        text = str(question).strip()
        if self.answer_format == ANSWER_FORMAT_EXACT:
            if options:
                raise ConfigError("an exactMatch family renders no options")
            return text
        displayed = [str(option).strip() for option in options]
        if not displayed or len(displayed) > int(self.max_options):
            raise ConfigError(f"a {self.answer_format} item needs 1..{self.max_options} options, "
                              f"got {len(displayed)}")
        letters = mc_letters(len(displayed))
        lines = [text, ""]
        lines.extend(f"{letters[i]}. {option}" for i, option in enumerate(displayed))
        return "\n".join(lines)


@dataclass(frozen=True)
class GradeResources:
    """The shape of one GRADE cell of this family (plan §2.5; consumed by ``kinds.cell_shape``).

    A MATH grader is sympy/lark on one core; MMLU-Pro and TruthfulQA are string comparisons
    and grade thousands of items in a cell; BigCodeBench must execute candidate code and
    therefore runs inside the pinned evaluator image (``container="apptainer"``, ``sif``
    resolved from ``legacy_sources.yaml``).  ``kinds.cell_shape`` indexes this mapping, so
    :meth:`as_mapping` is what a caller passes.
    """

    cpus: int
    mem: str
    time: str
    items_per_cell: int
    container: str
    sif: str | None

    def __post_init__(self) -> None:
        if self.container not in CONTAINERS:
            raise ConfigError(f"GradeResources.container {self.container!r} not in {CONTAINERS}")
        if (self.container == CONTAINER_APPTAINER) != bool(self.sif):
            raise ConfigError("GradeResources.sif is set exactly for an apptainer grader")
        if self.cpus < 1 or self.items_per_cell < 1:
            raise ConfigError("GradeResources needs cpus >= 1 and items_per_cell >= 1")

    def as_mapping(self) -> dict[str, Any]:
        return {"cpus": self.cpus, "mem": self.mem, "time": self.time,
                "items_per_cell": self.items_per_cell, "container": self.container, "sif": self.sif}


@dataclass(frozen=True)
class FamilySpec:
    """One task family, declared once (contract §2.3)."""

    name: str
    loader: LoaderSpec
    renderer: Renderer
    answer_instruction: str
    canonicalizer: str
    grader: str
    grade_resources: GradeResources

    def __post_init__(self) -> None:
        for field_name in ("canonicalizer", "grader"):
            value = getattr(self, field_name)
            module, sep, attr = str(value).partition(":")
            if not sep or not module.startswith("agents_scaling.nextdyn.") or not attr:
                raise ConfigError(f"FamilySpec.{field_name} must be "
                                  f"'agents_scaling.nextdyn.<module>:<callable>', got {value!r}")
        if not self.answer_instruction.strip():
            raise ConfigError(f"family {self.name!r} has no answer instruction")

    @property
    def answer_format(self) -> str:
        return self.renderer.answer_format


# --------------------------------------------------------------------------- answer instructions
# What the model is told to put in ``final_answer``.  One sentence per answer format, shared
# by every template that renders a task of that family (R1: the instruction is not re-typed in
# templates/).  No condition name, no graph description (R5).

INSTRUCTION_EXACT: Final[str] = (
    "Put only the final answer in the final_answer field, with no working and no units unless "
    "the question asks for them."
)
INSTRUCTION_MC: Final[str] = (
    "Put only the letter of the option you choose in the final_answer field (for example \"C\")."
)
INSTRUCTION_TRACE: Final[str] = (
    "Put only the final residue, as a decimal integer, in the final_answer field."
)
INSTRUCTION_CODE: Final[str] = (
    "Put the complete function implementation in the final_answer field, with no prose."
)

_G = "agents_scaling.nextdyn.graders."

MATH500: Final[str] = "MATH500"
MATH_EXPANDED: Final[str] = "MATH_EXPANDED"
MMLU_PRO: Final[str] = "MMLU_PRO"
TRUTHFULQA: Final[str] = "TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE"
MOD97: Final[str] = "MOD97"
BIGCODEBENCH: Final[str] = "BIGCODEBENCH"

#: The seven ``EleutherAI/hendrycks_math`` subjects cached at the pinned revision, in the
#: order their rows are read (plan D19: the realized subject set is recorded in the lock).
HENDRYCKS_SUBJECTS: Final[tuple[str, ...]] = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
    "prealgebra", "precalculus",
)
HENDRYCKS_REVISION: Final[str] = "21a5633873b6a120296cce3e2df9d5550074f4a3"
HENDRYCKS_TEST_FILE: Final[str] = "test-00000-of-00001.parquet"

#: Ten-option MMLU-Pro rows only (ARD36's eligibility, kept by D19) and the legacy MC1
#: option-count window (spec 5.1: 4 to 13 options).
MMLU_REQUIRED_OPTIONS: Final[int] = 10
TRUTHFULQA_MIN_OPTIONS: Final[int] = 4
TRUTHFULQA_MAX_OPTIONS: Final[int] = 13

_MATH_RENDERER = Renderer(answer_format=ANSWER_FORMAT_EXACT, max_options=None)
_MATH_GRADE = GradeResources(cpus=2, mem="4G", time="4:00:00", items_per_cell=64,
                             container=CONTAINER_NONE, sif=None)
_MC_GRADE = GradeResources(cpus=1, mem="2G", time="1:00:00", items_per_cell=512,
                           container=CONTAINER_NONE, sif=None)

FAMILIES: Final[Mapping[str, FamilySpec]] = {
    MATH500: FamilySpec(
        name=MATH500,
        loader=LoaderSpec(reader=READER_HF_ARROW, repo_id="HuggingFaceH4/MATH-500",
                          config_name=None, split="test",
                          revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", expected_rows=500,
                          benchmark_key="math"),
        renderer=_MATH_RENDERER,
        answer_instruction=INSTRUCTION_EXACT,
        canonicalizer=_G + "math:canonical_class",
        grader=_G + "math:grade",
        grade_resources=_MATH_GRADE,
    ),
    MATH_EXPANDED: FamilySpec(
        name=MATH_EXPANDED,
        loader=LoaderSpec(reader=READER_HUB_PARQUET, repo_id="EleutherAI/hendrycks_math",
                          config_name=None, split="test", revision=HENDRYCKS_REVISION,
                          expected_rows=5000, parts=HENDRYCKS_SUBJECTS),
        renderer=_MATH_RENDERER,
        answer_instruction=INSTRUCTION_EXACT,
        canonicalizer=_G + "math:canonical_class",
        grader=_G + "math:grade",
        grade_resources=_MATH_GRADE,
    ),
    MMLU_PRO: FamilySpec(
        name=MMLU_PRO,
        loader=LoaderSpec(reader=READER_HF_ARROW, repo_id="TIGER-Lab/MMLU-Pro", config_name=None,
                          split="test", revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
                          expected_rows=12032, benchmark_key="mmlu_pro"),
        renderer=Renderer(answer_format=ANSWER_FORMAT_MC, max_options=MMLU_REQUIRED_OPTIONS),
        answer_instruction=INSTRUCTION_MC,
        canonicalizer=_G + "mc:canonical_class",
        grader=_G + "mc:grade",
        grade_resources=_MC_GRADE,
    ),
    TRUTHFULQA: FamilySpec(
        name=TRUTHFULQA,
        loader=LoaderSpec(reader=READER_HF_ARROW, repo_id="truthfulqa/truthful_qa",
                          config_name="multiple_choice", split="validation",
                          revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490", expected_rows=817,
                          benchmark_key="truthfulqa"),
        renderer=Renderer(answer_format=ANSWER_FORMAT_MC, max_options=TRUTHFULQA_MAX_OPTIONS),
        answer_instruction=INSTRUCTION_MC,
        canonicalizer=_G + "mc:canonical_class",
        grader=_G + "mc:grade",
        grade_resources=_MC_GRADE,
    ),
    MOD97: FamilySpec(
        name=MOD97,
        loader=LoaderSpec(reader=READER_GENERATED, repo_id=None, config_name=None, split=None,
                          revision=None, expected_rows=None),
        renderer=_MATH_RENDERER,
        answer_instruction=INSTRUCTION_TRACE,
        canonicalizer=_G + "trace:canonical_class",
        grader=_G + "trace:grade",
        grade_resources=_MC_GRADE,
    ),
    BIGCODEBENCH: FamilySpec(
        name=BIGCODEBENCH,
        loader=LoaderSpec(reader=READER_LOCAL_PARQUET, repo_id="bigcode/bigcodebench",
                          config_name=None, split="v0.1.4", revision=None, expected_rows=1140,
                          local_path="study_v4:data/raw/bigcodebench_v0.1.4.parquet"),
        renderer=_MATH_RENDERER,
        answer_instruction=INSTRUCTION_CODE,
        canonicalizer=_G + "families:code_canonical_class",
        grader=_G + "families:code_grade",
        grade_resources=GradeResources(cpus=4, mem="8G", time="8:00:00", items_per_cell=32,
                                       container=CONTAINER_APPTAINER,
                                       sif="study_v4:data/raw/bigcodebench_evaluator.sif"),
    ),
}

#: dataset panel -> the ``study.yaml`` ``cohorts.families`` cohort it belongs to.  MATH-500 and
#: the broader hendrycks panel are two panels of one cohort family (plan D19); MOD97 and
#: BigCodeBench are not cohort families at all (E2b generates its own problems, BCB is E0 only).
TASK_FAMILY: Final[Mapping[str, str]] = {
    MATH500: "MATH",
    MATH_EXPANDED: "MATH",
    MMLU_PRO: "MMLU_PRO",
    TRUTHFULQA: "TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE",
    MOD97: "MOD97",
    BIGCODEBENCH: "BIGCODEBENCH",
}

#: Families that exist only for the legacy reanalysis and are never dispatched in E1-E3
#: (plan D11: BigCodeBench is E0 only).
E0_ONLY: Final[frozenset[str]] = frozenset({BIGCODEBENCH})

#: Families whose rows come from a pinned dataset and therefore need a ``datasets.lock.json``
#: entry (MOD97 is generated: its "lock" is the study seed and the mod97 generator version).
DATASET_FAMILIES: Final[tuple[str, ...]] = tuple(
    name for name, spec in FAMILIES.items() if spec.loader.reader != READER_GENERATED)

if set(TASK_FAMILY) != set(FAMILIES):  # pragma: no cover - import-time contract
    raise RuntimeError("every family needs exactly one cohort mapping")


def family(name: str) -> FamilySpec:
    try:
        return FAMILIES[name]
    except KeyError as exc:
        raise ConfigError(f"unknown family {name!r}; declared: {sorted(FAMILIES)}") from exc


def panels_of(task_family: str) -> tuple[str, ...]:
    """The dataset panels of one cohort family, in declaration order."""
    out = tuple(name for name, cohort in TASK_FAMILY.items() if cohort == task_family)
    if not out:
        raise ConfigError(f"no dataset panel serves cohort family {task_family!r}; "
                          f"declared cohorts: {sorted(set(TASK_FAMILY.values()))}")
    return out


def resolve(dotted: str):
    """Import ``"module:callable"`` lazily (writer side only: this is how a family reaches
    its grader without the generation import closure ever touching one, R7)."""
    import importlib

    module_name, _, attr = str(dotted).partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ConfigError(f"{dotted}: {module_name} has no attribute {attr!r}") from exc


__all__ = [
    "ANSWER_FORMATS", "ANSWER_FORMAT_EXACT", "ANSWER_FORMAT_MC", "BIGCODEBENCH", "CONTAINERS",
    "CONTAINER_APPTAINER", "CONTAINER_NONE", "DATASET_FAMILIES", "E0_ONLY", "FAMILIES",
    "FamilySpec", "GradeResources", "HENDRYCKS_REVISION", "HENDRYCKS_SUBJECTS",
    "HENDRYCKS_TEST_FILE", "INSTRUCTION_CODE", "INSTRUCTION_EXACT", "INSTRUCTION_MC",
    "INSTRUCTION_TRACE", "LoaderSpec", "MATH500", "MATH_EXPANDED", "MMLU_PRO",
    "MMLU_REQUIRED_OPTIONS", "MOD97", "READERS", "READER_GENERATED", "READER_HF_ARROW",
    "READER_HUB_PARQUET", "READER_LOCAL_PARQUET", "Renderer", "TASK_FAMILY", "TRUTHFULQA",
    "TRUTHFULQA_MAX_OPTIONS", "TRUTHFULQA_MIN_OPTIONS", "family", "panels_of", "resolve",
]
