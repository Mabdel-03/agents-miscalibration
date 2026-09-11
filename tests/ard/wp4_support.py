"""Shared helpers for the ARD36 WP4 (dispatch) tests.

Builds the smallest run root the dispatch path needs — a public item table with a handful of
synthetic tasks, an optional protected gold pair, a registered fake endpoint on the real
``ard36-8B`` serving profile — so each test states only what it is about.  Deliberately not
a ``conftest.py``: ``tests/ard/conftest.py`` would be shared ground for every ARD work
package and this file belongs to WP4 alone.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("HF_HOME", "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agents_scaling.ard import contract as K  # noqa: E402
from agents_scaling.ard import items as ard_items  # noqa: E402
from agents_scaling.ard.items import ArdTask  # noqa: E402
from agents_scaling.study.config import StudyConfig, load_config  # noqa: E402
from tests.study.fake_vllm import FakeTokenizer, FakeVllmServer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ARD_CONFIG_PATH = REPO_ROOT / "configs" / "ard36.yaml"
RUN_ROOT_SUBDIRS = ("servers", "cells", "requests", "data", "eval", "freeze", "logs")


def ard_config() -> StudyConfig:
    return load_config(ARD_CONFIG_PATH)


def make_run_root(tmp_path: Path, name: str = "run_root") -> Path:
    root = tmp_path / name
    for subdir in RUN_ROOT_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def synthetic_task(index: int, *, family: str = "math500", split: str = "test") -> ArdTask:
    is_math = family == K.Family.MATH500.value
    options = [] if is_math else [f"option {index}-{letter}" for letter in range(K.MMLU_REQUIRED_OPTIONS)]
    return ArdTask(
        item_id=f"{family}-{index:04d}",
        family=family,
        dataset_revision="0" * 40,
        source_id=f"{family}:src-{index}",
        split=split,
        group_id=f"{family}:src-{index}",
        prompt_hash="0" * 64,
        option_permutation=json.dumps(list(range(len(options)))),
        gold_location=f"protected/{family}_gold.jsonl#{family}-{index:04d}",
        grader_hash="0" * 64,
        subject="Algebra" if is_math else "biology",
        declared_difficulty_metadata="3" if is_math else "",
        n_options=len(options) if options else 0,
        answer_format="exactMatch" if is_math else "multipleChoice",
        stratum=f"{family}:s",
        rank=index,
        task_text=f"Task {index}: compute {index} + 1.",
        options_json=json.dumps(options),
    )


def synthetic_tasks(
    n_per_family: int = 2, splits: tuple[str, ...] = ("test",)
) -> list[ArdTask]:
    tasks: list[ArdTask] = []
    index = 0
    for split in splits:
        for family in (K.Family.MATH500.value, K.Family.MMLUPRO.value):
            for _ in range(n_per_family):
                tasks.append(synthetic_task(index, family=family, split=split))
                index += 1
    return tasks


def write_items(run_root: Path, tasks: list[ArdTask]) -> Path:
    """Write ``data/public/items.parquet`` directly (no dataset stack, no HF read)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [task.to_row() for task in tasks]
    table = pa.Table.from_pylist([{key: row[key] for key in ard_items.PUBLIC_COLUMNS}
                                  for row in rows])
    path = ard_items.items_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_gold(run_root: Path, tasks: list[ArdTask]) -> Path:
    """Write the protected gold pair with the 0700/0600 permissions the store demands."""
    directory = ard_items.protected_dir(run_root)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, ard_items.PROTECTED_DIR_MODE)
    math_rows, mmlu_rows = [], []
    for task in tasks:
        if task.family == K.Family.MATH500.value:
            math_rows.append({"item_id": task.item_id, "answer": str(task.rank + 1)})
        else:
            mmlu_rows.append({"item_id": task.item_id, "permuted_key": "C",
                              "options": task.options})
    for name, rows in ((ard_items.MATH_GOLD_FILE, math_rows), (ard_items.MMLU_GOLD_FILE, mmlu_rows)):
        path = directory / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        os.chmod(path, ard_items.PROTECTED_FILE_MODE)
    return directory


@contextmanager
def fake_server(run_root: Path, profile: str = K.PROFILE_SMALL, **kwargs: Any) -> Iterator[FakeVllmServer]:
    """A started, registered fake endpoint on an ARD36 serving profile."""
    from agents_scaling.serving.profiles import get_serving_profile

    spec = get_serving_profile(profile)
    kwargs.setdefault("served_model_name", spec.served_model_name)
    kwargs.setdefault("max_model_len", spec.max_model_len)
    kwargs.setdefault("tokenizer", FakeTokenizer())
    server = FakeVllmServer(**kwargs).start()
    server.register(run_root, profile)
    try:
        yield server
    finally:
        server.stop()


def runner_kwargs(**overrides: Any) -> dict[str, Any]:
    """Injection set for :class:`agents_scaling.ard.runner.ArdCellRunner` in tests.

    The tokenizer is a :class:`FakeTokenizer` (its ids are a pure function of the text, so
    any instance agrees with the fake server's) and the client is the real ``ArdChatClient``.
    """
    kwargs: dict[str, Any] = {
        "cfg": ard_config(),
        "tokenizer": FakeTokenizer(),
        "wait_timeout_s": 20.0,
        "run_id": "ard36-wp4-test",
    }
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------- #
# A genuine minimal controller artifact (WP4 fixtures + the offline dry run)
# --------------------------------------------------------------------------- #
#
# WP4 predates WP5, so its fixtures used to hand a stub payload to
# ``freeze_register``.  Once ard/controller.py landed, ``load_controller`` rightly
# refused it ("is a None artifact, not ARD36_CONTROLLER").  The fix is not to relax
# the validator: it is to have the fixture write a REAL artifact, so the GATE path
# these tests exercise is the same one the run will take.  The models are fitted on
# a tiny separable problem rather than hand-stuffed, so the artifact is valid by
# construction rather than by our say-so.

def minimal_controller_payload(feature_names=None):
    """A valid ``FrozenController`` payload with fitted p_private / p_peer heads."""
    import numpy as np

    from agents_scaling.ard import controller as CTRL
    from agents_scaling.study.neural import readout as R

    # The real frozen v1 names, not invented ones: the episode builds rows against
    # FEATURES_V1 and a head naming an absent column trips the feature firewall.
    from agents_scaling.ard import features as FE

    names = list(feature_names or FE.FEATURES_V1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, len(names)))
    heads = {}
    for head, sign in ((CTRL.HEAD_PRIVATE, +1.0), (CTRL.HEAD_PEER, -1.0)):
        y = (sign * X[:, 0] > 0).astype(float)
        model = R.LogisticModel(penalty=1.0, feature_names=list(names))
        model.fit(X, y)
        heads[head] = model
    return CTRL.FrozenController(
        feature_names=tuple(names),
        models=heads,
        meta={"provenance": "wp4_support.minimal_controller_payload",
              "note": "bring-up artifact for fixtures and the offline dry run; "
                      "never produced by the fitting path"},
    ).payload()


def freeze_minimal_controller(run_root):
    """Write the artifact through the one register writer and return its sha."""
    from agents_scaling.ard import runner as ard_runner

    return ard_runner.freeze_register(run_root, "controller", minimal_controller_payload())
