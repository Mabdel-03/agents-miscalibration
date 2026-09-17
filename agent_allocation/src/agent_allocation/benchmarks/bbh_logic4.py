"""BBH-logic4 adapter (master spec 2.3): the BBH four-subtask zero-shot subset, not the official full BBH score.

Source: ``suzgunmirac/BIG-Bench-Hard`` at the pinned commit, exactly the four task files ``bbh/<subtask>.json``
(250 examples each). Licence: MIT. The repository's few-shot CoT prompt files are never loaded.

Sources are fetched once (``fetch_sources``, GitHub only) into the shared cache
``/orcd/data/tpoggio/001/mabdel03/.cache/agent_allocation/bbh/<commit>/`` (override with
``ALLOCATION_BBH_CACHE``) with a sha256 ``MANIFEST.json``. ``load_inventory`` never touches the network: it
prefers a verified run copy in ``paths.bbh_sources_dir``, otherwise verifies the shared cache and copies the files
into ``paths.bbh_sources_dir`` (write-once, mode 0700 parent). Every byte is checked against the sha256 pins in
this module, so a changed upstream or cache file can never quietly replace the frozen version.

Config keys consumed: ``benchmarks[id=bbh_logic4].source`` (pinned repo) and ``.subtasks`` (must equal
``contracts.BBH_SUBTASKS``).

Preprocessing ``bbh_logic4-prompt-v1``: ``PublicTask.prompt`` is the example's ``input`` string exactly;
``item_id = "bbh_logic4/<subtask>/<index>"`` with the 0-based index in the file's ``examples`` list.
Choice subtasks (``output_kind=OPTION``): ``allowed_choices`` are the option letters listed as ``(X) ...`` lines
after the input's final ``Options:`` line (they must run A, B, C... in order, with distinct option texts); gold is
``target`` ``"(X)"`` normalized to ``"X"``. ``boolean_expressions`` (``output_kind=BOOLEAN``): ``allowed_choices=("True", "False")``
and gold is ``target`` exactly. A target that does not normalize to an allowed choice is an engineering exclusion.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.benchmarks.text_manifest import (
    SourceIntegrityError,
    benchmark_config,
    sha256_bytes,
)
from agent_allocation.contracts import (
    BBH_SUBTASKS,
    EvalKind,
    InventoryItem,
    OutputKind,
    PrivateEvalSpec,
    PublicTask,
    canonical_hash,
)
from agent_allocation.paths import PRIVATE_DIR_MODE, RunPaths

BENCHMARK = "bbh_logic4"
SOURCE_REPO = "suzgunmirac/BIG-Bench-Hard"
SOURCE_COMMIT = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
LICENSE = "MIT (suzgunmirac/BIG-Bench-Hard, LICENSE blob f3b47ea6da7ab08fd87b0cac96fbee655a211364)"
PREPROCESSING_VERSION = "bbh_logic4-prompt-v1"
REPORT_LABEL = "BBH four-subtask zero-shot subset"
BOOLEAN_SUBTASK = "boolean_expressions"
BOOLEAN_CHOICES = ("True", "False")
EXAMPLES_PER_SUBTASK = 250
SHARED_CACHE_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/.cache/agent_allocation/bbh")
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/bbh/{subtask}.json"
MANIFEST_NAME = "MANIFEST.json"

#: Pinned file facts at ``SOURCE_COMMIT`` (git blob sha1 cross-checked against the GitHub tree API on 2026-09-17).
SOURCE_FILES: Mapping[str, Mapping[str, Any]] = {
    "logical_deduction_seven_objects": {
        "sha256": "2896c7e3482eea318dd37bcc370d24ec3cc91e8374c1784287b5dbd38a529e33",
        "git_blob_sha1": "2bc3766619b76d9f4b379782b8c25d3e022025e8", "bytes": 198404},
    "tracking_shuffled_objects_seven_objects": {
        "sha256": "b59fd05b850c37cf57cb514ae1e6975c00b1638ade2291722056b2af3c54084d",
        "git_blob_sha1": "145cd15c39004738898746b2c1ecbb6ee561c0bf", "bytes": 214906},
    "boolean_expressions": {
        "sha256": "ea6c754ec005e2d3f2d085d349a740f593b5764f32ea4638ffed4cfc0061b12a",
        "git_blob_sha1": "dd7b27ce8a8d36933abe1fd21a2cae6d94945e65", "bytes": 17172},
    "temporal_sequences": {
        "sha256": "92a1189faaa84aacc4c405aaa13f13f48c2992eb884772eedb622505252baef5",
        "git_blob_sha1": "d5a0ebdbd8227bbf6ae0073fc76d3b1dad5450f3", "bytes": 148176},
}

_OPTION_LINE = re.compile(r"^\(([A-Z])\) ", re.MULTILINE)
_TARGET_LETTER = re.compile(r"\(([A-Z])\)")


def source_record() -> dict[str, Any]:
    return {"benchmark": BENCHMARK, "repo": SOURCE_REPO, "commit": SOURCE_COMMIT, "license": LICENSE,
            "report_label": REPORT_LABEL, "preprocessing_version": PREPROCESSING_VERSION,
            "files": {f"bbh/{subtask}.json": dict(SOURCE_FILES[subtask]) for subtask in BBH_SUBTASKS}}


def shared_cache_dir() -> Path:
    return Path(os.environ.get("ALLOCATION_BBH_CACHE") or SHARED_CACHE_ROOT / SOURCE_COMMIT)


def file_name(subtask: str) -> str:
    return f"{subtask}.json"


# ---------------------------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------------------------

def _verified(data: bytes, subtask: str, where: Path) -> bytes:
    pinned = SOURCE_FILES[subtask]["sha256"]
    observed = sha256_bytes(data)
    if observed != pinned:
        raise SourceIntegrityError(f"{where}: sha256 {observed} != pinned {pinned} for {subtask}")
    return data


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def manifest_payload(fetched_at: str | None = None) -> dict[str, Any]:
    payload = source_record()
    payload["urls"] = {subtask: RAW_URL.format(repo=SOURCE_REPO, commit=SOURCE_COMMIT, subtask=subtask)
                       for subtask in BBH_SUBTASKS}
    if fetched_at:
        payload["fetched_at"] = fetched_at
    return payload


def fetch_sources(dest: Path | None = None, *, timeout_s: float = 60.0) -> Path:
    """Download the four pinned files from GitHub into ``dest`` (default: the shared cache), verifying sha256.

    The only network access in this module; never called by ``load_inventory`` or tests. Existing verified
    files are kept; an existing file with other bytes raises rather than being replaced.
    """
    target = Path(dest) if dest is not None else shared_cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    for subtask in BBH_SUBTASKS:
        path = target / file_name(subtask)
        if path.exists():
            _verified(path.read_bytes(), subtask, path)
            continue
        url = RAW_URL.format(repo=SOURCE_REPO, commit=SOURCE_COMMIT, subtask=subtask)
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - pinned GitHub URL
            data = response.read()
        _atomic_write(path, _verified(data, subtask, Path(url)), mode=0o644)
    manifest = target / MANIFEST_NAME
    if not manifest.exists():
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write(manifest, json.dumps(manifest_payload(stamp), indent=2, sort_keys=True).encode(), 0o644)
    return target


def _read_dir(directory: Path) -> dict[str, bytes] | None:
    """All four files from ``directory`` verified against the pins, or ``None`` if any file is absent."""
    paths = {subtask: directory / file_name(subtask) for subtask in BBH_SUBTASKS}
    if not all(path.is_file() for path in paths.values()):
        return None
    return {subtask: _verified(path.read_bytes(), subtask, path) for subtask, path in paths.items()}


def materialize_sources(paths: RunPaths) -> dict[str, bytes]:
    """Verified source bytes; ensures a verified write-once copy exists in ``paths.bbh_sources_dir``."""
    run_dir = paths.bbh_sources_dir
    run_copy = _read_dir(run_dir)                  # verified against the pins, or None when incomplete
    if run_copy is not None:
        if not (run_dir / MANIFEST_NAME).exists():
            _atomic_write(run_dir / MANIFEST_NAME, json.dumps(manifest_payload(), indent=2, sort_keys=True).encode())
        return run_copy
    cache = _read_dir(shared_cache_dir())
    if cache is None:
        raise SourceIntegrityError(
            f"BBH sources are not in {shared_cache_dir()} or {run_dir}; run "
            "agent_allocation.benchmarks.bbh_logic4.fetch_sources() once (GitHub only)")
    paths.private_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(paths.private_dir, PRIVATE_DIR_MODE)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(run_dir, PRIVATE_DIR_MODE)
    for subtask, data in cache.items():
        target = run_dir / file_name(subtask)
        if target.exists():                      # partial earlier copy: must be the pinned bytes
            _verified(target.read_bytes(), subtask, target)
            continue
        _atomic_write(target, data)
    manifest = run_dir / MANIFEST_NAME
    if not manifest.exists():
        _atomic_write(manifest, json.dumps(manifest_payload(), indent=2, sort_keys=True).encode())
    return cache


# ---------------------------------------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------------------------------------

def option_letters(prompt: str) -> tuple[str, ...]:
    """Letters of the ``(X) ...`` option lines after the final ``Options:`` line of a BBH input."""
    marker = prompt.rfind("\nOptions:\n")
    if marker < 0:
        return ()
    return tuple(_OPTION_LINE.findall(prompt[marker + len("\nOptions:\n"):]))


def _option_texts(prompt: str) -> list[str]:
    marker = prompt.rfind("\nOptions:\n")
    if marker < 0:
        return []
    return [line[len("(X) "):].strip() for line in prompt[marker + len("\nOptions:\n"):].split("\n")
            if _OPTION_LINE.match(line)]


def build_item(subtask: str, index: int, example: Mapping[str, Any], *,
               source_revision: str = SOURCE_COMMIT) -> InventoryItem:
    if subtask not in BBH_SUBTASKS:
        raise ValueError(f"{subtask!r} is not one of {BBH_SUBTASKS}")
    if set(example) != {"input", "target"}:
        raise SourceIntegrityError(f"bbh/{subtask}[{index}] has fields {sorted(example)}, expected input/target")
    prompt = example["input"]
    target = example["target"]
    item_id = f"{BENCHMARK}/{subtask}/{index}"
    reason: str | None = None
    gold: str | None
    if subtask == BOOLEAN_SUBTASK:
        output_kind, allowed, eval_kind = OutputKind.BOOLEAN, BOOLEAN_CHOICES, EvalKind.TASK_EXACT
        gold = target if target in BOOLEAN_CHOICES else None
        if gold is None:
            reason = "boolean_target_not_true_false"
    else:
        output_kind, eval_kind = OutputKind.OPTION, EvalKind.TASK_EXACT
        allowed = option_letters(prompt)
        match = _TARGET_LETTER.fullmatch(target)
        gold = match.group(1) if match else None
        expected = tuple(chr(ord("A") + offset) for offset in range(len(allowed)))
        texts = _option_texts(prompt)
        if len(allowed) < 2 or allowed != expected:
            reason = "option_letters_not_consecutive_from_A"
        elif len(set(texts)) != len(texts):
            reason = "duplicate_option_text"
        elif gold is None:
            reason = "target_not_an_option_label"
        elif gold not in allowed:
            reason = "target_letter_not_among_options"
    if reason is None and not prompt.strip():
        reason = "empty_input"
    public = PublicTask(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision, prompt=prompt,
                        output_kind=output_kind, allowed_choices=tuple(allowed), subtask=subtask)
    private_record = {
        "benchmark": BENCHMARK, "item_id": item_id, "source_revision": source_revision,
        "preprocessing_version": PREPROCESSING_VERSION, "eval_kind": eval_kind.value, "gold_answer": gold,
        "allowed_choices": list(allowed), "subtask": subtask, "index": index,
        "source_record": {"input": prompt, "target": target},
    }
    private = PrivateEvalSpec(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                              eval_kind=eval_kind, gold_answer=gold, private_record_ref=None,
                              private_hash=canonical_hash(private_record), subtask=subtask)
    return InventoryItem(public=public, private=private, strata=(("subtask", subtask),),
                         eligible=reason is None, exclusion_reason=reason)


def items_from_bytes(sources: Mapping[str, bytes]) -> list[InventoryItem]:
    items: list[InventoryItem] = []
    for subtask in BBH_SUBTASKS:
        document = json.loads(sources[subtask].decode("utf-8"))
        examples = document.get("examples")
        if not isinstance(examples, list) or len(examples) != EXAMPLES_PER_SUBTASK:
            raise SourceIntegrityError(f"bbh/{subtask}: expected {EXAMPLES_PER_SUBTASK} examples")
        items.extend(build_item(subtask, index, example) for index, example in enumerate(examples))
    return items


def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]:
    entry = benchmark_config(cfg, BENCHMARK, source=SOURCE_REPO)
    if tuple(entry.get("subtasks", ())) != BBH_SUBTASKS:
        raise ValueError(f"{BENCHMARK}: config subtasks {entry.get('subtasks')!r} != {BBH_SUBTASKS}")
    return items_from_bytes(materialize_sources(paths))
