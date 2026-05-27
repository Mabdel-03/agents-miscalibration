"""Benchmark loaders -> normalized ``Question`` lists.

All loaders are deterministic given a seed (option order is shuffled with a per-question
RNG so the correct-answer position is not always 'A'). Datasets are pulled via the HF
``datasets`` library; weights/cache live under ``HF_HOME`` on scratch.
"""

from __future__ import annotations

import random

from agents_scaling.benchmarks.schema import AnswerType, Question

# Lazy import so `import agents_scaling` works without `datasets` installed (e.g. in CI).
def _load_hf(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _mcq(qid: str, bench: str, stem: str, options: list[str], correct_idx: int, rng: random.Random) -> Question:
    """Build an MCQ Question with options shuffled deterministically."""
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    new_correct = order.index(correct_idx)
    return Question(
        qid=qid,
        benchmark=bench,
        prompt_stem=stem,
        options=shuffled,
        answer_key=chr(ord("A") + new_correct),
        answer_type=AnswerType.MCQ,
    )


def load_gpqa(n: int | None = None, seed: int = 0, subset: str = "gpqa_diamond") -> list[Question]:
    ds = _load_hf("Idavidrein/gpqa", subset)["train"]
    out: list[Question] = []
    for i, row in enumerate(ds):
        rng = random.Random(f"{seed}-{i}")
        opts = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        out.append(_mcq(f"gpqa-{i}", "gpqa", row["Question"], opts, correct_idx=0, rng=rng))
        if n is not None and len(out) >= n:
            break
    return out


def load_mmlu_pro(n: int | None = None, seed: int = 0) -> list[Question]:
    ds = _load_hf("TIGER-Lab/MMLU-Pro")["test"]
    out: list[Question] = []
    for i, row in enumerate(ds):
        rng = random.Random(f"{seed}-{i}")
        opts = list(row["options"])
        correct_idx = row["answer_index"]
        out.append(_mcq(f"mmlupro-{i}", "mmlu_pro", row["question"], opts, correct_idx, rng))
        if n is not None and len(out) >= n:
            break
    return out


def load_truthfulqa(n: int | None = None, seed: int = 0) -> list[Question]:
    """TruthfulQA MC1: exactly one correct target among several. Calibration-native."""
    # Use the namespaced repo id; the bare "truthful_qa" alias is rejected by newer
    # huggingface_hub (requires 'namespace/name').
    ds = _load_hf("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out: list[Question] = []
    for i, row in enumerate(ds):
        rng = random.Random(f"{seed}-{i}")
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]  # 1 marks the single correct choice
        correct_idx = labels.index(1)
        out.append(_mcq(f"tqa-{i}", "truthfulqa", row["question"], choices, correct_idx, rng))
        if n is not None and len(out) >= n:
            break
    return out


def load_math(n: int | None = None, seed: int = 0) -> list[Question]:
    """MATH-500: free-form numeric, graded by final-answer match.

    Uses HuggingFaceH4/MATH-500 (parquet, no dataset script — the legacy
    hendrycks/competition_math script repo is no longer loadable). It has a clean
    ``answer`` column (the gold final answer); we fall back to the boxed value in
    ``solution`` if ``answer`` is missing.
    """
    from agents_scaling.benchmarks.grading import extract_boxed

    ds = _load_hf("HuggingFaceH4/MATH-500")["test"]
    out: list[Question] = []
    for i, row in enumerate(ds):
        gold = row.get("answer") or extract_boxed(row.get("solution", "")) or ""
        out.append(
            Question(
                qid=f"math-{i}",
                benchmark="math",
                prompt_stem=row["problem"],
                answer_key=gold,
                answer_type=AnswerType.NUMERIC,
            )
        )
        if n is not None and len(out) >= n:
            break
    return out


LOADERS = {
    "gpqa": load_gpqa,
    "mmlu_pro": load_mmlu_pro,
    "truthfulqa": load_truthfulqa,
    "math": load_math,
}


def load_benchmark(name: str, n: int | None = None, seed: int = 0) -> list[Question]:
    if name not in LOADERS:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name](n=n, seed=seed)
