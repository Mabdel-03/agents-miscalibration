"""Prompt-quality judging remains compatible with the exact generation contract."""

from types import SimpleNamespace

from agents_scaling.prompts.prompt_quality import llm_judge_quality
from agents_scaling.prompts import system_prompts
from agents_scaling.serving.client import ANSWER_GENERATION_TOKEN_ALLOWANCE


def test_llm_judge_uses_deterministic_registered_chat_allowance():
    observed = {}

    class Judge:
        def chat(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(
                text='{"score": 87, "rationale": "Clear and specific."}'
            )

    assert llm_judge_quality("Answer carefully.", Judge()) == 87.0
    assert observed["seed"] == 0
    assert observed["max_tokens"] == ANSWER_GENERATION_TOKEN_ALLOWANCE
    assert observed["temperature"] == 0.0
    assert observed["capture_logprobs"] is False


def test_explicit_release_prompt_root_does_not_use_package_layout(
    tmp_path, monkeypatch
):
    prompt_root = tmp_path / "immutable-release" / "configs" / "prompts"
    prompt_root.mkdir(parents=True)
    for level in range(system_prompts.N_LEVELS):
        (prompt_root / f"level{level}.txt").write_text(
            f"frozen release prompt {level}\n", encoding="utf-8"
        )
    monkeypatch.setattr(system_prompts, "_PROMPT_DIR", tmp_path / "site-packages")
    system_prompts._get_prompt.cache_clear()

    assert system_prompts.get_prompt(2, prompt_root=prompt_root) == (
        "frozen release prompt 2"
    )
    assert system_prompts.token_count(2, prompt_root=prompt_root) == 4
