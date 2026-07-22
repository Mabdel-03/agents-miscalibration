"""Exact chat-template context accounting and capacity preflight.

The server rejects a request when prompt tokens plus the requested output exceed the
served context window.  Estimating from words or raw text is unsafe because it misses
the model's chat-template control tokens.  These helpers call the *same Hugging Face
tokenizer chat template* with generation prompting and Qwen's ``enable_thinking`` flag,
then reject an oversized request before any HTTP submission.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Protocol, Sequence

from agents_scaling.serving.profiles import ServingProfile, get_serving_profile

CONTEXT_RESERVE_TOKENS = 128


class ChatTemplateTokenizer(Protocol):
    """Small tokenizer protocol, also convenient for deterministic unit tests."""

    def apply_chat_template(self, conversation: Sequence[Mapping[str, str]], **kwargs: Any) -> Any:
        ...


def _flat_token_ids(value: Any, *, source: str) -> tuple[int, ...]:
    """Normalize one unbatched tokenizer result into an immutable ID sequence."""
    if isinstance(value, Mapping):
        try:
            value = value["input_ids"]
        except KeyError as exc:
            raise TypeError(f"{source} tokenizer mapping has no input_ids") from exc
    if isinstance(value, str):
        raise TypeError(f"{source} tokenizer returned text instead of token ids")
    try:
        ids = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{source} tokenizer did not return a token-id sequence") from exc
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0 for token_id in ids):
        raise TypeError(f"{source} tokenizer returned invalid token ids")
    return ids


class CompletionPromptTokenizer(Protocol):
    """Tokenizer surface used by the raw OpenAI completions endpoint."""

    def encode(self, text: str, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class ContextPreflight:
    """Auditable token accounting for one prospective chat request."""

    profile_name: str
    prompt_tokens: int
    requested_output_tokens: int
    reserve_tokens: int
    served_context: int

    @property
    def required_tokens(self) -> int:
        return self.prompt_tokens + self.requested_output_tokens + self.reserve_tokens

    @property
    def fits(self) -> bool:
        return self.required_tokens <= self.served_context

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "serving_profile": self.profile_name,
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "context_reserve_tokens": self.reserve_tokens,
            "required_context_tokens": self.required_tokens,
            "effective_context_limit": self.served_context,
            "context_preflight_fits": self.fits,
        }


class ContextCapacityError(ValueError):
    """A deterministic request/profile mismatch; retrying another endpoint cannot help."""

    def __init__(self, preflight: ContextPreflight):
        self.preflight = preflight
        super().__init__(
            "request exceeds serving profile capacity: "
            f"profile={preflight.profile_name!r}, prompt={preflight.prompt_tokens}, "
            f"requested_output={preflight.requested_output_tokens}, "
            f"reserve={preflight.reserve_tokens}, required={preflight.required_tokens}, "
            f"served_context={preflight.served_context}"
        )


class TokenizerInitializationError(RuntimeError):
    """Exact context accounting cannot initialize its registered tokenizer.

    This is a harness/configuration failure, not an endpoint failure. Retrying the same
    cell cannot repair a missing or incompatible local tokenizer environment.
    """

    def __init__(self, profile_name: str, detail: str) -> None:
        self.profile_name = profile_name
        self.detail = detail
        super().__init__(
            "cannot initialize exact context tokenizer for serving profile "
            f"{profile_name!r}: {detail}"
        )


def build_chat_messages(system: str, user: str) -> list[dict[str, str]]:
    """Build exactly the message list sent by :meth:`LogprobClient.chat`."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def rendered_chat_token_ids(
    tokenizer: ChatTemplateTokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    enable_thinking: bool,
) -> tuple[int, ...]:
    """Return the exact rendered prompt IDs, including generation/control tokens.

    ``tokenize=True`` is essential: rendering to a string and re-encoding can differ for
    templates that deliberately manage special tokens.  ``add_generation_prompt=True``
    matches an OpenAI-compatible chat-completions request.  The Qwen template receives
    the same thinking toggle that the client places in ``chat_template_kwargs``.
    """
    token_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        # transformers 5.x otherwise defaults to BatchEncoding; force one flat sequence
        # so len(...) is the token count rather than the number of mapping keys.
        return_dict=False,
    )
    return _flat_token_ids(token_ids, source="chat")


def rendered_chat_token_count(
    tokenizer: ChatTemplateTokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    enable_thinking: bool,
) -> int:
    """Count the exact rendered prompt, including generation/control tokens."""
    return len(
        rendered_chat_token_ids(
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
        )
    )


def raw_completion_prompt_token_count(
    tokenizer: CompletionPromptTokenizer, prompt: str
) -> int:
    """Count the exact raw prompt tokenization used by ``/v1/completions``.

    vLLM's OpenAI completion request adds the model's normal special tokens by default;
    passing ``add_special_tokens=True`` explicitly keeps the local preflight aligned with
    that endpoint contract.
    """
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    return len(_flat_token_ids(token_ids, source="completion"))


def preflight_chat_context(
    tokenizer: ChatTemplateTokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    requested_output_tokens: int,
    served_context: int,
    profile_name: str,
    enable_thinking: bool,
    reserve_tokens: int = CONTEXT_RESERVE_TOKENS,
) -> ContextPreflight:
    """Validate ``rendered input + output + reserve <= served context``.

    Raises :class:`ContextCapacityError` before a request can reach an endpoint.  The
    exception intentionally is not an OpenAI connection error, so endpoint-fallback code
    cannot misclassify a deterministic configuration problem as server failure.
    """
    if requested_output_tokens < 0:
        raise ValueError("requested_output_tokens must be non-negative")
    if served_context <= 0:
        raise ValueError("served_context must be positive")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens must be non-negative")

    result = ContextPreflight(
        profile_name=profile_name,
        prompt_tokens=rendered_chat_token_count(
            tokenizer, messages, enable_thinking=enable_thinking
        ),
        requested_output_tokens=requested_output_tokens,
        reserve_tokens=reserve_tokens,
        served_context=served_context,
    )
    if not result.fits:
        raise ContextCapacityError(result)
    return result


def preflight_completion_context(
    tokenizer: CompletionPromptTokenizer,
    prompt: str,
    *,
    requested_output_tokens: int,
    served_context: int,
    profile_name: str,
    reserve_tokens: int = CONTEXT_RESERVE_TOKENS,
) -> ContextPreflight:
    """Validate a literal completion prompt before HTTP submission."""
    if requested_output_tokens < 0:
        raise ValueError("requested_output_tokens must be non-negative")
    if served_context <= 0:
        raise ValueError("served_context must be positive")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens must be non-negative")
    result = ContextPreflight(
        profile_name=profile_name,
        prompt_tokens=raw_completion_prompt_token_count(tokenizer, prompt),
        requested_output_tokens=requested_output_tokens,
        reserve_tokens=reserve_tokens,
        served_context=served_context,
    )
    if not result.fits:
        raise ContextCapacityError(result)
    return result


_TOKENIZER_INITIALIZATION_LOCK = threading.RLock()


@lru_cache(maxsize=None)
def _tokenizer_for_profile_cached(
    profile_name: str,
    model_contract_path: str | None,
    expected_model_contract_sha256: str | None,
) -> ChatTemplateTokenizer:
    """Unserialized cache body; callers enter through :func:`tokenizer_for_profile`."""
    profile = get_serving_profile(profile_name)
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts(
        model_contract_path, expected_sha256=expected_model_contract_sha256
    )
    identity = contracts.for_size(profile.model_size)
    if identity.hf_id != profile.hf_id:
        raise TokenizerInitializationError(
            profile_name,
            "serving profile HF id does not match the frozen model/tokenizer contract",
        )
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # lazy module imports can surface more than ImportError
        raise TokenizerInitializationError(
            profile_name,
            f"cannot import transformers.AutoTokenizer ({type(exc).__name__}: {exc})",
        ) from exc
    # A registered vLLM server has already loaded this same model into the shared HF_HOME.
    # Stay cache-only so hundreds of CPU clients do not stampede the Hub or become
    # dependent on login-node network/rate-limit availability during a sweep.
    try:
        return AutoTokenizer.from_pretrained(
            identity.tokenizer_id,
            revision=identity.tokenizer_revision,
            local_files_only=True,
        )
    except Exception as fast_error:
        # Some shared snapshots were produced by a newer Rust `tokenizers` schema than
        # the lightweight harness environment can deserialize.  Qwen ships vocab.json +
        # merges.txt as well, so the slow tokenizer is an exact cache-only fallback with
        # the same tokenizer_config/chat_template and token vocabulary.
        try:
            return AutoTokenizer.from_pretrained(
                identity.tokenizer_id,
                revision=identity.tokenizer_revision,
                local_files_only=True,
                use_fast=False,
            )
        except Exception as slow_error:
            raise TokenizerInitializationError(
                profile_name,
                "cached fast and slow tokenizer loads both failed; "
                f"fast={type(fast_error).__name__}: {fast_error}; "
                f"slow={type(slow_error).__name__}: {slow_error}",
            ) from slow_error


def tokenizer_for_profile(
    profile_name: str,
    *,
    model_contract_path: str | None = None,
    expected_model_contract_sha256: str | None = None,
) -> ChatTemplateTokenizer:
    """Load exactly one process-cached tokenizer per serving profile.

    ``lru_cache`` preserves cache integrity under threads but allows duplicate calls on a
    concurrent first miss. Transformers' lazy import is unsafe under that pattern in the
    topology thread pools, so both cache lookup and initialization are serialized.
    """
    with _TOKENIZER_INITIALIZATION_LOCK:
        return _tokenizer_for_profile_cached(
            profile_name,
            model_contract_path,
            expected_model_contract_sha256,
        )


def _clear_tokenizer_cache() -> None:
    """Testing/operator hook matching the former ``lru_cache`` public surface."""
    with _TOKENIZER_INITIALIZATION_LOCK:
        _tokenizer_for_profile_cached.cache_clear()


def _tokenizer_cache_info() -> Any:
    with _TOKENIZER_INITIALIZATION_LOCK:
        return _tokenizer_for_profile_cached.cache_info()


# Preserve the useful cache controls callers/tests already use while making clear safe.
tokenizer_for_profile.cache_clear = _clear_tokenizer_cache  # type: ignore[attr-defined]
tokenizer_for_profile.cache_info = _tokenizer_cache_info  # type: ignore[attr-defined]


def preflight_profile_chat(
    profile: ServingProfile | str,
    *,
    system: str,
    user: str,
    requested_output_tokens: int,
    enable_thinking: bool,
    tokenizer: ChatTemplateTokenizer | None = None,
    reserve_tokens: int = CONTEXT_RESERVE_TOKENS,
) -> ContextPreflight:
    """Convenience hook used by the HTTP client and experiment runner."""
    if isinstance(profile, str):
        profile = get_serving_profile(profile)
    tok = tokenizer if tokenizer is not None else tokenizer_for_profile(profile.name)
    return preflight_chat_context(
        tok,
        build_chat_messages(system, user),
        requested_output_tokens=requested_output_tokens,
        served_context=profile.max_model_len,
        profile_name=profile.name,
        enable_thinking=enable_thinking,
        reserve_tokens=reserve_tokens,
    )


def preflight_profile_completion(
    profile: ServingProfile | str,
    *,
    prompt: str,
    requested_output_tokens: int = 1,
    tokenizer: CompletionPromptTokenizer | None = None,
    reserve_tokens: int = CONTEXT_RESERVE_TOKENS,
) -> ContextPreflight:
    """Profile-aware raw-completion preflight used by option scoring."""
    if isinstance(profile, str):
        profile = get_serving_profile(profile)
    tok = tokenizer if tokenizer is not None else tokenizer_for_profile(profile.name)
    return preflight_completion_context(
        tok,  # type: ignore[arg-type] -- HF tokenizers expose both protocol surfaces.
        prompt,
        requested_output_tokens=requested_output_tokens,
        served_context=profile.max_model_len,
        profile_name=profile.name,
        reserve_tokens=reserve_tokens,
    )
