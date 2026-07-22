"""Serving profiles and deterministic cell-to-profile routing.

``ExperimentCell.model_size`` is a scientific configuration axis and must not change
when a model is served with a different runtime layout.  A :class:`ServingProfile`
therefore has a distinct registry key (``name``), while ``model_size`` and
``served_model_name`` retain the cell's model identity.

Every dense Qwen3 config exposes a native 40,960-token limit. Standard profiles stay
smaller for throughput; selective ``-long`` profiles preserve context-heavy treatments:

* ``32B``      -- the normal, one-GPU 16K profile;
* ``32B-long`` -- the two-GPU 40K profile used for coordinated intermediate/CoT cells.
* ``0.6B-long`` through ``14B-long`` -- one-GPU 40K profiles for large coordinated
  peer blocks at the highest reasoning/agent-count rungs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agents_scaling.config import ContextShareLevel, ExperimentCell, ReasoningLevel
from agents_scaling.models import REGISTRY, ModelSpec

STANDARD_32B_PROFILE = "32B"
LONG_32B_PROFILE = "32B-long"
LONG_CONTEXT_LIMIT = 40960
SELECTIVE_LONG_MODEL_SIZES = ("0.6B", "1.7B", "4B", "8B", "14B")
LONG_PROFILE_BY_MODEL = {
    model_size: f"{model_size}-long" for model_size in SELECTIVE_LONG_MODEL_SIZES
} | {"32B": LONG_32B_PROFILE}


@dataclass(frozen=True)
class ServingProfile:
    """One deployable layout for a model without changing its experiment identity."""

    name: str
    model_size: str
    hf_id: str
    tp_size: int
    max_model_len: int
    served_model_name: str

    @classmethod
    def from_model_spec(cls, spec: ModelSpec) -> "ServingProfile":
        return cls(
            name=spec.size,
            model_size=spec.size,
            hf_id=spec.hf_id,
            tp_size=spec.tp_size,
            max_model_len=spec.max_model_len,
            served_model_name=spec.size,
        )

    @property
    def registry_key(self) -> str:
        """Directory key used for endpoint discovery."""
        return self.name


SERVING_PROFILES: dict[str, ServingProfile] = {
    name: ServingProfile.from_model_spec(spec) for name, spec in REGISTRY.items()
}
for _model_size in SELECTIVE_LONG_MODEL_SIZES:
    _spec = REGISTRY[_model_size]
    _profile_name = LONG_PROFILE_BY_MODEL[_model_size]
    SERVING_PROFILES[_profile_name] = ServingProfile(
        name=_profile_name,
        model_size=_model_size,
        hf_id=_spec.hf_id,
        tp_size=1,
        max_model_len=LONG_CONTEXT_LIMIT,
        served_model_name=_model_size,
    )
SERVING_PROFILES[LONG_32B_PROFILE] = ServingProfile(
    name=LONG_32B_PROFILE,
    model_size="32B",
    hf_id=REGISTRY["32B"].hf_id,
    tp_size=2,
    # Qwen3-32B's native max_position_embeddings is 40,960.  The original 32K route
    # failed the exact n=7 dense-peer audit once the full 4,096-token answer allowance
    # was enforced; the native limit leaves the scientific treatment unchanged.
    max_model_len=LONG_CONTEXT_LIMIT,
    # Both profiles expose the same OpenAI model name.  This is what prevents a runtime
    # layout choice from leaking into result rows or changing the capacity treatment.
    served_model_name="32B",
)


def get_serving_profile(name: str) -> ServingProfile:
    """Return a named serving profile, raising a useful error for unknown names."""
    try:
        return SERVING_PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown serving profile {name!r}; known: {sorted(SERVING_PROFILES)}"
        ) from exc


def serving_profile_for_cell(cell: ExperimentCell) -> ServingProfile:
    """Route a cell to exactly one profile.

    Long-context routing is deliberately deterministic. Every coordinated 32B cell that
    shares intermediate results or CoT uses ``32B-long``. For 0.6B--14B, long routing is
    limited to the exact dense-peer risk boundary: n>=6 at b8192/unlimited, or n>=7 at
    b2048, with a context treatment that shares intermediate results or CoT.
    """
    if (
        cell.topology.shares_context
        and cell.context_share_level.rank
        >= ContextShareLevel.PLUS_INTERMEDIATE.rank
    ):
        if cell.model_size == "32B":
            return get_serving_profile(LONG_32B_PROFILE)
        high_reasoning = cell.reasoning_level in {
            ReasoningLevel.B8192,
            ReasoningLevel.UNLIMITED,
        }
        mid_reasoning_n7 = (
            cell.n_agents >= 7
            and cell.reasoning_level is ReasoningLevel.B2048
        )
        if (
            cell.model_size in SELECTIVE_LONG_MODEL_SIZES
            and ((cell.n_agents >= 6 and high_reasoning) or mid_reasoning_n7)
        ):
            return get_serving_profile(LONG_PROFILE_BY_MODEL[cell.model_size])
    return get_serving_profile(cell.model_size)


def serving_metadata(profile: ServingProfile | str) -> Mapping[str, Any]:
    """Metadata fields that make the effective serving capacity auditable.

    The helper returns a new immutable-by-convention mapping on each call so callers can
    safely merge it into a cell metadata record.
    """
    if isinstance(profile, str):
        profile = get_serving_profile(profile)
    return {
        "serving_profile": profile.name,
        "served_model_name": profile.served_model_name,
        "effective_context_limit": profile.max_model_len,
        "tensor_parallel_size": profile.tp_size,
    }
