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
* ``ard36-8B`` / ``ard36-32B`` -- the Agents Round Dynamics 36h study's 16K profiles
  (plan §4.2); see the ARD36 block below for the context and port-base reasoning.
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


# --------------------------------------------------------------------------- ARD36
# Agents Round Dynamics 36h (plan §4.2): two dedicated profiles so the study never shares
# a replica pool -- or a port -- with a study_v4 fleet.
#
# ``max_model_len`` is ARD36_MAX_MODEL_LEN = 16,384: the frozen prompt cap (~3.3k) plus the
# largest output allowance (1 + 4,096 + 1 + 256 = 4,354) plus the 128-token reserve, with
# margin.  Not 40,960 (2.5x the KV waste) and not 8,192 (the 4,096-thinking arms would not
# fit).  The names are chosen for their crc32 port bases (8925 / 8992), which are >= 16
# clear of every existing profile -- ``32B-ard36`` would land on 8830, four ports from
# ``32B-long``'s 8826, a silent SO_REUSEPORT split if a study_v4 replica were live.
# ``launch_server.assert_port_base_clearance`` re-checks this at import of the launcher.
# Every value below comes from the frozen ARD36 contract; nothing is re-declared here.
# ``agents_scaling.ard.contract`` is a leaf module (stdlib only), so this import adds no
# dependency cycle and no runtime cost.
from agents_scaling.ard.contract import (  # noqa: E402
    CHECKPOINT_LARGE as _ARD36_LARGE_SIZE,
    DTYPE as ARD36_DTYPE,
    CHECKPOINT_SMALL as _ARD36_SMALL_SIZE,
    MAX_MODEL_LEN as ARD36_MAX_MODEL_LEN,
    MAX_NUM_SEQS_LARGE as _ARD36_SEQS_LARGE,
    MAX_NUM_SEQS_SMALL as _ARD36_SEQS_SMALL,
    PROFILE_LARGE as ARD36_LARGE_PROFILE,
    PROFILE_SMALL as ARD36_SMALL_PROFILE,
)

ARD36_PROFILES = (ARD36_SMALL_PROFILE, ARD36_LARGE_PROFILE)
#: vLLM ``--max-num-seqs`` per profile (spec §6.1 throughput; a serve-template placeholder).
ARD36_MAX_NUM_SEQS = {
    ARD36_SMALL_PROFILE: _ARD36_SEQS_SMALL,
    ARD36_LARGE_PROFILE: _ARD36_SEQS_LARGE,
}
# ARD36_DTYPE (spec §6.1: one precision throughout, never quantize only the larger model)
# is re-exported above straight from the contract.

for _ard_name, _ard_size, _ard_tp in (
    (ARD36_SMALL_PROFILE, _ARD36_SMALL_SIZE, 1),
    (ARD36_LARGE_PROFILE, _ARD36_LARGE_SIZE, 2),
):
    SERVING_PROFILES[_ard_name] = ServingProfile(
        name=_ard_name,
        model_size=_ard_size,
        hf_id=REGISTRY[_ard_size].hf_id,
        tp_size=_ard_tp,
        max_model_len=ARD36_MAX_MODEL_LEN,
        # The OpenAI model name stays the scientific model size, so a runtime layout
        # choice never leaks into a result row.
        served_model_name=_ard_size,
    )


# --------------------------------------------------------------------------- nextdyn
# task_conditioned_dynamics_v1 (plan D5): two dedicated 32K profiles so both compared
# models are served at the SAME window (spec 5.5: never quietly give only 32B a smaller
# window).  ``max_model_len`` is 32,768: worst-case revision prompt (~3.1k) + 1 + 2,048
# thinking + 1 + 1,024 final + 256 reserve is ~6.5k, so the window never binds; it is the
# declared served window, not a KV budget.  32B is TP2 everywhere: TP1 at 32K is the
# documented KV OOM, and ``tp`` enters ``engine_digest`` so one profile has one digest.
# The names are chosen for their crc32 port bases (nd1-8B -> 8884, nd1-32B -> 8083), each
# >= 25 clear of every existing profile; ``nextdyn-32B`` would have landed on 8301, one
# port from ``0.6B`` (8302).  ``--max-num-seqs`` is NOT a profile property here: it varies
# by GPU type and is passed per replica by ``nextdyn.fleet`` (``compute.yaml``).
NEXTDYN_MAX_MODEL_LEN = 32768
NEXTDYN_SMALL_PROFILE = "nd1-8B"
NEXTDYN_LARGE_PROFILE = "nd1-32B"
NEXTDYN_PROFILES = (NEXTDYN_SMALL_PROFILE, NEXTDYN_LARGE_PROFILE)

for _nd_name, _nd_size, _nd_tp in (
    (NEXTDYN_SMALL_PROFILE, "8B", 1),
    (NEXTDYN_LARGE_PROFILE, "32B", 2),
):
    SERVING_PROFILES[_nd_name] = ServingProfile(
        name=_nd_name,
        model_size=_nd_size,
        hf_id=REGISTRY[_nd_size].hf_id,
        tp_size=_nd_tp,
        max_model_len=NEXTDYN_MAX_MODEL_LEN,
        # The OpenAI model name stays the scientific model size (same rule as ARD36).
        served_model_name=_nd_size,
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
