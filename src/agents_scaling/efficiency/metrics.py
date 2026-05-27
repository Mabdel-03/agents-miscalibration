"""Kim et al. coordination metrics, computed vs a matched Single-Agent baseline.

Symbol meanings CONFIRMED against arXiv:2512.08296 v1 methodology:
  * T = conversational reasoning turns (reasoning-response exchanges), NOT tokens.
  * E = error rate (failure probability), NOT a token/compute cost.
  * S = success rate.

Formulas:
  * Coordination efficiency  Ec = S / (T / T_SAS)      (success per relative turn cost)
  * Error amplification      Ae = E_MAS / E_SAS         (ratio of failure probabilities)
  * Overhead                 O% = (T_MAS - T_SAS)/T_SAS * 100
  * Message density          c  = inter-agent messages per reasoning turn
  * Redundancy               R  = mean pairwise cosine similarity of agent output embeddings

We also log token-based efficiency variants in parallel (the result schema records both
turns and tokens), since open-weight token accounting is exact and useful.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EfficiencyMetrics:
    success_rate: float
    error_rate: float
    mean_turns: float
    mean_messages: float
    message_density: float            # c
    coordination_efficiency: float    # Ec
    error_amplification: float        # Ae
    overhead_pct: float               # O%
    redundancy: float = 0.0           # R (0 if embeddings unavailable / single agent)
    mean_total_tokens: float = 0.0    # token-based parallel accounting

    def to_dict(self) -> dict:
        return vars(self)


def coordination_metrics(
    *,
    success_rate: float,
    error_rate: float,
    mean_turns: float,
    mean_messages: float,
    sas_turns: float,
    sas_error_rate: float,
    redundancy: float = 0.0,
    mean_total_tokens: float = 0.0,
) -> EfficiencyMetrics:
    """Compute Kim's metrics given the MAS aggregates and the matched SAS baseline."""
    rel_turns = (mean_turns / sas_turns) if sas_turns > 0 else float("inf")
    ec = (success_rate / rel_turns) if rel_turns not in (0, float("inf")) else 0.0
    ae = (error_rate / sas_error_rate) if sas_error_rate > 0 else float("inf")
    overhead = ((mean_turns - sas_turns) / sas_turns * 100.0) if sas_turns > 0 else 0.0
    c = (mean_messages / mean_turns) if mean_turns > 0 else 0.0
    return EfficiencyMetrics(
        success_rate=success_rate,
        error_rate=error_rate,
        mean_turns=mean_turns,
        mean_messages=mean_messages,
        message_density=c,
        coordination_efficiency=ec,
        error_amplification=ae,
        overhead_pct=overhead,
        redundancy=redundancy,
        mean_total_tokens=mean_total_tokens,
    )
