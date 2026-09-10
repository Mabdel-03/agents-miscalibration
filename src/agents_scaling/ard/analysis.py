"""ARD36 analysis: the three primary contrasts under Holm, the registered secondary families,
the within-item discrimination panel, and the seal that keeps labels out of the held-out
predictions (spec §15; plan §8, §5.8).

What this module is responsible for

1. **Primaries** (spec §15.2, frozen in ``FROZEN.yaml`` before any test-label join):
   ``GATE - PEER`` and ``GATE - RANDOM_PROTECT`` round-3 selected accuracy on the natural test
   panel, and ``PEER - PRIVATE`` current-round coverage change from root to round 3.  Exactly
   three contrasts, Holm-adjusted with ``n_primary = 3`` (``stats.holm`` asserts the count).
2. **Secondary/exploratory families**, each labelled and Holm-adjusted *inside* the family:
   challenge C0 levels, the receiver count x checkability x budget cube, the 8B/32B protocol
   differences, suffix utility reversal, calibration contrasts, the neural increment, and the
   registered decision-matched ``GATE - FRESH@20`` / ``PEER - FRESH@20`` family (plan §8:
   decision-matched at 20 logical decisions per item per arm, explicitly **not** FLOP-matched).
3. **Within-item discrimination** (spec §15.1): the bundle's ``within_item_auc`` per
   mixed-correctness item, averaged over items, with the eligible-item count reported.  An
   all-correct or all-wrong item returns ``None`` and increments an excluded counter, so a
   panel row's AUC denominator can never equal its row count.  Missing confidence is *missing*
   - never 0.5 (plan §5.7).
4. **Order**: held-out predictions are written and hashed **before** ``labels`` is joined.
   :func:`join_labels` refuses a join whose prediction file is unhashed or has changed since
   it was sealed (:class:`LabelJoinRefused`), which is the ``aggregate.Correctness.of``
   JoinRefused pattern applied to the analysis stage.

Every estimate goes through :mod:`agents_scaling.ard.stats`, whose domain axis is the family
axis ``("math500", "mmlu_pro")`` with 0.5/0.5 weights (spec §15.1 prespecified equal-family
macro average).  The generalization unit is the original task: rows are averaged to the item
before any resampling, and the cluster bootstrap resamples items.

This module reads gold-derived labels and therefore belongs to the evaluator namespace only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from agents_scaling.ard import contract as K
from agents_scaling.ard import stats as S
from agents_scaling.ard.reference_metrics import within_item_auc
from agents_scaling.experiment import io
from agents_scaling.study.aggregate import _write_parquet  # reused verbatim (.empty sentinel)

TABLES_DIR: str = "tables"
PREDICTIONS_DIR: str = "predictions"
PREDICTIONS_MANIFEST: str = "MANIFEST.json"
STATS_FILE: str = "ard_stats.json"
STATS_MARKDOWN: str = "ard_stats.md"

#: The long-form selected-correctness table written by ``ard/compact.py``.
SELECTION_TABLE: str = "round_selection"

#: Columns a held-out prediction file may never carry (contract §0.2 and spec §15.2: the
#: predictions are sealed before the label join, so a label column here is a seal violation).
LABEL_COLUMNS: frozenset[str] = K.FORBIDDEN_FEATURE_SOURCES | {
    "is_correct", "evaluator_status", "y", "y_true", "target", "outcome", "graded",
}

#: Families below this many clusters are reported as estimation-only (plan §8).
ESTIMATION_ONLY_CLUSTERS: int = 30


class AnalysisError(RuntimeError):
    """The tables do not support the estimand that was asked for."""


class LabelJoinRefused(AnalysisError):
    """A label join was attempted against predictions that were not sealed first."""


# --------------------------------------------------------------------------- tables


def read_table(path: str | os.PathLike) -> list[dict[str, Any]]:
    """One compaction table as a row list.  A ``.empty`` sentinel reads as no rows."""
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def load_tables(run_root: str | os.PathLike, *, tables_dir: str = TABLES_DIR) -> dict[str, list[dict[str, Any]]]:
    """Every table ``compact.py`` writes, by name (missing tables read as no rows)."""
    root = Path(run_root) / tables_dir
    names = (*K.ALL_TABLES, SELECTION_TABLE)
    return {name: read_table(root / f"{name}.parquet") for name in names}


def episode_index(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """``episode_id -> {item_id, family, split, cohort, protocol, checkpoint, ...}``.

    ``round_selection`` and ``round_summary`` are keyed by episode; the split and family axes
    live on ``episodes``, so every panel builder joins through here.
    """
    index: dict[str, dict[str, Any]] = {}
    for row in tables.get("episodes", ()):
        index[str(row["episode_id"])] = {
            "item_id": row.get("item_id"), "family": row.get("family"),
            "split": row.get("split"), "cohort": row.get("cohort"),
            "protocol": row.get("protocol"), "checkpoint": row.get("checkpoint"),
            "team_id": row.get("team_id"), "reasoning_allowance": row.get("reasoning_allowance"),
            "parent_ref": row.get("parent_ref"), "fallback": bool(row.get("fallback")),
        }
    # ``round_summary`` carries family/split directly; use it to fill episodes we did not see.
    for row in tables.get("round_summary", ()):
        eid = str(row.get("episode_id"))
        entry = index.setdefault(eid, {})
        for key in ("item_id", "family", "split", "cohort", "protocol", "checkpoint",
                    "team_id", "reasoning_allowance", "parent_ref"):
            entry.setdefault(key, row.get(key))
    return index


# --------------------------------------------------------------------------- panels


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(out) else out


def arm_values(
    rows: Iterable[Mapping[str, Any]],
    index: Mapping[str, Mapping[str, Any]],
    *,
    keep: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
    value_of: Callable[[Mapping[str, Any]], float | None],
) -> dict[str, tuple[str, float]]:
    """``item_id -> (family, mean value)`` for one arm.

    Spec §15.1: the independent unit is the original task, so several episodes of one item
    (challenge teams, budget arms, episode repetitions) are averaged **to the item** before any
    resampling touches them.
    """
    acc: dict[str, list[float]] = defaultdict(list)
    family: dict[str, str] = {}
    for row in rows:
        meta = index.get(str(row.get("episode_id")), {})
        if not meta or not keep(row, meta):
            continue
        value = value_of(row)
        if value is None:
            continue
        item = str(meta.get("item_id"))
        acc[item].append(value)
        family[item] = str(meta.get("family"))
    return {item: (family[item], float(np.mean(vals))) for item, vals in acc.items() if vals}


def paired_panel(
    treatment: Mapping[str, tuple[str, float]],
    control: Mapping[str, tuple[str, float]],
    *,
    domains: Sequence[str] = S.DOMAINS,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Per-item ``treatment - control`` differences by family, plus the paired item ids.

    Every declared domain is a key even when it holds no item, so
    :func:`stats.paired_contrast`'s axis assertion sees the full axis and the failure mode is
    "no domain has >= 2 clusters", never a silent drop.
    """
    by_domain: dict[str, list[float]] = {d: [] for d in domains}
    items = sorted(set(treatment) & set(control))
    for item in items:
        fam_t, value_t = treatment[item]
        fam_c, value_c = control[item]
        if fam_t != fam_c:
            raise AnalysisError(f"item {item} has two families: {fam_t!r} and {fam_c!r}")
        if fam_t not in by_domain:
            raise S.StatsAxisError(
                f"item {item} carries family {fam_t!r}, which is not on the declared axis "
                f"{list(domains)}")
        by_domain[fam_t].append(value_t - value_c)
    return {d: np.asarray(v, dtype=float) for d, v in by_domain.items()}, items


def selected_accuracy_arm(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    index: Mapping[str, Mapping[str, Any]],
    *,
    protocol: str,
    round_: int = K.UPDATE_ROUNDS,
    selector: str = "plurality",
    pool: str = "current",
    cohort: str = K.Cohort.NATURAL.value,
    split: str = "test",
    checkpoint: str = K.CHECKPOINT_SMALL,
) -> dict[str, tuple[str, float]]:
    """Selected correctness of one arm, per item (the endpoint of primaries 1 and 2)."""
    def keep(row: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
        return (str(row.get("protocol")) == protocol
                and int(row.get("round", -1)) == int(round_)
                and str(row.get("selector")) == selector
                and str(row.get("pool")) == pool
                and str(row.get("cohort")) == cohort
                and str(meta.get("split")) == split
                and str(meta.get("checkpoint")) == checkpoint
                and not meta.get("fallback"))

    return arm_values(tables.get(SELECTION_TABLE, ()), index, keep=keep,
                      value_of=lambda row: _as_float(row.get("selected_correct")))


def coverage_change_arm(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    index: Mapping[str, Mapping[str, Any]],
    *,
    protocol: str,
    cohort: str = K.Cohort.NATURAL.value,
    split: str = "test",
    checkpoint: str = K.CHECKPOINT_SMALL,
    from_round: int = 0,
    to_round: int = K.UPDATE_ROUNDS,
    column: str = "coverage_current",
) -> dict[str, tuple[str, float]]:
    """Root -> round-3 change in current-round coverage, per item (primary 3).

    Coverage is a per-episode quantity, so the change is formed **within** the episode and only
    then averaged to the item; differencing item means of two rounds would silently pair
    different episodes.
    """
    per_episode: dict[str, dict[int, float]] = defaultdict(dict)
    for row in tables.get("round_summary", ()):
        eid = str(row.get("episode_id"))
        meta = index.get(eid, {})
        if (str(row.get("protocol")) != protocol or str(row.get("cohort")) != cohort
                or str(meta.get("split")) != split
                or str(meta.get("checkpoint")) != checkpoint or meta.get("fallback")):
            continue
        value = _as_float(row.get(column))
        if value is not None:
            per_episode[eid][int(row["round"])] = value
    acc: dict[str, list[float]] = defaultdict(list)
    family: dict[str, str] = {}
    for eid, by_round in per_episode.items():
        if from_round not in by_round or to_round not in by_round:
            continue
        meta = index.get(eid, {})
        item = str(meta.get("item_id"))
        acc[item].append(by_round[to_round] - by_round[from_round])
        family[item] = str(meta.get("family"))
    return {item: (family[item], float(np.mean(v))) for item, v in acc.items() if v}


# --------------------------------------------------------------------------- contrasts


def contrast(
    treatment: Mapping[str, tuple[str, float]],
    control: Mapping[str, tuple[str, float]],
    *,
    label: str,
    study_seed: str,
    n_resamples: int = K.N_RESAMPLES,
    domains: Sequence[str] = S.DOMAINS,
    weights: Mapping[str, float] = S.WEIGHTS,
    n_primary: int = K.N_PRIMARY,
    alternative: str = "two-sided",
    components: int = 1,
) -> dict[str, Any]:
    """One paired cluster-bootstrap contrast with its own deterministic resample seed."""
    diffs, items = paired_panel(treatment, control, domains=domains)
    rs = S.Resampler({d: len(v) for d, v in diffs.items()}, n_resamples,
                     S.analysis_seed(study_seed, label), domains=tuple(domains))
    result = S.paired_contrast(diffs, rs, domains=domains, weights=weights,
                               n_primary=n_primary, alternative=alternative,
                               components=components)
    n_clusters = sum(len(v) for v in diffs.values())
    result.update({
        "label": label,
        "n_items": n_clusters,
        "n_paired_items": len(items),
        "estimation_only": bool(n_clusters < ESTIMATION_ONLY_CLUSTERS),
    })
    if result["estimation_only"]:
        result["note"] = (f"fewer than {ESTIMATION_ONLY_CLUSTERS} clusters: estimation only, "
                          "no test is claimed (plan §8)")
    return result


def primary_analysis(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    study_seed: str,
    n_resamples: int = K.N_RESAMPLES,
    checkpoint: str = K.CHECKPOINT_SMALL,
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    """The three frozen primary contrasts (spec §15.2) under Holm with ``n_primary = 3``."""
    index = episode_index(tables)
    natural = K.Cohort.NATURAL.value
    builders: dict[str, Callable[[], dict[str, Any]]] = {
        K.PRIMARY_CONTRASTS[0]: lambda: contrast(
            selected_accuracy_arm(tables, index, protocol=K.Protocol.GATE.value,
                                  cohort=natural, checkpoint=checkpoint),
            selected_accuracy_arm(tables, index, protocol=K.Protocol.PEER.value,
                                  cohort=natural, checkpoint=checkpoint),
            label=K.PRIMARY_CONTRASTS[0], study_seed=study_seed, n_resamples=n_resamples),
        K.PRIMARY_CONTRASTS[1]: lambda: contrast(
            selected_accuracy_arm(tables, index, protocol=K.Protocol.GATE.value,
                                  cohort=natural, checkpoint=checkpoint),
            selected_accuracy_arm(tables, index, protocol=K.Protocol.RANDOM_PROTECT.value,
                                  cohort=natural, checkpoint=checkpoint),
            label=K.PRIMARY_CONTRASTS[1], study_seed=study_seed, n_resamples=n_resamples),
        K.PRIMARY_CONTRASTS[2]: lambda: contrast(
            coverage_change_arm(tables, index, protocol=K.Protocol.PEER.value,
                                cohort=natural, checkpoint=checkpoint),
            coverage_change_arm(tables, index, protocol=K.Protocol.PRIVATE.value,
                                cohort=natural, checkpoint=checkpoint),
            label=K.PRIMARY_CONTRASTS[2], study_seed=study_seed, n_resamples=n_resamples),
    }
    contrasts: dict[str, Any] = {}
    for name, build in builders.items():
        try:
            contrasts[name] = build()
        except (S.StatsError, AnalysisError) as exc:
            if not allow_unavailable:
                raise
            contrasts[name] = {"label": name, "status": "unavailable", "estimate": None,
                               "p": 1.0, "error": str(exc)}
    pvalues = {name: float(res["p"]) for name, res in contrasts.items()}
    holm = S.holm(pvalues, K.ALPHA, n_primary=K.N_PRIMARY)
    complete = all(res.get("status") != "unavailable" for res in contrasts.values())
    return {
        "family": "primary",
        "label": "confirmatory",
        "n_primary": K.N_PRIMARY,
        "contrasts": contrasts,
        "holm": holm,
        "complete": complete,
        "note": ("Holm across exactly the three pre-registered contrasts of "
                 "contract.PRIMARY_CONTRASTS; a failed contrast is never redefined as a "
                 "success after its sign is seen (spec §15.2)."),
    }


# --------------------------------------------------------------------------- secondary


def _family(label: str, kind: str, contrasts: Mapping[str, Mapping[str, Any]],
            note: str = "") -> dict[str, Any]:
    """One registered family: Holm applied only inside it (spec §15.2)."""
    usable = {name: float(res["p"]) for name, res in contrasts.items()
              if res.get("status") != "unavailable"}
    return {
        "family": label,
        "label": kind,
        "contrasts": dict(contrasts),
        "holm": S.holm(usable, K.ALPHA) if usable else {},
        "n_contrasts": len(contrasts),
        "note": note,
    }


def _safe(build: Callable[[], dict[str, Any]], label: str) -> dict[str, Any]:
    try:
        return build()
    except (S.StatsError, AnalysisError) as exc:
        return {"label": label, "status": "unavailable", "estimate": None, "p": 1.0,
                "error": str(exc)}


def challenge_family(tables, index, *, study_seed, n_resamples, checkpoint=K.CHECKPOINT_SMALL):
    """Challenge C0 = 1 and C0 = 4 outcomes (spec §15.2 secondary; plan §8)."""
    cohort = K.Cohort.CHALLENGE.value
    c0_of = {str(row["episode_id"]): row.get("constructed_C0")
             for row in tables.get("round_summary", ())}

    def arm(protocol: str, level: int) -> dict[str, tuple[str, float]]:
        def keep(row: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
            return (str(row.get("protocol")) == protocol
                    and int(row.get("round", -1)) == K.UPDATE_ROUNDS
                    and str(row.get("selector")) == "plurality"
                    and str(row.get("pool")) == "current"
                    and str(row.get("cohort")) == cohort
                    and str(meta.get("checkpoint")) == checkpoint
                    and c0_of.get(str(row.get("episode_id"))) == level)
        return arm_values(tables.get(SELECTION_TABLE, ()), index, keep=keep,
                          value_of=lambda row: _as_float(row.get("selected_correct")))

    contrasts: dict[str, Any] = {}
    for c0 in K.CHALLENGE_C0_LEVELS:
        for treat, ctrl in ((K.Protocol.GATE, K.Protocol.PEER),
                            (K.Protocol.GATE, K.Protocol.RANDOM_PROTECT)):
            label = f"challenge_C0{c0}_{treat.value}_minus_{ctrl.value}"
            contrasts[label] = _safe(
                lambda t=treat, c=ctrl, lab=label, level=c0: contrast(
                    arm(t.value, level), arm(c.value, level), label=lab,
                    study_seed=study_seed, n_resamples=n_resamples), label)
    return _family("challenge_c0", "secondary", contrasts,
                   "Label-conditioned cohort; never pooled with the natural cohort.  Levels "
                   f"C0 in {list(K.CHALLENGE_C0_LEVELS)}.")


def checkpoint_family(tables, index, *, study_seed, n_resamples):
    """8B vs 32B protocol differences on the shared 32B subset (spec §15.2 secondary)."""
    contrasts: dict[str, Any] = {}
    for protocol in (K.Protocol.PRIVATE, K.Protocol.PEER, K.Protocol.BALANCED):
        label = f"checkpoint_32B_minus_8B_{protocol.value}"
        contrasts[label] = _safe(lambda p=protocol, lab=label: contrast(
            selected_accuracy_arm(tables, index, protocol=p.value,
                                  checkpoint=K.CHECKPOINT_LARGE),
            selected_accuracy_arm(tables, index, protocol=p.value,
                                  checkpoint=K.CHECKPOINT_SMALL),
            label=lab, study_seed=study_seed, n_resamples=n_resamples), label)
    return _family("checkpoint", "secondary", contrasts,
                   "Native checkpoint comparisons do not isolate pretraining scale "
                   "(spec §15.4).")


def decision_matched_family(tables, index, *, study_seed, n_resamples,
                            checkpoint=K.CHECKPOINT_SMALL):
    """The registered ``GATE - FRESH@20`` / ``PEER - FRESH@20`` family (plan §8).

    FRESH@20 is FRESH's cumulative round-3 **archive** pool: 20 logical decisions per item, the
    same count the three-round interacting protocols spend.  Decision-matched, explicitly not
    FLOP-matched - the realized cost sits in ``cost_accuracy``.
    """
    contrasts: dict[str, Any] = {}
    for selector in ("plurality", "confidence_weighted", "CURRENT"):
        fresh = selected_accuracy_arm(
            tables, index, protocol=K.Protocol.FRESH.value, selector=selector,
            pool="archive", checkpoint=checkpoint)
        for protocol in (K.Protocol.GATE, K.Protocol.PEER):
            label = f"{protocol.value}_minus_FRESH20_{selector}"
            contrasts[label] = _safe(lambda p=protocol, s=selector, lab=label: contrast(
                selected_accuracy_arm(tables, index, protocol=p.value, selector=s,
                                      pool="current", checkpoint=checkpoint),
                fresh, label=lab, study_seed=study_seed, n_resamples=n_resamples), label)
    return _family(
        "decision_matched_independent_sampling", "secondary", contrasts,
        "Decision-matched at 20 logical decisions per item per arm, NOT FLOP-matched.  A GATE "
        "gain that does not exceed FRESH@20 is the spec §22 insufficient/boundary outcome.")


def calibration_family(tables, *, study_seed, n_resamples, checkpoint=K.CHECKPOINT_SMALL):
    """Cross-protocol calibration contrasts - exploratory, Holm only inside this family."""
    panel = tables.get("calibration_panel", ())
    contrasts: dict[str, Any] = {}
    pairs = ((K.Protocol.PEER, K.Protocol.PRIVATE), (K.Protocol.GATE, K.Protocol.PEER),
             (K.Protocol.BALANCED, K.Protocol.PEER))
    for treat, ctrl in pairs:
        label = f"ece_{treat.value}_minus_{ctrl.value}"
        rows = {(str(r.get("protocol")), int(r.get("round", -1))): r for r in panel
                if str(r.get("checkpoint")) == checkpoint
                and str(r.get("signal")) == "self_reported"
                and str(r.get("cohort")) == K.Cohort.NATURAL.value}
        a = rows.get((treat.value, K.UPDATE_ROUNDS))
        b = rows.get((ctrl.value, K.UPDATE_ROUNDS))
        if a is None or b is None:
            contrasts[label] = {"label": label, "status": "unavailable", "estimate": None,
                                "p": 1.0, "error": "calibration_panel has no round-3 row "
                                                   f"for {treat.value}/{ctrl.value}"}
            continue
        contrasts[label] = {
            "label": label, "status": "estimation_only",
            "estimate": (_as_float(a.get("ece_primary")) or 0.0) - (_as_float(b.get("ece_primary")) or 0.0),
            "p": 1.0,
            "n": {"treatment_valid_n": a.get("valid_n"), "control_valid_n": b.get("valid_n")},
            "note": ("Plug-in ECEs are never compared across cells with different valid n; "
                     "use the debiased estimator or matched-n at the minimum shared n."),
        }
    return _family("calibration", "exploratory", contrasts,
                   "Miscalibration is a user-requested addition beyond the spec; high "
                   "confidence may be copied certainty, and aggregate calibration is not "
                   "within-item selection ability (spec §15.4).")


def _table_family(rows: Sequence[Mapping[str, Any]], *, label: str, kind: str,
                  note: str) -> dict[str, Any]:
    """A family whose producing work package has not landed: registered, not silently absent."""
    if not rows:
        return {"family": label, "label": kind, "contrasts": {}, "holm": {},
                "n_contrasts": 0, "status": "unexecuted",
                "note": note + "  Recorded as unexecuted in the amendment register."}
    return {"family": label, "label": kind, "contrasts": {}, "holm": {},
            "n_contrasts": 0, "status": "descriptive", "n_rows": len(rows), "note": note}


def secondary_analysis(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    study_seed: str,
    n_resamples: int = K.N_RESAMPLES,
    checkpoint: str = K.CHECKPOINT_SMALL,
) -> dict[str, Any]:
    """Every registered secondary/exploratory family, each with its own Holm scope."""
    index = episode_index(tables)
    return {
        "challenge_c0": challenge_family(tables, index, study_seed=study_seed,
                                         n_resamples=n_resamples, checkpoint=checkpoint),
        "receiver": _table_family(
            tables.get("receiver", ()), label="receiver", kind="secondary",
            note="Receiver count x checkability x budget interactions (spec §11); produced by "
                 "the WP5 receiver assay."),
        "checkpoint": checkpoint_family(tables, index, study_seed=study_seed,
                                        n_resamples=n_resamples),
        "suffix_reversal": _table_family(
            tables.get("suffix_summary", ()), label="suffix_reversal", kind="secondary",
            note="Common-suffix utility reversal (spec §13): immediate selection gain versus "
                 "future utility after a shared continuation."),
        "calibration": calibration_family(tables, study_seed=study_seed,
                                          n_resamples=n_resamples, checkpoint=checkpoint),
        "neural_increment": _table_family(
            tables.get("neural_geometry", ()), label="neural_increment", kind="exploratory",
            note="Incremental value of captured pre-update states over behaviour-only features "
                 "(spec §14.4).  A useful predictor is not a causal neural mechanism."),
        "decision_matched_independent_sampling": decision_matched_family(
            tables, index, study_seed=study_seed, n_resamples=n_resamples,
            checkpoint=checkpoint),
    }


# --------------------------------------------------------------------------- within-item AUC


def within_item_rows(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    score_column: str = "confidence",
) -> list[dict[str, Any]]:
    """Per (protocol, checkpoint, cohort, round): within-item AUC averaged over mixed items.

    Spec §15.1: average pairwise correct-versus-wrong ranking **within** mixed-correctness
    items, then average items, and report how many items are eligible.  All-correct and
    all-wrong items return ``None`` from :func:`reference_metrics.within_item_auc` and are
    counted in ``excluded_single_class`` instead of being folded into a pooled AUC - which is
    why ``eligible_items`` can never equal the row count of the panel it summarises.

    Missing confidence excludes that candidate from the item's ranking; it is never imputed as
    0.5 (plan §5.7 - the atlas probe script's ``fillna(0.5)`` is not lifted).
    """
    labels = {str(row["decision_id"]): row for row in tables.get("labels", ())}
    index = episode_index(tables)
    groups: dict[tuple[str, str, str, int], dict[str, list[tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list))
    missing_score: dict[tuple[str, str, str, int], int] = defaultdict(int)
    for row in tables.get("decisions", ()):
        meta = index.get(str(row.get("episode_id")), {})
        if meta.get("fallback"):
            continue
        label = labels.get(str(row.get("decision_id")))
        if label is None or label.get("is_correct") is None:
            continue
        key = (str(row.get("protocol")), str(row.get("checkpoint")), str(row.get("cohort")),
               int(row.get("round", -1)))
        score = _as_float(row.get(score_column))
        if score is None:
            missing_score[key] += 1
            continue
        item = str(meta.get("item_id") or row.get("episode_id"))
        groups[key][item].append((score, int(bool(label["is_correct"]))))
    rows: list[dict[str, Any]] = []
    for key in sorted(set(groups) | set(missing_score)):
        protocol, checkpoint, cohort, round_ = key
        aucs: list[float] = []
        excluded = 0
        n_candidates = 0
        for _item, pairs in sorted(groups.get(key, {}).items()):
            n_candidates += len(pairs)
            value = within_item_auc([p[0] for p in pairs], [p[1] for p in pairs])
            if value is None:
                excluded += 1
            else:
                aucs.append(float(value))
        rows.append({
            "protocol": protocol, "checkpoint": checkpoint, "cohort": cohort, "round": round_,
            "signal": score_column,
            "within_item_auc": (float(np.mean(aucs)) if aucs else None),
            "eligible_items": len(aucs),
            "excluded_single_class": excluded,
            "n_items": len(groups.get(key, {})),
            "n_candidates": n_candidates,
            "n_missing_score": missing_score.get(key, 0),
            "note": ("mixed-correctness items only; all-correct/all-wrong items are excluded, "
                     "never pooled (spec §15.1).  Missing confidence is excluded, never 0.5."),
        })
    return rows


# --------------------------------------------------------------------------- calibration


def calibration_metrics(conf: Sequence[float | None], correct: Sequence[Any],
                        *, n_bins: int = 10) -> dict[str, Any]:
    """Brier, NLL, signed gap and equal-mass 10-bin L1 ECE with the missing-is-missing policy.

    Pairs whose confidence is missing are dropped and reported through ``conf_missing_rate``;
    they are never imputed.  The equal-mass binning is the plan's primary ECE; the plug-in
    value is only comparable across cells with the same valid ``n``.
    """
    pairs = [(float(c), int(bool(y))) for c, y in zip(conf, correct)
             if c is not None and math.isfinite(float(c)) and y is not None]
    n_total = len(list(conf))
    if not pairs:
        return {"n": n_total, "valid_n": 0, "conf_missing_rate": 1.0 if n_total else None,
                "mean_confidence": None, "accuracy": None, "signed_gap": None,
                "brier": None, "nll": None, "ece_primary": None}
    p = np.array([x[0] for x in pairs], dtype=float)
    y = np.array([x[1] for x in pairs], dtype=float)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, min(n_bins, len(order)))
    ece = float(sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)) / len(p))
    return {
        "n": n_total,
        "valid_n": len(pairs),
        "conf_missing_rate": (n_total - len(pairs)) / n_total if n_total else None,
        "mean_confidence": float(p.mean()),
        "accuracy": float(y.mean()),
        "signed_gap": float(p.mean() - y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "ece_primary": ece,
        "ece_bins": int(min(n_bins, len(order))),
    }


# --------------------------------------------------------------------------- seal


def predictions_dir(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / PREDICTIONS_DIR


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_held_out_predictions(
    run_root: str | os.PathLike,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Write ``predictions/<name>.parquet`` with **no label column** and hash it.

    Plan §7 hour 30-34 and §8: held-out predictions are written and hashed *before*
    ``labels.parquet`` is joined.  A row carrying a label-ish column is refused here, so the
    seal cannot be produced from a frame that already saw the outcome.
    """
    folder = predictions_dir(run_root)
    folder.mkdir(parents=True, exist_ok=True)
    offending = sorted({key for row in rows for key in row} & LABEL_COLUMNS)
    if offending:
        raise LabelJoinRefused(
            f"held-out predictions {name!r} carry label column(s) {offending}; predictions are "
            "sealed before any label join (spec §15.2)")
    path = folder / f"{name}.parquet"
    _write_parquet(path, list(rows))
    sealed = path if path.exists() else path.with_suffix(".empty")
    entry = {
        "name": name,
        "file": sealed.name,
        "sha256": _sha256_file(sealed),
        "n_rows": len(rows),
        "columns": sorted({key for row in rows for key in row}),
        "sealed_at": float(clock() if clock else time.time()),
    }
    manifest_path = folder / PREDICTIONS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[name] = entry
    io.write_json(manifest_path, manifest)
    return entry


def predictions_manifest(run_root: str | os.PathLike) -> dict[str, Any]:
    path = predictions_dir(run_root) / PREDICTIONS_MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def join_labels(
    run_root: str | os.PathLike,
    name: str,
    labels: Mapping[str, Mapping[str, Any]],
    *,
    key: str = "decision_id",
) -> list[dict[str, Any]]:
    """Join ``labels`` onto the sealed predictions ``name``, or refuse.

    The refusal is the point: an unsealed prediction file, or one whose bytes changed after it
    was hashed, means the ordering guarantee was broken and the join must not happen.
    """
    manifest = predictions_manifest(run_root)
    entry = manifest.get(name)
    if entry is None:
        raise LabelJoinRefused(
            f"no sealed predictions named {name!r}: write and hash the held-out predictions "
            "before joining labels (plan §8)")
    path = predictions_dir(run_root) / entry["file"]
    if not path.exists():
        raise LabelJoinRefused(f"sealed prediction file {path.name} is missing")
    digest = _sha256_file(path)
    if digest != entry["sha256"]:
        raise LabelJoinRefused(
            f"predictions {name!r} changed after they were sealed "
            f"({digest[:12]} != {entry['sha256'][:12]}); the label join is refused")
    rows = read_table(path)
    joined: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        label = labels.get(str(row.get(key)))
        out["is_correct"] = None if label is None else label.get("is_correct")
        joined.append(out)
    return joined


# --------------------------------------------------------------------------- driver


def run_analysis(
    run_root: str | os.PathLike,
    *,
    study_seed: str | None = None,
    n_resamples: int = K.N_RESAMPLES,
    checkpoint: str = K.CHECKPOINT_SMALL,
    tables_dir: str = TABLES_DIR,
    allow_unavailable: bool = False,
    with_secondary: bool = True,
) -> dict[str, Any]:
    """Compute every registered readout and write ``tables/ard_stats.json`` + ``.md``.

    ``allow_unavailable`` is off by default: a primary contrast whose panel is missing raises
    rather than being recorded with ``p = 1.0``.  Turning it on is how a partial run root is
    inspected mid-flight, and it stamps ``primary.complete = False``.
    """
    run_root = Path(run_root)
    tables = load_tables(run_root, tables_dir=tables_dir)
    seed = study_seed or _study_seed(run_root)
    result: dict[str, Any] = {
        "study_id": K.STUDY_ID,
        "run_root": str(run_root),
        "study_seed": seed,
        "n_resamples": int(n_resamples),
        "alpha": K.ALPHA,
        "domains": list(S.DOMAINS),
        "weights": dict(S.WEIGHTS),
        "checkpoint": checkpoint,
        "primary": primary_analysis(tables, study_seed=seed, n_resamples=n_resamples,
                                    checkpoint=checkpoint,
                                    allow_unavailable=allow_unavailable),
        "within_item_discrimination": within_item_rows(tables),
        "predictions_sealed": sorted(predictions_manifest(run_root)),
        "nonclaims": list(NONCLAIMS),
    }
    if with_secondary:
        result["secondary"] = secondary_analysis(tables, study_seed=seed,
                                                 n_resamples=n_resamples,
                                                 checkpoint=checkpoint)
    out = run_root / tables_dir
    out.mkdir(parents=True, exist_ok=True)
    io.write_json(out / STATS_FILE, result)
    io.atomic_write_text(out / STATS_MARKDOWN, render_markdown(result))
    return result


#: Carried verbatim into ``results_summary.md`` beside every table (spec §15.4).
NONCLAIMS: tuple[str, ...] = (
    "Identical isolated roots matching pass@k is not novel.",
    "Positive expected drift over a few rounds does not prove eventual convergence to truth.",
    "A low-rank state or an outlier is not inherently correct.",
    "High confidence may reflect copied certainty; calibration is not within-item selection "
    "ability.",
    "A useful predictor is not a causal neural mechanism.",
    "Native checkpoint comparisons do not isolate pretraining scale.",
    "Conditional correct-count transition tables are not a validated Markov law unless "
    "history-sufficiency is tested.",
    "More rounds are not independent samples; their cumulative coverage is not iid pass@k.",
    "No claim of superiority over optimised single-agent inference is made: the cost-matched "
    "long single-agent baseline (spec §17 item 9) was not executed.",
)


def _study_seed(run_root: Path) -> str:
    frozen = run_root / "FROZEN.yaml"
    if frozen.exists():
        for line in frozen.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("study_seed:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    return K.SEED_NAMESPACE


def _fmt(value: Any, nd: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [f"# ARD36 statistics ({result.get('study_id')})", "",
             f"Resamples: {result.get('n_resamples')}; alpha {result.get('alpha')}; "
             f"family axis {result.get('domains')} weighted {result.get('weights')}.", "",
             "## Primary (confirmatory, Holm over exactly three contrasts)", "",
             "| contrast | estimate | 95% CI | p | p (Holm) | reject | n items |",
             "|---|---|---|---|---|---|---|"]
    primary = result.get("primary", {})
    holm = primary.get("holm", {})
    for name, res in primary.get("contrasts", {}).items():
        ci = res.get("ci95_studentized") or [None, None]
        adjusted = holm.get(name, {})
        lines.append(
            f"| {name} | {_fmt(res.get('estimate'))} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(res.get('p'))} | "
            f"{_fmt(adjusted.get('p_holm'))} | {adjusted.get('reject')} | "
            f"{res.get('n_items', '-')} |")
    lines += ["", f"_{primary.get('note', '')}_", "", "## Within-item discrimination", "",
              "| protocol | checkpoint | cohort | round | AUC | eligible items | excluded |",
              "|---|---|---|---|---|---|---|"]
    for row in result.get("within_item_discrimination", []):
        lines.append(f"| {row['protocol']} | {row['checkpoint']} | {row['cohort']} | "
                     f"{row['round']} | {_fmt(row['within_item_auc'])} | "
                     f"{row['eligible_items']} | {row['excluded_single_class']} |")
    lines += ["", "## Registered nonclaims", ""]
    lines += [f"- {claim}" for claim in result.get("nonclaims", [])]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.ard.analysis",
                                description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--run-root", required=True)
    p.add_argument("--n-resamples", type=int, default=K.N_RESAMPLES)
    p.add_argument("--checkpoint", default=K.CHECKPOINT_SMALL)
    p.add_argument("--allow-unavailable", action="store_true",
                   help="record an unavailable primary contrast instead of failing "
                        "(mid-flight inspection; stamps primary.complete = False)")
    p.add_argument("--no-secondary", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analysis(args.run_root, n_resamples=args.n_resamples,
                          checkpoint=args.checkpoint,
                          allow_unavailable=args.allow_unavailable,
                          with_secondary=not args.no_secondary)
    print(json.dumps({"primary": {k: v.get("p") for k, v in
                                  result["primary"]["contrasts"].items()}}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AnalysisError", "LabelJoinRefused", "NONCLAIMS",
    "calibration_metrics", "contrast", "coverage_change_arm", "episode_index", "join_labels",
    "load_tables", "paired_panel", "predictions_manifest", "primary_analysis", "read_table",
    "run_analysis", "secondary_analysis", "selected_accuracy_arm", "within_item_rows",
    "write_held_out_predictions",
]
