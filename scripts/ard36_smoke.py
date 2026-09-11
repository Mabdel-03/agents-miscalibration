#!/usr/bin/env python3
"""Turn completed ARD36 pilot cells into the launch gate's ``--measured`` JSON.

Spec §16.3 requires the gate to be fed *measured* fields, stratified by call type,
"including prefill and split-generation overhead" — one aggregate tokens/s hides the lane
that actually binds.  This reads the decisions the pilot already wrote and emits exactly the
schema ``ard/gates.py`` consumes.

It measures; it decides nothing.  A call type the frozen tier does not run must be declared
with ``--not-in-tier HUB_LEADER=reason``, because the gate treats silence as a failure.

    PYTHONPATH=src python scripts/ard36_smoke.py --run-root <root> \
        --not-in-tier HUB_LEADER="HUB5 dropped by 16.4 step 2" \
        --not-in-tier HUB_WORKER="HUB5 dropped by 16.4 step 2" \
        --grader-items-per-s 26 --replay-sequences-per-s 3.0 --out measured.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from agents_scaling.ard import contract as K
from agents_scaling.ard import episode as E

#: The phase vocabulary is WP2's, imported rather than spelled out: this table used to key
#: on "revise"/"hub", which ``episode.py`` never emits (it writes "update"/"leader"), so every
#: revision fell through the lookup and was counted as a ROOT.  That left PEER_REVISE and
#: BALANCED_REVISE with no measured row at all and silently mixed revisions into the root
#: parse-validity the launch gate reads.
#:
#: A FRESH update is a new independent draw on the same task, so it costs what a root costs.
PROTOCOL_CALL_TYPE = {"BALANCED": "BALANCED_REVISE", "FRESH": "ROOT"}


def call_type(dec: dict) -> str:
    phase = str(dec.get("phase") or "")
    proto = dec.get("protocol")
    proto = None if proto in (None, "NONE", "") else str(proto)
    budget = int(dec.get("budget_tokens") or 0)
    if phase == K.PURPOSE_RECEIVER:
        return "RECEIVER"
    if budget > K.THINKING_TOKENS_LOW:
        return "HIGH_BUDGET"
    if phase == E.PHASE_ROOT:
        return "ROOT"
    if phase == E.PHASE_LEADER:
        return "HUB_LEADER"
    if phase == E.PHASE_WORKER:
        return "HUB_WORKER"
    if phase == E.PHASE_UPDATE:
        return PROTOCOL_CALL_TYPE.get(proto or "", "PEER_REVISE")
    raise SystemExit(
        f"unknown decision phase {phase!r}: the gate's call-type table and WP2's phase "
        "vocabulary have diverged; fix the mapping rather than let it default"
    )


def load_decisions(run_root: Path, cells: list[str] | None) -> list[dict]:
    out: list[dict] = []
    pattern = cells or ["*"]
    for pat in pattern:
        for f in glob.glob(str(run_root / "cells" / pat / "items" / "*.json")):
            doc = json.loads(Path(f).read_text())
            for ep in doc.get("episodes", []):
                for dec in ep.get("decisions", []):
                    dec = dict(dec)
                    dec["_family"] = doc.get("family")
                    out.append(dec)
    return out


def _row(decs: list[dict]) -> dict:
    lat = sorted(float(d["latency_s"]) for d in decs if d.get("latency_s"))
    prompt = [int(d["prompt_tokens"]) for d in decs if d.get("prompt_tokens") is not None]
    gen = [int(d.get("thinking_tokens") or 0) + int(d.get("final_tokens") or 0) + 2 for d in decs]
    # Effective tokens/s per request stream: generated tokens over the request's own wall time.
    # Summed over concurrent streams this is what a replica sustains; the gate compares it to
    # the per-replica requirement, so a per-request rate is the honest unit here.
    rates = [g / float(d["latency_s"]) for g, d in zip(gen, decs)
             if d.get("latency_s") and float(d["latency_s"]) > 0]
    return {
        "tokens_per_s": round(st.mean(rates), 2) if rates else 0.0,
        "latency_mean_s": round(st.mean(lat), 3) if lat else 0.0,
        "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 3) if lat else 0.0,
        "prompt_tokens_mean": round(st.mean(prompt), 1) if prompt else 0.0,
        "generated_tokens_mean": round(st.mean(gen), 1) if gen else 0.0,
        "n_requests": len(decs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--cells", action="append", help="cell-id glob (repeatable); default all")
    ap.add_argument("--not-in-tier", action="append", default=[],
                    help="CALL_TYPE=reason for a call type the frozen tier does not run")
    ap.add_argument("--grader-items-per-s", type=float, required=True)
    ap.add_argument("--required-grader-items-per-s", type=float, default=0.0)
    ap.add_argument("--replay-sequences-per-s", type=float, required=True)
    ap.add_argument("--replay-hours", type=float, default=None)
    ap.add_argument("--http-error-rate", type=float, default=0.0)
    ap.add_argument("--small-tps-per-replica", type=float, required=True)
    ap.add_argument("--large-tps-per-replica", type=float, required=True)
    ap.add_argument("--projected-critical-path-hours", type=float, required=True)
    ap.add_argument("--projected-total-hours", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    decs = load_decisions(args.run_root, args.cells)
    if not decs:
        raise SystemExit(f"no decisions under {args.run_root}/cells; run a pilot cell first")

    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in decs:
        by_type[call_type(d)].append(d)

    table = {k: _row(v) for k, v in by_type.items()}
    for item in args.not_in_tier:
        name, _, reason = item.partition("=")
        table[name.strip()] = {"not_in_tier": True, "reason": reason.strip() or "not in the frozen tier"}

    roots = by_type.get("ROOT", [])
    status = Counter(str(d.get("parse_status")) for d in roots)
    # Validity is the contract's set, not the string "ok": a span cut after </final> closed
    # still delivers its answer (amendment I7).  Hard-coding "ok" here would report a
    # validity the parser, the vote and the graders all disagree with.
    valid_statuses = {st.value for st in K.VALID_PARSE_STATUSES}
    n_valid = sum(c for st, c in status.items() if st in valid_statuses)
    validity = (n_valid / len(roots)) if roots else 0.0
    cap = sum(1 for d in roots if d.get("boundary_injected"))

    measured = {
        "root_parse_validity": round(validity, 4),
        "http_error_rate": args.http_error_rate,
        "grader_items_per_s": args.grader_items_per_s,
        "required_grader_items_per_s": args.required_grader_items_per_s,
        "replay_sequences_per_s": args.replay_sequences_per_s,
        "replay_hours": args.replay_hours,
        "throughput_by_call_type": table,
        "small_tps_per_replica": args.small_tps_per_replica,
        "large_tps_per_replica": args.large_tps_per_replica,
        "projected_critical_path_hours": args.projected_critical_path_hours,
        "projected_total_hours": args.projected_total_hours,
        "provenance": {
            "n_decisions": len(decs),
            "n_root_decisions": len(roots),
            "root_parse_status": dict(status),
            "root_valid_statuses": sorted(valid_statuses),
            "root_conf_missing_rate": (
                round(sum(1 for d in roots if d.get("confidence") is None) / len(roots), 4)
                if roots else None
            ),
            "root_cap_hit_rate": round(cap / len(roots), 4) if roots else None,
            "call_types_measured": sorted(k for k, v in table.items() if "not_in_tier" not in v),
            "call_types_not_in_tier": sorted(k for k, v in table.items() if "not_in_tier" in v),
            "note": "tokens_per_s is per request stream (generated tokens / that request's wall "
                    "time), which is the unit the gate's per-replica requirement compares against",
        },
    }
    args.out.write_text(json.dumps(measured, indent=2) + "\n")
    print(json.dumps(measured["provenance"], indent=2))
    print(f"\nwrote {args.out}")
    for name in sorted(table):
        row = table[name]
        if "not_in_tier" in row:
            print(f"  {name:16s} not in tier: {row['reason']}")
        else:
            print(f"  {name:16s} {row['tokens_per_s']:7.1f} tok/s  lat {row['latency_mean_s']:5.2f}/"
                  f"{row['latency_p95_s']:5.2f}s  n={row['n_requests']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
