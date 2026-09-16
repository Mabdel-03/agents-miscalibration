#!/usr/bin/env python3
"""nextdyn hour-0 preflight: what the fleet is allowed to do, captured live.

Plan §3.5 S1 ("ports, QOS caps incl. mem, free GPUs per (partition, gpu_type), drained
nodes"), D20 (``fleet up`` refuses any spec whose per-(partition, gpu_type) GPU sum exceeds
the MaxTRESPU captured here).  Read-only: it submits nothing and cancels nothing.

The output is not a report for a human to read and forget -- ``fleet.assert_within_caps``
loads ``preflight/h0.json`` and refuses an over-subscribed spec against *these* numbers.  That
is the whole point of capturing them: ``compute.yaml``'s partition table is a planning
snapshot from the day the plan was written, and a QOS that has been tightened since is exactly
the case a launch must not discover by leaving twelve replicas pending for six hours.

    PYTHONPATH=src python scripts/nextdyn_preflight.py --json-out $RUN/preflight/h0.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agents_scaling.nextdyn import config as C  # noqa: E402
from agents_scaling.nextdyn import fleet as FL  # noqa: E402

PORT_BASE = 8000
PORT_MODULUS = 1000
#: How many replicas of one profile we may ever run: two profiles whose crc32 port bases are
#: closer than this would silently split traffic across one another's ports.
MAX_REPLICAS = 16


def sh(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as exc:  # pragma: no cover - operational tool
        return f"<failed: {exc}>"


def port_base(name: str) -> int:
    return PORT_BASE + (zlib.crc32(name.encode()) % PORT_MODULUS)


def check_ports(profiles: list[str]) -> dict:
    """A new profile whose crc32 base lands near a live one silently splits traffic."""
    from agents_scaling.serving.profiles import SERVING_PROFILES

    ours = {name: port_base(name) for name in profiles}
    existing = {name: port_base(name) for name in SERVING_PROFILES if name not in ours}
    collisions = [
        {"ours": ours_name, "ours_base": base, "other": other, "other_base": other_base,
         "gap": abs(base - other_base)}
        for ours_name, base in ours.items()
        for other, other_base in existing.items()
        if abs(base - other_base) <= MAX_REPLICAS
    ]
    return {"ok": not collisions, "nextdyn_bases": ours, "existing_bases": existing,
            "collisions": collisions}


def check_capacity(compute) -> dict:
    """Idle GPUs per (partition, gpu_type), and the nodes that are drained or excluded."""
    rows = []
    drained: set[str] = set()
    for partition, block in compute.partitions.items():
        for gpu_type in block.gpu:
            census = FL.free_gpus(partition, gpu_type,
                                  exclude=compute.serving.exclude_nodes)
            drained |= set(census.drained)
            rows.append({"partition": partition, "gpu_type": gpu_type, "total": census.total,
                         "used": census.used, "free": census.free,
                         "preemptable": bool(block.preemptable),
                         "drained": list(census.drained)})
    return {"ok": True, "partitions": rows, "drained_nodes": sorted(drained)}


def check_qos(compute) -> dict:
    """The live MaxTRESPU caps -- GPU **per type**, CPU and MEM -- per partition QOS.

    CPU and memory are captured as well as GPUs because ``mit_normal``'s per-user 96 CPU /
    386 G is what actually bounds the deadline lane's cell workers, and a fleet that only
    counted GPUs would queue 200 workers against a 96-CPU ceiling.
    """
    caps: list[dict] = []
    raw: dict[str, str] = {}
    for qos, block in compute.partitions.items():
        text = sh(["sacctmgr", "-nP", "show", "qos", qos,
                   "format=name,maxtrespu,grptres,maxsubmitjobspu,maxwall"])
        raw[qos] = text.strip()
        parsed = _parse_tres(text)
        for gpu_type, planned in block.gpu.items():
            live = parsed.get(f"gres/gpu:{gpu_type}", parsed.get("gres/gpu"))
            caps.append({"partition": qos, "gpu_type": gpu_type,
                         "gpu": int(live if live is not None else planned),
                         "planned": int(planned),
                         "drift": live is not None and int(live) != int(planned)})
        caps.append({"partition": qos, "gpu_type": "", "cpu": parsed.get("cpu", block.cpu),
                     "mem_gb": parsed.get("mem", block.mem_gb), "gpu": parsed.get("gres/gpu", 0)})
    assoc = sh(["sacctmgr", "-nP", "show", "assoc", f"user={os.environ.get('USER', '')}",
                "format=account,qos,maxsubmitjobs"])
    return {"ok": True, "caps": caps, "raw": raw, "association": assoc.strip()}


def _parse_tres(text: str) -> dict:
    """``cpu=96,mem=386G,gres/gpu:h100=8`` -> a dict of ints (GB for memory)."""
    out: dict[str, int] = {}
    for field in text.replace("|", ",").split(","):
        key, _, value = field.strip().partition("=")
        if not key or not value:
            continue
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            continue
        number = int(digits)
        if key == "mem" and value.strip().upper().endswith("T"):
            number *= 1024
        out[key] = number
    return out


def check_other_jobs() -> dict:
    """The user's other project must never be cancelled or counted as nextdyn capacity."""
    raw = sh(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%P|%j|%T|%b|%N|%e"])
    jobs = []
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) < 7:
            continue
        jobs.append({"jobid": fields[0], "partition": fields[1], "name": fields[2],
                     "state": fields[3], "tres": fields[4], "node": fields[5], "end": fields[6],
                     "is_nextdyn": fields[2].startswith(FL.JOB_PREFIX) or "nextdyn" in fields[2]})
    foreign = [j for j in jobs if not j["is_nextdyn"] and "gpu" in j["tres"]]
    return {"ok": True, "n_jobs": len(jobs), "foreign_gpu_jobs": foreign, "jobs": jobs}


def check_filesystem(run_root: Path) -> dict:
    usage = shutil.disk_usage(run_root) if run_root.exists() else None
    inodes = sh(["df", "-i", str(run_root)]).splitlines()[-1:] or [""]
    return {"ok": run_root.exists() or run_root.parent.exists(), "run_root": str(run_root),
            "free_tb": round(usage.free / 1e12, 2) if usage else None,
            "inodes_line": inodes[0].strip()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=C.DEFAULT_STUDY_CONFIG)
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = C.load_study(args.config)
    compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                             or C.DEFAULT_COMPUTE_CONFIG)
    run_root = args.run_root or (Path(os.environ.get("ASYS_RESULTS_ROOT", ".")) / cfg.study.id)
    profiles = sorted({model.profile for model in cfg.models.values()})

    report = {
        "study_id": cfg.study.id,
        "run_root": str(run_root),
        "ports": check_ports(profiles),
        "capacity": check_capacity(compute),
        "qos": check_qos(compute),
        "other_jobs": check_other_jobs(),
        "filesystem": check_filesystem(run_root),
    }
    hard = {k: v for k, v in report.items() if isinstance(v, dict) and "ok" in v}
    report["all_ok"] = all(v["ok"] for v in hard.values())

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")

    print("\n=== SUMMARY ===", file=sys.stderr)
    for name, value in hard.items():
        print(f"  {'PASS' if value['ok'] else 'FAIL'}  {name}", file=sys.stderr)
    drift = [row for row in report["qos"]["caps"] if row.get("drift")]
    if drift:
        print(f"  NOTE  {len(drift)} QOS cap(s) drifted from compute.yaml's snapshot; the LIVE "
              "cap is what `fleet up` enforces", file=sys.stderr)
    if report["other_jobs"]["foreign_gpu_jobs"]:
        print(f"  NOTE  {len(report['other_jobs']['foreign_gpu_jobs'])} non-nextdyn GPU job(s) "
              "of this user are running; never cancel them", file=sys.stderr)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
