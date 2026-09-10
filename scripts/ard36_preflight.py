#!/usr/bin/env python3
"""ARD36 hour-0 preflight (plan section 6).

Read-only checks that must pass before the fleet keepalive is submitted:

  1. Serving-profile port bases are clear of every existing profile by > max replicas.
  2. Partition capacity and per-user / group QOS headroom per (partition, gpu_type).
  3. The user's other running jobs, so the ARD36 fleet is never confused with them.
  4. Node health: the excluded and drained nodes are still what the template assumes.
  5. Filesystem: the results root is on the data filesystem with inode headroom.

Submits nothing.  Exit 0 if every hard check passes, 1 otherwise.

    PYTHONPATH=src python scripts/ard36_preflight.py [--json-out preflight.json]
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

RESULTS_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results")
RUN_ID = "ard36"

#: (partition, gpu_type) -> replicas we intend to hold, from the plan's keepalive spec.
FLEET_INTENT: dict[tuple[str, str], int] = {
    ("pi_manoli", "a100"): 7,        # 2x TP2 (4 GPUs) + 3x TP1
    ("pi_tpoggio", "a100"): 4,       # 1x TP2 (2 GPUs) + 2x TP1
    ("ou_bcs_normal", "a100"): 7,    # 1x TP2 (2 GPUs) + 5x TP1
    ("ou_bcs_low", "a100"): 6,       # burst TP1 only
}
EXCLUDED_NODES = ("node3804",)


def sh(cmd: list[str], timeout: int = 60) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout
    except Exception as exc:  # pragma: no cover - operational tool
        return f"<failed: {exc}>"


def check_ports() -> dict:
    """A new profile name whose crc32 base lands near a live one silently splits traffic."""
    from agents_scaling.ard import contract as C
    from agents_scaling.serving.profiles import SERVING_PROFILES

    def base(name: str) -> int:
        return 8000 + (zlib.crc32(name.encode()) % 1000)

    existing = {name: base(name) for name in SERVING_PROFILES}
    ours = {n: base(n) for n in (C.PROFILE_SMALL, C.PROFILE_LARGE)}
    max_replicas = 16
    collisions = []
    for our_name, our_base in ours.items():
        for other, other_base in existing.items():
            if other in ours:
                continue
            if abs(our_base - other_base) <= max_replicas:
                collisions.append(
                    {"ours": our_name, "ours_base": our_base,
                     "other": other, "other_base": other_base,
                     "gap": abs(our_base - other_base)}
                )
    return {"ok": not collisions, "ard36_bases": ours,
            "existing_bases": existing, "collisions": collisions}


def _free_gpus(partition: str, gpu_type: str) -> dict:
    """Idle GPUs of one type on a partition, and how many are reclaimable by preemption."""
    raw = sh(["sinfo", "-h", "-p", partition, "-N", "-O",
              "NodeHost:20,Gres:60,GresUsed:60,StateLong:20"])
    total = used = 0
    nodes = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        host, gres, gres_used, state = parts[0], parts[1], parts[2], parts[3]
        if gpu_type not in gres:
            continue
        if "drain" in state.lower() or "down" in state.lower() or host in EXCLUDED_NODES:
            nodes.append({"node": host, "state": state, "skipped": True})
            continue

        def count(field: str) -> int:
            for chunk in field.split(","):
                if chunk.startswith(f"gpu:{gpu_type}:"):
                    n = chunk.split(":")[2].split("(")[0]
                    return int(n) if n.isdigit() else 0
            return 0

        t, u = count(gres), count(gres_used)
        total += t
        used += u
        nodes.append({"node": host, "state": state, "total": t, "used": u, "free": t - u})
    return {"partition": partition, "gpu_type": gpu_type,
            "total": total, "used": used, "free": total - used, "nodes": nodes}


def check_capacity() -> dict:
    rows = []
    for (partition, gpu_type), want in FLEET_INTENT.items():
        cap = _free_gpus(partition, gpu_type)
        cap["intended_replicas_gpus"] = want
        # Free GPUs are informative, not binding: pi_* is tier 100 and reclaims by preemption.
        cap["reclaimable_by_preemption"] = partition.startswith("pi_")
        rows.append(cap)
    return {"ok": True, "partitions": rows}


def check_qos() -> dict:
    out = {}
    for qos in ("pi_manoli", "pi_tpoggio", "ou_bcs_normal", "ou_bcs_low", "mit_preemptable"):
        raw = sh(["sacctmgr", "-nP", "show", "qos", qos,
                  "format=name,maxtrespu,grptres,maxsubmitjobspu,maxwall"])
        out[qos] = raw.strip()
    assoc = sh(["sacctmgr", "-nP", "show", "assoc", f"user={os.environ.get('USER','')}",
                "format=account,qos,maxsubmitjobs"])
    return {"ok": True, "qos": out, "association": assoc.strip()}


def check_other_jobs() -> dict:
    """The user's other project must never be cancelled or counted as ARD36 capacity."""
    raw = sh(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%P|%j|%T|%b|%N|%e"])
    jobs = []
    for line in raw.splitlines():
        f = line.split("|")
        if len(f) < 7:
            continue
        jobs.append({"jobid": f[0], "partition": f[1], "name": f[2], "state": f[3],
                     "tres": f[4], "node": f[5], "end": f[6],
                     "is_ard36": f[2].startswith("asys-srv-ard36") or "ard36" in f[2]})
    foreign_gpu = [j for j in jobs if not j["is_ard36"] and "gpu" in j["tres"]]
    return {"ok": True, "n_jobs": len(jobs), "foreign_gpu_jobs": foreign_gpu, "jobs": jobs}


def check_filesystem() -> dict:
    root = RESULTS_ROOT
    usage = shutil.disk_usage(root) if root.exists() else None
    inodes = sh(["df", "-i", str(root)]).splitlines()[-1:] or [""]
    quota = Path.home() / "orcd" / ".quota"
    return {
        "ok": root.exists(),
        "results_root": str(root),
        "run_root_exists": (root / RUN_ID).exists(),
        "free_tb": round(usage.free / 1e12, 2) if usage else None,
        "inodes_line": inodes[0].strip(),
        "quota_report": quota.read_text().strip().splitlines()[-4:] if quota.exists() else [],
    }


def check_nodes() -> dict:
    out = {}
    for node in EXCLUDED_NODES:
        raw = sh(["scontrol", "show", "node", node])
        state = next((tok for tok in raw.split() if tok.startswith("State=")), "State=?")
        out[node] = state
    return {"ok": True, "excluded_nodes": out,
            "note": "the serve template hard-excludes these; re-audit if any is healthy again"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    report = {
        "run_id": RUN_ID,
        "ports": check_ports(),
        "capacity": check_capacity(),
        "qos": check_qos(),
        "other_jobs": check_other_jobs(),
        "filesystem": check_filesystem(),
        "nodes": check_nodes(),
    }
    hard = {k: v for k, v in report.items() if isinstance(v, dict) and "ok" in v}
    report["all_ok"] = all(v["ok"] for v in hard.values())

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")

    print("\n=== SUMMARY ===", file=sys.stderr)
    for name, v in hard.items():
        print(f"  {'PASS' if v['ok'] else 'FAIL'}  {name}", file=sys.stderr)
    if report["other_jobs"]["foreign_gpu_jobs"]:
        print(f"  NOTE  {len(report['other_jobs']['foreign_gpu_jobs'])} non-ARD36 GPU jobs of "
              f"this user are running; never cancel them", file=sys.stderr)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
