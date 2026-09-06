"""Render + submit study_v4 neural capture shards from slurm/study_neural.sbatch.tmpl.

    python slurm/study_neural_launch.py --run-id study_v4 --stage native --checkpoint 32B \
        --num-shards 4 --partition pi_manoli [--shards 0,1] [--extra "--blocks 15,31,47 --fidelity 20"] [--dry-run]

Renders one sbatch per shard under <run_root>/neural/sbatch/ and submits it (unless
--dry-run).  GPUS = 2 for 32B, else 1.  Re-submitting a shard is safe (the capture CLI is
resumable and skips committed keys).  Never uses --requeue.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results"
DEFAULT_PYTHON = "/orcd/data/tpoggio/001/mabdel03/envs/neural_env/bin/python"
DEFAULT_HF_HOME = "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface"


def render(template: str, values: dict[str, str]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    left = re.search(r"(?<!\$)\{[A-Z_]+\}", out)
    if left:
        raise SystemExit(f"unrendered placeholder {left.group(0)}")
    return out


def freeze_work_list(args, run_root: Path) -> Path:
    """Run the capture CLI's --dry-run once (num_shards=1) and persist its selected work items."""
    import json
    out_dir = run_root / "neural" / "worklists"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.stage}_{args.freeze_items}.json"
    if target.exists():
        raise SystemExit(f"{target} exists; frozen work lists are immutable (pick another label)")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), HF_HOME=args.hf_home, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cmd = [args.python, "-m", "agents_scaling.study.neural.capture", "--run-id", args.run_id, "--stage", args.stage,
           "--checkpoint", args.checkpoint, "--shard", "0", "--num-shards", "1", "--results-root", args.results_root, "--dry-run"]
    extra = [a for a in args.extra.split() if a]
    # keep selection-affecting args (blocks/methods/modules/panel/cells-file/generated-ks) but drop run-only ones
    skip = {"--fidelity", "--fidelity-tokens", "--limit", "--max-batch-tokens", "--max-batch-rows", "--mask-policy", "--chunk-rows"}
    i = 0
    while i < len(extra):
        if extra[i] in skip:
            i += 2
            continue
        cmd.append(extra[i])
        i += 1
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"dry-run failed: {proc.stderr.strip()[-800:]}")
    sel = run_root / "neural" / args.stage / f"selection.{args.stage}.s000of001.json"
    doc = json.loads(sel.read_text())
    items = doc.get("shard_items") or []
    if not items:
        raise SystemExit(f"dry-run selected no work items ({sel})")
    target.write_text(json.dumps(items, indent=1))
    (out_dir / f"{args.stage}_{args.freeze_items}.meta.json").write_text(json.dumps(
        {"n_items": len(items), "work_sha256": doc.get("work_sha256"), "selection_file": str(sel), "checkpoint": args.checkpoint,
         "panel_items": doc.get("panel_items"), "methods": doc.get("methods"), "cmd": cmd}, indent=1))
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", required=True, choices=("native", "report"))
    ap.add_argument("--checkpoint", default="32B")
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--shards", default=None, help="comma list of shard indices (default: all)")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--time", default=None, help="override the template walltime (e.g. 8:00:00)")
    ap.add_argument("--mem", default=None, help="override the template memory request (e.g. 80G)")
    ap.add_argument("--cpus", type=int, default=None, help="override the template CPU request")
    ap.add_argument("--extra", default="", help="extra capture CLI args (quoted)")
    ap.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    ap.add_argument("--freeze-items", default=None, metavar="LABEL",
                    help="freeze the stage's work list once (capture --dry-run --num-shards 1) into "
                         "<run_root>/neural/worklists/<stage>_<LABEL>.json and pass it as --items-file to every shard, so "
                         "shards launched together partition the SAME snapshot (never duplicate keys across shards)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    run_root = Path(args.results_root) / args.run_id
    log_dir = run_root / "logs"
    out_dir = run_root / "neural" / "sbatch"
    out_dir.mkdir(parents=True, exist_ok=True)
    template = (REPO / "slurm" / "study_neural.sbatch.tmpl").read_text()
    if args.freeze_items:
        items_path = freeze_work_list(args, run_root)
        args.extra = (args.extra + f" --items-file {items_path}").strip()
        print(f"[freeze] {args.stage} work list frozen at {items_path}")
    if args.time:
        template = re.sub(r"^#SBATCH --time=.*$", f"#SBATCH --time={args.time}", template, flags=re.M)
    if args.mem:
        template = re.sub(r"^#SBATCH --mem=.*$", f"#SBATCH --mem={args.mem}", template, flags=re.M)
    if args.cpus:
        template = re.sub(r"^#SBATCH --cpus-per-task=.*$", f"#SBATCH --cpus-per-task={args.cpus}", template, flags=re.M)
    gpus = "2" if args.checkpoint == "32B" else "1"
    shards = [int(s) for s in args.shards.split(",")] if args.shards else list(range(args.num_shards))
    for shard in shards:
        if not 0 <= shard < args.num_shards:
            raise SystemExit(f"shard {shard} outside 0..{args.num_shards - 1}")
        values = {
            "RUN_ID": args.run_id, "STAGE": args.stage, "SHARD": str(shard), "NUM_SHARDS": str(args.num_shards),
            "PARTITION": args.partition, "GPUS": gpus, "LOG_DIR": str(log_dir), "REPO": str(REPO), "HF_HOME": args.hf_home,
            "PYTHON": args.python, "CHECKPOINT": args.checkpoint, "ARGS": args.extra.strip(),
        }
        text = render(template, values)
        path = out_dir / f"neural_{args.stage}_{args.checkpoint}_s{shard}of{args.num_shards}.sbatch"
        path.write_text(text)
        if args.dry_run:
            print(f"[dry-run] rendered {path}")
            continue
        proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[launch] sbatch failed for shard {shard}: {proc.stderr.strip()[:300]}", file=sys.stderr)
            return 1
        print(f"[launch] {args.stage} {args.checkpoint} shard {shard}/{args.num_shards} on {args.partition}: {proc.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
