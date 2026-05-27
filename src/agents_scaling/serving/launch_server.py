"""Launch (and register) a vLLM server for one model size on SLURM.

Two roles, selected by flag:

* default — render ``slurm/serve_qwen.sbatch.tmpl`` for a model size and ``sbatch`` it.
  Picks a deterministic port from the model size so re-launches are stable.
* ``--register`` — run *inside* the serving job after vLLM starts: healthcheck the local
  server, then write its ``node:port`` to the registry so cell jobs can discover it.
"""

from __future__ import annotations

import argparse
import subprocess
import zlib
from pathlib import Path

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.models import get_model
from agents_scaling.serving import healthcheck, registry

REPO = Path(__file__).resolve().parents[3]  # .../agents_scaling
TEMPLATE = REPO / "slurm" / "serve_qwen.sbatch.tmpl"

# A100-80GB nodes; pi_tpoggio is the dedicated 7-day partition.
_PARTITION_DEFAULT = "pi_tpoggio"
_GPU_TYPE_DEFAULT = "a100"


def _port_for(model_size: str) -> int:
    """Deterministic port in [8000, 8999] from the model size string."""
    return 8000 + (zlib.crc32(model_size.encode()) % 1000)


def render_sbatch(
    model_size: str,
    run_root: str,
    partition: str,
    gpu_type: str,
    time_limit: str,
    log_dir: str,
) -> str:
    spec = get_model(model_size)
    port = _port_for(model_size)
    cpus = max(8, spec.tp_size * 8)
    mem = f"{spec.tp_size * 120}G"
    text = TEMPLATE.read_text()
    repl = {
        "MODEL_SIZE": model_size,
        "HF_ID": spec.hf_id,
        "TP_SIZE": str(spec.tp_size),
        "MAX_MODEL_LEN": str(spec.max_model_len),
        "PARTITION": partition,
        "GPU_TYPE": gpu_type,
        "CPUS": str(cpus),
        "MEM": mem,
        "TIME": time_limit,
        "PORT": str(port),
        "RUN_ROOT": run_root,
        "LOG_DIR": log_dir,
        "REPO": str(REPO),
    }
    for k, v in repl.items():
        text = text.replace("{" + k + "}", v)
    return text


def submit(model_size: str, run_root: str, partition: str, gpu_type: str, time_limit: str) -> str:
    log_dir = Path(run_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sbatch_text = render_sbatch(model_size, run_root, partition, gpu_type, time_limit, str(log_dir))
    sbatch_path = Path(run_root) / "servers" / f"serve_{model_size}.sbatch"
    sbatch_path.parent.mkdir(parents=True, exist_ok=True)
    sbatch_path.write_text(sbatch_text)
    out = subprocess.run(
        ["sbatch", str(sbatch_path)], capture_output=True, text=True, check=True
    ).stdout.strip()
    # "Submitted batch job 12345"
    job_id = out.split()[-1]
    print(f"[launch] model_size={model_size} -> job {job_id} (sbatch: {sbatch_path})")
    return job_id


def _register_role(run_root: str, model_size: str, hf_id: str, port: int) -> None:
    """Run inside the serving job: wait for local vLLM, then publish endpoint."""
    healthcheck.wait_until_ready("localhost", port, timeout_s=2400.0)
    entry = registry.register_server(run_root, model_size, hf_id, port)
    print(f"[register] {model_size} ready at {entry.base_url}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch or register a vLLM server.")
    ap.add_argument("--register", action="store_true", help="run inside the serving job")
    ap.add_argument("--model-size", required=True)
    ap.add_argument("--run-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--hf-id", help="(register role) the HF id being served")
    ap.add_argument("--port", type=int, help="(register role) local vLLM port")
    ap.add_argument("--partition", default=_PARTITION_DEFAULT)
    ap.add_argument("--gpu-type", default=_GPU_TYPE_DEFAULT)
    ap.add_argument("--time", dest="time_limit", default="2-00:00:00")
    args = ap.parse_args()

    if args.register:
        if not args.hf_id or args.port is None:
            ap.error("--register requires --hf-id and --port")
        _register_role(args.run_root, args.model_size, args.hf_id, args.port)
    else:
        submit(args.model_size, args.run_root, args.partition, args.gpu_type, args.time_limit)


if __name__ == "__main__":
    main()
