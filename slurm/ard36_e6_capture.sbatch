#!/bin/bash
#SBATCH --job-name=asys-ard36-e6-capture
#SBATCH --partition=pi_manoli
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36/logs/e6_capture_%j.out
#SBATCH --no-requeue
#SBATCH --export=NONE

# ARD36 E6 capture: teacher-forced replay of the frozen 144-item capture panel (spec §14.2,
# §14.5) on ONE A100.  No vLLM, no endpoint, no generation: the request store already holds
# prompt_token_ids + response.token_ids for every decision, so this job replays stored tokens
# and writes fp16 residual-stream vectors at {PROMPT_END, PRE_FINAL, ANSWER_END, REPORT_END}
# × blocks (8, 17, 26).  Nothing here is authorised to produce a probe or a prediction — the
# panel is sized for descriptive geometry only.
#
# Environment block mirrors tests/study_neural/gpu_smoke.sbatch, which is the configuration
# the CaptureEngine was verified on (offline HF, node-local compile/caches, causal mask
# policy, expandable_segments).  It deliberately does NOT source slurm/common.sh: that
# activates the harness env and prepends the serve env's libstdc++ for vLLM/FlashInfer,
# neither of which this job wants.
#
# Measured cost of the whole panel (this run root, 2026-09-13, from the frozen manifest and
# the request store — not an estimate): 18 cells × 8 items × 400 decisions = 7,200 replays,
# 9.57 M tokens, 86,400 stored vectors ≈ 708 MB of fp16 + ~120 MB of ledger.  Max sequence
# 3,136 tokens, so every replay fits under --max-batch-tokens.  The 4-hour walltime is
# ~3-4x the expected time for the default all-18-cell range and leaves room for a cold NFS
# weight load per cell.
#
# Template variables: --partition (a partition with an a100; set to pi_manoli). Everything else is an
# environment variable with a default, so the orchestrator can submit this file as-is:
#   ARD36_CELLS_FILE  cells file under the run root      (default cells_E6_8B.json)
#   ARD36_INDEX_START first cell index to run            (default 0)
#   ARD36_INDEX_END   last cell index, inclusive         (default: the last cell in the file)
#   ARD36_PYTHON      interpreter                        (default the neural env)
#   ARD36_DRY_RUN     1 = resolve and print, load nothing
#
# DO NOT SUBMIT WITHOUT REVIEW.  The capture worker now refuses to write a complete cell
# that captured nothing (ard.neural.capture.CaptureCellError), so a misconfigured job fails
# loudly here instead of leaving an empty native stage that looks finished.

set -euo pipefail
umask 027

REPO=/orcd/data/tpoggio/001/mabdel03/agents_scaling
RESULTS_ROOT=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
RUN_ID=ard36
RUN_ROOT="$RESULTS_ROOT/$RUN_ID"
CELLS_FILE="${ARD36_CELLS_FILE:-cells_E6_8B.json}"
PYTHON="${ARD36_PYTHON:-/orcd/data/tpoggio/001/mabdel03/envs/neural_env/bin/python}"

export PATH=/usr/bin:/bin
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1
export ASYS_RESULTS_ROOT="$RESULTS_ROOT"

# Node-local scratch: chunks are staged here and consolidated into the run root atomically
# (spec §14.5).  ARD36_LOCAL_SCRATCH is what capture.staging_root reads first.
ASYS_LOCAL_CACHE="/tmp/asys_ard36_e6_${SLURM_JOB_ID:-$$}"
mkdir -p "$ASYS_LOCAL_CACHE"/{xdg,triton,torchinductor,tmp,staging}
trap 'rm -rf "$ASYS_LOCAL_CACHE"' EXIT
export XDG_CACHE_HOME="$ASYS_LOCAL_CACHE/xdg"
export TRITON_CACHE_DIR="$ASYS_LOCAL_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$ASYS_LOCAL_CACHE/torchinductor"
export TMPDIR="$ASYS_LOCAL_CACHE/tmp"
export ARD36_LOCAL_SCRATCH="$ASYS_LOCAL_CACHE/staging"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_ROOT/logs"
echo "[e6] job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
cd "$REPO"
"$PYTHON" -c "import torch, transformers; print('[e6] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

# Preflight: read the frozen manifest, verify every cell carries the options the worker
# consumes, and report the work list size.  This is the check that would have caught the
# wiring defect before a GPU hour was spent.
PREFLIGHT=$("$PYTHON" - "$RUN_ROOT" "$CELLS_FILE" <<'PY'
import sys
from pathlib import Path
from agents_scaling.ard.cells import load_ard_cells_file
from agents_scaling.ard.neural import capture as C

run_root, name = Path(sys.argv[1]), sys.argv[2]
path = Path(name) if Path(name).is_absolute() else run_root / name
cells = load_ard_cells_file(path)
total = 0
for cell in cells:
    for key in ("snapshot_path",):
        if not cell.options.get(key):
            raise SystemExit(f"{cell.cell_id}: options carry no {key!r}")
    for item in cell.items:
        total += len(C.resolve_cell_work(run_root, item, cell.options))   # raises on a bad options block
print(f"{len(cells)} {total}")
PY
)
N_CELLS=${PREFLIGHT% *}
N_DECISIONS=${PREFLIGHT#* }
echo "[e6] cells_file=$CELLS_FILE cells=$N_CELLS decisions=$N_DECISIONS"

START="${ARD36_INDEX_START:-0}"
END="${ARD36_INDEX_END:-$((N_CELLS - 1))}"
if [ "${ARD36_DRY_RUN:-0}" = "1" ]; then
  echo "[e6] dry run: would replay indices $START..$END; exiting before the model load"
  exit 0
fi

# One cell per run_one invocation (the frozen CLI), sequentially on the single GPU.  Exit 5
# is EXIT_DEPS_UNMET — no meta.json, no work done — which the lane driver re-queues; any
# other non-zero is a real failure and must stop the job so it is not mistaken for a
# complete capture.
for index in $(seq "$START" "$END"); do
  echo "[e6] cell index $index start=$(date -Is)"
  set +e
  "$PYTHON" -m agents_scaling.ard.run_one \
    --run-id "$RUN_ID" \
    --cells-file "$CELLS_FILE" \
    --index "$index"
  rc=$?
  set -e
  if [ "$rc" -eq 5 ]; then
    echo "[e6] cell index $index: dependencies unmet (EXIT_DEPS_UNMET); nothing written, the lane loop will re-queue it"
    continue
  fi
  if [ "$rc" -ne 0 ]; then
    echo "[e6] cell index $index FAILED rc=$rc" >&2
    exit "$rc"
  fi
  echo "[e6] cell index $index done=$(date -Is)"
done

echo "[e6] end=$(date -Is) indices=$START..$END"
