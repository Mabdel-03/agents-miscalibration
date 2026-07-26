#!/usr/bin/env bash
# Refresh analysis caches. Safe to re-run as sweep cells land.
#
# The default is the authoritative schema-5 primary dataset and cannot be pointed at a
# subset or at legacy directories.  Historical data requires the explicit invocation:
#
#   ANALYSIS_MODE=supplementary-legacy INCLUDE_UNMANIFESTED=1 bash analysis/refresh.sh
#
# which writes a separate mixed-protocol cache by default.
# Production analysis executes the ingest implementation from the frozen release with
# the immutable harness interpreter recorded in schema-5 control.  The mutable checkout
# is used only as the default output location.
set -euo pipefail

CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RESULTS_ROOT="${ASYS_RESULTS_ROOT:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results}"
CONTROL_STATE_DIR="${SCHEMA5_CONTROL_STATE_DIR:-$RESULTS_ROOT/.dispatcher-schema5-v1}"
CONTROL_JSON="$CONTROL_STATE_DIR/control.json"
BOOTSTRAP_PYTHON="${SCHEMA5_BOOTSTRAP_PYTHON:-python3}"
if [[ ! -f "$CONTROL_JSON" ]]; then
  echo "[refresh] missing schema-5 control authority: $CONTROL_JSON" >&2
  exit 2
fi
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH LD_PRELOAD

# Bootstrap only the interpreter/release paths, then ask the frozen control
# implementation under that interpreter to verify every immutable file pin.  No code
# from the mutable checkout participates in the trusted validation or ingest process.
BOOTSTRAP="$("$BOOTSTRAP_PYTHON" -I - "$CONTROL_JSON" <<'PY'
import json
from pathlib import Path
import sys

control_path = Path(sys.argv[1])
payload = json.loads(control_path.read_text(encoding="utf-8"))
immutable = payload.get("immutable")
if not isinstance(immutable, dict):
    raise SystemExit("control.json lacks immutable pins")
for field in ("release_worktree", "harness_environment_prefix"):
    value = immutable.get(field)
    path = Path(value) if isinstance(value, str) else Path("")
    if not value or not path.is_absolute() or "\n" in value or "\r" in value:
        raise SystemExit(f"invalid bootstrap pin {field}")
    print(path.resolve())
PY
)"
mapfile -t BOOTSTRAP_FIELDS <<< "$BOOTSTRAP"
if [[ "${#BOOTSTRAP_FIELDS[@]}" -ne 2 ]]; then
  echo "[refresh] failed to bootstrap exact schema-5 runtime" >&2
  exit 2
fi
RELEASE_WORKTREE="${BOOTSTRAP_FIELDS[0]}"
HARNESS_PREFIX="${BOOTSTRAP_FIELDS[1]}"
PY="$HARNESS_PREFIX/bin/python"
if [[ ! -x "$PY" || ! -f "$RELEASE_WORKTREE/slurm/schema5_control.py" ]]; then
  echo "[refresh] pinned analysis runtime is incomplete" >&2
  exit 2
fi

VERIFIED="$(
  LD_LIBRARY_PATH="$HARNESS_PREFIX/lib" "$PY" -I - \
    "$RELEASE_WORKTREE" "$CONTROL_STATE_DIR" <<'PY'
from pathlib import Path
import sys

release = Path(sys.argv[1]).resolve()
state_dir = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(release))
from slurm.schema5_control import load_control

control = load_control(state_dir, verify_files=True)
immutable = control["immutable"]
if Path(immutable["release_worktree"]).resolve() != release:
    raise SystemExit("release bootstrap/control mismatch")
fields = (
    immutable["release_worktree"],
    immutable["harness_environment_prefix"],
    immutable["hf_home"],
    immutable["results_root"],
    immutable["harness_environment_sha256"],
    control["immutable_sha256"],
    immutable["model_contract_path"],
    immutable["model_contract_sha256"],
    immutable["release_id"],
    immutable["git_commit"],
    immutable["source_tree_sha256"],
)
for value in fields:
    text = str(value)
    if "\n" in text or "\r" in text:
        raise SystemExit("unsafe newline in immutable analysis pin")
    print(text)
PY
)"
mapfile -t VERIFIED_FIELDS <<< "$VERIFIED"
if [[ "${#VERIFIED_FIELDS[@]}" -ne 11 ]]; then
  echo "[refresh] frozen control validation returned incomplete pins" >&2
  exit 2
fi
RELEASE_WORKTREE="${VERIFIED_FIELDS[0]}"
HARNESS_PREFIX="${VERIFIED_FIELDS[1]}"
HF_HOME_PIN="${VERIFIED_FIELDS[2]}"
RESULTS_ROOT="${VERIFIED_FIELDS[3]}"
HARNESS_SHA256="${VERIFIED_FIELDS[4]}"
IMMUTABLE_SHA256="${VERIFIED_FIELDS[5]}"
MODEL_CONTRACT_PATH="${VERIFIED_FIELDS[6]}"
MODEL_CONTRACT_SHA256="${VERIFIED_FIELDS[7]}"
RELEASE_ID="${VERIFIED_FIELDS[8]}"
GIT_COMMIT="${VERIFIED_FIELDS[9]}"
SOURCE_TREE_SHA256="${VERIFIED_FIELDS[10]}"
PY="$HARNESS_PREFIX/bin/python"
INGEST="$RELEASE_WORKTREE/analysis/nb_lib/ingest.py"
if [[ ! -f "$INGEST" ]]; then
  echo "[refresh] frozen ingest entry point is missing: $INGEST" >&2
  exit 2
fi

ANALYSIS_MODE="${ANALYSIS_MODE:-primary-schema5}"
if [[ "$ANALYSIS_MODE" == "primary-schema5" ]]; then
  DEFAULT_RUN_IDS="full_sweep_schema5_v1 full_sweep_agent_counts_schema5_v1 full_sweep_agent_count_7_schema5_v1"
  DEFAULT_OUT_DIR="analysis/cache"
elif [[ "$ANALYSIS_MODE" == "interim-schema5" ]]; then
  DEFAULT_RUN_IDS="full_sweep_schema5_v1 full_sweep_agent_counts_schema5_v1 full_sweep_agent_count_7_schema5_v1"
  DEFAULT_OUT_DIR="analysis/cache/interim_schema5"
elif [[ "$ANALYSIS_MODE" == "supplementary-legacy" ]]; then
  DEFAULT_RUN_IDS="full_sweep_v1 full_sweep_agent_counts_v1 full_sweep_agent_count_7_v1"
  DEFAULT_OUT_DIR="analysis/cache/supplementary_legacy"
else
  echo "[refresh] invalid ANALYSIS_MODE=$ANALYSIS_MODE" >&2
  exit 2
fi
read -r -a RUN_IDS <<< "${RUN_IDS:-$DEFAULT_RUN_IDS}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$CHECKOUT/$OUT_DIR"
fi
cd "$CHECKOUT"

# Canonical item/agent/cell caches.  The ingestion CLI independently enforces the exact
# three-run set, artifact policies, and manifest/benchmark hashes.
INGEST_ARGS=(
  --mode "$ANALYSIS_MODE"
  --run-id "${RUN_IDS[@]}"
  --out-dir "$OUT_DIR"
)
if [[ "$ANALYSIS_MODE" == "primary-schema5" ]]; then
  if [[ -n "${ASYS_TRUSTED_GENERATION_CATALOG:-}" || -n "${ASYS_SERVER_POOL_ROOT:-}" ]]; then
    if [[ -z "${ASYS_TRUSTED_GENERATION_CATALOG:-}" || -z "${ASYS_SERVER_POOL_ROOT:-}" ]]; then
      echo "[refresh] catalog marker and server-pool root must be supplied together" >&2
      exit 2
    fi
    CATALOG_MARKER="$ASYS_TRUSTED_GENERATION_CATALOG"
    SERVER_POOL_ROOT="$ASYS_SERVER_POOL_ROOT"
  else
    CATALOG_AUTHORITY="$(
      LD_LIBRARY_PATH="$HARNESS_PREFIX/lib" "$PY" -I - \
        "$RELEASE_WORKTREE" "$CONTROL_STATE_DIR" <<'PY'
from pathlib import Path
import sys

release = Path(sys.argv[1]).resolve()
state_dir = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(release))
from slurm.schema5_control import load_control
from agents_scaling.serving.generation_catalog import (
    load_current_trusted_generation_catalog,
)

control = load_control(state_dir, verify_files=True)
pool = Path(control["immutable"]["server_pool_root"]).resolve()
catalog = load_current_trusted_generation_catalog(
    state_dir, server_pool_root=pool, required=True
)
assert catalog is not None
print(catalog.marker_path)
print(pool)
PY
    )"
    mapfile -t CATALOG_FIELDS <<< "$CATALOG_AUTHORITY"
    if [[ "${#CATALOG_FIELDS[@]}" -ne 2 ]]; then
      echo "[refresh] trusted-generation catalog authority is incomplete" >&2
      exit 2
    fi
    CATALOG_MARKER="${CATALOG_FIELDS[0]}"
    SERVER_POOL_ROOT="${CATALOG_FIELDS[1]}"
  fi
  INGEST_ARGS+=(
    --trusted-generation-catalog "$CATALOG_MARKER"
    --server-pool-root "$SERVER_POOL_ROOT"
  )
fi
if [[ "${INCLUDE_UNMANIFESTED:-0}" == "1" ]]; then
  INGEST_ARGS+=(--include-unmanifested)
fi
if [[ -n "${PRE_REPAIR_SNAPSHOT_ROOT:-}" ]]; then
  INGEST_ARGS+=(--pre-repair-snapshot-root "$PRE_REPAIR_SNAPSHOT_ROOT")
fi
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH LD_PRELOAD
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$HF_HOME_PIN"
export ASYS_RESULTS_ROOT="$RESULTS_ROOT"
export ASYS_RELEASE_WORKTREE="$RELEASE_WORKTREE"
export ASYS_HARNESS_ENVIRONMENT_PREFIX="$HARNESS_PREFIX"
export ASYS_HARNESS_ENVIRONMENT_SHA256="$HARNESS_SHA256"
export ASYS_IMMUTABLE_PINS_SHA256="$IMMUTABLE_SHA256"
export ASYS_MODEL_CONTRACT="$MODEL_CONTRACT_PATH"
export ASYS_MODEL_CONTRACT_SHA256="$MODEL_CONTRACT_SHA256"
export ASYS_RELEASE_ID="$RELEASE_ID"
export ASYS_RELEASE_GIT_COMMIT="$GIT_COMMIT"
export ASYS_SOURCE_TREE_SHA256="$SOURCE_TREE_SHA256"
LD_LIBRARY_PATH="$HARNESS_PREFIX/lib" "$PY" -I "$INGEST" "${INGEST_ARGS[@]}"

echo "[refresh] done: $(date) — mode=$ANALYSIS_MODE out=$OUT_DIR"
