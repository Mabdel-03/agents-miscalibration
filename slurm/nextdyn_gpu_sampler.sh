#!/bin/bash
# nextdyn per-replica GPU sampler (WP1; spec 18 resources.parquet, plan §3.7).
#
# A fork of slurm/ard36_gpu_sampler.sh under the nextdyn_ prefix: nothing here may edit an
# ARD36 file, and the two runs' serve jobs must be able to sample side by side without one
# function definition shadowing the other.
#
# Sourced by a nextdyn serve job right after the server starts:
#
#     source "$REPO/slurm/nextdyn_gpu_sampler.sh"
#     nextdyn_start_gpu_sampler "$RUN_ROOT/resources/gpu_$SLURM_JOB_ID.csv"
#
# Writes one CSV row per GPU per interval:
#   timestamp,uuid,name,memory.total,memory.used,utilization.gpu,power.draw
# which agents_scaling.nextdyn.telemetry.parse_nvidia_smi_csv (= ard.resources.parse_nvidia_smi_csv,
# imported through the ARD_IMPORTS allowlist) reads back, joined to the registry on
# (host, slurm_job_id).  The query string is exactly ard/resources.NVIDIA_SMI_FIELDS and the
# reader is that module's parser, so the two cannot drift apart:
# tests/nextdyn/test_telemetry.py::test_the_sampler_query_is_the_declared_field_list pins it.
#
# The sampled utilization is the cross-check in conservation_report.json: attributed GPU
# seconds <= union busy seconds <= allocated GPU seconds, per replica, with a replica that was
# never sampled recorded as "unsampled" rather than silently counted as idle.
nextdyn_start_gpu_sampler() {
  local out="${1:?usage: nextdyn_start_gpu_sampler <csv path> [interval_s]}"
  local interval="${2:-15}"
  mkdir -p "$(dirname "$out")"
  ( nvidia-smi \
      --query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits --loop="$interval" \
    | awk -v OFS=, '{ print systime(), $0 }' >> "$out" ) &
  NEXTDYN_GPU_SAMPLER_PID=$!
  echo "[nextdyn] gpu sampler pid $NEXTDYN_GPU_SAMPLER_PID -> $out (every ${interval}s)"
}

nextdyn_stop_gpu_sampler() {
  if [ -n "${NEXTDYN_GPU_SAMPLER_PID:-}" ]; then
    kill "$NEXTDYN_GPU_SAMPLER_PID" 2>/dev/null || true
    echo "[nextdyn] gpu sampler $NEXTDYN_GPU_SAMPLER_PID stopped"
  fi
}
