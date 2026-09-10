#!/bin/bash
# ARD36 per-replica GPU sampler (WP4; spec §18 resources.parquet, plan §5.8).
#
# Sourced by an ARD36 serve job right after the server starts:
#
#     source "$REPO/slurm/ard36_gpu_sampler.sh"
#     ard36_start_gpu_sampler "$RUN_ROOT/resources/gpu_$SLURM_JOB_ID.csv"
#
# Writes one CSV row per GPU per interval:
#   timestamp,uuid,name,memory.total,memory.used,utilization.gpu,power.draw
# which agents_scaling.ard.resources.parse_nvidia_smi_csv reads back, joined to the
# registry on (host, slurm_job_id).  The query string is the one frozen in
# ard/resources.NVIDIA_SMI_FIELDS — keep the two in step.
ard36_start_gpu_sampler() {
  local out="${1:?usage: ard36_start_gpu_sampler <csv path> [interval_s]}"
  local interval="${2:-15}"
  mkdir -p "$(dirname "$out")"
  ( nvidia-smi \
      --query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits --loop="$interval" \
    | awk -v OFS=, '{ print systime(), $0 }' >> "$out" ) &
  ARD36_GPU_SAMPLER_PID=$!
  echo "[ard36] gpu sampler pid $ARD36_GPU_SAMPLER_PID -> $out (every ${interval}s)"
}

ard36_stop_gpu_sampler() {
  if [ -n "${ARD36_GPU_SAMPLER_PID:-}" ]; then
    kill "$ARD36_GPU_SAMPLER_PID" 2>/dev/null || true
    echo "[ard36] gpu sampler $ARD36_GPU_SAMPLER_PID stopped"
  fi
}
