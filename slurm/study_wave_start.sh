#!/bin/bash
# study_v4 rolling wave, step 1: seal pools -> plan the wave (complete items only) ->
# JUDGE_BEST cells (wave-tagged) -> dispatch on the judge fleet.  Idempotent per wave tag.
# usage: slurm/study_wave_start.sh <generate manifest> <tier> <wave> [min_items] [throttle]
set -euo pipefail
MANIFEST="$1"; TIER="$2"; WAVE="$3"; MIN="${4:-1}"; THROTTLE="${5:-16}"
RUN="${ASYS_STUDY_RUN:-study_v4}"; JUDGE_RUN="${ASYS_STUDY_JUDGE_RUN:-study_v4_judge}"
export ASYS_RESULTS_ROOT="${ASYS_RESULTS_ROOT:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results}"
ROOT="$ASYS_RESULTS_ROOT/$RUN"; PY="${ASYS_HARNESS_PY:-/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python}"
cd "$(dirname "$0")/.."; export PYTHONPATH=src
SEAL=$(sha256sum "$ROOT/$MANIFEST" | cut -c1-64)
echo "[wave $WAVE] manifest $MANIFEST seal ${SEAL:0:12}"
"$PY" -m agents_scaling.study.selection.seal --run-id "$RUN" --cells-file "$MANIFEST" --kind pools | tail -4
WFILE="waves/${SEAL:0:8}_w${WAVE}.json"
if [ ! -s "$ROOT/$WFILE" ]; then
  "$PY" -m agents_scaling.study.waves --run-id "$RUN" --cells-file "$MANIFEST" --seal "$SEAL" --wave "$WAVE" --min-items "$MIN" || { echo "[wave $WAVE] fewer than $MIN new complete items; nothing dispatched"; exit 3; }
fi
OUT="cells_${TIER}-select_32B.${SEAL:0:8}.w${WAVE}.json"
if [ ! -s "$ROOT/$OUT" ]; then
  "$PY" -m agents_scaling.study.cells --run-id "$RUN" --tier "${TIER}-select" --lane 32B --seal "$SEAL" --items-file "$WFILE" --wave "$WAVE" --out "$OUT" | grep -E '"n_cells"|"n_items"|sha256'
fi
N=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['cells']))" "$ROOT/$OUT")
[ "$N" -gt 0 ] || { echo "[wave $WAVE] no JUDGE_BEST cells"; exit 0; }
"$PY" -u slurm/study_launch_chunked.py --allow-legacy-admission --run-id "$RUN" --server-run-id "$JUDGE_RUN" --cells-file "$OUT" --lane 32B \
  --chunk-size "$N" --throttle "$THROTTLE" --submit-cap 440 --qos-limit 460 --cell-partition mit_preemptable --cell-time 1-00:00:00 --cell-mem 4G --cpus 1 --poll-s 60 | tail -3
echo "[wave $WAVE] JUDGE_BEST dispatched: $N cells; next: slurm/study_wave_finish.sh $MANIFEST $TIER $WAVE"
