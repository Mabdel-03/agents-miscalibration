#!/bin/bash
# study_v4 rolling wave, step 2 (after every JUDGE_BEST cell of the wave has meta.json):
# seal selections (VOTE + JUDGE_BEST) -> JUDGE_HLE cells (judge fleet) + EVAL_BCB cells
# (eval lane, 2 CPU / 8G) -> dispatch.  Idempotent per wave tag.
# usage: slurm/study_wave_finish.sh <generate manifest> <tier> <wave> [judge_throttle] [eval_throttle]
set -euo pipefail
MANIFEST="$1"; TIER="$2"; WAVE="$3"; JT="${4:-16}"; ET="${5:-20}"
RUN="${ASYS_STUDY_RUN:-study_v4}"; JUDGE_RUN="${ASYS_STUDY_JUDGE_RUN:-study_v4_judge}"
export ASYS_RESULTS_ROOT="${ASYS_RESULTS_ROOT:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results}"
ROOT="$ASYS_RESULTS_ROOT/$RUN"; PY="${ASYS_HARNESS_PY:-/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python}"
cd "$(dirname "$0")/.."; export PYTHONPATH=src
SEAL=$(sha256sum "$ROOT/$MANIFEST" | cut -c1-64)
WFILE="waves/${SEAL:0:8}_w${WAVE}.json"; SEL="cells_${TIER}-select_32B.w${WAVE}.json"
MISSING=$("$PY" -c "
import json,sys,os
cells=json.load(open(sys.argv[1]))['cells']; root=sys.argv[2]
print(sum(1 for c in cells if not os.path.exists(os.path.join(root,'cells',c['cell_id'],'meta.json'))))" "$ROOT/$SEL" "$ROOT")
[ "$MISSING" = "0" ] || { echo "[wave $WAVE] $MISSING JUDGE_BEST cells still incomplete"; exit 3; }
"$PY" -m agents_scaling.study.selection.seal --run-id "$RUN" --cells-file "$MANIFEST" --kind selections --select-cells-file "$SEL" | tail -6
for LANE in 32B eval; do
  OUT="cells_${TIER}-eval_${LANE}.w${WAVE}.json"
  if [ ! -s "$ROOT/$OUT" ]; then
    "$PY" -m agents_scaling.study.cells --run-id "$RUN" --tier "${TIER}-eval" --lane "$LANE" --seal "$SEAL" --items-file "$WFILE" --wave "$WAVE" --out "$OUT" | grep -E '"n_cells"|"n_items"'
  fi
  N=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['cells']))" "$ROOT/$OUT")
  if [ "$N" -eq 0 ]; then echo "[wave $WAVE] lane $LANE: no cells"; continue; fi
  if [ "$LANE" = "eval" ]; then
    "$PY" -u slurm/study_launch_chunked.py --allow-legacy-admission --run-id "$RUN" --server-run-id "$RUN" --cells-file "$OUT" --lane eval \
      --chunk-size "$N" --throttle "$ET" --submit-cap 380 --qos-limit 460 --cell-partition mit_preemptable --cell-time 1-00:00:00 --cell-mem 8G --cpus 2 --poll-s 60 | tail -2
  else
    "$PY" -u slurm/study_launch_chunked.py --allow-legacy-admission --run-id "$RUN" --server-run-id "$JUDGE_RUN" --cells-file "$OUT" --lane 32B \
      --chunk-size "$N" --throttle "$JT" --submit-cap 440 --qos-limit 460 --cell-partition mit_preemptable --cell-time 1-00:00:00 --cell-mem 4G --cpus 1 --poll-s 60 | tail -2
  fi
  echo "[wave $WAVE] lane $LANE dispatched: $N cells"
done
echo "[wave $WAVE] evaluation dispatched; aggregate when the eval cells finish"
