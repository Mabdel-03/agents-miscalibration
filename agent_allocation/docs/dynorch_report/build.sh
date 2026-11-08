#!/usr/bin/env bash
# dynorch_v1 report build (contract section 7): analysis (numbers.json, tables, figures) -> render -> lint -> compile.
#   ./build.sh                 full rebuild
#   ./build.sh --no-analysis   re-render and compile from the existing numbers.json/figures
# Inputs are read-only: the sealed ladder merge (LADDER) and the dynorch run root (RUN) with its tables/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-$HERE/../../.venv/bin/python}
RUN=${RUN:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/dynorch_v1/artifacts}
LADDER=${LADDER:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts/merged}
CFG=${CFG:-$HERE/../../configs/dynorch_v1.json}
LADDER_NUMBERS=${LADDER_NUMBERS:-$HERE/../forced_v2_report/numbers.json}
SCRATCH=${SCRATCH:-/orcd/data/tpoggio/001/mabdel03/dynorch_scratch}
# TeX must not write under /home (inode quota): font caches and formats go to scratch.
export TEXMFVAR="$SCRATCH/texmf-var" TEXMFCONFIG="$SCRATCH/texmf-config" TMPDIR="$SCRATCH/tmp"
mkdir -p "$TEXMFVAR" "$TEXMFCONFIG" "$TMPDIR"
export PATH="/home/mabdel03/.local/bin:$PATH"   # TinyTeX (read-only use)
cd "$HERE"
if [ "${1:-}" != "--no-analysis" ]; then
  echo "== analysis =="
  "$PY" analysis.py --run "$RUN" --ladder "$LADDER" --config "$CFG" --ladder-numbers "$LADDER_NUMBERS" --out "$HERE"
fi
echo "== render (every number in the prose comes from numbers.json) =="
"$PY" render.py
echo "== lint =="
if grep -n -P '\x{2014}|---' report.tex tables/*.tex; then echo "LINT FAIL: em dash or --- found"; exit 1; fi
if grep -n '??' report.tex; then echo "LINT FAIL: unresolved placeholder"; exit 1; fi
echo "clean"
echo "== compile =="
latexmk -pdf -interaction=nonstopmode report.tex >/dev/null
echo "== log check =="
bad=0
grep -n 'Undefined control sequence\|LaTeX Warning: Reference\|LaTeX Warning: Citation\|File .* not found' report.log && bad=1
echo "overfull hboxes: $(grep -c 'Overfull \\hbox' report.log || true)"
[ "$bad" -eq 0 ] || { echo "LOG CHECK FAIL"; exit 1; }
echo "OK -> $HERE/report.pdf ($(pdfinfo report.pdf | awk '/Pages/{print $2}') pages)"
