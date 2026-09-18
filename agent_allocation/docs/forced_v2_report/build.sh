#!/usr/bin/env bash
# Regenerate every number, table and figure from the merged sealed tables, render the prose, compile.
#   ./build.sh                 full rebuild (analysis ~2 min)
#   ./build.sh --no-analysis   re-render and compile from the existing numbers.json/figures
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-$HERE/../../.venv/bin/python}
RUN=${RUN:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts}
V1=${V1:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1}
export PATH="/home/mabdel03/.local/bin:$PATH"   # TinyTeX
cd "$HERE"
if [ "${1:-}" != "--no-analysis" ]; then
  echo "== analysis =="
  "$PY" analysis.py --run "$RUN/merged" --v1 "$V1" --calibration "$RUN/calibration.json"
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
