#!/usr/bin/env bash
# passk_cont_v1 report: numbers/tables/figures from the run root, render the prose, lint, compile.
#   ./build.sh                 full rebuild (analysis reads RUN: e0/, scored/, sealed/, analysis_inputs/)
#   ./build.sh --no-analysis   re-render and compile from the existing numbers.json/figures
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-$HERE/../../.venv/bin/python}
RUN=${RUN:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/orcd/data/tpoggio/001/mabdel03/.cache/matplotlib}
export PATH="/home/mabdel03/.local/bin:$PATH"   # TinyTeX
cd "$HERE"
if [ "${1:-}" != "--no-analysis" ]; then
  echo "== analysis =="
  "$PY" analysis.py --run-root "$RUN"
fi
echo "== render (every number in the prose comes from numbers.json) =="
"$PY" render.py
echo "== lint =="
if grep -n -P '\x{2014}|---' report.tex tables/*.tex 2>/dev/null; then echo "LINT FAIL: em dash or --- found"; exit 1; fi
if grep -n '??' report.tex; then echo "LINT FAIL: unresolved placeholder"; exit 1; fi
echo "clean"
if command -v latexmk >/dev/null; then
  echo "== compile =="
  latexmk -pdf -interaction=nonstopmode report.tex >/dev/null
  echo "OK -> $HERE/report.pdf"
else
  echo "latexmk not found: report.tex rendered, not compiled"
fi
