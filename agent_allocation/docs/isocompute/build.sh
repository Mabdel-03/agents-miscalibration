#!/usr/bin/env bash
# Regenerate every figure and number from the sealed tables, then compile the report.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The analysis imports agent_allocation, which lives in the allocation-v1 worktree's venv.
# Works both in that worktree (../../.venv) and from this copy in the main checkout.
WORKTREE_VENV=/orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1/agent_allocation/.venv
if [ -z "${PY:-}" ]; then
  if [ -x "$HERE/../../.venv/bin/python" ]; then PY="$HERE/../../.venv/bin/python"
  elif [ -x "$WORKTREE_VENV/bin/python" ]; then PY="$WORKTREE_VENV/bin/python"
  else echo "no interpreter with agent_allocation installed; set PY=..." >&2; exit 1; fi
fi
export PATH="/home/mabdel03/.local/bin:$PATH"   # TinyTeX, TeX Live 2026

cd "$HERE"

echo "== analysis =="
"$PY" isocompute.py --out "$HERE" "$@"

echo "== lint =="
# The report is written without em dashes, in prose and in generated tables alike.
if grep -n -P '\x{2014}|---' report.tex tables/*.tex; then
  echo "LINT FAIL: em dash or --- found"; exit 1
fi
echo "no em dashes"
"$PY" check_numbers.py

echo "== compile =="
latexmk -pdf -interaction=nonstopmode report.tex >/dev/null

echo "== log check =="
bad=0
grep -n 'Undefined control sequence\|LaTeX Warning: Reference\|LaTeX Warning: Citation\|File .* not found' report.log && bad=1
ov=$(grep -c 'Overfull \\hbox' report.log || true)
echo "overfull hboxes: $ov"
[ "$bad" -eq 0 ] || { echo "LOG CHECK FAIL"; exit 1; }
echo "OK -> $HERE/report.pdf"
