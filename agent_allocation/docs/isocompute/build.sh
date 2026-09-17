#!/usr/bin/env bash
# Regenerate every figure and number from the sealed tables, then compile the report.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HERE/../../.venv/bin/python}"
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
