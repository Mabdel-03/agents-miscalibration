#!/usr/bin/env bash
# Build the scaffolding atlas. TinyTeX must come first on PATH: the conda pdflatex
# earlier on PATH has an empty texmf tree and fails with "file not found" on every package.
set -euo pipefail
export PATH=/orcd/home/002/mabdel03/.TinyTeX/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error atlas.tex >/dev/null
bibtex atlas >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error atlas.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error atlas.tex | tail -20
echo "--- warnings ---"
grep -E "LaTeX Warning: (Citation|Reference|There were)" atlas.log || echo "none"
echo "--- built: $(ls -la atlas.pdf | awk '{print $5" bytes"}') ---"
