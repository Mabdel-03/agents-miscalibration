#!/usr/bin/env bash
# Build the full_sweep_v1 technical report.
#
# The PATH export is mandatory: the miniforge pdflatex that appears first on the
# default PATH has no format file and fails immediately.  The working toolchain is
# the TinyTeX TeX Live 2026 install in ~/.local/bin.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="/home/mabdel03/.local/bin:$PATH"
exec make "${@:-all}"
