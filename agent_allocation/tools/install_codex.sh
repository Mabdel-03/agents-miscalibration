#!/bin/bash
# Idempotent install of OpenAI Codex CLI 0.156.1 into the dynorch scratch prefix (plan §6, "Codex (H3)").
#   tools/install_codex.sh [--force]
# Uses npm from the consortium conda env, keeps the npm cache on scratch (never ~/.npm: /home is at its inode
# quota), installs without root, then verifies `codex --version` and that the bundled bwrap (the host's 0.4.0 lacks
# --perms, so Codex silently uses its own 0.11.2) runs on this kernel with unprivileged user namespaces.
set -euo pipefail
SCRATCH=${DYNORCH_SCRATCH:-/orcd/data/tpoggio/001/mabdel03/dynorch_scratch}
PREFIX=${CODEX_PREFIX:-$SCRATCH/npm-prefix}
VERSION=${CODEX_VERSION:-0.156.1}
export npm_config_cache=$SCRATCH/npm-cache
export TMPDIR=$SCRATCH/tmp
export CODEX_HOME=$SCRATCH/doctor-probe-home/codex      # a bare `codex --version` writes ~/.codex otherwise
mkdir -p "$PREFIX" "$npm_config_cache" "$TMPDIR" "$CODEX_HOME"
case "${1:-}" in --force) FORCE=1 ;; "") FORCE=0 ;; *) echo "usage: $0 [--force]"; exit 2 ;; esac
BIN="$PREFIX/bin/codex"
have() { [ -x "$BIN" ] && "$BIN" --version 2>/dev/null | grep -q "codex-cli $VERSION"; }
if have && [ "$FORCE" = 0 ]; then
  echo "[install_codex] already installed: $("$BIN" --version 2>/dev/null) at $BIN"
else
  command -v npm >/dev/null || { echo "[install_codex] npm not on PATH (source the consortium env)"; exit 1; }
  echo "[install_codex] npm $(npm --version) -> $PREFIX (cache $npm_config_cache)"
  npm install -g --prefix "$PREFIX" "@openai/codex@$VERSION"
fi
have || { echo "[install_codex] $BIN does not report codex-cli $VERSION"; "$BIN" --version || true; exit 1; }
NATIVE=$(find "$PREFIX/lib/node_modules/@openai" -path '*x86_64-unknown-linux-musl/bin/codex' -type f | head -1)
BWRAP=$(find "$PREFIX/lib/node_modules/@openai" -path '*codex-resources/bwrap' -type f | head -1)
[ -n "$NATIVE" ] || { echo "[install_codex] native musl binary not found under $PREFIX"; exit 1; }
[ -n "$BWRAP" ] || { echo "[install_codex] bundled bwrap not found under $PREFIX"; exit 1; }
chmod u+x "$BWRAP" 2>/dev/null || true
if "$BWRAP" --unshare-user --unshare-pid --die-with-parent --ro-bind / / --dev /dev --proc /proc /bin/true; then
  echo "[install_codex] bundled bwrap runs ($("$BWRAP" --version 2>/dev/null || echo 'version unknown'))"
else
  echo "[install_codex] bundled bwrap FAILED on this kernel (user namespaces?)"; exit 1
fi
"$BWRAP" --help 2>/dev/null | grep -q -- '--perms' && echo "[install_codex] bundled bwrap supports --perms (Codex will use it)"
echo "[install_codex] OK: $("$BIN" --version) native=$NATIVE"
echo "[install_codex] add to PATH: export PATH=\"$PREFIX/bin:\$PATH\"  (slurm/dynorch_v1.sbatch does this)"
FILES=$(find "$HOME/.npm" -type f 2>/dev/null | wc -l || echo 0)
[ "$FILES" -gt 0 ] && echo "[install_codex] WARNING: $HOME/.npm holds $FILES files; purge it (/home inode quota)" || true
