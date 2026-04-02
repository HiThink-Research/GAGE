#!/usr/bin/env bash
set -euo pipefail

CODEX_HOME="${CODEX_HOME:-/agent}"
mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME" || true

if [[ ! -f "$CODEX_HOME/auth.json" && -n "${GAGE_CODEX_HOST_HOME:-}" && -d "${GAGE_CODEX_HOST_HOME}" ]]; then
  cp -R "${GAGE_CODEX_HOST_HOME}/." "$CODEX_HOME/" 2>/dev/null || true
  chmod 700 "$CODEX_HOME" || true
  chmod 600 "$CODEX_HOME/auth.json" 2>/dev/null || true
fi

if [[ ! -f "$CODEX_HOME/auth.json" && -n "${OPENAI_API_KEY:-}" ]]; then
  umask 077
  cat >"$CODEX_HOME/auth.json" <<EOF
{
  "OPENAI_API_KEY": "${OPENAI_API_KEY}"
}
EOF
fi

if [[ "${GAGE_CODEX_VALIDATE_ON_BOOT:-0}" == "1" ]]; then
  codex --version
fi

if [[ "$#" -eq 0 ]]; then
  exec sleep infinity
fi

exec "$@"
