#!/usr/bin/env bash
set -euo pipefail

CODEX_HOME="${CODEX_HOME:-/agent}"
mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME" || true

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
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
