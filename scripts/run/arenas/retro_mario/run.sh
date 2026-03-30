#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
MODE="${MODE:-dummy_ws}"
CONFIG="${CONFIG:-}"
RUN_ID="${RUN_ID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"
RETRO_WS_RGB_PORT="${RETRO_WS_RGB_PORT:-5800}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run/arenas/retro_mario/run.sh [options]

Options:
  --mode <mode>         dummy_ws | openai_ws | human_ws | dummy_headless | openai_headless
                        Default: dummy_ws
  --config <path>       Explicit config path. Overrides --mode mapping.
  --run-id <run_id>     Run id. Default is auto-generated from mode/timestamp.
  --output-dir <dir>    Output directory. Default: runs
  --python-bin <path>   Python interpreter path
  -h, --help            Show this help

Examples:
  bash scripts/run/arenas/retro_mario/run.sh
  bash scripts/run/arenas/retro_mario/run.sh --mode openai_ws
  bash scripts/run/arenas/retro_mario/run.sh --mode human_ws
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[retro_mario][error] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXEC="${PYTHON_BIN}"
else
  PYTHON_EXEC="$(gage_default_python)"
fi

resolve_config() {
  if [[ -n "${CONFIG}" ]]; then
    printf '%s\n' "${CONFIG}"
    return 0
  fi

  case "${MODE}" in
    dummy_ws)
      printf '%s\n' "${ROOT}/config/custom/retro_mario/retro_mario_phase1_dummy_ws.yaml"
      ;;
    openai_ws)
      printf '%s\n' "${ROOT}/config/custom/retro_mario/retro_mario_openai_ws_rgb_auto_eval.yaml"
      ;;
    human_ws)
      printf '%s\n' "${ROOT}/config/custom/retro_mario/retro_mario_phase1_human_ws.yaml"
      ;;
    dummy_headless)
      printf '%s\n' "${ROOT}/config/custom/retro_mario/retro_mario_phase1_dummy_headless_auto_eval.yaml"
      ;;
    openai_headless)
      printf '%s\n' "${ROOT}/config/custom/retro_mario/retro_mario_openai_headless_auto_eval.yaml"
      ;;
    *)
      echo "[retro_mario][error] Unsupported mode: ${MODE}" >&2
      exit 1
      ;;
  esac
}

CFG="$(resolve_config)"
if [[ ! -f "${CFG}" ]]; then
  echo "[retro_mario][error] Config not found: ${CFG}" >&2
  exit 1
fi

case "${MODE}" in
  openai_ws|openai_headless)
    if [[ -z "${OPENAI_API_KEY:-}" && -n "${LITELLM_API_KEY:-}" ]]; then
      export OPENAI_API_KEY="${LITELLM_API_KEY}"
    fi
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "[retro_mario][error] OPENAI_API_KEY or LITELLM_API_KEY is required for ${MODE}." >&2
      exit 1
    fi
    ;;
esac

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="retro_mario_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${OUTPUT_DIR}"

cat <<MSG
[retro_mario] Python: ${PYTHON_EXEC}
[retro_mario] Mode: ${MODE}
[retro_mario] Config: ${CFG}
[retro_mario] Output: ${OUTPUT_DIR}
[retro_mario] Run ID: ${RUN_ID}
MSG

case "${MODE}" in
  dummy_ws|openai_ws)
    cat <<MSG
[retro_mario] Viewer: http://127.0.0.1:${RETRO_WS_RGB_PORT}/ws_rgb/viewer
MSG
    ;;
  human_ws)
    cat <<'MSG'
[retro_mario] Viewer: http://127.0.0.1:5800/ws_rgb/viewer
[retro_mario] Input queue: http://127.0.0.1:8001
[retro_mario] Browser keys:
  Movement -> W/A/S/D or arrow keys
  Jump -> J, Space, Z, C
  Run -> K, X
  Select -> L, Shift
  Start -> Enter
MSG
    ;;
esac

PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
RETRO_WS_RGB_PORT="${RETRO_WS_RGB_PORT}" \
"${PYTHON_EXEC}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"
