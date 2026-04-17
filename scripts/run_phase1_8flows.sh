#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_PREFIX="${RUN_PREFIX:-phase1_${RUN_DATE}}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
RUN_TIME_SUFFIX="${RUN_TIME_SUFFIX-__AUTO__}"

if [[ "$RUN_TIME_SUFFIX" == "__AUTO__" ]]; then
  RUN_TIME_SUFFIX="$(date +%H%M%S)"
fi

RUN_ID_SUFFIX=""
if [[ -n "$RUN_TIME_SUFFIX" ]]; then
  RUN_ID_SUFFIX="_${RUN_TIME_SUFFIX}"
fi

OUTPUT_DIR_BASE="${OUTPUT_DIR:-$ROOT_DIR/runs/${RUN_PREFIX}_8flows}"
OUTPUT_DIR_EXACT="${OUTPUT_DIR_EXACT:-0}"
OUTPUT_DIR="$OUTPUT_DIR_BASE"
if [[ "$OUTPUT_DIR_EXACT" != "1" && -n "$RUN_ID_SUFFIX" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR_BASE}${RUN_ID_SUFFIX}"
fi

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3-vl:2b-instruct}"
OLLAMA_API_KEY="${OLLAMA_API_KEY:-dummy}"

SWEBENCH_LOCAL_PATH="${SWEBENCH_LOCAL_PATH:-$ROOT_DIR/local-datasets/swebench_pro}"
TAU2_DATA_DIR="${TAU2_DATA_DIR:-/home/amyjx/work/tau2-bench/data}"
TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-12}"
TAU2_MAX_ERRORS="${TAU2_MAX_ERRORS:-4}"
TAU2_MAX_TURNS="${TAU2_MAX_TURNS:-12}"
TAU2_USER_MODEL="${TAU2_USER_MODEL:-ollama_chat/${OLLAMA_MODEL}}"

export PYTHONPATH="${ROOT_DIR}/src"
export OLLAMA_BASE_URL
export OLLAMA_MODEL
export OLLAMA_API_KEY
export SWEBENCH_LOCAL_PATH
export TAU2_DATA_DIR
export TAU2_MAX_STEPS
export TAU2_MAX_ERRORS
export TAU2_MAX_TURNS
export TAU2_USER_MODEL

CODEX_SERVICE_URL="${GAGE_CODEX_CLIENT_URL:-${CODEX_CLIENT_URL:-${GAGE_INSTALLED_CLIENT_URL:-}}}"
CODEX_DEFAULT_SERVICE_URL="${CODEX_DEFAULT_SERVICE_URL:-http://127.0.0.1:8787}"
CODEX_AVAILABLE=1
CODEX_SKIP_REASON=""

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required in PATH" >&2
  exit 1
fi

if [[ -z "$CODEX_SERVICE_URL" ]]; then
  if curl -fsS "${CODEX_DEFAULT_SERVICE_URL%/}/healthz" >/dev/null 2>&1; then
    CODEX_SERVICE_URL="$CODEX_DEFAULT_SERVICE_URL"
    export GAGE_CODEX_CLIENT_URL="$CODEX_SERVICE_URL"
  fi
fi

if [[ -z "$CODEX_SERVICE_URL" ]]; then
  CODEX_AVAILABLE=0
  CODEX_SKIP_REASON="installed-client service URL is missing (set GAGE_CODEX_CLIENT_URL/CODEX_CLIENT_URL/GAGE_INSTALLED_CLIENT_URL or run the local proxy on ${CODEX_DEFAULT_SERVICE_URL})"
fi

mkdir -p "$OUTPUT_DIR"

run_workflow() {
  local run_id="$1"
  local config_path="$2"

  if [[ ! -f "$config_path" ]]; then
    echo "config file not found: $config_path" >&2
    exit 1
  fi

  echo
  echo "==> Running ${run_id}"
  "$PYTHON_BIN" "$ROOT_DIR/run.py" \
    --config "$config_path" \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$run_id" \
    --max-samples "$MAX_SAMPLES"
}

run_installed_client_workflow() {
  local run_id="$1"
  local config_path="$2"

  if [[ "$CODEX_AVAILABLE" != "1" ]]; then
    echo
    echo "==> Skipping ${run_id}"
    echo "    reason: ${CODEX_SKIP_REASON}"
    return 0
  fi

  run_workflow "$run_id" "$config_path"
}

# STEP 1: Run terminal-bench workflows
run_workflow \
  "${RUN_PREFIX}_terminal_framework_loop${RUN_ID_SUFFIX}" \
  "$ROOT_DIR/config/custom/terminal_bench/terminal_bench_framework_loop_ollama.yaml"
run_installed_client_workflow \
  "${RUN_PREFIX}_terminal_installed_client${RUN_ID_SUFFIX}" \
  "$ROOT_DIR/config/custom/terminal_bench/terminal_bench_installed_client_ollama.yaml"

# STEP 2: Run SWE-bench workflows
run_workflow \
  "${RUN_PREFIX}_swebench_framework_loop${RUN_ID_SUFFIX}" \
  "$ROOT_DIR/config/custom/swebench_pro/swebench_pro_smoke_runtime_ollama_local.yaml"
run_installed_client_workflow \
  "${RUN_PREFIX}_swebench_installed_client${RUN_ID_SUFFIX}" \
  "$ROOT_DIR/config/custom/swebench_pro/swebench_pro_smoke_installed_client_ollama_local.yaml"

# STEP 3: Run Tau2 workflows
run_workflow \
  "${RUN_PREFIX}_tau2_framework_loop${RUN_ID_SUFFIX}" \
  "$ROOT_DIR/config/custom/tau2/tau2_telecom_runtime_ollama.yaml"
run_installed_client_workflow \
  "${RUN_PREFIX}_tau2_installed_client${RUN_ID_SUFFIX}" \
  "$ROOT_DIR/config/custom/tau2/tau2_telecom_installed_client_ollama.yaml"

# STEP 4: Run AppWorld workflows
run_workflow \
  "${RUN_PREFIX}_appworld_framework_loop${RUN_ID_SUFFIX}" \
  "$ROOT_DIR/config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml"
run_installed_client_workflow \
  "${RUN_PREFIX}_appworld_installed_client${RUN_ID_SUFFIX}" \
  "$ROOT_DIR/config/custom/appworld/appworld_agent_demo_installed_client_ollama.yaml"

echo
echo "All 8 workflows finished."
echo "Output directory: $OUTPUT_DIR"
