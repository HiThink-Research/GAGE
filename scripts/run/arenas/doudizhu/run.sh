#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"

MODE="${MODE:-llm_visual}"
CONFIG="${CONFIG:-}"
RUN_ID="${RUN_ID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
EXTRA_RUN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run/arenas/doudizhu/run.sh [options] [-- run.py options]

Options:
  --mode <mode>         dummy | dummy_visual | llm_headless | llm_visual |
                        human_visual | human_acceptance |
                        llm_headless_openai | llm_visual_openai |
                        human_visual_openai | human_acceptance_openai
                        Default: llm_visual
  --config <path>       Explicit config path. Overrides --mode mapping.
  --run-id <run_id>     Run id. Default is auto-generated from mode/timestamp.
  --output-dir <dir>    Output directory. Default: GAGE_RUNS_DIR or workspace runs.
  --max-samples <n>     Forwarded to run.py.
  --python-bin <path>   Python interpreter path. Default: PYTHON_BIN or detected Python.
  -h, --help            Show this help.

Examples:
  bash scripts/run/arenas/doudizhu/run.sh --mode llm_visual --max-samples 1
  OPENAI_API_KEY=sk-... bash scripts/run/arenas/doudizhu/run.sh --mode llm_visual_openai --max-samples 1
  bash scripts/run/arenas/doudizhu/run.sh --mode human_visual
  bash scripts/run/arenas/doudizhu/run.sh --mode dummy_visual
EOF
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "[doudizhu][error] ${flag} requires a value." >&2
    usage >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      require_value "$1" "${2:-}"
      MODE="${2:-}"
      shift 2
      ;;
    --mode=*)
      VALUE="${1#*=}"
      require_value "--mode" "${VALUE}"
      MODE="${VALUE}"
      shift
      ;;
    --config)
      require_value "$1" "${2:-}"
      CONFIG="${2:-}"
      shift 2
      ;;
    --config=*)
      VALUE="${1#*=}"
      require_value "--config" "${VALUE}"
      CONFIG="${VALUE}"
      shift
      ;;
    --run-id)
      require_value "$1" "${2:-}"
      RUN_ID="${2:-}"
      shift 2
      ;;
    --run-id=*)
      VALUE="${1#*=}"
      require_value "--run-id" "${VALUE}"
      RUN_ID="${VALUE}"
      shift
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --output-dir=*)
      VALUE="${1#*=}"
      require_value "--output-dir" "${VALUE}"
      OUTPUT_DIR="${VALUE}"
      shift
      ;;
    --max-samples)
      require_value "$1" "${2:-}"
      MAX_SAMPLES="${2:-}"
      shift 2
      ;;
    --max-samples=*)
      VALUE="${1#*=}"
      require_value "--max-samples" "${VALUE}"
      MAX_SAMPLES="${VALUE}"
      shift
      ;;
    --python-bin)
      require_value "$1" "${2:-}"
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --python-bin=*)
      VALUE="${1#*=}"
      require_value "--python-bin" "${VALUE}"
      PYTHON_BIN="${VALUE}"
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_RUN_ARGS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_RUN_ARGS+=("$1")
      shift
      ;;
  esac
done

config_path() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT}/${path}"
  fi
}

resolve_config() {
  if [[ -n "${CONFIG}" ]]; then
    config_path "${CONFIG}"
    return 0
  fi

  case "${MODE}" in
    dummy|dummy_headless|headless)
      config_path "config/custom/doudizhu/doudizhu_dummy_gamekit.yaml"
      ;;
    dummy_visual|visual)
      config_path "config/custom/doudizhu/doudizhu_dummy_visual_gamekit.yaml"
      ;;
    llm|llm_headless)
      config_path "config/custom/doudizhu/doudizhu_llm_headless_gamekit.yaml"
      ;;
    openai|llm_openai|llm_headless_openai|openai_headless)
      config_path "config/custom/doudizhu/doudizhu_llm_headless_openai_gamekit.yaml"
      ;;
    llm_visual)
      config_path "config/custom/doudizhu/doudizhu_llm_visual_gamekit.yaml"
      ;;
    llm_visual_openai|openai_visual)
      config_path "config/custom/doudizhu/doudizhu_llm_visual_openai_gamekit.yaml"
      ;;
    human|human_visual)
      config_path "config/custom/doudizhu/doudizhu_human_visual_gamekit.yaml"
      ;;
    human_visual_openai|human_openai|openai_human_visual)
      config_path "config/custom/doudizhu/doudizhu_human_visual_openai_gamekit.yaml"
      ;;
    human_acceptance|acceptance)
      config_path "config/custom/doudizhu/doudizhu_human_visual_acceptance_gamekit.yaml"
      ;;
    human_acceptance_openai|acceptance_openai)
      config_path "config/custom/doudizhu/doudizhu_human_visual_acceptance_openai_gamekit.yaml"
      ;;
    *)
      echo "[doudizhu][error] Unsupported mode: ${MODE}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

PYTHON_EXEC="${PYTHON_BIN:-$(gage_default_python)}"
CFG="$(resolve_config)"
if [[ ! -f "${CFG}" ]]; then
  echo "[doudizhu][error] Config not found: ${CFG}" >&2
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="doudizhu_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${OUTPUT_DIR}"

RUN_ARGS=(--config "${CFG}" --output-dir "${OUTPUT_DIR}" --run-id "${RUN_ID}")
if [[ -n "${MAX_SAMPLES}" ]]; then
  RUN_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if (( ${#EXTRA_RUN_ARGS[@]} > 0 )); then
  RUN_ARGS+=("${EXTRA_RUN_ARGS[@]}")
fi

cat <<MSG
[doudizhu] Python: ${PYTHON_EXEC}
[doudizhu] Mode: ${MODE}
[doudizhu] Config: ${CFG}
[doudizhu] Output: ${OUTPUT_DIR}
[doudizhu] Run ID: ${RUN_ID}
MSG
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_EXEC}" "${ROOT}/run.py" "${RUN_ARGS[@]}"
