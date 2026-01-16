#!/usr/bin/env bash
set -euo pipefail

# Validate that AppWorld evaluation JSON can be read from exported task outputs.
#
# Example 1: basic usage (point to a single task export directory)
#   bash gage-eval-main/docker/appworld/check_eval_output.sh \
#     --image appworld-mcp:latest \
#     --experiment appworld-experiment-20260115-165435-089949 \
#     --task-id 50e1ac9_1 \
#     --task-output-dir runs/appworld_official_jsonl/appworld_official_jsonl_run_xxx/appworld_artifacts/50e1ac9_1
#
# Example 2: keep the temporary workspace for inspection
#   bash gage-eval-main/docker/appworld/check_eval_output.sh \
#     --image appworld-mcp:latest \
#     --experiment appworld-experiment-20260115-165435-089949 \
#     --task-id 50e1ac9_1 \
#     --task-output-dir runs/appworld_official_jsonl/appworld_official_jsonl_run_xxx/appworld_artifacts/50e1ac9_1 \
#     --keep-workdir

usage() {
  cat <<'USAGE'
Usage:
  check_eval_output.sh --experiment NAME --task-id TASK --task-output-dir DIR [options]

Required:
  --experiment        AppWorld experiment name (matches /run/experiments/outputs/<name>)
  --task-id           Task ID to evaluate
  --task-output-dir   Exported task output directory (from GAGE appworld_artifacts/<task_id>)

Options:
  --image             AppWorld image name (default: appworld-mcp:latest)
  --with-setup        Run appworld evaluate with --with-setup (downloads data if missing)
  --keep-workdir      Keep the temporary workspace for debugging
  -h, --help          Show this help
USAGE
}

IMAGE="appworld-mcp:latest"
EXPERIMENT_NAME=""
TASK_ID=""
TASK_OUTPUT_DIR=""
WITH_SETUP="false"
KEEP_WORKDIR="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --task-id)
      TASK_ID="$2"
      shift 2
      ;;
    --task-output-dir)
      TASK_OUTPUT_DIR="$2"
      shift 2
      ;;
    --with-setup)
      WITH_SETUP="true"
      shift
      ;;
    --keep-workdir)
      KEEP_WORKDIR="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${EXPERIMENT_NAME}" || -z "${TASK_ID}" || -z "${TASK_OUTPUT_DIR}" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${TASK_OUTPUT_DIR}" ]]; then
  echo "Task output directory not found: ${TASK_OUTPUT_DIR}" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
cleanup() {
  if [[ "${KEEP_WORKDIR}" != "true" ]]; then
    rm -rf "${WORK_DIR}"
  else
    echo "[check] Keeping workspace: ${WORK_DIR}"
  fi
}
trap cleanup EXIT

TARGET_DIR="${WORK_DIR}/experiments/outputs/${EXPERIMENT_NAME}/tasks/${TASK_ID}"
mkdir -p "${TARGET_DIR}"
cp -a "${TASK_OUTPUT_DIR}/." "${TARGET_DIR}/"

EVAL_FLAGS=""
if [[ "${WITH_SETUP}" == "true" ]]; then
  EVAL_FLAGS="--with-setup"
fi

RESULT_PATH="/run/experiments/outputs/${EXPERIMENT_NAME}/evaluations/on_only_${TASK_ID}.json"

docker run --rm \
  --entrypoint /bin/sh \
  -v "${WORK_DIR}/experiments/outputs:/run/experiments/outputs" \
  "${IMAGE}" \
  -lc "set -eu
    appworld evaluate \"${EXPERIMENT_NAME}\" --task-id \"${TASK_ID}\" --root /run ${EVAL_FLAGS}
    echo \"\"
    echo \"[check] Evaluation JSON: ${RESULT_PATH}\"
    cat \"${RESULT_PATH}\"
  "
