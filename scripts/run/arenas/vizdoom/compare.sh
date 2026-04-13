#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"
ROUNDS="${ROUNDS:-1}"
MODE_A="${MODE_A:-llm_headless}"
MODE_B="${MODE_B:-llm_headless}"
CONFIG_A="${CONFIG_A:-${CFG_A:-}}"
CONFIG_B="${CONFIG_B:-${CFG_B:-}}"

if ! [[ "${ROUNDS}" =~ ^[0-9]+$ ]] || [[ "${ROUNDS}" -le 0 ]]; then
  echo "[vizdoom][compare][error] ROUNDS must be a positive integer, got: ${ROUNDS}" >&2
  exit 1
fi

run_one() {
  local tag="$1"
  local mode="$2"
  local config="$3"
  local idx="$4"
  local run_id="vizdoom_${tag}_r${idx}_$(date +%Y%m%d_%H%M%S)"
  local args=(--run-id "${run_id}" --output-dir "${OUTPUT_DIR}")

  if [[ -n "${config}" ]]; then
    args+=(--config "${config}")
  else
    args+=(--mode "${mode}")
  fi
  if [[ -n "${MAX_SAMPLES:-}" ]]; then
    args+=(--max-samples "${MAX_SAMPLES}")
  fi

  echo "[vizdoom][compare] tag=${tag} round=${idx} run_id=${run_id}"
  bash "${ROOT}/scripts/run/arenas/vizdoom/run.sh" "${args[@]}"
}

for ((i=1; i<=ROUNDS; i++)); do
  run_one "a" "${MODE_A}" "${CONFIG_A}" "${i}"
  run_one "b" "${MODE_B}" "${CONFIG_B}" "${i}"
done
