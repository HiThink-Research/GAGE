#!/usr/bin/env bash

if [[ -n "${GAGE_RUN_COMMON_ENV_SH:-}" ]]; then
  return 0
fi
GAGE_RUN_COMMON_ENV_SH=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAGE_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GAGE_WORKSPACE_ROOT="$(cd "${GAGE_REPO_ROOT}" && pwd)"

gage_default_local_env_file() {
  if [[ -n "${GAGE_LOCAL_ENV_FILE:-}" ]]; then
    printf '%s\n' "${GAGE_LOCAL_ENV_FILE}"
    return 0
  fi
  if [[ -f "${GAGE_WORKSPACE_ROOT}/env/scripts/run.env" ]]; then
    printf '%s\n' "${GAGE_WORKSPACE_ROOT}/env/scripts/run.env"
    return 0
  fi
  if [[ -f "${GAGE_WORKSPACE_ROOT}/env/localenv" ]]; then
    printf '%s\n' "${GAGE_WORKSPACE_ROOT}/env/localenv"
    return 0
  fi
  printf '%s\n' "${GAGE_WORKSPACE_ROOT}/env/scripts/run.env"
}

gage_default_state_dir() {
  printf '%s\n' "${GAGE_SCRIPT_STATE_DIR:-${GAGE_WORKSPACE_ROOT}/env/scripts/generated}"
}

gage_default_runs_dir() {
  printf '%s\n' "${GAGE_RUNS_DIR:-${GAGE_WORKSPACE_ROOT}/runs}"
}

gage_default_venv_path() {
  if [[ -n "${VENV_PATH:-}" ]]; then
    printf '%s\n' "${VENV_PATH}"
    return 0
  fi
  if [[ -d "${GAGE_WORKSPACE_ROOT}/env/.venv" ]]; then
    printf '%s\n' "${GAGE_WORKSPACE_ROOT}/env/.venv"
    return 0
  fi
  if [[ -d "${GAGE_REPO_ROOT}/.venv311" ]]; then
    printf '%s\n' "${GAGE_REPO_ROOT}/.venv311"
    return 0
  fi
  printf '%s\n' "${GAGE_REPO_ROOT}/.venv"
}

gage_default_python() {
  local venv_path
  venv_path="$(gage_default_venv_path)"
  if [[ -x "${venv_path}/bin/python" ]]; then
    printf '%s\n' "${venv_path}/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  command -v python
}

gage_activate_venv() {
  local venv_path
  venv_path="${1:-$(gage_default_venv_path)}"
  if [[ -d "${venv_path}" ]]; then
    # shellcheck disable=SC1091
    source "${venv_path}/bin/activate"
  fi
}

gage_load_local_env() {
  local env_file
  env_file="${1:-$(gage_default_local_env_file)}"
  if [[ -f "${env_file}" ]]; then
    # shellcheck disable=SC1090
    source "${env_file}"
  fi
}

export GAGE_REPO_ROOT
export GAGE_WORKSPACE_ROOT
