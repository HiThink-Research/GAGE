#!/usr/bin/env bash

gage_is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

gage_pick_port() {
  local base_port="$1"
  local max_tries="${2:-50}"
  local reserved="${3:-}"
  local port="${base_port}"
  local idx=0
  while [[ "${idx}" -lt "${max_tries}" ]]; do
    if [[ -n "${reserved}" && "${port}" == "${reserved}" ]]; then
      port=$((port + 1))
      idx=$((idx + 1))
      continue
    fi
    if ! gage_is_port_in_use "${port}"; then
      printf '%s\n' "${port}"
      return 0
    fi
    port=$((port + 1))
    idx=$((idx + 1))
  done
  return 1
}
