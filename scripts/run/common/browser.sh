#!/usr/bin/env bash

gage_open_url() {
  local url="$1"
  local auto_open="${2:-1}"
  if [[ "${auto_open}" == "0" ]]; then
    return 0
  fi
  if command -v open >/dev/null 2>&1; then
    open "${url}" >/dev/null 2>&1 || true
    return 0
  fi
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${url}" >/dev/null 2>&1 || true
    return 0
  fi
  return 1
}

gage_wait_for_viewer() {
  local url="$1"
  local timeout_s="$2"
  local waited=0
  while [[ "${waited}" -lt "${timeout_s}" ]]; do
    if curl -sf "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}
