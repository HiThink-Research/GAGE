#!/usr/bin/env bash

gage_run_pipeline() {
  local repo_root="$1"
  local config_path="$2"
  local output_dir="$3"
  shift 3
  python "${repo_root}/run.py" \
    --config "${config_path}" \
    --output-dir "${output_dir}" \
    "$@"
}
