#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /workspace /logs/agent /logs/verifier /artifacts

log() {
  printf '[gage-swebench-init] %s\n' "$*" >&2
}

start_docker_if_available() {
  if ! command -v dockerd >/dev/null 2>&1; then
    log "dockerd is not available; continuing without Docker daemon"
    return 0
  fi

  if docker info >/dev/null 2>&1; then
    log "Docker daemon is already available"
    return 0
  fi

  log "attempting to start Docker daemon"
  dockerd > /logs/verifier/dockerd.log 2>&1 &
  docker_pid="$!"

  for _ in $(seq 1 30); do
    if docker info >/dev/null 2>&1; then
      log "Docker daemon started"
      return 0
    fi
    if ! kill -0 "$docker_pid" >/dev/null 2>&1; then
      log "dockerd exited before becoming ready; see /logs/verifier/dockerd.log"
      return 0
    fi
    sleep 1
  done

  log "Docker daemon did not become ready; continuing without privileged DIND"
}

start_docker_if_available

log "wrapper ready"
tail -f /dev/null
