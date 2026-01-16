#!/usr/bin/env bash
# Export AppWorld dataset subsets from a container or image into minimal JSONL files.
#
# Classic usage:
#   ./export_datasets.sh --image appworld-mcp:latest --output local-datasets/appworld
#   ./export_datasets.sh --container appworld-mcp --output local-datasets/appworld --subsets train,dev
#
# NOTE: This script avoids importing the appworld Python package on the host to prevent version drift.
set -euo pipefail

# Keep usage details close to the entry point for quick CLI discovery.
usage() {
  cat <<'USAGE'
Export AppWorld dataset subsets to JSONL without importing appworld on host.

Usage:
  export_datasets.sh --image IMAGE [--output PATH] [--subsets LIST] [--root PATH]
  export_datasets.sh --container NAME [--output PATH] [--subsets LIST] [--root PATH]

Options:
  --image IMAGE       Docker image name (used when no --container is supplied)
  --container NAME    Running container name or ID (optional)
  --output PATH       Host output directory (default: ../local-datasets/appworld)
  --subsets LIST      Comma-separated subsets (default: train,dev,test_normal,test_challenge)
  --root PATH         AppWorld root inside container (default: /run)
  -h, --help          Show this help text
USAGE
}

# Resolve paths relative to this script so it works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${GAGE_ROOT}/.." && pwd)"
# Default to workspace-local datasets to avoid polluting the repo while keeping data discoverable.
DEFAULT_OUTPUT="${WORKSPACE_ROOT}/local-datasets/appworld"

IMAGE=""
CONTAINER=""
OUTPUT=""
SUBSETS="train,dev,test_normal,test_challenge"
# AppWorld images typically place data under /run; allow override for custom images.
APPWORLD_ROOT="/run"

# Parse flags early so we can decide between container or image execution.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --container)
      CONTAINER="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --subsets)
      SUBSETS="$2"
      shift 2
      ;;
    --root)
      APPWORLD_ROOT="$2"
      shift 2
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

# Require an explicit source so we do not accidentally run against an unintended image.
if [[ -z "${CONTAINER}" && -z "${IMAGE}" ]]; then
  echo "Either --image or --container is required." >&2
  usage >&2
  exit 1
fi

# Hard fail if docker is not present so users get a clear actionable error.
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found in PATH." >&2
  exit 1
fi

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${DEFAULT_OUTPUT}"
fi

# Normalize output to an absolute path for reliable docker bind mounts.
mkdir -p "${OUTPUT}"
OUTPUT="$(cd "${OUTPUT}" && pwd)"

IMAGE_NAME="${IMAGE}"
if [[ -n "${CONTAINER}" ]]; then
  # Detect the image behind the container so the manifest can record provenance.
  INSPECTED_IMAGE="$(docker inspect -f '{{.Config.Image}}' "${CONTAINER}" 2>/dev/null || true)"
  if [[ -n "${INSPECTED_IMAGE}" ]]; then
    IMAGE_NAME="${INSPECTED_IMAGE}"
  fi
fi

# Run Python inside the container to access AppWorld's on-disk data without host dependencies.
PYTHON_SNIPPET="$(cat <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

root = os.environ.get("APPWORLD_ROOT", "/run")
subsets = [s.strip() for s in os.environ.get("SUBSETS", "").split(",") if s.strip()]
output_dir = os.environ["OUTPUT_DIR"]
image_name = os.environ.get("IMAGE_NAME", "")

# AppWorld stores dataset lists and task specs under a fixed data layout.
data_root = os.path.join(root, "data")
datasets_dir = os.path.join(data_root, "datasets")
tasks_dir = os.path.join(data_root, "tasks")

if not os.path.isdir(datasets_dir) or not os.path.isdir(tasks_dir):
    raise SystemExit(f"AppWorld data not found under: {data_root}")

os.makedirs(output_dir, exist_ok=True)

counts = {}
missing_specs = []

for subset in subsets:
    # Each subset lists task IDs, which map to task specs in data/tasks/<task_id>/specs.json.
    list_path = os.path.join(datasets_dir, f"{subset}.txt")
    if not os.path.exists(list_path):
        raise SystemExit(f"Missing dataset list: {list_path}")
    with open(list_path, "r", encoding="utf-8") as handle:
        task_ids = [line.strip() for line in handle if line.strip()]
    counts[subset] = len(task_ids)
    out_path = os.path.join(output_dir, f"{subset}.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for task_id in task_ids:
            spec_path = os.path.join(tasks_dir, task_id, "specs.json")
            if not os.path.exists(spec_path):
                missing_specs.append(spec_path)
                continue
            with open(spec_path, "r", encoding="utf-8") as spec_handle:
                spec = json.load(spec_handle)
            instruction = spec.get("instruction") or ""
            allowed_apps = spec.get("allowed_apps")
            # Export minimal fields only, keeping ground truth data out of host artifacts.
            record = {
                "id": task_id,
                "task_id": task_id,
                "instruction": instruction,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": instruction}]}
                ],
                "metadata": {"appworld": {"task_id": task_id, "subset": subset}},
            }
            if allowed_apps:
                record["metadata"]["appworld"]["allowed_apps"] = allowed_apps
            out.write(json.dumps(record, ensure_ascii=True) + "\n")

# Write a manifest so downstream tooling can confirm data provenance and counts.
manifest = {
    "image": image_name,
    "appworld_root": root,
    "exported_at": datetime.now(timezone.utc).isoformat(),
    "subsets": subsets,
    "records": counts,
}
if missing_specs:
    manifest["missing_specs"] = missing_specs[:20]
    manifest["missing_specs_count"] = len(missing_specs)

manifest_path = os.path.join(output_dir, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=True, indent=2)

# Fail loudly if specs are missing so the caller does not assume a full export.
if missing_specs:
    raise SystemExit(f"Missing specs for {len(missing_specs)} tasks. See manifest.json.")
PY
)"

if [[ -n "${CONTAINER}" ]]; then
  # Use a temporary export directory to avoid writing into the container's main data tree.
  EXPORT_DIR="/tmp/appworld_export_${RANDOM}"
  docker exec "${CONTAINER}" /bin/sh -lc "mkdir -p '${EXPORT_DIR}'"
  docker exec -i \
    -e APPWORLD_ROOT="${APPWORLD_ROOT}" \
    -e SUBSETS="${SUBSETS}" \
    -e OUTPUT_DIR="${EXPORT_DIR}" \
    -e IMAGE_NAME="${IMAGE_NAME}" \
    "${CONTAINER}" /bin/sh -lc "python -" <<< "${PYTHON_SNIPPET}"
  # Copy results back to the host so the loader can read them without docker.
  docker cp "${CONTAINER}:${EXPORT_DIR}/." "${OUTPUT}"
  docker exec "${CONTAINER}" /bin/sh -lc "rm -rf '${EXPORT_DIR}'"
  echo "Exported JSONL to ${OUTPUT} from container ${CONTAINER}"
  exit 0
fi

# When no container is provided, run a short-lived container and bind-mount the output.
docker run --rm -i \
  --entrypoint /bin/sh \
  -e APPWORLD_ROOT="${APPWORLD_ROOT}" \
  -e SUBSETS="${SUBSETS}" \
  -e OUTPUT_DIR="/export" \
  -e IMAGE_NAME="${IMAGE_NAME}" \
  -v "${OUTPUT}:/export" \
  "${IMAGE}" \
  -lc "python -" <<< "${PYTHON_SNIPPET}"

echo "Exported JSONL to ${OUTPUT} from image ${IMAGE}"
