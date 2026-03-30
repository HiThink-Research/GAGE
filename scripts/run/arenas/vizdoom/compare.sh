#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
PYTHON_BIN="${PYTHON_BIN:-$(gage_default_python)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"
ROUNDS="${ROUNDS:-1}"

CFG_A="${CFG_A:-${ROOT}/config/custom/vizdoom/vizdoom_llm_vs_llm_s1_vs_s2.yaml}"
CFG_B="${CFG_B:-${ROOT}/config/custom/vizdoom/vizdoom_llm_vs_llm_s2_vs_s1.yaml}"

if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${LITELLM_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${LITELLM_API_KEY}"
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[compare][error] OPENAI_API_KEY or LITELLM_API_KEY is required." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if ! [[ "${ROUNDS}" =~ ^[0-9]+$ ]] || [ "${ROUNDS}" -le 0 ]; then
  echo "[compare][error] ROUNDS must be a positive integer, got: ${ROUNDS}" >&2
  exit 1
fi

run_ids=()

run_one() {
  local cfg="$1"
  local tag="$2"
  local idx="$3"
  local run_id="vizdoom_${tag}_r${idx}_$(date +%Y%m%d_%H%M%S)"

  echo "[compare] running tag=${tag} round=${idx}"
  echo "[compare] config=${cfg}"
  echo "[compare] run_id=${run_id}"

  PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" "${ROOT}/run.py" \
    --config "${cfg}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-id "${run_id}"

  run_ids+=("${run_id}")
}

for ((i=1; i<=ROUNDS; i++)); do
  run_one "${CFG_A}" "s1_vs_s2" "${i}"
  run_one "${CFG_B}" "s2_vs_s1" "${i}"
done

echo
echo "[compare] finished. run ids:"
for rid in "${run_ids[@]}"; do
  echo "  - ${rid}"
done

RUN_IDS_CSV="$(IFS=,; echo "${run_ids[*]}")"
export RUN_IDS_CSV OUTPUT_DIR

python - <<'PY'
import json
import os
from pathlib import Path

run_ids = [item for item in os.getenv("RUN_IDS_CSV", "").split(",") if item]
output_dir = Path(os.getenv("OUTPUT_DIR", "."))

print("\n[compare] quick summary")
print("run_id\twinner\treason\tsteps\tduration_s")
for rid in run_ids:
    summary_path = output_dir / rid / "summary.json"
    if not summary_path.exists():
        print(f"{rid}\t<missing>\t<missing>\t<missing>\t<missing>")
        continue
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    arena_overall = (payload.get("arena_summary") or {}).get("overall") or {}
    winner_map = (payload.get("arena_summary") or {}).get("winner_player_id") or {}
    reason_map = (payload.get("arena_summary") or {}).get("termination_reason") or {}
    winner = ",".join(sorted(winner_map.keys())) if winner_map else "draw_or_unknown"
    reason = ",".join(sorted(reason_map.keys())) if reason_map else "unknown"
    steps = arena_overall.get("avg_episode_length_steps", "n/a")
    duration_ms = arena_overall.get("avg_episode_duration_ms")
    duration_s = "n/a" if duration_ms is None else f"{float(duration_ms)/1000.0:.3f}"
    print(f"{rid}\t{winner}\t{reason}\t{steps}\t{duration_s}")
PY
