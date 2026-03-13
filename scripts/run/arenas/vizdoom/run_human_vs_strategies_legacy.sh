#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
PYTHON_BIN="${PYTHON_BIN:-$(gage_default_python)}"
CFG="${CFG:-${ROOT}/config/custom/vizdoom_human_vs_llm_record_ws_rgb_strategy.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"
P1_SCHEME_ID="${P1_SCHEME_ID:-S3_text_image_current}"
RUN_ID="${RUN_ID:-vizdoom_human_p0_vs_${P1_SCHEME_ID}_$(date +%Y%m%d_%H%M%S)}"
WS_RGB_HOST="${WS_RGB_HOST:-127.0.0.1}"
WS_RGB_PORT="${WS_RGB_PORT:-5800}"

if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${LITELLM_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${LITELLM_API_KEY}"
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[oneclick][error] OPENAI_API_KEY or LITELLM_API_KEY is required." >&2
  exit 1
fi

if [ ! -f "${CFG}" ]; then
  echo "[oneclick][error] config not found: ${CFG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cat <<MSG
[vizdoom][human_vs_strategies] p0 uses websocket human input.
[vizdoom][human_vs_strategies] p1 uses backend LLM with one scheme_id per run.
[vizdoom][human_vs_strategies] Viewer: http://${WS_RGB_HOST}:${WS_RGB_PORT}/ws_rgb/viewer
[vizdoom][human_vs_strategies] Keys: A/Left=2, D/Right=3, Space/J=1
[vizdoom][human_vs_strategies] Python: ${PYTHON_BIN}
[vizdoom][human_vs_strategies] Config: ${CFG}
[vizdoom][human_vs_strategies] p1 scheme: ${P1_SCHEME_ID}
[vizdoom][human_vs_strategies] run_id: ${RUN_ID}
MSG

print_one_summary() {
  local run_id="$1"
  local scheme="$2"
  python - "${OUTPUT_DIR}" "${run_id}" "${scheme}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
run_id = sys.argv[2]
scheme = sys.argv[3]
summary_path = output_dir / run_id / "summary.json"

if not summary_path.exists():
    print(f"[result] run_id={run_id} scheme={scheme} summary=missing")
    sys.exit(0)

payload = json.loads(summary_path.read_text(encoding="utf-8"))
arena_summary = payload.get("arena_summary") or {}
overall = arena_summary.get("overall") or {}
winner_map = arena_summary.get("winner_player_id") or {}
reason_map = arena_summary.get("termination_reason") or {}

winner = ",".join(sorted(winner_map.keys())) if winner_map else "draw_or_unknown"
reason = ",".join(sorted(reason_map.keys())) if reason_map else "unknown"
steps = overall.get("avg_episode_length_steps", "n/a")
duration_ms = overall.get("avg_episode_duration_ms")
duration_s = "n/a" if duration_ms is None else f"{float(duration_ms)/1000.0:.3f}"
p1_killed_p0 = "yes" if "p1" in winner_map else "no"

print(
    f"[result] run_id={run_id} scheme={scheme} "
    f"p1_killed_p0={p1_killed_p0} winner={winner} reason={reason} "
    f"steps={steps} duration_s={duration_s}"
)
PY
}

echo
echo "[run] scheme=${P1_SCHEME_ID}"
echo "[run] run_id=${RUN_ID}"

VIZDOOM_P1_SCHEME_ID="${P1_SCHEME_ID}" \
WS_RGB_HOST="${WS_RGB_HOST}" \
WS_RGB_PORT="${WS_RGB_PORT}" \
PYTHONPATH="${ROOT}/src" \
"${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"

print_one_summary "${RUN_ID}" "${P1_SCHEME_ID}"
