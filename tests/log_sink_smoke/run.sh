#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_REL="tests/log_sink_smoke/config/log_sink.yaml"
CONFIG_PATH="${REPO_ROOT}/${CONFIG_REL}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "配置文件不存在: ${CONFIG_PATH}" >&2
  exit 1
fi

pushd "${REPO_ROOT}" >/dev/null
trap 'popd >/dev/null' EXIT

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export GAGE_EVAL_SAVE_DIR="${REPO_ROOT}/runs/log_sink_smoke"
rm -rf "${GAGE_EVAL_SAVE_DIR}"
mkdir -p "${GAGE_EVAL_SAVE_DIR}"

python "${REPO_ROOT}/run.py" --config "${CONFIG_REL}" >&2

LATEST_RUN="$(python - <<'PY'
import os
import sys

root = os.environ.get("GAGE_EVAL_SAVE_DIR")
if not root or not os.path.isdir(root):
    sys.exit(1)
children = [os.path.join(root, item) for item in os.listdir(root)]
children = [path for path in children if os.path.isdir(path)]
if not children:
    sys.exit(2)
children.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print(children[0])
PY
)"

if [[ -z "${LATEST_RUN}" ]]; then
  echo "未找到 log sink 运行目录" >&2
  exit 1
fi

EVENT_FILE="${LATEST_RUN}/events.jsonl"
if [[ ! -f "${EVENT_FILE}" ]]; then
  echo "缺少事件文件: ${EVENT_FILE}" >&2
  exit 1
fi
export EVENT_FILE

python - <<'PY'
import json
import os
import sys

event_file = os.environ.get("EVENT_FILE")
if not event_file or not os.path.exists(event_file):
    print("EVENT_FILE 未定义或不存在", file=sys.stderr)
    sys.exit(1)

log_events = 0
with open(event_file, "r", encoding="utf-8") as handle:
    for line in handle:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "log":
            log_events += 1

if log_events == 0:
    print("未捕获任何 log 事件", file=sys.stderr)
    sys.exit(1)
PY

echo "Log sink 冒烟测试通过：事件文件中捕获到 log 事件。"
