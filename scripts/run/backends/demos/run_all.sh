#!/usr/bin/env bash
set -euo pipefail

# 串行跑完 dummy 冒烟 + multi_provider demo（依赖环境变量指定 provider）

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

echo "[oneclick] step 1/2: dummy echo smoke"
bash "${ROOT}/scripts/run/backends/demos/run_demo_echo.sh"

echo "[oneclick] step 2/2: multi_provider_http demo"
bash "${ROOT}/scripts/run/backends/demos/run_multi_provider_http_demo.sh"

echo "[oneclick] all done."
