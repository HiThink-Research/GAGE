#!/usr/bin/env bash
set -euo pipefail

# 一键跑 demo_echo（hf_inference_endpoint），无需手改 YAML。
# 需设置 HUGGINGFACEHUB_API_TOKEN；未提供时使用默认 endpoint/model 自动创建（token 无效会 401）。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
BASE_DIR="${ROOT}/scripts/oneclick/backends/hf_inference_endpoint/demo_echo"
TEMPLATE="${BASE_DIR}/template.yaml"
GEN_DIR="${ROOT}/scripts/oneclick/.generated/backends/hf_inference_endpoint/demo_echo"
MODEL_MATRIX="${GEN_DIR}/model_matrix.yaml"

mkdir -p "${GEN_DIR}"

ENDPOINT_NAME="${ENDPOINT_NAME:-demo-endpoint}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

VENDOR="${VENDOR:-aws}"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-nvidia-a10g}"
INSTANCE_SIZE="${INSTANCE_SIZE:-x1}"
REVISION="${REVISION:-main}"
REUSE_EXISTING="${REUSE_EXISTING:-false}"
AUTO_START="${AUTO_START:-true}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
ENABLE_ASYNC="${ENABLE_ASYNC:-false}"
ASYNC_MAX_CONCURRENCY="${ASYNC_MAX_CONCURRENCY:-0}"

cat > "${MODEL_MATRIX}" <<EOF
models:
  - name: demo_endpoint
    endpoint_name: ${ENDPOINT_NAME:-null}
    model_name: ${MODEL_NAME:-null}
    revision: ${REVISION}
    vendor: ${VENDOR}
    region: ${REGION}
    instance_type: ${INSTANCE_TYPE}
    instance_size: ${INSTANCE_SIZE}
    reuse_existing: ${REUSE_EXISTING}
    auto_start: ${AUTO_START}
    wait_timeout: ${WAIT_TIMEOUT}
    poll_interval: ${POLL_INTERVAL}
    max_new_tokens: ${MAX_NEW_TOKENS}
    max_samples: ${MAX_SAMPLES}
    concurrency: ${CONCURRENCY}
    enable_async: ${ENABLE_ASYNC}
    async_max_concurrency: ${ASYNC_MAX_CONCURRENCY}
EOF

MODEL_MATRIX="${MODEL_MATRIX}" \
TEMPLATE="${TEMPLATE}" \
GEN_DIR="${GEN_DIR}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/hf_endpoint_demo}" \
bash "${ROOT}/scripts/oneclick/backends/hf_inference_endpoint/run_all_models.sh"
