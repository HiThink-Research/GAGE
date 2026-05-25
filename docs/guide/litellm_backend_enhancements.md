# LiteLLM Backend Enhancement Guide

English | [中文](litellm_backend_enhancements_zh.md)

This guide describes the enhanced GAGE `litellm` backend capabilities, common
vLLM deployment profiles, and validation points for thinking, multimodal input,
tool calling, streaming, async calls, and request pass-through. Commands assume
they are run from the `gage-eval-main/` repository root.

This document is for evaluation authors who use GAGE through LiteLLM to call
OpenAI-compatible services, vLLM servers, LiteLLM Gateway, or LiteLLM Router.

## 0. Related Entry Points

- Config schema: `src/gage_eval/role/model/config/litellm.py`
- Backend entry point: `src/gage_eval/role/model/backends/litellm_backend.py`
- LiteLLM backend modules:
  - `src/gage_eval/role/model/backends/litellm/request_builder.py`
  - `src/gage_eval/role/model/backends/litellm/message_normalizer.py`
  - `src/gage_eval/role/model/backends/litellm/response_normalizer.py`
  - `src/gage_eval/role/model/backends/litellm/policies.py`
  - `src/gage_eval/role/model/backends/litellm/router_factory.py`
  - `src/gage_eval/role/model/backends/litellm/model_server_lifecycle.py`
- Config validation:

```bash
PYTHONPATH=src python -m gage_eval.tools.config_checker \
  --config config/custom/examples/qwen3_vllm_piqa_messages_no_thinking.yaml
```

## 1. Capability Overview

The `litellm` backend can be used as a normal remote-model backend and as the
unified adapter for vLLM OpenAI-compatible servers.

| Capability | Supported | Key configuration |
| --- | --- | --- |
| Single endpoint | Yes | `route_mode: direct`, `provider: hosted_vllm`, `api_base` |
| Multiple endpoints | Yes | `route_mode: router`, `model_list`, `router_settings` |
| External LiteLLM Gateway | Yes | Gateway `api_base`, `vllm.topology.kind: external_litellm_gateway` |
| Thinking / no-thinking | Yes | `generation_parameters.thinking_mode`, `thinking_policy`, `vllm.reasoning_parser` |
| `reasoning_content` normalization | Yes | Extracted by the response normalizer |
| Multimodal content blocks | Yes | `multimodal.*`, `vllm.topology.language_model_only` |
| Tool calling | Yes | `tool_calling.*`, sample-level `tools`, `tool_choice` |
| Structured output | Yes | `litellm_request.response_format` |
| Sync streaming | Yes | `streaming: true`, `litellm_request.stream_options` |
| Async calls | Yes | Backend `ainvoke` through LiteLLM `acompletion` |
| Request pass-through | Yes | `litellm_request.*` |
| Retry ownership | Yes | Router/Gateway profiles collapse the outer retry budget |
| Experimental server lifecycle | Command mode | `model_server.enabled`, `startup_command`, `health.urls` |

## 2. Recommended Deployment Profiles

### 2.1 P1 Direct: One vLLM Endpoint

Use this profile for one exposed vLLM OpenAI-compatible endpoint, whether the
server is single-GPU, tensor-parallel, or a multi-node deployment with a single
head API.

vLLM server example:

```bash
vllm serve /mnt/model/Qwen3.6-35B-A3B \
  --served-model-name qwen3-p1 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": true}'
```

GAGE backend example:

```yaml
backends:
  - backend_id: qwen3_litellm_direct
    type: litellm
    config:
      provider: hosted_vllm
      custom_llm_provider: hosted_vllm
      model: hosted_vllm/qwen3-p1
      api_base: http://127.0.0.1:8000/v1
      api_key: EMPTY
      streaming: false
      drop_params: true
      max_retries: 2
      generation_parameters:
        max_new_tokens: 256
        temperature: 0.0
      vllm:
        reasoning_parser: qwen3
        topology:
          kind: external_endpoint
          language_model_only: false
```

Notes:

- Keep `api_base` at `/v1`; do not point it at `/v1/chat/completions`.
- Prefer `hosted_vllm/<served-model-name>` and keep it aligned with the vLLM
  `--served-model-name`.
- `drop_params: true` is compatible with thinking mode because GAGE places
  vLLM thinking controls under `extra_body.chat_template_kwargs`.

### 2.2 P2 Internal Router: Multiple vLLM Endpoints

Use this profile when GAGE should create an in-process LiteLLM Router for
multiple vLLM deployments.

```yaml
backends:
  - backend_id: qwen3_litellm_router
    type: litellm
    config:
      provider: hosted_vllm
      custom_llm_provider: hosted_vllm
      route_mode: router
      model: hosted_vllm/qwen3-router
      api_key: EMPTY
      streaming: false
      drop_params: true
      generation_parameters:
        max_new_tokens: 128
        temperature: 0.0
      model_list:
        - model_name: qwen3-router
          litellm_params:
            model: hosted_vllm/qwen3-r1
            api_base: http://127.0.0.1:8001/v1
            api_key: EMPTY
        - model_name: qwen3-router
          litellm_params:
            model: hosted_vllm/qwen3-r2
            api_base: http://127.0.0.1:8002/v1
            api_key: EMPTY
      router_settings:
        routing_strategy: simple-shuffle
        allowed_fails: 1
        cooldown_time: 10
        num_retries: 2
      vllm:
        reasoning_parser: qwen3
        topology:
          kind: internal_router
          language_model_only: false
```

Keep `reasoning_parser`, chat template defaults, and
`language_model_only` consistent within a `model_name` group. If per-deployment
metadata is supplied, GAGE validates these assumptions and fails fast when the
deployments conflict.

### 2.3 External LiteLLM Gateway

Use this profile when a standalone LiteLLM Gateway already owns routing and
retry behavior.

```yaml
backends:
  - backend_id: qwen3_litellm_gateway
    type: litellm
    config:
      provider: litellm_gateway
      custom_llm_provider: hosted_vllm
      model: hosted_vllm/qwen3-gateway
      api_base: http://127.0.0.1:4000/v1
      api_key: EMPTY
      max_retries: 6
      generation_parameters:
        max_new_tokens: 128
        temperature: 0.0
      vllm:
        topology:
          kind: external_litellm_gateway
```

GAGE treats the Gateway as the external routing layer and collapses this
backend's outer retry budget to avoid multiplying Gateway retries.

## 3. Thinking / No-Thinking

For Qwen3 plus vLLM, thinking mode should be sent through
`extra_body.chat_template_kwargs.enable_thinking`. Evaluation authors usually
only need to set `generation_parameters.thinking_mode`.

```yaml
config:
  provider: hosted_vllm
  custom_llm_provider: hosted_vllm
  model: hosted_vllm/qwen3-p1
  api_base: http://127.0.0.1:8000/v1
  drop_params: true
  generation_parameters:
    max_new_tokens: 256
    temperature: 0.6
    thinking_mode: disabled
  thinking_policy:
    capability: supported
    on_unsupported: fail_fast
    on_mismatch: warn_and_continue
    record_effective_state: true
  vllm:
    reasoning_parser: qwen3
```

GAGE converts `thinking_mode: disabled` to:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

The response normalizer extracts reasoning from `message.reasoning_content`,
`message.reasoning_details`, `<think>...</think>` tags, and usage or raw-response
signals such as `reasoning_tokens`. If the observed response conflicts with the
configured expectation, GAGE records `metadata.thinking_mismatch` and applies
`thinking_policy.on_mismatch`.

## 4. Multimodal Input

The backend advertises `text`, `vision`, and `audio` modalities. Message
normalization preserves OpenAI-style content blocks and rejects unsupported media
when `fallback: fail` is used.

```yaml
multimodal:
  fallback: fail
  audio_url_mapping: input_audio
  default_audio_format: wav
  max_media_bytes: 5242880
vllm:
  topology:
    language_model_only: false
```

Important boundaries:

- Prefer `fallback: fail` so unsupported media cannot silently degrade to text.
- `max_media_bytes` is a lightweight guard for large base64 media payloads.
- If the vLLM server runs with `--language-model-only`, set
  `vllm.topology.language_model_only: true`; GAGE rejects multimodal requests
  before sending them.

## 5. Tool Calling

Tool definitions can come from sample-level `tools`. `tool_choice` and
`parallel_tool_calls` can come from either the sample or the backend config.

```yaml
tool_calling:
  enabled: true
  auto_tool_choice: true
  parallel_tool_calls: true
  require_server_parser: true
  expected_tool_parser: qwen3_xml
  expected_server_flags:
    enable_auto_tool_choice: true
    tool_call_parser: qwen3_xml
litellm_request:
  parallel_tool_calls: true
```

The vLLM server should match the declared parser settings:

```bash
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

GAGE normalizes returned `tool_calls`. When `function.arguments` is valid JSON,
it is parsed to a dictionary; malformed JSON is preserved as `raw_arguments` so
debugging information is not lost.

## 6. Structured Output, Streaming, and Async

### 6.1 JSON Mode

```yaml
litellm_request:
  response_format:
    type: json_object
generation_parameters:
  max_new_tokens: 128
  temperature: 0.0
```

GAGE no longer forces `response_format: {"type": "text"}`. Whether the response
is valid JSON still depends on the model and server-side structured-output
support.

### 6.2 Sync Streaming

```yaml
streaming: true
litellm_request:
  stream_options:
    include_usage: true
```

Streaming chunks are folded into a single GAGE result with `answer`,
`raw_response`, `usage`, and `finish_reason`.

### 6.3 Async

`LiteLLMBackend.ainvoke()` uses LiteLLM `acompletion`. Normal pipeline use
depends on the upper runtime path, but direct backend smoke tests can call
`ainvoke` explicitly.

```python
import asyncio
from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend


async def main():
    backend = LiteLLMBackend({
        "provider": "hosted_vllm",
        "custom_llm_provider": "hosted_vllm",
        "model": "hosted_vllm/qwen3-p1",
        "api_base": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
        "streaming": True,
        "generation_parameters": {"max_new_tokens": 64, "temperature": 0.0},
    })
    result = await backend.ainvoke({
        "sample": {
            "messages": [{"role": "user", "content": "Reply with OK."}]
        }
    })
    print(result["answer"])


asyncio.run(main())
```

## 7. Request Pass-Through

`litellm_request` is the explicit pass-through section for LiteLLM completion
parameters.

| Field | Purpose |
| --- | --- |
| `response_format` | JSON mode or other structured-output controls |
| `max_completion_tokens` | Provider-compatible token limit |
| `stream_options` | Streaming options such as `include_usage` |
| `parallel_tool_calls` | Parallel tool-call control |
| `logit_bias` | Token bias |
| `user` | End-user identifier |
| `metadata` | Request metadata |
| `fallbacks` | LiteLLM fallback configuration |
| `context_window_fallback_dict` | Context-window fallback configuration |
| `extra_body` | Provider-specific body, such as `chat_template_kwargs` |
| `drop_params` | Request-level LiteLLM drop-params override |

Sample payloads or `sampling_params` can override these fields for local
experiments on a subset of samples.

## 8. Retry, Errors, and Observability

GAGE avoids stacked retry loops:

| Scenario | Retry owner | Backend behavior |
| --- | --- | --- |
| Direct endpoint | GAGE LiteLLM backend | Uses `max_retries` |
| Internal Router | LiteLLM Router | Outer retry budget is collapsed to 1 |
| External LiteLLM Gateway | Gateway | Outer retry budget is collapsed to 1 |

Common normalized error types include:

| Error type | Common source |
| --- | --- |
| `dependency_unavailable` | Endpoint unavailable, server health timeout, missing dependency |
| `invalid_request` | `api_base` points at a concrete endpoint, invalid request parameter |
| `unsupported_capability` | Multimodal request to a language-only server, unsupported tool capability |
| `response_mismatch` | Observed thinking state conflicts with expectation |
| `invalid_media_payload` / `media_payload_too_large` | Invalid or oversized media content |
| `invalid_tool_arguments` | Tool-call arguments are not valid JSON |

The response normalizer also records a redacted observability summary, including
provider, model, hashed API base, route mode, topology, modality counts, tool
counts, retry owner, finish reason, usage, and latency. Full API keys and full
base64 media payloads are not emitted.

## 9. Experimental ModelServerLifecycle

`model_server` can run a user-provided bash command before evaluation, then wait
for configured health URLs before sending requests.

```yaml
model_server:
  enabled: true
  startup_command: |
    mkdir -p ${GAGE_VLLM_LOG_DIR:-/tmp/gage-vllm}
    nohup vllm serve ${GAGE_MODEL_PATH:-/mnt/model/Qwen3.6-35B-A3B} \
      --served-model-name ${GAGE_VLLM_P1_MODEL:-qwen3-p1} \
      --host 0.0.0.0 \
      --port ${GAGE_VLLM_P1_PORT:-8000} \
      --trust-remote-code \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_xml \
      > ${GAGE_VLLM_LOG_DIR:-/tmp/gage-vllm}/vllm.log 2>&1 &
  health:
    urls:
      - http://127.0.0.1:8000/v1/models
    timeout_seconds: 900
    interval_seconds: 5
```

Limitations:

- Commands run through `bash -lc`.
- GAGE handles startup and health-gating only; it does not manage log rotation,
  resource cleanup, port governance, or process orchestration.
- Basic dangerous command patterns such as `curl | bash`, `wget | sh`,
  `rm -rf /`, and `mkfs` are blocked.
- Health timeout is classified as `dependency_unavailable`.

Use an external process manager, container platform, or job system for production
or long-running stress tests.

## 10. Configuration Skeleton

```yaml
backends:
  - backend_id: qwen3_litellm
    type: litellm
    config:
      provider: hosted_vllm
      custom_llm_provider: hosted_vllm
      route_mode: direct
      model: hosted_vllm/qwen3-p1
      api_base: http://127.0.0.1:8000/v1
      api_key: EMPTY
      streaming: false
      drop_params: true
      timeout: 120
      max_retries: 2
      generation_parameters:
        max_new_tokens: 256
        temperature: 0.0
        top_p: 0.95
        thinking_mode: disabled
      litellm_request:
        response_format: null
        stream_options: null
        parallel_tool_calls: null
        extra_body: {}
        drop_params: null
      vllm:
        reasoning_parser: qwen3
        topology:
          kind: external_endpoint
          language_model_only: false
      thinking_policy:
        capability: auto
        on_unsupported: warn_and_continue
        on_mismatch: warn_and_continue
        record_effective_state: true
      multimodal:
        fallback: fail
        audio_url_mapping: input_audio
        default_audio_format: wav
        max_media_bytes: 5242880
      tool_calling:
        enabled: false
        auto_tool_choice: false
        parallel_tool_calls: null
        require_server_parser: false
        expected_tool_parser: null
      model_server:
        enabled: false
        startup_command: null
        health:
          urls: []
```

## 11. Validation Checklist

Before a real run, check dependency versions:

```bash
python - <<'PY'
import importlib.metadata as md

for pkg in ("litellm", "vllm"):
    print(pkg, md.version(pkg))
PY
```

Minimum dependency baseline:

```text
litellm >= 1.85.1
vllm >= 0.21.0
```

Recommended smoke coverage:

| Scenario | Purpose |
| --- | --- |
| Basic chat | Validate endpoint, served model, usage, and latency |
| Router distribution | Validate multi-deployment routing and isolation |
| Thinking on/off | Validate `extra_body.chat_template_kwargs` survives `drop_params` |
| Multimodal input | Validate image/audio content block preservation |
| Tool calling | Validate vLLM parser output normalization |
| JSON mode | Validate `response_format` pass-through |
| Sync streaming / async | Validate chunk folding and `acompletion` |
| Error classification | Validate dead endpoint, 4xx, and unsupported capability cases |
| Concurrency | Validate parallel behavior after removing the global call lock |
| ModelServerLifecycle | Validate startup command and health-gate behavior |

macOS local environments can use `vllm 0.21.0+cpu` for imports and config
validation. Real server behavior should still be validated in a Linux + GPU
environment.
