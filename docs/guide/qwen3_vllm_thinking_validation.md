# Qwen3 vLLM Thinking Validation

## Overview

This change adds a reproducible local validation path for Qwen3 `thinking` versus `no-thinking`
 behavior on the native `vllm` backend.

The validation uses:

- Two dedicated pipeline configs:
  - `config/custom/examples/qwen3_vllm_piqa_messages_thinking.yaml`
  - `config/custom/examples/qwen3_vllm_piqa_messages_no_thinking.yaml`
- One raw-message fixture:
  - `tests/fixtures/piqa_10_vllm_messages.jsonl`
- One local model:
  - `/Users/shuo/models/Qwen3-0.6B`

The goal is to isolate the effect of `chat_template_kwargs.enable_thinking` while keeping the
 model, sampling parameters, and dataset fixed.

## What This Enables

Before this change, the repository could not reliably validate `thinking` versus `no-thinking`
 through backend-level configuration on `vllm`.

After this change, the local `vllm` backend can be validated with two configs that differ only in:

```yaml
chat_template_kwargs:
  enable_thinking: true
```

versus:

```yaml
chat_template_kwargs:
  enable_thinking: false
```

This allows A/B testing with the same dataset and model while observing:

- reasoning content presence
- answer text differences
- latency differences

## Root Cause 1: Backend Config Did Not Reach Prompt Rendering

### Problem

`VLLMBackend.prepare_inputs()` only used request-level `chat_template_kwargs` coming from the
 payload normalization path.

That meant backend-level configuration such as:

```yaml
config:
  chat_template_kwargs:
    enable_thinking: true
```

was parsed by `BaseBackend`, but never merged into the actual request prepared for
 `tokenizer.apply_chat_template(...)`.

### Why It Broke Thinking Validation

The Qwen3 `thinking` switch is applied during chat template rendering, not inside the vLLM engine
 itself. If `enable_thinking` does not reach `apply_chat_template`, the rendered prompt is the same
 as a normal request and the experiment becomes invalid.

### Fix

`src/gage_eval/role/model/backends/vllm_backend.py`

The backend now merges:

- resolved backend-level thinking kwargs
- request-level `chat_template_kwargs`

with request-level values taking precedence.

In practice, this means backend config can now provide a default `enable_thinking` value while
 still allowing per-request overrides.

### Tests

`tests/backends/test_vllm_backend_chat_template.py`

Added coverage for:

- backend default `chat_template_kwargs` are applied
- request payload `chat_template_kwargs` override backend defaults

## Root Cause 2: String Message Content Was Split Into Characters

### Problem

`sample_from_dict()` treated `message.content` as an iterable unless it was already a list or dict.
When `content` was a plain string, it was iterated character by character.

For example:

```json
{"role": "user", "content": "hello world"}
```

was effectively converted into fragments like:

- `"h"`
- `"e"`
- `"l"`
- `"l"`
- `"o"`

instead of one text fragment.

### Why It Broke Thinking Validation

The raw-message PIQA fixture is meant to test chat-template behavior directly. Once the user message
 is broken into hundreds of one-character fragments, the request no longer reflects the intended
 chat input and any thinking/no-thinking comparison becomes unreliable.

### Fix

`src/gage_eval/assets/datasets/sample.py`

Added explicit normalization rules for `message.content`:

- `None -> []`
- `dict -> [dict]`
- `list -> list`
- scalar values, including `str -> [scalar]`

This preserves plain string content as one logical text fragment.

### Tests

`tests/preprocessors/test_dataclass_adapter.py`

Added coverage for:

- plain string content remains a single fragment
- dict content is wrapped correctly

## Validation Dataset Design

The raw-message fixture was introduced to avoid interference from preprocessors that may flatten,
 pre-render, or otherwise normalize prompts before the backend can apply the model chat template.

Fixture:

- `tests/fixtures/piqa_10_vllm_messages.jsonl`

Each record includes:

- standardized `messages`
- `references`
- `label`
- `metadata.correct_choice`

This makes it possible to:

- inspect backend outputs directly
- recompute choice accuracy from cached samples
- verify whether `reasoning_content` appears only when expected

## How To Run

Thinking:

```bash
VLLM_CPU_KVCACHE_SPACE=4 /Users/shuo/code/GAGE_workspace/env/.venv/bin/python run.py \
  --config config/custom/examples/qwen3_vllm_piqa_messages_thinking.yaml \
  --output-dir runs/thinking_mode_validation_vllm \
  --run-id qwen3_vllm_piqa_messages_thinking_final_v2 \
  --concurrency 1 --gpus 0 --cpus 4
```

No-thinking:

```bash
VLLM_CPU_KVCACHE_SPACE=4 /Users/shuo/code/GAGE_workspace/env/.venv/bin/python run.py \
  --config config/custom/examples/qwen3_vllm_piqa_messages_no_thinking.yaml \
  --output-dir runs/thinking_mode_validation_vllm \
  --run-id qwen3_vllm_piqa_messages_no_thinking_final_v2 \
  --concurrency 1 --gpus 0 --cpus 4
```

## Expected Output Artifacts

Each run writes:

- `events.jsonl`
- `samples.jsonl`
- `summary.json`
- per-sample cached JSON under `samples/`

Example result locations:

- `runs/thinking_mode_validation_vllm/qwen3_vllm_piqa_messages_thinking_final_v2/`
- `runs/thinking_mode_validation_vllm/qwen3_vllm_piqa_messages_no_thinking_final_v2/`

Comparison artifact:

- `runs/thinking_mode_validation_vllm/qwen3_vllm_piqa_messages_comparison_final_v2.json`

## Notes

- The built-in `multi_choice_accuracy` summary is currently not a reliable indicator for this
  experiment. Real accuracy should be recomputed from cached sample outputs and
  `metadata.correct_choice`.
- On this machine, local vLLM runs on CPU, so the validation is correct but not fast.
