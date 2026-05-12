from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
SOURCE_ROOT = (
    WORKSPACE_ROOT
    / "agent-eval"
    / "0421"
    / "tau"
    / "samples"
    / "runtime"
    / "tau2_telecom_vllm_full"
)
TAU2_SOURCE_ROOT = (
    WORKSPACE_ROOT
    / "agent-eval"
    / "0421"
    / "tau 2"
    / "samples"
    / "runtime"
    / "tau2_telecom_vllm_full"
)
GEMMA4_ALL_SOURCE_ROOT = REPO_ROOT / "runs" / "gemma4_all" / "gemma4_all" / "tau" / "samples"
QWEN_ALL_TAU_ROOT = WORKSPACE_ROOT / "agent-eval" / "0421" / "qwen3_6_35B_all" / "tau"
QWEN_GPT_TAU_ROOT = WORKSPACE_ROOT / "agent-eval" / "0421" / "qwen3_6_35B_all_gpt" / "tau"
SOURCE_ROOTS = {
    "tau": SOURCE_ROOT,
    "tau2": TAU2_SOURCE_ROOT,
    "gemma4_all": GEMMA4_ALL_SOURCE_ROOT,
    "qwen_all_tau": QWEN_ALL_TAU_ROOT,
    "qwen_gpt_tau": QWEN_GPT_TAU_ROOT,
}
OUTPUT_ROOT = Path(__file__).resolve().parent / "runtime_samples"


TRACE_FIXTURES = [
    {
        "output": "0421_tau_plain_text_missing_tool_response.txt",
        "sample_dir": "telecom_[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]__trial_0",
        "trace_index": 0,
        "description": "Real first assistant response that did not call any tool in the failed 2026-04-21 run.",
    },
    {
        "output": "0421_tau_harmony_xml_respond_response.txt",
        "sample_dir": "telecom_[mobile_data_issue]airplane_mode_on|bad_network_preference[PERSONA:Hard]__trial_0",
        "trace_index": 1,
        "description": "Real assistant response containing a harmony-style XML respond tool call from the failed 2026-04-21 run.",
    },
    {
        "output": "0421_tau2_think_tail_bare_json_respond_response.txt",
        "source_run": "tau2",
        "sample_dir": "telecom_[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_saver_mode_on[PERSONA:Hard]__trial_0",
        "trace_index": 0,
        "description": "Real assistant response from the latest 2026-04-21 tau 2 run with a bare JSON respond call immediately after </think>.",
    },
    {
        "output": "0421_gemma4_airline_bare_call_respond_response.txt",
        "source_run": "gemma4_all",
        "source": "judge_tau2_airline_vllm_full/tau2_airline_vllm_full_airline_0__trial_0.json",
        "source_field": "model_output.agent_trace[0].output.answer",
        "description": "Real Gemma 4 assistant response from the 2026-04-21 airline run with a bare call: respond tool call.",
    },
    {
        "output": "0421_qwen_gpt_airline_plain_text_response.txt",
        "source_run": "qwen_gpt_tau",
        "source": "samples.jsonl",
        "sample_id": "tau2_airline_vllm_full:airline_0__trial_2",
        "source_field": "model_output.answer",
        "description": "Real Qwen3.6/GPT-simulator airline response from 2026-04-21 that returned plain text instead of calling respond.",
    },
]

ARTIFACT_FIXTURES = [
    {
        "output": "0421_tau_old_failed_cost.json",
        "source": "telecom_[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]__trial_0/artifacts/tau2_cost.json",
        "description": "Original failed cost artifact documenting the old too_many_errors classification.",
    }
]

JSON_FIXTURES = [
    {
        "output": "0421_tau2_unknown_user_side_tool_error.json",
        "source_run": "tau2",
        "sample_dir": "telecom_[mms_issue]break_apn_mms_setting|user_abroad_roaming_enabled_off[PERSONA:Hard]__trial_0",
        "source_field": "scheduler_result.agent_output.agent_trace[8]",
        "description": "Real failed tool trace from the latest 2026-04-21 tau 2 run where the DUT called a user-side device tool directly.",
    },
    {
        "output": "0421_qwen_simulation_terminated_tool_trace.json",
        "source_run": "qwen_all_tau",
        "source": "samples.jsonl",
        "sample_id": "tau2_retail_vllm_full:retail_0__trial_0",
        "source_field": "model_output.agent_trace[15]",
        "description": "Real failed retail tool trace from 2026-04-21 where tau2 had already terminated and returned final_answer=simulation_terminated.",
    },
    {
        "output": "0421_qwen_real_terminal_signals.json",
        "source_run": "qwen_all_tau",
        "source": "samples.jsonl",
        "description": "Real user simulator terminal-token outputs from the 2026-04-21 Qwen3.6 all-subsets run.",
        "terminal_messages": [
            {
                "sample_id": "tau2_telecom_vllm_full:telecom_[mobile_data_issue]airplane_mode_on|data_saver_mode_on|user_abroad_roaming_disabled_on[PERSONA:None]__trial_1",
                "trace_index": 15,
                "field": "user_message",
                "kind": "STOP",
            },
            {
                "sample_id": "tau2_telecom_vllm_full:telecom_[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]__trial_0",
                "trace_index": 13,
                "field": "user_message",
                "kind": "STOP",
            },
            {
                "sample_id": "tau2_telecom_vllm_full:telecom_[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]__trial_0",
                "trace_index": 17,
                "field": "user_message",
                "kind": "TRANSFER",
            },
            {
                "sample_id": "tau2_telecom_vllm_full:telecom_[service_issue]airplane_mode_on|overdue_bill_suspension[PERSONA:None]__trial_1",
                "trace_index": 21,
                "field": "user_message",
                "kind": "TRANSFER",
            },
            {
                "sample_id": "tau2_retail_vllm_full:retail_5__trial_0",
                "trace_index": 1,
                "field": "user_message",
                "kind": "OUT_OF_SCOPE",
            },
            {
                "sample_id": "tau2_telecom_vllm_full:telecom_[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]__trial_0",
                "trace_index": 12,
                "field": "user_message",
                "kind": "STOP",
            },
        ],
    }
]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"source_root": _relative(SOURCE_ROOT), "fixtures": []}
    existing_fixtures = _read_existing_fixture_entries()

    for spec in TRACE_FIXTURES:
        if str(spec.get("source") or "").endswith(".jsonl"):
            metadata_path = _source_root(spec) / spec["source"]
            if not metadata_path.exists():
                _append_existing_fixture(manifest, existing_fixtures, spec["output"], metadata_path)
                continue
            metadata = _read_jsonl_sample(metadata_path, spec["sample_id"])
            source_field = spec["source_field"]
            answer = _read_source_field(metadata, source_field)
        elif "source" in spec:
            metadata_path = _source_root(spec) / spec["source"]
            if not metadata_path.exists():
                _append_existing_fixture(manifest, existing_fixtures, spec["output"], metadata_path)
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_field = spec["source_field"]
            answer = _read_source_field(metadata, source_field)
        else:
            metadata_path = _source_root(spec) / spec["sample_dir"] / "runtime_metadata.json"
            if not metadata_path.exists():
                _append_existing_fixture(manifest, existing_fixtures, spec["output"], metadata_path)
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            trace = metadata["scheduler_result"]["agent_output"]["agent_trace"]
            trace_index = int(spec["trace_index"])
            source_field = f"scheduler_result.agent_output.agent_trace[{trace_index}].output.answer"
            answer = trace[trace_index]["output"]["answer"]
        output_path = OUTPUT_ROOT / spec["output"]
        output_path.write_text(answer, encoding="utf-8")
        manifest["fixtures"].append(
            {
                "file": spec["output"],
                "description": spec["description"],
                "source": _relative(metadata_path),
                "source_field": source_field,
                "sha256": _sha256(answer.encode("utf-8")),
            }
        )

    for spec in JSON_FIXTURES:
        if "terminal_messages" in spec:
            metadata_path = _source_root(spec) / spec["source"]
            if not metadata_path.exists():
                _append_existing_fixture(manifest, existing_fixtures, spec["output"], metadata_path)
                continue
            payload = _build_terminal_signal_fixture(metadata_path, spec["terminal_messages"])
            source_field = "derived:model_output.agent_trace[].output.user_message"
        elif str(spec.get("source") or "").endswith(".jsonl"):
            metadata_path = _source_root(spec) / spec["source"]
            if not metadata_path.exists():
                _append_existing_fixture(manifest, existing_fixtures, spec["output"], metadata_path)
                continue
            metadata = _read_jsonl_sample(metadata_path, spec["sample_id"])
            source_field = spec["source_field"]
            payload = _read_source_field(metadata, source_field)
        else:
            metadata_path = _source_root(spec) / spec["sample_dir"] / "runtime_metadata.json"
            if not metadata_path.exists():
                _append_existing_fixture(manifest, existing_fixtures, spec["output"], metadata_path)
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_field = spec["source_field"]
            payload = _read_source_field(metadata, source_field)
        output_path = OUTPUT_ROOT / spec["output"]
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        manifest["fixtures"].append(
            {
                "file": spec["output"],
                "description": spec["description"],
                "source": _relative(metadata_path),
                "source_field": source_field,
                "sha256": _sha256(json.dumps(payload, sort_keys=True).encode("utf-8")),
            }
        )

    for spec in ARTIFACT_FIXTURES:
        source_path = SOURCE_ROOT / spec["source"]
        if not source_path.exists():
            _append_existing_fixture(manifest, existing_fixtures, spec["output"], source_path)
            continue
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        output_path = OUTPUT_ROOT / spec["output"]
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        manifest["fixtures"].append(
            {
                "file": spec["output"],
                "description": spec["description"],
                "source": _relative(source_path),
                "sha256": _sha256(json.dumps(payload, sort_keys=True).encode("utf-8")),
            }
        )

    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_existing_fixture_entries() -> dict[str, dict[str, Any]]:
    manifest_path = OUTPUT_ROOT / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = manifest.get("fixtures", [])
    if not isinstance(fixtures, list):
        return {}
    return {
        fixture["file"]: fixture
        for fixture in fixtures
        if isinstance(fixture, dict) and isinstance(fixture.get("file"), str)
    }


def _append_existing_fixture(
    manifest: dict[str, Any],
    existing_fixtures: dict[str, dict[str, Any]],
    output: str,
    missing_source: Path,
) -> None:
    fixture = existing_fixtures.get(output)
    if fixture is None:
        raise FileNotFoundError(missing_source)
    manifest["fixtures"].append(fixture)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def _source_root(spec: dict[str, Any]) -> Path:
    source_run = str(spec.get("source_run") or "tau")
    return SOURCE_ROOTS[source_run]


def _read_source_field(payload: Any, source_field: str) -> Any:
    current = payload
    for part in source_field.split("."):
        name, indexes = _split_field_part(part)
        if name:
            current = current[name]
        for index in indexes:
            current = current[index]
    return current


def _read_jsonl_sample(path: Path, sample_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        if sample.get("sample_id") == sample_id:
            return sample
    raise KeyError(f"{sample_id} not found in {path}")


def _build_terminal_signal_fixture(
    path: Path,
    message_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for spec in message_specs:
        sample = _read_jsonl_sample(path, spec["sample_id"])
        source_field = (
            f"model_output.agent_trace[{int(spec['trace_index'])}].output.{spec['field']}"
        )
        content = _read_source_field(sample, source_field)
        if not isinstance(content, str):
            raise TypeError(f"{source_field} is not a string")
        messages.append(
            {
                "source_sample_id": spec["sample_id"],
                "trace_index": int(spec["trace_index"]),
                "field": spec["field"],
                "kind": spec["kind"],
                "raw": _extract_terminal_token_line(content),
                "content": content,
            }
        )
    return {"source": _relative(path), "messages": messages}


def _extract_terminal_token_line(content: str) -> str:
    for line in content.splitlines() or [content]:
        stripped = line.strip()
        if stripped.startswith("###") and stripped.endswith("###"):
            return stripped
    raise ValueError(f"No terminal token line found in {content!r}")


def _split_field_part(part: str) -> tuple[str, list[int]]:
    name = part.split("[", 1)[0]
    indexes = [int(match) for match in re.findall(r"\[(\d+)\]", part)]
    return name, indexes


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
