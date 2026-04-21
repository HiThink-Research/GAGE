from __future__ import annotations

import hashlib
import json
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
]

ARTIFACT_FIXTURES = [
    {
        "output": "0421_tau_old_failed_cost.json",
        "source": "telecom_[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]__trial_0/artifacts/tau2_cost.json",
        "description": "Original failed cost artifact documenting the old too_many_errors classification.",
    }
]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"source_root": _relative(SOURCE_ROOT), "fixtures": []}

    for spec in TRACE_FIXTURES:
        metadata_path = SOURCE_ROOT / spec["sample_dir"] / "runtime_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trace = metadata["scheduler_result"]["agent_output"]["agent_trace"]
        trace_index = int(spec["trace_index"])
        answer = trace[trace_index]["output"]["answer"]
        output_path = OUTPUT_ROOT / spec["output"]
        output_path.write_text(answer, encoding="utf-8")
        manifest["fixtures"].append(
            {
                "file": spec["output"],
                "description": spec["description"],
                "source": _relative(metadata_path),
                "source_field": f"scheduler_result.agent_output.agent_trace[{trace_index}].output.answer",
                "sha256": _sha256(answer.encode("utf-8")),
            }
        )

    for spec in ARTIFACT_FIXTURES:
        source_path = SOURCE_ROOT / spec["source"]
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


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
