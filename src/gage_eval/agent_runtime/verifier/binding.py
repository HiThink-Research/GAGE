from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from gage_eval.agent_runtime.serialization import to_json_compatible


@dataclass(frozen=True)
class JudgeBinding:
    """Captures the cold-path verifier binding for one runtime plan."""

    judge_mode: Literal["role_adapter", "runtime_verifier", "none"]
    judge_adapter_id: str | None = None
    benchmark_kit_id: str | None = None
    verifier_kind: Literal["native", "judge_adapter"] | None = None
    verifier_resource_refs: dict[str, Any] | None = None
    failure_policy: Literal["bind_failure", "raise"] = "bind_failure"
    result_normalizer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {
            "judge_mode": self.judge_mode,
            "judge_adapter_id": self.judge_adapter_id,
            "benchmark_kit_id": self.benchmark_kit_id,
            "verifier_kind": self.verifier_kind,
            "verifier_resource_refs": to_json_compatible(self.verifier_resource_refs or {}),
            "failure_policy": self.failure_policy,
            "result_normalizer": self.result_normalizer,
        }
