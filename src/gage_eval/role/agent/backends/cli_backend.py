"""CLI-based agent backend implementation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from gage_eval.role.agent.backends.base import AgentBackend, normalize_agent_output


class CliBackend(AgentBackend):
    """Agent backend that executes a local command."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._command = config.get("command")
        if not self._command:
            raise ValueError("command is required for CliBackend")
        self._workdir = config.get("workdir")
        self._input_path = config.get("input_path")
        self._output_path = config.get("output_path")
        self._trace_path = config.get("trace_path")
        self._timeout_s = config.get("timeout_s", 60)

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._input_path:
            self._write_json(Path(self._input_path), payload)
        result = subprocess.run(
            self._command,
            shell=True,
            cwd=self._workdir,
            text=True,
            capture_output=True,
            timeout=self._timeout_s,
            check=False,
        )
        raw_output = self._load_output(result.stdout)
        normalized = normalize_agent_output(raw_output)
        normalized.setdefault("raw_stdout", result.stdout)
        normalized.setdefault("raw_stderr", result.stderr)
        return normalized

    def _load_output(self, stdout: str) -> Any:
        if self._output_path:
            output_file = Path(self._output_path)
            if output_file.exists():
                data = self._read_json(output_file)
                if data is not None:
                    return data
        parsed = _try_parse_json(stdout)
        if parsed is not None:
            return parsed
        return stdout.strip()

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


def _try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None
