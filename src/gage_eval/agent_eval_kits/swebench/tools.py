from __future__ import annotations

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_messages, normalize_tools
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry


_DEFAULT_WORKING_DIR = "/app"

_SWEBENCH_SYSTEM_PROMPT_TEMPLATE = """\
You are a software engineering agent fixing a real-world repository bug.

WORKFLOW:
1. Explore the repository to find the relevant source file(s).
2. Reproduce the bug locally when practical, using a tiny repro script if helpful.
3. Implement a minimal source-code fix.
4. Re-run the repro or targeted checks to confirm the fix.
5. Consider edge cases before submitting.

CRITICAL RULES:
- DO NOT modify test files. The harness applies the official tests itself.
- DO NOT cat or list huge files; use grep, head, tail, or sed line ranges for navigation.
- Use absolute repository paths starting with {working_dir}/ for str_replace_editor and view_file_window.
- Keep changes minimal and focused on the bug.
- You MUST call submit_patch_tool before terminating, even if you think the fix is complete; it captures submission.patch.
- DO NOT finish with a prose explanation; the harness only scores your submitted patch.
"""


def build_swebench_runtime_context(sample: dict[str, object]) -> dict[str, object]:
    """Build reusable SWE-bench runtime context."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return {
        "instruction": build_swebench_instruction(sample),
        "repo": metadata.get("repo"),
        "base_commit": metadata.get("base_commit"),
        "test_command": metadata.get("test_command"),
    }


def build_swebench_instruction(
    sample: dict[str, object],
    *,
    working_dir: str = _DEFAULT_WORKING_DIR,
) -> str:
    """Build the SWE-bench instruction with submission contract."""

    system_prompt = _build_swebench_system_prompt(working_dir=working_dir)
    instruction = extract_instruction(sample)
    if not instruction:
        return system_prompt
    if "submit_patch_tool" in instruction and "DO NOT modify test files" in instruction:
        return instruction
    return f"{instruction.rstrip()}\n\n{system_prompt}"


def build_swebench_messages(
    sample: dict[str, object],
    *,
    working_dir: str = _DEFAULT_WORKING_DIR,
) -> list[dict[str, object]]:
    """Build framework-loop messages for SWE-bench."""

    system_prompt = _build_swebench_system_prompt(working_dir=working_dir)
    messages = normalize_messages(sample, fallback_text=extract_instruction(sample))
    if not messages:
        return [{"role": "system", "content": system_prompt}]
    return [
        {"role": "system", "content": system_prompt},
        *messages,
    ]


def build_swebench_tools(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build the canonical SWE-bench tool surface."""

    default_tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a bounded shell command in the repository workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_s": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "View, create, or edit files using exact string replacement.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        },
                        "path": {"type": "string"},
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "file_text": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "insert_line": {"type": "integer"},
                    },
                    "required": ["command", "path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "view_file_window",
                "description": "Read a small line-numbered window from a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "line_count": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_in_repo",
                "description": "Search repository files for a text pattern and return bounded line-numbered matches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_file",
                "description": "Find files by name under a repository path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_patch_tool",
                "description": "Review and then capture the final staged repository diff.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timeout_s": {"type": "integer"},
                        "force": {"type": "boolean"},
                    },
                },
            },
            "x-gage": {"final_answer_from": "stdout"},
        },
    ]
    return normalize_tools(sample, default_tools)


def build_tool_registry() -> RuntimeToolRegistry:
    registry = RuntimeToolRegistry()
    for schema in build_swebench_tools({}):
        registry.register_provider_schema(schema, provider="swebench", provider_kind="environment")
    return registry


def _build_swebench_system_prompt(*, working_dir: str) -> str:
    return _SWEBENCH_SYSTEM_PROMPT_TEMPLATE.format(
        working_dir=_normalize_working_dir(working_dir),
    )


def _normalize_working_dir(working_dir: str) -> str:
    value = str(working_dir or "").strip()
    if not value.startswith("/"):
        value = _DEFAULT_WORKING_DIR
    return value.rstrip("/") or "/"
