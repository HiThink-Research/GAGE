from gage_eval.config.builtin_compiler import (
    RuntimeParameterError,
    substitute_runtime_placeholders,
)


def _definition_with_placeholder(path: str) -> dict:
    return {
        "metadata": {"name": "compiler_prefix_guard"},
        "tasks": [
            {
                "task_id": "alpha",
                "concurrency": "${" + path + "}",
            }
        ],
    }


def test_force_merge_requires_task_id():
    definition = _definition_with_placeholder("runtime.tasks.concurrency")
    runtime_params = {"tasks": {"alpha": {"concurrency": 3}}}
    try:
        substitute_runtime_placeholders(definition, runtime_params, force_merge=True, task_ids=("alpha",))
    except RuntimeParameterError as exc:
        text = str(exc)
        assert "task id" in text
        assert "definition.tasks[0].concurrency" in text
    else:
        raise AssertionError("expected RuntimeParameterError when task id is missing in runtime.tasks path")


def test_force_merge_unknown_task_id_rejected():
    definition = _definition_with_placeholder("runtime.tasks.ghost.concurrency")
    runtime_params = {"tasks": {"alpha": {"concurrency": 3}}}
    try:
        substitute_runtime_placeholders(definition, runtime_params, force_merge=True, task_ids=("alpha", "beta"))
    except RuntimeParameterError as exc:
        message = str(exc)
        assert "ghost" in message
        assert "alpha" in message and "beta" in message
        assert "definition.tasks[0].concurrency" in message
    else:
        raise AssertionError("expected RuntimeParameterError when runtime path references unknown task id")


def test_force_merge_with_valid_task_id_passes():
    definition = _definition_with_placeholder("runtime.tasks.alpha.concurrency")
    runtime_params = {"tasks": {"alpha": {"concurrency": 5}}}
    result = substitute_runtime_placeholders(definition, runtime_params, force_merge=True, task_ids=("alpha",))
    assert result["tasks"][0]["concurrency"] == 5


def test_type_preservation_for_full_placeholder():
    definition = {"datasets": [{"limit": "${runtime.datasets.limit}"}]}
    runtime_params = {"datasets": {"limit": 500}}
    result = substitute_runtime_placeholders(definition, runtime_params)
    assert isinstance(result["datasets"][0]["limit"], int)
    assert result["datasets"][0]["limit"] == 500


def test_inline_string_interpolation_keeps_string():
    definition = {"path": "/data/${runtime.name}.json"}
    runtime_params = {"name": "foo"}
    result = substitute_runtime_placeholders(definition, runtime_params)
    assert result["path"] == "/data/foo.json"


def test_error_context_shows_field_path():
    definition = {"datasets": [{"limit": "${runtime.datasets.limit}"}]}
    runtime_params = {"datasets": {}}
    try:
        substitute_runtime_placeholders(definition, runtime_params)
    except RuntimeParameterError as exc:
        text = str(exc)
        assert "definition.datasets[0].limit" in text
    else:
        raise AssertionError("expected RuntimeParameterError when key is missing with field context")


if __name__ == "__main__":
    test_force_merge_requires_task_id()
    test_force_merge_unknown_task_id_rejected()
    test_force_merge_with_valid_task_id_passes()
    test_type_preservation_for_full_placeholder()
    test_inline_string_interpolation_keeps_string()
    test_error_context_shows_field_path()
    print("builtin_compiler runtime prefix guard tests passed")
