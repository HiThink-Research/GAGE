from gage_eval.tools.distill import DistillError, DistillTaskAnalysis, analyze_tasks_for_distill


def _base_payload() -> dict:
    return {
        "metadata": {"name": "builtin_distill_guard"},
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
        "datasets": [
            {
                "dataset_id": "dummy_dataset",
                "loader": "jsonl",
                "params": {"path": "dummy.jsonl"},
            }
        ],
        "role_adapters": [
            {
                "adapter_id": "dut",
                "role_type": "dut_model",
            }
        ],
    }


def test_single_task_passes():
    payload = _base_payload()
    payload["tasks"] = [
        {
            "task_id": "task_single",
            "dataset_id": "dummy_dataset",
            "steps": [{"step": "inference", "adapter_id": "dut"}],
        }
    ]
    result = analyze_tasks_for_distill(payload)
    assert isinstance(result, DistillTaskAnalysis)
    assert result.mode == "ATOMIC"
    assert result.is_monolithic is False
    assert result.task_ids == ("task_single",)


def test_zero_task_is_treated_as_singleton():
    payload = _base_payload()
    payload["tasks"] = []
    result = analyze_tasks_for_distill(payload)
    assert result.mode == "ATOMIC"
    assert result.is_monolithic is False
    assert result.task_ids == ()


def test_multi_task_rejected_without_force():
    payload = _base_payload()
    payload["tasks"] = [
        {
            "task_id": "task_a",
            "dataset_id": "dummy_dataset",
            "steps": [{"step": "inference", "adapter_id": "dut"}],
        },
        {
            "task_id": "task_b",
            "dataset_id": "dummy_dataset",
            "steps": [{"step": "inference", "adapter_id": "dut"}],
        },
    ]
    try:
        analyze_tasks_for_distill(payload, force_merge=False)
    except DistillError as exc:
        message = str(exc)
        assert "multi-task config detected" in message
        assert "task_a" in message and "task_b" in message
        assert hasattr(exc, "context")
        ctx = exc.context  # type: ignore[attr-defined]
        assert isinstance(ctx, DistillTaskAnalysis)
        assert ctx.mode == "REJECTED"
        assert ctx.task_ids == ("task_a", "task_b")
    else:
        raise AssertionError("expected DistillError when multi task and force_merge=False")


def test_multi_task_allowed_with_force_merge():
    payload = _base_payload()
    payload["tasks"] = [
        {
            "task_id": "task_a",
            "dataset_id": "dummy_dataset",
            "steps": [{"step": "inference", "adapter_id": "dut"}],
        },
        {
            "task_id": "task_b",
            "dataset_id": "dummy_dataset",
            "steps": [{"step": "inference", "adapter_id": "dut"}],
        },
    ]
    result = analyze_tasks_for_distill(payload, force_merge=True)
    assert result.mode == "MONOLITHIC"
    assert result.is_monolithic is True
    assert result.task_ids == ("task_a", "task_b")


if __name__ == "__main__":
    test_single_task_passes()
    test_zero_task_is_treated_as_singleton()
    test_multi_task_rejected_without_force()
    test_multi_task_allowed_with_force_merge()
    print("builtin_distill task guard tests passed")
