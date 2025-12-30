from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _ArenaAdapter:
    role_type = "arena"
    resource_requirement = {}
    backend = None

    def __init__(self) -> None:
        self.adapter_id = "arena"

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):
        return {
            "winner": "Black",
            "result": "win",
            "reason": "test",
            "move_count": 1,
            "illegal_move_count": 0,
            "final_board": "board",
            "game_log": [],
        }


def test_sample_loop_runs_arena_step():
    samples = [
        {
            "id": "s1",
            "messages": [],
            "choices": [],
        }
    ]
    loop = SampleLoop(samples, concurrency=1)
    planner = TaskPlanner()
    planner.configure_custom_steps([{"step": "arena", "adapter_id": "arena"}])
    rm = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]))
    rm.register_role_adapter("arena", _ArenaAdapter())
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="arena-loop"))

    loop.run(planner, rm, trace)

    assert loop.processed_count == 1
    assert samples[0].get("predict_result")
    assert samples[0]["predict_result"][0]["winner"] == "Black"
