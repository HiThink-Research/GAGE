from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Any

from gage_eval.role.arena.runtime_services import ArenaRuntimeServiceHub
from gage_eval.role.arena.visualization.contracts import ControlCommand


class _StubVisualizer:
    def __init__(self) -> None:
        self.action_queue = object()
        self.bind_calls: list[tuple[Any, str]] = []
        self.clear_calls: list[str] = []
        self.stopped = False

    def bind_action_queue(self, action_router: Any, *, sample_id: str) -> None:
        self.bind_calls.append((action_router, sample_id))

    def clear_action_queue(self, *, sample_id: str | None = None) -> None:
        self.clear_calls.append(str(sample_id))

    def stop(self) -> None:
        self.stopped = True


class _StubActionServer:
    def __init__(self) -> None:
        self.action_queue = object()
        self.chat_queue: Queue[dict[str, str]] = Queue()
        self.register_calls: list[tuple[str, Any]] = []
        self.unregister_calls: list[str] = []
        self.stopped = False

    def register_action_queue(self, sample_id: str, action_router: Any) -> None:
        self.register_calls.append((sample_id, action_router))

    def unregister_action_queue(self, sample_id: str) -> None:
        self.unregister_calls.append(sample_id)

    def stop(self) -> None:
        self.stopped = True


class _StubControlActionServer(_StubActionServer):
    def __init__(self) -> None:
        super().__init__()
        self.control_queue: Queue[dict[str, object]] = Queue()


class _StubWsHub:
    def __init__(self) -> None:
        self.registrations: list[Any] = []
        self.unregistered: list[str] = []
        self.stopped = False

    def register_display(self, registration: Any) -> None:
        self.registrations.append(registration)

    def unregister_display(self, display_id: str) -> None:
        self.unregistered.append(display_id)

    def stop(self) -> None:
        self.stopped = True


class _StubLiveControlSource:
    def __init__(self, related_event_seq: int = 0) -> None:
        self.related_event_seq = int(related_event_seq)
        self.commands: list[ControlCommand] = []

    def apply_control_command(self, command: ControlCommand) -> int:
        self.commands.append(command)
        return self.related_event_seq


class _StubLiveVisualizer(_StubVisualizer):
    def __init__(self, live_source: _StubLiveControlSource | None = None) -> None:
        super().__init__()
        self.live_source = live_source
        self.resolve_calls: list[tuple[str, str | None]] = []

    def resolve_live_session(self, session_id: str, *, run_id: str | None = None) -> Any:
        self.resolve_calls.append((session_id, run_id))
        return self.live_source


def test_runtime_service_hub_initializes_each_shared_service_once() -> None:
    hub = ArenaRuntimeServiceHub(adapter_id="arena")
    visualizer_creations = 0
    action_server_creations = 0
    ws_hub_creations = 0
    results: list[tuple[Any, Any, Any]] = []
    lock = threading.Lock()

    def _visualizer_factory() -> _StubVisualizer:
        nonlocal visualizer_creations
        time.sleep(0.02)
        visualizer_creations += 1
        return _StubVisualizer()

    def _action_server_factory() -> _StubActionServer:
        nonlocal action_server_creations
        time.sleep(0.02)
        action_server_creations += 1
        return _StubActionServer()

    def _ws_hub_factory() -> _StubWsHub:
        nonlocal ws_hub_creations
        time.sleep(0.02)
        ws_hub_creations += 1
        return _StubWsHub()

    def _worker() -> None:
        visualizer = hub.ensure_visualizer(_visualizer_factory)
        action_server = hub.ensure_action_server(_action_server_factory)
        ws_hub = hub.ensure_ws_rgb_hub(_ws_hub_factory)
        with lock:
            results.append((visualizer, action_server, ws_hub))

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert len(results) == 6
    assert visualizer_creations == 1
    assert action_server_creations == 1
    assert ws_hub_creations == 1
    assert len({id(item[0]) for item in results}) == 1
    assert len({id(item[1]) for item in results}) == 1
    assert len({id(item[2]) for item in results}) == 1


def test_runtime_service_hub_routes_bindings_and_shutdown() -> None:
    hub = ArenaRuntimeServiceHub(adapter_id="arena")
    visualizer = hub.ensure_visualizer(_StubVisualizer)
    action_server = hub.ensure_action_server(_StubActionServer)
    ws_hub = hub.ensure_ws_rgb_hub(_StubWsHub)
    action_router = object()
    registration = object()

    hub.bind_sample_routes(
        sample_id="sample-1",
        action_server=action_server,
        action_router=action_router,
        visualizer=visualizer,
    )
    hub.register_display(
        display_id="display-1",
        hub=ws_hub,
        registration=registration,
    )
    hub.clear_sample_routes(
        sample_id="sample-1",
        action_server=action_server,
        visualizer=visualizer,
    )
    hub.shutdown()

    assert action_server.register_calls == [("sample-1", action_router)]
    assert visualizer.bind_calls == [(action_router, "sample-1")]
    assert action_server.unregister_calls == ["sample-1"]
    assert visualizer.clear_calls == ["sample-1"]
    assert ws_hub.registrations == [registration]
    assert ws_hub.unregistered == ["display-1"]
    assert visualizer.stopped is True
    assert action_server.stopped is True
    assert ws_hub.stopped is True
    assert hub.registered_displays() == set()


def test_runtime_service_hub_submits_chat_and_control_messages_with_control_sink() -> None:
    hub = ArenaRuntimeServiceHub(adapter_id="arena")
    action_server = hub.ensure_action_server(_StubControlActionServer)

    chat_receipt = hub.submit_chat_message(
        "session-1",
        None,
        {"playerId": "p0", "text": "hi"},
    )
    control_receipt = hub.submit_control_command(
        "session-1",
        None,
        {"commandType": "pause"},
    )

    assert chat_receipt.state == "accepted"
    assert control_receipt.state == "accepted"
    assert action_server.control_queue.get_nowait() == {"commandType": "pause"}


def test_runtime_service_hub_routes_live_playback_control_to_visualizer() -> None:
    hub = ArenaRuntimeServiceHub(adapter_id="arena")
    live_source = _StubLiveControlSource(related_event_seq=5)
    visualizer = hub.ensure_visualizer(lambda: _StubLiveVisualizer(live_source))

    control_receipt = hub.submit_control_command(
        "session-1",
        "run-live",
        {"commandType": "replay"},
    )

    assert control_receipt.state == "accepted"
    assert control_receipt.reason == "playback_applied"
    assert control_receipt.related_event_seq == 5
    assert visualizer.resolve_calls == [("session-1", "run-live")]
    assert live_source.commands == [ControlCommand(command_type="replay")]


def test_runtime_service_hub_rejects_control_command_when_control_sink_missing() -> None:
    hub = ArenaRuntimeServiceHub(adapter_id="arena")
    hub.ensure_action_server(_StubActionServer)

    control_receipt = hub.submit_control_command(
        "session-1",
        None,
        {"commandType": "pause"},
    )

    assert control_receipt.state == "rejected"
    assert control_receipt.reason == "control_queue_not_available"
