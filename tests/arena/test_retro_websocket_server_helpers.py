import asyncio
import json
from queue import Queue

from gage_eval.role.arena.games.retro.keyboard_input import KeyState, build_default_key_map
from gage_eval.role.arena.games.retro.websocket_attached_server import _apply_ws_input as apply_attached_ws_input
from gage_eval.role.arena.games.retro.websocket_attached_server import _enqueue_move
from gage_eval.role.arena.games.retro.websocket_server import RetroGameLoop, ServerConfig
from gage_eval.role.arena.games.retro.websocket_server import _apply_ws_input as apply_ws_input


class DummyLoop:
    def call_soon_threadsafe(self, callback):
        callback()


class DummyGameLoop:
    def __init__(self):
        self.reset_requested = False

    def request_reset(self) -> None:
        self.reset_requested = True


def test_websocket_server_apply_ws_input_handles_reset_and_key_events():
    key_state = KeyState(build_default_key_map())
    game_loop = DummyGameLoop()

    apply_ws_input({"reset": True}, key_state, game_loop)
    assert game_loop.reset_requested is True

    apply_ws_input({"type": "keydown", "key": "d"}, key_state, game_loop)
    assert key_state.snapshot()["right"] is True

    apply_ws_input({"type": "keyup", "key": "d"}, key_state, game_loop)
    assert key_state.snapshot()["right"] is False

    apply_ws_input({"keys": {"d": True, "k": True}}, key_state, game_loop)
    snapshot = key_state.snapshot()
    assert snapshot["right"] is True
    assert snapshot["run"] is True


def test_websocket_server_retro_game_loop_publish_frame_drops_oldest_when_full():
    frame_queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=1)
    frame_queue.put_nowait("old")

    game_loop = RetroGameLoop(
        config=ServerConfig(frame_queue_size=1),
        frame_queue=frame_queue,
        key_state=KeyState(build_default_key_map()),
    )
    game_loop._loop = DummyLoop()  # type: ignore[attr-defined]

    game_loop._publish_frame("new")  # noqa: SLF001
    assert frame_queue.get_nowait() == "new"


def test_attached_websocket_server_enqueue_move_emits_json_payload():
    queue: Queue[str] = Queue()
    _enqueue_move(queue, move="right", hold_ticks=7)

    payload = json.loads(queue.get_nowait())
    assert payload == {"move": "right", "hold_ticks": 7}


def test_attached_websocket_server_apply_ws_input_updates_key_state():
    key_state = KeyState(build_default_key_map())
    apply_attached_ws_input({"type": "keydown", "key": "d"}, key_state)
    assert key_state.snapshot()["right"] is True

    apply_attached_ws_input({"type": "keyup", "key": "d"}, key_state)
    assert key_state.snapshot()["right"] is False

    apply_attached_ws_input({"keys": {"ArrowRight": True}}, key_state)
    assert key_state.snapshot()["right"] is True
