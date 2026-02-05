import pytest

from gage_eval.role.adapters.arena import ArenaRoleAdapter


class DummyFrameEnv:
    def get_last_frame(self):
        return None


def test_arena_role_adapter_retro_ws_server_uses_action_queue_for_human(monkeypatch: pytest.MonkeyPatch):
    from gage_eval.role.arena.games.retro.websocket_attached_server import RetroAttachedWebSocketServer

    monkeypatch.setattr(RetroAttachedWebSocketServer, "start", lambda *_: None)

    adapter = ArenaRoleAdapter(
        adapter_id="arena_test",
        environment={"impl": "retro_env_v1", "legal_moves": ["noop", "right"]},
        human_input={"enabled": True, "host": "127.0.0.1", "port": 5800, "fps": 30, "hold_ticks_default": 9},
    )

    server, action_queue = adapter._ensure_retro_websocket_server(  # type: ignore[attr-defined]
        [{"type": "human"}],
        environment=DummyFrameEnv(),
    )

    assert server is not None
    assert action_queue is not None
    assert getattr(server, "_config").hold_ticks_default == 9
    assert getattr(server, "_config").legal_moves == ["noop", "right"]


def test_arena_role_adapter_retro_ws_server_stream_only_without_human(monkeypatch: pytest.MonkeyPatch):
    from gage_eval.role.arena.games.retro.websocket_attached_server import RetroAttachedWebSocketServer

    monkeypatch.setattr(RetroAttachedWebSocketServer, "start", lambda *_: None)

    adapter = ArenaRoleAdapter(
        adapter_id="arena_test",
        environment={"impl": "retro_env_v1", "legal_moves": ["noop", "right"]},
        human_input={"enabled": True, "host": "127.0.0.1", "port": 5800, "fps": 30},
    )

    server, action_queue = adapter._ensure_retro_websocket_server(  # type: ignore[attr-defined]
        [{"type": "backend"}],
        environment=DummyFrameEnv(),
    )

    assert server is not None
    assert action_queue is None
