import asyncio
import json
import sys
import types

import pytest

from gage_eval.role.arena.games.retro import websocket_server as ws_server
from gage_eval.role.arena.types import ArenaAction


class DummyHTMLResponse:
    def __init__(self, body: str):
        self.body = body
        self.status_code = 200


class DummyPlainTextResponse:
    def __init__(self, body: str, status_code: int = 200):
        self.body = body
        self.status_code = status_code


class DummyFileResponse:
    def __init__(self, path):
        self.path = path
        self.status_code = 200


class DummyFastAPI:
    def __init__(self):
        self.events: dict[str, list[object]] = {}
        self.get_routes: dict[str, object] = {}
        self.ws_routes: dict[str, object] = {}

    def on_event(self, name: str):
        def decorator(fn):
            self.events.setdefault(name, []).append(fn)
            return fn

        return decorator

    def get(self, path: str):
        def decorator(fn):
            self.get_routes[path] = fn
            return fn

        return decorator

    def websocket(self, path: str):
        def decorator(fn):
            self.ws_routes[path] = fn
            return fn

        return decorator


class DummyLoop:
    def call_soon_threadsafe(self, callback):
        callback()


def _install_fake_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = DummyFastAPI
    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = DummyFileResponse
    responses.HTMLResponse = DummyHTMLResponse
    responses.PlainTextResponse = DummyPlainTextResponse
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses)


def test_load_web_deps_raises_import_error_when_fastapi_missing(monkeypatch: pytest.MonkeyPatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("fastapi"):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="dependencies missing"):
        ws_server._load_web_deps()  # noqa: SLF001


def test_create_app_registers_routes_and_startup_shutdown(monkeypatch: pytest.MonkeyPatch):
    _install_fake_fastapi(monkeypatch)

    calls: dict[str, object] = {}

    def fake_start(self, loop):  # noqa: ANN001
        calls["started"] = True
        calls["loop"] = loop

    def fake_stop(self):  # noqa: ANN001
        calls["stopped"] = True

    monkeypatch.setattr(ws_server.RetroGameLoop, "start", fake_start)
    monkeypatch.setattr(ws_server.RetroGameLoop, "stop", fake_stop)

    app = ws_server.create_app(ws_server.ServerConfig(frame_queue_size=1))
    assert isinstance(app, DummyFastAPI)

    asyncio.run(app.events["startup"][0]())
    asyncio.run(app.events["shutdown"][0]())
    assert calls.get("started") is True
    assert calls.get("stopped") is True

    index_resp = asyncio.run(app.get_routes["/"]())
    assert index_resp.status_code == 200
    assert "<html" in index_resp.body.lower()

    js_resp = asyncio.run(app.get_routes["/client.js"]())
    assert str(js_resp.path).endswith("client.js")


def test_create_app_returns_404_when_client_bundle_missing(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _install_fake_fastapi(monkeypatch)
    monkeypatch.setattr(ws_server, "_INDEX_PATH", tmp_path / "missing.html")
    monkeypatch.setattr(ws_server, "_CLIENT_JS_PATH", tmp_path / "missing.js")

    app = ws_server.create_app(ws_server.ServerConfig(frame_queue_size=1))
    index_resp = asyncio.run(app.get_routes["/"]())
    assert index_resp.status_code == 404

    js_resp = asyncio.run(app.get_routes["/client.js"]())
    assert js_resp.status_code == 404


def test_ws_endpoint_sends_error_payload_when_encode_fails(monkeypatch: pytest.MonkeyPatch):
    _install_fake_fastapi(monkeypatch)
    created: dict[str, object] = {}
    original_queue = asyncio.Queue

    def queue_factory(*, maxsize: int):
        queue = original_queue(maxsize=maxsize)
        created["queue"] = queue
        return queue

    monkeypatch.setattr(ws_server.asyncio, "Queue", queue_factory)
    monkeypatch.setattr(ws_server, "encode_jpeg", lambda frame, quality=80: (None, "encode_failed"))
    monkeypatch.setattr(ws_server, "parse_key_payload", lambda message: None)

    app = ws_server.create_app(ws_server.ServerConfig(frame_queue_size=1))
    frame_queue = created["queue"]
    frame_queue.put_nowait(object())

    class StubWebSocket:
        def __init__(self):
            self.accepted = False
            self.sent_text: list[str] = []

        async def accept(self) -> None:
            self.accepted = True

        async def receive_text(self) -> str:
            await asyncio.Event().wait()
            return ""

        async def send_text(self, text: str) -> None:
            self.sent_text.append(text)

        async def send_bytes(self, data: bytes) -> None:
            raise AssertionError("bytes not expected")

    ws = StubWebSocket()
    asyncio.run(app.ws_routes["/ws"](ws))
    assert ws.accepted is True
    assert json.loads(ws.sent_text[0])["error"] == "encode_failed"


def test_ws_endpoint_sends_bytes_and_handles_disconnect(monkeypatch: pytest.MonkeyPatch):
    _install_fake_fastapi(monkeypatch)
    created: dict[str, object] = {}
    original_queue = asyncio.Queue

    def queue_factory(*, maxsize: int):
        queue = original_queue(maxsize=maxsize)
        created["queue"] = queue
        return queue

    monkeypatch.setattr(ws_server.asyncio, "Queue", queue_factory)
    monkeypatch.setattr(ws_server, "encode_jpeg", lambda frame, quality=80: (b"jpeg", None))
    monkeypatch.setattr(ws_server, "parse_key_payload", lambda message: None)

    class Disconnect(Exception):
        pass

    monkeypatch.setattr(ws_server, "FastAPIWebSocketDisconnect", Disconnect)

    app = ws_server.create_app(ws_server.ServerConfig(frame_queue_size=1))
    frame_queue = created["queue"]
    frame_queue.put_nowait(object())

    class StubWebSocket:
        def __init__(self):
            self.sent_bytes: list[bytes] = []

        async def accept(self) -> None:
            return None

        async def receive_text(self) -> str:
            await asyncio.Event().wait()
            return ""

        async def send_text(self, text: str) -> None:
            raise AssertionError("text not expected")

        async def send_bytes(self, data: bytes) -> None:
            self.sent_bytes.append(data)
            raise Disconnect()

    ws = StubWebSocket()
    asyncio.run(app.ws_routes["/ws"](ws))
    assert ws.sent_bytes == [b"jpeg"]


def test_retro_game_loop_run_advances_env_and_publishes_frames(monkeypatch: pytest.MonkeyPatch):
    resets: list[object] = []
    applied: list[ArenaAction] = []

    class StubEnv:
        def __init__(self, **kwargs):  # noqa: ANN003
            self._terminal = False
            self._frame = None

        def reset(self) -> None:
            resets.append(object())
            self._terminal = False
            self._frame = f"reset_{len(resets)}"

        def get_last_frame(self):
            return self._frame

        def apply(self, action: ArenaAction) -> None:
            applied.append(action)
            self._terminal = True
            self._frame = "frame_after_apply"

        def is_terminal(self) -> bool:
            return self._terminal

    monkeypatch.setattr(ws_server, "StableRetroArenaEnvironment", StubEnv)
    monkeypatch.setattr(ws_server.time, "sleep", lambda _: None)
    counter = {"t": 0.0}

    def fake_monotonic() -> float:
        counter["t"] += 0.01
        return counter["t"]

    monkeypatch.setattr(ws_server.time, "monotonic", fake_monotonic)

    frame_queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=1)
    key_state = ws_server.KeyState(ws_server.build_default_key_map())
    game_loop = ws_server.RetroGameLoop(
        config=ws_server.ServerConfig(frame_queue_size=1, fps=60),
        frame_queue=frame_queue,
        key_state=key_state,
    )
    game_loop._loop = DummyLoop()  # type: ignore[attr-defined]
    game_loop._reset_event.set()  # type: ignore[attr-defined]

    def resolve_move() -> str:
        game_loop._stop_event.set()  # type: ignore[attr-defined]
        return "noop"

    game_loop._key_state.resolve_move = resolve_move  # type: ignore[method-assign]
    game_loop._run()  # noqa: SLF001

    assert len(resets) >= 2
    assert len(applied) == 1
    assert frame_queue.get_nowait() == "frame_after_apply"


def test_websocket_server_main_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class DummyParser:
        def parse_args(self):
            return types.SimpleNamespace(
                game="g",
                state="s",
                host="127.0.0.1",
                port=5801,
                fps=30,
                frame_queue_size=2,
            )

    monkeypatch.setattr(ws_server, "build_arg_parser", lambda: DummyParser())
    monkeypatch.setattr(ws_server, "create_app", lambda config: {"app": "ok"})

    def fake_run(app, host, port, log_level):  # noqa: ANN001
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port
        calls["log_level"] = log_level

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=fake_run))
    ws_server.main()

    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 5801
