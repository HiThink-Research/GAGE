import asyncio
import json
import sys
import types
from queue import Queue

import pytest

from gage_eval.role.arena.games.retro import websocket_attached_server as ws_attached


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


def test_attached_load_web_deps_raises_when_uvicorn_missing(monkeypatch: pytest.MonkeyPatch):
    _install_fake_fastapi(monkeypatch)
    monkeypatch.setattr(ws_attached, "uvicorn", None)
    with pytest.raises(ImportError, match="requires uvicorn"):
        ws_attached._load_web_deps()  # noqa: SLF001


def test_attached_server_run_registers_routes_and_index(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _install_fake_fastapi(monkeypatch)

    class StubConfig:
        def __init__(self, app, **kwargs):  # noqa: ANN003
            self.app = app
            self.kwargs = kwargs

    class StubServer:
        def __init__(self, config):  # noqa: ANN001
            self.config = config
            self.should_exit = False
            self.ran = False

        def run(self) -> None:
            self.ran = True

    stub_uvicorn = types.SimpleNamespace(Config=StubConfig, Server=StubServer)
    monkeypatch.setattr(ws_attached, "uvicorn", stub_uvicorn)

    async def fake_frame_publisher(self, loop, frame_queue):  # noqa: ANN001
        return None

    monkeypatch.setattr(ws_attached.RetroAttachedWebSocketServer, "_frame_publisher", fake_frame_publisher)

    server = ws_attached.RetroAttachedWebSocketServer(
        config=ws_attached.AttachedServerConfig(port=5802),
        frame_source=lambda: None,
        action_queue=None,
    )
    server._run()  # noqa: SLF001

    assert server._server is not None  # noqa: SLF001
    assert server._server.ran is True  # noqa: SLF001
    app = server._server.config.app  # noqa: SLF001
    assert isinstance(app, DummyFastAPI)

    index_resp = asyncio.run(app.get_routes["/"]())
    assert index_resp.status_code in {200, 404}


def test_attached_server_stop_sets_should_exit_and_joins_thread():
    server = ws_attached.RetroAttachedWebSocketServer(
        config=ws_attached.AttachedServerConfig(),
        frame_source=lambda: None,
        action_queue=None,
    )

    class StubThread:
        def __init__(self):
            self.joined = False

        def join(self, timeout: float) -> None:
            self.joined = True

    class StubServer:
        def __init__(self):
            self.should_exit = False

    server._thread = StubThread()  # type: ignore[attr-defined]
    server._server = StubServer()  # type: ignore[attr-defined]
    server.stop()

    assert server._server.should_exit is True  # type: ignore[attr-defined]
    assert server._thread.joined is True  # type: ignore[attr-defined]


def test_attached_frame_publisher_drops_oldest_when_queue_full():
    action_queue: Queue[str] = Queue()
    server = ws_attached.RetroAttachedWebSocketServer(
        config=ws_attached.AttachedServerConfig(frame_queue_size=1, fps=60),
        frame_source=lambda: "frame",
        action_queue=action_queue,
    )
    frame_queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=1)
    frame_queue.put_nowait("old")

    async def run_once() -> None:
        task = asyncio.create_task(server._frame_publisher(DummyLoop(), frame_queue))  # noqa: SLF001
        await asyncio.sleep(0)
        assert await asyncio.wait_for(frame_queue.get(), timeout=1.0) == "frame"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_once())


def test_attached_ws_endpoint_enqueues_moves_and_sends_error(monkeypatch: pytest.MonkeyPatch):
    _install_fake_fastapi(monkeypatch)

    created: dict[str, object] = {}
    original_queue = asyncio.Queue

    def queue_factory(*, maxsize: int):
        queue = original_queue(maxsize=maxsize)
        created["queue"] = queue
        return queue

    monkeypatch.setattr(ws_attached.asyncio, "Queue", queue_factory)
    monkeypatch.setattr(ws_attached, "encode_jpeg", lambda frame, quality=80: (None, "encode_failed"))

    class StubConfig:
        def __init__(self, app, **kwargs):  # noqa: ANN003
            self.app = app

    class StubServer:
        def __init__(self, config):  # noqa: ANN001
            self.config = config
            self.should_exit = False

        def run(self) -> None:
            return None

    stub_uvicorn = types.SimpleNamespace(Config=StubConfig, Server=StubServer)
    monkeypatch.setattr(ws_attached, "uvicorn", stub_uvicorn)

    processed = asyncio.Event()
    original_enqueue = ws_attached._enqueue_move

    def enqueue_wrapper(action_queue: Queue[str], *, move: str, hold_ticks: int) -> None:
        original_enqueue(action_queue, move=move, hold_ticks=hold_ticks)
        processed.set()

    monkeypatch.setattr(ws_attached, "_enqueue_move", enqueue_wrapper)

    action_queue: Queue[str] = Queue()
    server = ws_attached.RetroAttachedWebSocketServer(
        config=ws_attached.AttachedServerConfig(hold_ticks_default=5),
        frame_source=lambda: None,
        action_queue=action_queue,
    )
    server._run()  # noqa: SLF001

    app = server._server.config.app  # type: ignore[attr-defined]
    frame_queue = created["queue"]

    class StubWebSocket:
        def __init__(self):
            self.incoming: "asyncio.Queue[str]" = original_queue()
            self.sent_text: list[str] = []

        async def accept(self) -> None:
            return None

        async def receive_text(self) -> str:
            return await self.incoming.get()

        async def send_text(self, text: str) -> None:
            self.sent_text.append(text)

        async def send_bytes(self, data: bytes) -> None:
            raise AssertionError("bytes not expected")

    async def run_ws() -> None:
        ws = StubWebSocket()
        ws.incoming.put_nowait('{"type":"keydown","key":"d"}')
        task = asyncio.create_task(app.ws_routes["/ws"](ws))
        await asyncio.wait_for(processed.wait(), timeout=1.0)
        frame_queue.put_nowait(object())
        await asyncio.wait_for(task, timeout=1.0)
        payload = json.loads(action_queue.get_nowait())
        assert payload == {"move": "right", "hold_ticks": 5}
        assert json.loads(ws.sent_text[0])["error"] == "encode_failed"

    asyncio.run(run_ws())
