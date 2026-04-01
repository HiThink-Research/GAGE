from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from unittest.mock import patch

import yaml

from gage_eval.config import build_default_registry
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VisualBrowserProbeResult:
    viewer_url: str
    session_id: str
    run_id: str | None
    host_html: str
    session_payload: dict[str, Any]
    timeline_payload: dict[str, Any]
    first_scene: dict[str, Any]
    last_scene: dict[str, Any]
    first_media_payload: dict[str, Any] | None = None
    last_media_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeExecutionResult:
    config_path: Path
    output_dir: Path
    run_id: str
    sample: dict[str, Any]
    output: dict[str, Any]
    browser_urls: tuple[str, ...] = ()
    browser_probe: VisualBrowserProbeResult | None = None

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id


def materialize_gamekit_config(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str | None = None,
    gpus: int = 0,
    cpus: int = 1,
) -> None:
    path = Path(config_path).expanduser().resolve()
    payload = _load_pipeline_payload(path)
    config = PipelineConfig.from_dict(payload)
    resolved_output_dir = _resolve_output_dir(output_dir)
    resolved_run_id = _resolve_run_id(path, run_id=run_id)

    with _pushd(REPO_ROOT), patch.dict(
        os.environ,
        {"GAGE_EVAL_SAVE_DIR": str(resolved_output_dir)},
        clear=False,
    ):
        build_runtime(
            config,
            registry=build_default_registry(),
            resource_profile=ResourceProfile(
                [NodeResource(node_id="local", gpus=gpus, cpus=cpus)]
            ),
            trace=ObservabilityTrace(run_id=resolved_run_id),
        )


def run_gamekit_config(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str | None = None,
    gpus: int = 0,
    cpus: int = 1,
    max_samples: int | None = 1,
    sample_record: dict[str, Any] | None = None,
    launch_browser: bool | None = None,
    linger_after_finish_s: float | None = None,
    verify_visual: bool = False,
    expect_plugin: str | None = None,
    expect_live_scene_scheme: str | None = None,
    probe_timeout_s: float = 20.0,
) -> RuntimeExecutionResult:
    path = Path(config_path).expanduser().resolve()
    if max_samples not in (None, 1):
        raise ValueError("gamekit acceptance helpers only support max_samples=1")

    resolved_output_dir = _resolve_output_dir(output_dir)
    resolved_run_id = _resolve_run_id(path, run_id=run_id)
    payload = _load_pipeline_payload(path)

    if verify_visual and expect_plugin is None:
        raise ValueError("expect_plugin is required when verify_visual=True")
    if verify_visual and launch_browser is None:
        launch_browser = True
    _override_visualizer_config(
        payload,
        launch_browser=launch_browser,
        linger_after_finish_s=(
            1.0 if verify_visual and linger_after_finish_s is None else linger_after_finish_s
        ),
    )
    _assign_visualizer_port(payload)

    sample_id = str((sample_record or {}).get("id") or f"{path.stem}_sample_001")
    merged_sample = {
        "schema_version": "gage.v1",
        "id": sample_id,
        "messages": [{"role": "system", "content": "GameArena runtime fixture sample."}],
        "choices": [],
        **dict(sample_record or {}),
    }

    browser_urls: list[str] = []
    browser_event = threading.Event()
    stop_event = threading.Event()
    probe_results: list[VisualBrowserProbeResult] = []
    probe_errors: list[str] = []
    worker: threading.Thread | None = None

    def _capture_browser_open(url: str) -> bool:
        browser_urls.append(str(url))
        browser_event.set()
        return True

    if verify_visual:
        worker = threading.Thread(
            target=_browser_probe_worker,
            kwargs={
                "browser_urls": browser_urls,
                "browser_event": browser_event,
                "stop_event": stop_event,
                "expect_plugin": str(expect_plugin),
                "expect_live_scene_scheme": expect_live_scene_scheme,
                "timeout_s": float(probe_timeout_s),
                "probe_results": probe_results,
                "probe_errors": probe_errors,
            },
            daemon=True,
        )
        worker.start()

    with tempfile.TemporaryDirectory(prefix=f"{path.stem}_") as temp_dir:
        sample_path = Path(temp_dir) / f"{path.stem}.jsonl"
        sample_path.write_text(
            json.dumps(merged_sample, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        payload["datasets"][0]["params"]["path"] = str(sample_path)
        config = PipelineConfig.from_dict(payload)
        processed_samples: list[dict[str, Any]] = []

        try:
            with _pushd(REPO_ROOT), patch.dict(
                os.environ,
                {"GAGE_EVAL_SAVE_DIR": str(resolved_output_dir)},
                clear=False,
            ):
                patcher = (
                    patch(
                        "gage_eval.role.arena.core.game_session.webbrowser.open",
                        _capture_browser_open,
                    )
                    if verify_visual
                    else _null_patcher()
                )
                with patcher:
                    runtime = build_runtime(
                        config,
                        registry=build_default_registry(),
                        resource_profile=ResourceProfile(
                            [NodeResource(node_id="local", gpus=gpus, cpus=cpus)]
                        ),
                        trace=ObservabilityTrace(run_id=resolved_run_id),
                    )
                    runtime.sample_loop.register_hook(
                        lambda sample: processed_samples.append(dict(sample))
                    )
                    runtime.run()
        finally:
            stop_event.set()
            if worker is not None:
                worker.join(timeout=2.0)

    if not processed_samples:
        raise AssertionError(f"Runtime for '{path}' did not process any samples")
    processed_sample = processed_samples[0]
    predict_result = processed_sample.get("predict_result")
    if not isinstance(predict_result, list) or not predict_result:
        raise AssertionError(f"Runtime for '{path}' did not produce predict_result")
    output = predict_result[0]
    if not isinstance(output, dict):
        raise AssertionError(f"Runtime for '{path}' produced a non-mapping predict_result")

    if verify_visual and not probe_results:
        if probe_errors:
            raise AssertionError(probe_errors[-1])
        raise AssertionError(f"Visual browser probe did not complete for '{path}'")

    return RuntimeExecutionResult(
        config_path=path,
        output_dir=resolved_output_dir,
        run_id=resolved_run_id,
        sample=processed_sample,
        output=output,
        browser_urls=tuple(browser_urls),
        browser_probe=probe_results[0] if probe_results else None,
    )


def verify_visual_gamekit_config(
    config_path: str | Path,
    *,
    expect_plugin: str,
    output_dir: str | Path,
    run_id: str | None = None,
    gpus: int = 0,
    cpus: int = 1,
    max_samples: int | None = 1,
    expect_live_scene_scheme: str | None = None,
    linger_after_finish_s: float | None = None,
    probe_timeout_s: float = 20.0,
) -> RuntimeExecutionResult:
    return run_gamekit_config(
        config_path,
        output_dir=output_dir,
        run_id=run_id,
        gpus=gpus,
        cpus=cpus,
        max_samples=max_samples,
        launch_browser=True,
        linger_after_finish_s=linger_after_finish_s,
        verify_visual=True,
        expect_plugin=expect_plugin,
        expect_live_scene_scheme=expect_live_scene_scheme,
        probe_timeout_s=probe_timeout_s,
    )


def _browser_probe_worker(
    *,
    browser_urls: list[str],
    browser_event: threading.Event,
    stop_event: threading.Event,
    expect_plugin: str,
    expect_live_scene_scheme: str | None,
    timeout_s: float,
    probe_results: list[VisualBrowserProbeResult],
    probe_errors: list[str],
) -> None:
    last_error = "runtime did not trigger browser open"
    while not stop_event.is_set():
        if browser_event.wait(timeout=0.05):
            break
    if not browser_urls:
        probe_errors.append(last_error)
        return

    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline and not stop_event.is_set():
        try:
            probe_results.append(
                _probe_visual_session(
                    viewer_url=browser_urls[0],
                    expect_plugin=expect_plugin,
                    expect_live_scene_scheme=expect_live_scene_scheme,
                )
            )
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.1)

    probe_errors.append(last_error)


def _probe_visual_session(
    *,
    viewer_url: str,
    expect_plugin: str,
    expect_live_scene_scheme: str | None,
) -> VisualBrowserProbeResult:
    parsed = urlsplit(viewer_url)
    host_html = _read_text(viewer_url)
    lowered_html = host_html.lower()
    if "<html" not in lowered_html and "<!doctype html" not in lowered_html:
        raise AssertionError(f"Viewer shell did not return HTML for {viewer_url}")

    session_id = str(parsed.path.rstrip("/").rsplit("/", 1)[-1]).strip()
    if not session_id:
        raise AssertionError(f"Unable to resolve session_id from viewer_url={viewer_url}")
    run_id = parse_qs(parsed.query).get("run_id", [None])[0]
    origin = f"{parsed.scheme}://{parsed.netloc}"
    session_root = f"{origin}/arena_visual/sessions/{quote(session_id, safe='')}"

    session_payload = _read_json(_url_with_query(session_root, run_id=run_id))
    if session_payload.get("pluginId") != expect_plugin:
        raise AssertionError(
            f"Unexpected pluginId for {viewer_url}: {session_payload.get('pluginId')} != {expect_plugin}"
        )

    timeline_payload = _read_json(
        _url_with_query(f"{session_root}/timeline", run_id=run_id)
    )
    events = timeline_payload.get("events")
    if not isinstance(events, list):
        raise AssertionError(f"Timeline payload missing events for {viewer_url}")
    snapshot_seqs = [
        int(event["seq"])
        for event in events
        if isinstance(event, dict) and event.get("type") == "snapshot" and "seq" in event
    ]
    if len(snapshot_seqs) < 2:
        raise AssertionError(
            f"Expected at least two snapshot seqs for {viewer_url}, got {snapshot_seqs}"
        )

    first_scene = _read_json(
        _url_with_query(f"{session_root}/scene", seq=snapshot_seqs[0], run_id=run_id)
    )
    last_scene = _read_json(
        _url_with_query(f"{session_root}/scene", seq=snapshot_seqs[-1], run_id=run_id)
    )
    kind = str(first_scene.get("kind") or last_scene.get("kind") or "").strip()
    _assert_scene_core_content(first_scene)
    _assert_scene_core_content(last_scene)
    _assert_scene_progression(first_scene, last_scene, kind=kind)

    first_media_payload = _load_primary_media_payload(
        scene=first_scene,
        origin=origin,
        session_root=session_root,
        run_id=run_id,
        expected_scheme=expect_live_scene_scheme,
    )
    last_media_payload = _load_primary_media_payload(
        scene=last_scene,
        origin=origin,
        session_root=session_root,
        run_id=run_id,
        expected_scheme=expect_live_scene_scheme,
    )

    return VisualBrowserProbeResult(
        viewer_url=viewer_url,
        session_id=session_id,
        run_id=run_id,
        host_html=host_html,
        session_payload=session_payload,
        timeline_payload=timeline_payload,
        first_scene=first_scene,
        last_scene=last_scene,
        first_media_payload=first_media_payload,
        last_media_payload=last_media_payload,
    )


def _load_primary_media_payload(
    *,
    scene: dict[str, Any],
    origin: str,
    session_root: str,
    run_id: str | None,
    expected_scheme: str | None,
) -> dict[str, Any] | None:
    media = _mapping_or_empty(scene.get("media"))
    primary = _mapping_or_empty(media.get("primary"))
    if not primary:
        return None

    media_id = str(primary.get("mediaId") or "").strip()
    if not media_id:
        raise AssertionError("Primary media payload is missing mediaId")
    media_payload = _read_json(
        _url_with_query(
            f"{session_root}/media/{quote(media_id, safe='')}",
            run_id=run_id,
        )
    )
    transport = str(primary.get("transport") or media_payload.get("transport") or "").strip()
    if expected_scheme is not None and transport != expected_scheme:
        raise AssertionError(
            f"Unexpected live scene transport: {transport} != {expected_scheme}"
        )

    if transport == "low_latency_channel":
        stream_url = str(primary.get("url") or media_payload.get("url") or "").strip()
        if not stream_url:
            raise AssertionError("low_latency_channel media payload is missing stream url")
        if stream_url.startswith("/"):
            stream_url = f"{origin}{stream_url}"
        content_type, prefix = _read_stream_prefix(stream_url)
        if not content_type.startswith("multipart/x-mixed-replace"):
            raise AssertionError(
                f"Unexpected stream content-type for {stream_url}: {content_type}"
            )
        if not prefix:
            raise AssertionError(f"low_latency_channel stream returned no content: {stream_url}")
        return media_payload

    content = _read_bytes(
        _url_with_query(
            f"{session_root}/media/{quote(media_id, safe='')}",
            run_id=run_id,
            content=1,
        )
    )
    if not content:
        raise AssertionError(f"Media payload returned empty content: {media_id}")
    return media_payload


def _assert_scene_core_content(scene: dict[str, Any]) -> None:
    kind = str(scene.get("kind") or "").strip()
    body = _mapping_or_empty(scene.get("body"))
    if not body:
        raise AssertionError(f"Scene body is empty for kind={kind}")

    if kind == "board":
        board = _mapping_or_empty(body.get("board"))
        players = body.get("players")
        if not board.get("cells"):
            raise AssertionError("Board scene is missing projected cells")
        if not isinstance(players, list) or not players:
            raise AssertionError("Board scene is missing players")
        return

    if kind == "table":
        table = _mapping_or_empty(body.get("table"))
        seats = table.get("seats")
        status = _mapping_or_empty(body.get("status"))
        if not isinstance(seats, list) or not seats:
            raise AssertionError("Table scene is missing seats")
        if not status:
            raise AssertionError("Table scene is missing status")
        return

    if kind == "frame":
        frame = _mapping_or_empty(body.get("frame"))
        primary = _mapping_or_empty(_mapping_or_empty(scene.get("media")).get("primary"))
        if not frame.get("title"):
            raise AssertionError("Frame scene is missing frame metadata")
        if not primary:
            raise AssertionError("Frame scene is missing primary media")
        return

    if kind == "rts":
        frame = _mapping_or_empty(body.get("frame"))
        rts = _mapping_or_empty(body.get("rts"))
        primary = _mapping_or_empty(_mapping_or_empty(scene.get("media")).get("primary"))
        units = rts.get("units")
        if not frame.get("title"):
            raise AssertionError("RTS scene is missing frame metadata")
        if not primary:
            raise AssertionError("RTS scene is missing primary media")
        if not isinstance(units, list) or not units:
            raise AssertionError("RTS scene is missing unit payloads")
        return

    raise AssertionError(f"Unsupported visual kind in browser probe: {kind}")


def _assert_scene_progression(
    first_scene: dict[str, Any],
    last_scene: dict[str, Any],
    *,
    kind: str,
) -> None:
    first_seq = int(first_scene.get("seq") or 0)
    last_seq = int(last_scene.get("seq") or 0)
    if first_seq == last_seq:
        raise AssertionError(f"Expected changing scene seqs, got {first_seq}")

    first_body = json.dumps(first_scene.get("body") or {}, sort_keys=True, ensure_ascii=False)
    last_body = json.dumps(last_scene.get("body") or {}, sort_keys=True, ensure_ascii=False)
    if first_body != last_body:
        return

    if kind in {"frame", "rts"}:
        first_media_id = _mapping_or_empty(
            _mapping_or_empty(first_scene.get("media")).get("primary")
        ).get("mediaId")
        last_media_id = _mapping_or_empty(
            _mapping_or_empty(last_scene.get("media")).get("primary")
        ).get("mediaId")
        if first_media_id != last_media_id:
            return

    raise AssertionError(
        f"Scene content did not change between seq={first_seq} and seq={last_seq}"
    )


def _load_pipeline_payload(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config '{config_path}' must load as a mapping")
    return payload


def _override_visualizer_config(
    payload: dict[str, Any],
    *,
    launch_browser: bool | None,
    linger_after_finish_s: float | None,
) -> None:
    if launch_browser is None and linger_after_finish_s is None:
        return
    adapters = payload.get("role_adapters")
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("PipelineConfig must contain at least one role_adapters entry")
    adapter = adapters[0]
    if not isinstance(adapter, dict):
        raise TypeError("Primary role adapter must be a mapping")
    params = adapter.setdefault("params", {})
    if not isinstance(params, dict):
        raise TypeError("Primary role adapter params must be a mapping")
    visualizer = dict(params.get("visualizer") or {})
    if launch_browser is not None:
        visualizer["launch_browser"] = bool(launch_browser)
    if linger_after_finish_s is not None:
        visualizer["linger_after_finish_s"] = float(linger_after_finish_s)
    params["visualizer"] = visualizer


def _assign_visualizer_port(payload: dict[str, Any]) -> None:
    adapters = payload.get("role_adapters")
    if not isinstance(adapters, list) or not adapters:
        return
    adapter = adapters[0]
    if not isinstance(adapter, dict):
        return
    params = adapter.get("params")
    if not isinstance(params, dict):
        return
    visualizer = params.get("visualizer")
    if not isinstance(visualizer, dict):
        return
    if visualizer.get("enabled") is not True:
        return
    visualizer["port"] = _allocate_free_local_port()


def _allocate_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _resolve_output_dir(output_dir: str | Path) -> Path:
    resolved = Path(output_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_run_id(config_path: Path, *, run_id: str | None) -> str:
    text = str(run_id or config_path.stem).strip()
    if not text:
        raise ValueError("run_id must not be empty")
    return text


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _url_with_query(base_url: str, **params: object) -> str:
    normalized = {
        key: value
        for key, value in params.items()
        if value is not None and value != ""
    }
    if not normalized:
        return base_url
    return f"{base_url}?{urlencode(normalized)}"


def _read_json(url: str) -> dict[str, Any]:
    payload = _read_text(url)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object from {url}")
    return data


def _read_text(url: str) -> str:
    request = Request(url, headers={"Accept": "application/json,text/html,*/*"})
    with urlopen(request, timeout=2.0) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="ignore")


def _read_bytes(url: str) -> bytes:
    with urlopen(url, timeout=2.0) as response:  # noqa: S310
        return response.read()


def _read_stream_prefix(url: str) -> tuple[str, bytes]:
    with urlopen(url, timeout=2.0) as response:  # noqa: S310
        content_type = str(response.headers.get("Content-Type") or "")
        prefix_parts: list[bytes] = []
        for _ in range(4):
            try:
                line = response.readline()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                break
            if not line:
                break
            prefix_parts.append(line)
            if line in {b"\r\n", b"\n"}:
                break
        return content_type, b"".join(prefix_parts)


@contextmanager
def _pushd(target: Path):
    previous = Path.cwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _null_patcher():
    yield
