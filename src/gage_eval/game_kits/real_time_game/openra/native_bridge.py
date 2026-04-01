"""Native OpenRA process bridge for real frame capture and input relay."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Sequence
from urllib.request import urlopen
import zipfile

OPENRA_REFERENCE_ROOT_ENV = "GAGE_OPENRA_REFERENCE_ROOT"
OPENRA_BRIDGE_DIR_ENV = "GAGE_OPENRA_BRIDGE_DIR"
OPENRA_BRIDGE_FPS_ENV = "GAGE_OPENRA_BRIDGE_FPS"
OPENRA_STARTUP_ORDERS_ENV = "GAGE_OPENRA_STARTUP_ORDERS"
DEFAULT_FRAME_RATE_HZ = 4.0
DEFAULT_STARTUP_TIMEOUT_S = 20.0
DEFAULT_WINDOW_SIZE = {"width": 1280, "height": 720}


def locate_openra_reference_root() -> Path | None:
    env_override = os.environ.get(OPENRA_REFERENCE_ROOT_ENV)
    candidate_paths: list[Path] = []
    if env_override:
        candidate_paths.append(Path(env_override).expanduser())
    for ancestor in Path(__file__).resolve().parents:
        candidate_paths.append(ancestor / "reference" / "gamearena" / "OpenRA-bleed")
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    return None


def locate_workspace_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _normalize_startup_orders(startup_orders: Sequence[str] | None) -> list[str]:
    if startup_orders is None:
        return []
    resolved: list[str] = []
    for item in startup_orders:
        normalized = str(item or "").strip()
        if normalized:
            resolved.append(normalized)
    return resolved


def _encode_startup_orders(startup_orders: Sequence[str] | None) -> str | None:
    normalized = _normalize_startup_orders(startup_orders)
    if not normalized:
        return None
    return json.dumps(normalized, ensure_ascii=False)


def _build_launch_arguments(
    *,
    launch_script: Path,
    mod_id: str,
    support_dir: Path,
    viewport: dict[str, int],
    map_ref: str | None,
    launch_mode: str | None,
) -> list[str]:
    args = [
        str(launch_script),
        f"Game.Mod={mod_id}",
        f"Engine.SupportDir={support_dir}",
        "Graphics.Mode=Windowed",
        f"Graphics.WindowedSize={viewport['width']},{viewport['height']}",
        "Graphics.VSync=False",
        "Graphics.UIScale=1",
        "--just-die",
    ]
    launch_map = _resolve_launch_map_name(map_ref)
    if launch_map:
        args.append(f"Launch.Map={launch_map}")
    normalized_launch_mode = str(launch_mode or "").strip().lower()
    if normalized_launch_mode:
        args.append(f"Launch.Mode={normalized_launch_mode}")
    return args


class OpenRANativeBridge:
    """Owns one native OpenRA runtime and exposes the latest rendered frame."""

    def __init__(
        self,
        *,
        env_id: str,
        mod_id: str,
        map_ref: str | None,
        viewport: dict[str, int] | None = None,
        reference_root: Path | None = None,
        support_dir: Path | None = None,
        bridge_dir: Path | None = None,
        launch_mode: str | None = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
        startup_orders: Sequence[str] | None = None,
    ) -> None:
        self._env_id = str(env_id or "openra")
        self._mod_id = str(mod_id or "").strip()
        self._map_ref = str(map_ref or "").strip() or None
        self._viewport = {
            "width": int((viewport or DEFAULT_WINDOW_SIZE).get("width", DEFAULT_WINDOW_SIZE["width"])),
            "height": int((viewport or DEFAULT_WINDOW_SIZE).get("height", DEFAULT_WINDOW_SIZE["height"])),
        }
        self._reference_root = (reference_root or locate_openra_reference_root())
        if self._reference_root is None:
            raise RuntimeError("openra_reference_root_not_found")
        workspace_root = locate_workspace_root()
        runtime_root = workspace_root / ".runtime" / "openra"
        bridge_dir_override = os.environ.get(OPENRA_BRIDGE_DIR_ENV)
        self._support_dir = (support_dir or (runtime_root / "support")).resolve()
        self._bridge_dir = (
            bridge_dir
            or (
                Path(bridge_dir_override).expanduser()
                if str(bridge_dir_override or "").strip()
                else runtime_root / "bridge" / f"{self._env_id}-{int(time.time() * 1000)}"
            )
        ).resolve()
        self._launch_mode = str(launch_mode or "").strip().lower() or None
        self._startup_timeout_s = max(1.0, float(startup_timeout_s))
        self._frame_rate_hz = max(1.0, float(frame_rate_hz))
        self._startup_orders = tuple(_normalize_startup_orders(startup_orders))
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_handle = None
        self._stderr_handle = None
        self._last_state: dict[str, Any] = {}
        self._last_frame_signature: tuple[str, int] | None = None
        self._last_frame_data_url: str | None = None

        self._support_dir.mkdir(parents=True, exist_ok=True)
        self._bridge_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._bridge_dir / "state.json"
        self._frame_path = self._bridge_dir / "frame.png"
        self._input_path = self._bridge_dir / "input.ndjson"
        self._stdout_path = self._bridge_dir / "stdout.log"
        self._stderr_path = self._bridge_dir / "stderr.log"

    def read_state(self) -> dict[str, Any]:
        self._ensure_started()
        loaded = self._load_state()
        if loaded is not None:
            self._last_state = loaded
        if self._last_state:
            return dict(self._last_state)
        raise RuntimeError("openra_native_frame_not_ready")

    def submit_input(self, payload: dict[str, Any]) -> None:
        self._ensure_started()
        normalized = dict(payload or {})
        normalized.setdefault("timestamp_ms", int(time.time() * 1000))
        with self._input_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            handle.flush()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if self._stdout_handle is not None:
            self._stdout_handle.close()
            self._stdout_handle = None
        if self._stderr_handle is not None:
            self._stderr_handle.close()
            self._stderr_handle = None

    def _ensure_started(self) -> None:
        if self._process is not None:
            return
        self._ensure_runtime_built()
        self._ensure_mod_content()
        self._stdout_handle = self._stdout_path.open("wb")
        self._stderr_handle = self._stderr_path.open("wb")
        env = os.environ.copy()
        dotnet_bin = _resolve_dotnet_bin()
        env["PATH"] = f"{dotnet_bin.parent}:{env.get('PATH', '')}"
        env[OPENRA_BRIDGE_DIR_ENV] = str(self._bridge_dir)
        env[OPENRA_BRIDGE_FPS_ENV] = str(self._frame_rate_hz)
        startup_orders_payload = _encode_startup_orders(self._startup_orders)
        if startup_orders_payload is not None:
            env[OPENRA_STARTUP_ORDERS_ENV] = startup_orders_payload
        env.setdefault("SDL_AUDIODRIVER", "dummy")
        launch_script = self._reference_root / "launch-game.sh"
        args = _build_launch_arguments(
            launch_script=launch_script,
            mod_id=self._mod_id,
            support_dir=self._support_dir,
            viewport=self._viewport,
            map_ref=self._map_ref,
            launch_mode=self._launch_mode,
        )
        self._process = subprocess.Popen(
            args,
            cwd=self._reference_root,
            env=env,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
        )
        deadline = time.monotonic() + self._startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"openra_process_exited_early:{self._process.returncode}"
                )
            loaded = self._load_state()
            if loaded is not None:
                self._last_state = loaded
                return
            time.sleep(0.2)
        raise RuntimeError("openra_native_bridge_startup_timeout")

    def _load_state(self) -> dict[str, Any] | None:
        if not self._state_path.exists():
            return None
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if "is_ready" in payload and not bool(payload.get("is_ready")):
            return None
        data_url = self._read_frame_data_url()
        if data_url is None:
            return None
        viewport = payload.get("viewport")
        resolved_viewport = (
            {
                "width": int(viewport.get("width", self._viewport["width"])),
                "height": int(viewport.get("height", self._viewport["height"])),
            }
            if isinstance(viewport, dict)
            else dict(self._viewport)
        )
        return {
            "frame_data_url": data_url,
            "mime_type": str(payload.get("mime_type") or "image/png"),
            "engine_tick": _coerce_optional_int(payload.get("engine_tick")),
            "render_frame": _coerce_optional_int(payload.get("render_frame")),
            "viewport": resolved_viewport,
            "updated_at_ms": _coerce_optional_int(payload.get("updated_at_ms")),
            "is_ready": bool(payload.get("is_ready", True)),
            "ready_reason": str(payload.get("ready_reason") or "").strip() or None,
            "playable_player_count": _coerce_optional_int(payload.get("playable_player_count")),
            "local_player_actor_count": _coerce_optional_int(payload.get("local_player_actor_count")),
            "camera_center": (
                {
                    "x": _coerce_optional_int(payload.get("camera_center", {}).get("x")),
                    "y": _coerce_optional_int(payload.get("camera_center", {}).get("y")),
                    "z": _coerce_optional_int(payload.get("camera_center", {}).get("z")),
                }
                if isinstance(payload.get("camera_center"), dict)
                else None
            ),
        }

    def _read_frame_data_url(self) -> str | None:
        if not self._frame_path.exists():
            return None
        try:
            stat = self._frame_path.stat()
            signature = (str(self._frame_path), int(stat.st_mtime_ns))
            if signature == self._last_frame_signature and self._last_frame_data_url is not None:
                return self._last_frame_data_url
            raw_bytes = self._frame_path.read_bytes()
        except OSError:
            return None
        self._last_frame_signature = signature
        self._last_frame_data_url = (
            f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('ascii')}"
        )
        return self._last_frame_data_url

    def _ensure_runtime_built(self) -> None:
        binary_path = self._reference_root / "bin" / "OpenRA.dll"
        if binary_path.exists():
            return
        env = os.environ.copy()
        dotnet_bin = _resolve_dotnet_bin()
        env["PATH"] = f"{dotnet_bin.parent}:{env.get('PATH', '')}"
        subprocess.run(
            ["make", "-j4"],
            cwd=self._reference_root,
            env=env,
            check=True,
        )

    def _ensure_mod_content(self) -> None:
        downloads_path = self._reference_root / "mods" / f"{self._mod_id}-content" / "installer" / "downloads.yaml"
        mod_yaml_path = self._reference_root / "mods" / f"{self._mod_id}-content" / "mod.yaml"
        if not downloads_path.exists() or not mod_yaml_path.exists():
            return
        quick_download = _extract_quick_download_name(mod_yaml_path)
        if not quick_download:
            return
        definition = _parse_download_definition(downloads_path.read_text(encoding="utf-8"), quick_download)
        if not definition or not definition["extract"]:
            return
        extract_targets = [
            _resolve_support_target_path(self._support_dir, target)
            for target in definition["extract"].keys()
        ]
        if extract_targets and all(target.exists() for target in extract_targets):
            return
        download_cache_dir = self._support_dir / ".downloads"
        download_cache_dir.mkdir(parents=True, exist_ok=True)
        archive_path = download_cache_dir / f"{self._mod_id}-{quick_download}.zip"
        archive_bytes = None
        if archive_path.exists() and _matches_sha1(archive_path, definition.get("sha1")):
            archive_bytes = archive_path.read_bytes()
        if archive_bytes is None:
            archive_bytes = _download_archive(definition)
            archive_path.write_bytes(archive_bytes)
            if not _matches_sha1(archive_path, definition.get("sha1")):
                raise RuntimeError(f"openra_content_sha1_mismatch:{self._mod_id}:{quick_download}")
        with zipfile.ZipFile(archive_path) as archive:
            for target, source in definition["extract"].items():
                resolved_target = _resolve_support_target_path(self._support_dir, target)
                resolved_target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(source) as source_handle, resolved_target.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)


def _resolve_dotnet_bin() -> Path:
    dotnet = shutil.which("dotnet")
    candidates = [
        dotnet,
        "/opt/homebrew/opt/dotnet@8/libexec/bin/dotnet",
        "/usr/local/share/dotnet/dotnet",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate).resolve()
    raise RuntimeError("dotnet_not_found")


def _resolve_launch_map_name(map_ref: str | None) -> str | None:
    if not map_ref:
        return None
    return Path(str(map_ref)).name or None


def _extract_quick_download_name(mod_yaml_path: Path) -> str | None:
    for raw_line in mod_yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("QuickDownload:"):
            value = line.split(":", 1)[1].strip()
            return value or None
    return None


def _parse_download_definition(raw_text: str, package_name: str) -> dict[str, Any] | None:
    in_package = False
    in_extract = False
    definition: dict[str, Any] = {"extract": {}}
    for raw_line in raw_text.splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip("\t "))
        line = raw_line.strip()
        if indent == 0 and ":" in line:
            current_name = line.split(":", 1)[0].strip()
            if in_package and current_name != package_name:
                break
            in_package = current_name == package_name
            in_extract = False
            continue
        if not in_package:
            continue
        if indent <= 1:
            in_extract = line.startswith("Extract:")
            if in_extract:
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                definition[key.strip().lower()] = value.strip() or None
            continue
        if in_extract and ":" in line:
            target, source = line.split(":", 1)
            definition["extract"][target.strip()] = source.strip()
    return definition if in_package else None


def _resolve_support_target_path(support_dir: Path, raw_target: str) -> Path:
    normalized = str(raw_target).strip()
    if normalized.startswith("^SupportDir|"):
        normalized = normalized.replace("^SupportDir|", "", 1)
    return (support_dir / normalized).resolve()


def _download_archive(definition: dict[str, Any]) -> bytes:
    mirrors: list[str] = []
    mirror_list = definition.get("mirrorlist")
    url = definition.get("url")
    if mirror_list:
        with urlopen(str(mirror_list), timeout=30.0) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
        mirrors.extend(line.strip() for line in raw.splitlines() if line.strip())
    if url:
        mirrors.append(str(url))
    last_error: Exception | None = None
    for candidate in mirrors:
        try:
            with urlopen(candidate, timeout=60.0) as response:  # noqa: S310
                return response.read()
        except Exception as exc:  # pragma: no cover - network instability path
            last_error = exc
    raise RuntimeError(f"openra_content_download_failed:{last_error}")


def _matches_sha1(path: Path, expected_sha1: str | None) -> bool:
    if not expected_sha1:
        return True
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    return digest.lower() == str(expected_sha1).strip().lower()


def _coerce_optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_FRAME_RATE_HZ",
    "DEFAULT_STARTUP_TIMEOUT_S",
    "OpenRANativeBridge",
    "locate_openra_reference_root",
]
