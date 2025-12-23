"""Cache helpers used by AutoEvalStep/ReportStep."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from loguru import logger
from gage_eval.evaluation.buffered_writer import BufferedResultWriter


class EvalCache:
    """File-based cache aligned with the configured save directory and run id."""

    def __init__(self, base_dir: Optional[str] = None, run_id: Optional[str] = None) -> None:
        env_base = os.environ.get("GAGE_EVAL_SAVE_DIR") or None
        default_base = env_base or "./runs"
        self._base_dir = Path(base_dir or default_base).expanduser()
        self._run_id = run_id or self._generate_run_id()
        self._run_dir = self._base_dir / self._run_id
        self._samples_dir = self._run_dir / "samples"
        self._samples_jsonl = self._run_dir / "samples.jsonl"
        self._lock = threading.Lock()
        self._sample_count = 0
        self._namespace_counts: Dict[str, int] = {}
        self._timings: Dict[str, float] = {}
        self._metadata: Dict[str, Any] = {}
        self._ensure_dirs()
        disable_buffer = _env_flag("GAGE_EVAL_DISABLE_BUFFERED_WRITER", default=False)
        force_buffer = _env_flag("GAGE_EVAL_ENABLE_BUFFERED_WRITER", default=False)
        self._buffer_threshold = _env_int("GAGE_EVAL_BUFFER_THRESHOLD", default=1000)
        self._buffer_batch_size = _env_int("GAGE_EVAL_BUFFER_BATCH_SIZE", default=64)
        self._buffer_flush_interval = _env_float("GAGE_EVAL_BUFFER_FLUSH_S", default=2.0)
        self._buffer_auto_mode = False
        if force_buffer and not disable_buffer:
            self._use_buffered_writes = True
        elif disable_buffer:
            self._use_buffered_writes = False
        else:
            self._use_buffered_writes = False
            self._buffer_auto_mode = self._buffer_threshold > 0
        self._writers: Dict[str, BufferedResultWriter] = {}
        self._writer_lock = threading.Lock()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def samples_dir(self) -> Path:
        return self._samples_dir

    @property
    def samples_jsonl(self) -> Path:
        return self._samples_jsonl

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def namespace_counts(self) -> Dict[str, int]:
        return dict(self._namespace_counts)

    def record_timing(self, phase: str, seconds: float) -> None:
        with self._lock:
            self._timings[phase] = float(seconds)

    def get_timing(self, phase: str) -> Optional[float]:
        return self._timings.get(phase)

    def set_metadata(self, key: str, value: Any) -> None:
        with self._lock:
            self._metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        return self._metadata.get(key)

    def write_sample(self, sample_id: str, payload: Dict, *, namespace: Optional[str] = None) -> Path:
        """Persist the per-sample payload to disk."""

        namespace = self._sanitize_namespace(namespace)
        safe_sample_id = self._sanitize_sample_id(sample_id)
        with self._lock:
            self._sample_count += 1
            self._namespace_counts[namespace] = self._namespace_counts.get(namespace, 0) + 1
            payload_with_meta = {
                "sample_id": sample_id,
                "sequence_id": self._sample_count,
                "namespace": namespace,
                **payload,
            }
            if self._buffer_auto_mode and not self._use_buffered_writes and self._sample_count >= self._buffer_threshold:
                logger.info(
                    "EvalCache enabling buffered writer (samples={} threshold={})",
                    self._sample_count,
                    self._buffer_threshold,
                )
                self._use_buffered_writes = True
        self._append_jsonl(payload_with_meta)
        if self._use_buffered_writes:
            writer = self._get_writer(namespace)
            writer.record(payload_with_meta)
            target = writer.target_path
        else:
            target = self._write_sample_legacy(namespace, safe_sample_id, payload_with_meta)
        logger.debug("Cached sample_id={} to {}", sample_id, target)
        return target

    def write_summary(self, payload: Dict) -> Path:
        """Persist the aggregated summary to disk."""

        target = self._run_dir / "summary.json"
        self.flush_writers()
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        logger.info("Wrote summary for run_id={} to {}", self._run_id, target)
        return target

    def iter_samples(self) -> Iterator[Dict]:
        """Iterate over cached sample payloads."""

        if self._samples_jsonl.exists():
            with self._samples_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return

        if not self._samples_dir.exists():
            return
        files = sorted(self._samples_dir.rglob("*.json"))
        for file in files:
            try:
                yield json.loads(file.read_text())
            except json.JSONDecodeError:
                continue

    def snapshot(self) -> Dict[str, str]:
        return {
            "run_id": self._run_id,
            "run_dir": str(self._run_dir),
            "samples_dir": str(self._samples_dir),
            "samples_jsonl": str(self._samples_jsonl),
            "namespaces": dict(self._namespace_counts),
            "timings": dict(self._timings),
            "metadata": dict(self._metadata),
        }

    def _ensure_dirs(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._samples_dir.mkdir(parents=True, exist_ok=True)

    def _generate_run_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return f"run-{timestamp}-{suffix}"

    @staticmethod
    def _sanitize_sample_id(sample_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(sample_id))

    def _append_jsonl(self, payload: Dict) -> None:
        self._samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self._samples_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
            handle.write("\n")

    def _get_writer(self, namespace: str) -> BufferedResultWriter:
        with self._writer_lock:
            writer = self._writers.get(namespace)
            if writer is None:
                target = self._samples_dir / namespace / "samples.jsonl"
                writer = BufferedResultWriter(
                    target=target,
                    max_batch_size=self._buffer_batch_size,
                    flush_interval_s=self._buffer_flush_interval,
                )
                self._writers[namespace] = writer
            return writer

    def flush_writers(self) -> None:
        if not self._use_buffered_writes:
            return
        with self._writer_lock:
            for writer in self._writers.values():
                writer.flush()

    def close(self) -> None:
        self.flush_writers()

    def _write_sample_legacy(self, namespace: str, sample_id: str, payload: Dict) -> Path:
        target = self._samples_dir / namespace / f"{sample_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        formatted = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        target.write_text(formatted)
        return target

    @staticmethod
    def _sanitize_namespace(namespace: Optional[str]) -> str:
        return (namespace or "default").replace("/", "_")


def _json_default(obj: Any) -> Any:
    """Best-effort serializer that falls back to str() for unknown objects."""

    if isinstance(obj, (Path,)):
        return str(obj)
    try:
        import PIL.Image  # type: ignore

        if isinstance(obj, PIL.Image.Image):
            return f"<PIL.Image mode={obj.mode} size={obj.size}>"
    except ImportError:  # pragma: no cover
        pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:  # pragma: no cover - defensive
            return str(obj)
    if hasattr(obj, "__dict__"):
        try:
            return obj.__dict__
        except Exception:  # pragma: no cover
            return str(obj)
    return str(obj)


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
