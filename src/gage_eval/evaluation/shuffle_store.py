"""Shuffle helpers for bounded-memory sample iteration."""

from __future__ import annotations

import heapq
import pickle
import random
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def try_resolve_length(samples: Iterable[dict[str, Any]]) -> int | None:
    """Return a cheap length hint when the iterable exposes one."""

    try:
        size = len(samples)  # type: ignore[arg-type]
    except TypeError:
        return None
    return int(size)


def iter_reservoir_samples(
    samples: Iterable[dict[str, Any]],
    *,
    max_samples: int,
    seed: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield a deterministic random subset using O(K) memory.

    The implementation assigns each incoming sample a random priority and keeps
    the ``K`` smallest priorities. Sorting the retained set by priority gives a
    deterministic pseudo-random order for the final subset.
    """

    if max_samples <= 0:
        return
    rng = random.Random(seed)
    reservoir: list[tuple[float, int, dict[str, Any]]] = []
    for source_index, sample in enumerate(samples):
        priority = rng.random()
        entry = (-priority, source_index, sample)
        if len(reservoir) < max_samples:
            heapq.heappush(reservoir, entry)
            continue
        if priority < -reservoir[0][0]:
            heapq.heapreplace(reservoir, entry)

    selected = [(-priority, source_index, sample) for priority, source_index, sample in reservoir]
    selected.sort(key=lambda item: (item[0], item[1]))
    for logical_idx, (_, _, sample) in enumerate(selected):
        yield logical_idx, sample


def iter_external_shuffle_samples(
    samples: Iterable[dict[str, Any]],
    *,
    seed: int,
    artifact_root: Path | str | None = None,
    keep_artifacts: bool = False,
    chunk_size: int = 2048,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield all samples in pseudo-random order using external chunk files."""

    resolved_chunk_size = max(1, int(chunk_size))
    with _shuffle_workspace(artifact_root, keep_artifacts=keep_artifacts) as workspace:
        chunk_paths = _write_sorted_chunks(
            samples,
            seed=seed,
            workspace=workspace,
            chunk_size=resolved_chunk_size,
        )
        yield from _merge_sorted_chunks(chunk_paths)


@contextmanager
def _shuffle_workspace(
    artifact_root: Path | str | None,
    *,
    keep_artifacts: bool,
) -> Iterator[Path]:
    if artifact_root is None:
        root = Path(tempfile.mkdtemp(prefix="gage-shuffle-"))
        should_cleanup = True
    else:
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        should_cleanup = not keep_artifacts
    try:
        yield root
    finally:
        if should_cleanup:
            shutil.rmtree(root, ignore_errors=True)


def _write_sorted_chunks(
    samples: Iterable[dict[str, Any]],
    *,
    seed: int,
    workspace: Path,
    chunk_size: int,
) -> Sequence[Path]:
    rng = random.Random(seed)
    chunk_paths: list[Path] = []
    buffer: list[tuple[float, int, dict[str, Any]]] = []
    chunk_index = 0

    # STEP 1: Spill the incoming iterator into sorted chunk files.
    for source_index, sample in enumerate(samples):
        buffer.append((rng.random(), source_index, sample))
        if len(buffer) < chunk_size:
            continue
        chunk_paths.append(_flush_chunk(buffer, workspace=workspace, chunk_index=chunk_index))
        chunk_index += 1
        buffer = []

    # STEP 2: Flush the trailing partial chunk.
    if buffer:
        chunk_paths.append(_flush_chunk(buffer, workspace=workspace, chunk_index=chunk_index))
    return tuple(chunk_paths)


def _flush_chunk(
    records: list[tuple[float, int, dict[str, Any]]],
    *,
    workspace: Path,
    chunk_index: int,
) -> Path:
    records.sort(key=lambda item: (item[0], item[1]))
    target = workspace / f"chunk-{chunk_index:05d}.pkl"
    with target.open("wb") as handle:
        for record in records:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def _merge_sorted_chunks(
    chunk_paths: Sequence[Path],
) -> Iterator[tuple[int, dict[str, Any]]]:
    handles = [path.open("rb") for path in chunk_paths]
    heap: list[tuple[float, int, int, dict[str, Any]]] = []

    try:
        # STEP 1: Prime the merge heap with the first record from each chunk.
        for chunk_id, handle in enumerate(handles):
            record = _read_chunk_record(handle)
            if record is None:
                continue
            priority, source_index, sample = record
            heapq.heappush(heap, (priority, source_index, chunk_id, sample))

        # STEP 2: Merge all sorted chunks into a single ordered stream.
        logical_idx = 0
        while heap:
            priority, source_index, chunk_id, sample = heapq.heappop(heap)
            yield logical_idx, sample
            logical_idx += 1
            record = _read_chunk_record(handles[chunk_id])
            if record is None:
                continue
            next_priority, next_source_index, next_sample = record
            heapq.heappush(
                heap,
                (next_priority, next_source_index, chunk_id, next_sample),
            )
    finally:
        for handle in handles:
            handle.close()


def _read_chunk_record(handle) -> tuple[float, int, dict[str, Any]] | None:
    try:
        return pickle.load(handle)
    except EOFError:
        return None

