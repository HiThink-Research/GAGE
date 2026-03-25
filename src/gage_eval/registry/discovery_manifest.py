"""Manifest-backed registry discovery primitives."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

_MANIFEST_VERSION = 1
_ROOT_ENV_VAR = "GAGE_EVAL_DISCOVERY_MANIFEST_ROOTS"
_MANIFEST_FILE_ORDER = (
    "core.json",
    "datasets.json",
    "arena.json",
    "backends.json",
    "metrics.json",
)


@dataclass(frozen=True, slots=True)
class DiscoveryManifestEntry:
    """Single manifest-backed registry asset record."""

    kind: str
    name: str
    module: str
    load_phase: str = "prepare_only"
    declared_in: str = ""
    aliases: tuple[str, ...] = ()
    optional: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DiscoveryManifestEntry":
        kind = str(payload.get("kind") or "").strip()
        name = str(payload.get("name") or "").strip()
        module = str(payload.get("module") or "").strip()
        if not kind or not name or not module:
            raise ValueError(f"Invalid manifest entry: {payload!r}")
        aliases = tuple(
            str(alias).strip()
            for alias in payload.get("aliases", ())
            if str(alias).strip()
        )
        return cls(
            kind=kind,
            name=name,
            module=module,
            load_phase=str(payload.get("load_phase") or "prepare_only").strip() or "prepare_only",
            declared_in=str(payload.get("declared_in") or "").strip(),
            aliases=aliases,
            optional=bool(payload.get("optional", False)),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "module": self.module,
            "load_phase": self.load_phase,
            "declared_in": self.declared_in,
            "aliases": list(self.aliases),
            "optional": self.optional,
        }


class DiscoveryManifestRepository:
    """Read-only view over merged manifest shards."""

    def __init__(
        self,
        *,
        entries: Mapping[tuple[str, str], DiscoveryManifestEntry],
        aliases: Mapping[tuple[str, str], tuple[str, str]],
        roots: Sequence[Path],
    ) -> None:
        self._entries = dict(entries)
        self._aliases = dict(aliases)
        self._roots = tuple(Path(root).resolve() for root in roots)

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    def resolve(self, kind: str, name: str) -> DiscoveryManifestEntry | None:
        key = (str(kind).strip(), str(name).strip())
        entry = self._entries.get(key)
        if entry is not None:
            return entry
        alias_target = self._aliases.get(key)
        if alias_target is None:
            return None
        return self._entries.get(alias_target)

    def require(self, kind: str, name: str) -> DiscoveryManifestEntry:
        entry = self.resolve(kind, name)
        if entry is None:
            raise KeyError(f"Unknown manifest asset '{kind}:{name}'")
        return entry

    def entries_for_kind(self, kind: str) -> tuple[DiscoveryManifestEntry, ...]:
        lookup_kind = str(kind).strip()
        return tuple(
            entry
            for (entry_kind, _), entry in self._entries.items()
            if entry_kind == lookup_kind
        )

    def modules_for_kind(self, kind: str) -> tuple[str, ...]:
        modules: dict[str, None] = {}
        for entry in self.entries_for_kind(kind):
            modules.setdefault(entry.module, None)
        return tuple(modules.keys())

    def manifest(self) -> Dict[str, list[Dict[str, Any]]]:
        payload: Dict[str, list[Dict[str, Any]]] = {}
        for entry in self._entries.values():
            payload.setdefault(entry.kind, []).append(entry.to_payload())
        for values in payload.values():
            values.sort(key=lambda item: (item["name"], item["module"]))
        return payload

    @classmethod
    def from_roots(cls, roots: Sequence[Path]) -> "DiscoveryManifestRepository":
        entries: Dict[tuple[str, str], DiscoveryManifestEntry] = {}
        aliases: Dict[tuple[str, str], tuple[str, str]] = {}
        root_paths = tuple(Path(root).resolve() for root in roots)
        for payload in _iter_merged_manifest_payloads(root_paths):
            entry = DiscoveryManifestEntry.from_payload(payload)
            key = (entry.kind, entry.name)
            entries[key] = entry
            for alias in entry.aliases:
                aliases[(entry.kind, alias)] = key
        return cls(entries=entries, aliases=aliases, roots=root_paths)

    @classmethod
    def default(cls) -> "DiscoveryManifestRepository":
        return load_default_manifest_repository()


def load_default_manifest_repository() -> DiscoveryManifestRepository:
    return _load_default_manifest_repository(tuple(str(path) for path in _resolve_manifest_roots()))


@lru_cache(maxsize=8)
def _load_default_manifest_repository(root_strings: tuple[str, ...]) -> DiscoveryManifestRepository:
    roots = tuple(Path(root) for root in root_strings)
    return DiscoveryManifestRepository.from_roots(roots)


def clear_manifest_repository_cache() -> None:
    _load_default_manifest_repository.cache_clear()


def default_manifest_root() -> Path:
    return Path(__file__).resolve().parent / "manifests"


def iter_manifest_paths(root: Path) -> Iterator[Path]:
    resolved_root = Path(root).resolve()
    for filename in _MANIFEST_FILE_ORDER:
        path = resolved_root / filename
        if path.is_file():
            yield path
    plugins_root = resolved_root / "plugins"
    if plugins_root.is_dir():
        for path in sorted(plugins_root.glob("*.json")):
            yield path
    overrides = resolved_root / "manual_overrides.json"
    if overrides.is_file():
        yield overrides


def _resolve_manifest_roots() -> tuple[Path, ...]:
    roots: list[Path] = [default_manifest_root()]
    raw = str(os.environ.get(_ROOT_ENV_VAR) or "").strip()
    if raw:
        for item in raw.split(os.pathsep):
            candidate = str(item).strip()
            if candidate:
                roots.append(Path(candidate).expanduser())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return tuple(deduped)


def _iter_merged_manifest_payloads(roots: Sequence[Path]) -> Iterator[Mapping[str, Any]]:
    non_override_paths: list[Path] = []
    override_paths: list[Path] = []
    for root in roots:
        for path in iter_manifest_paths(root):
            target = override_paths if path.name == "manual_overrides.json" else non_override_paths
            target.append(path)
    for path in (*non_override_paths, *override_paths):
        yield from _read_manifest_entries(path)


def _read_manifest_entries(path: Path) -> Iterator[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        entries = payload
        version = _MANIFEST_VERSION
    else:
        version = int(payload.get("manifest_version") or _MANIFEST_VERSION)
        entries = payload.get("entries", ())
    if version != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version in {path}: {version}")
    for entry in entries:
        if isinstance(entry, Mapping):
            yield entry
