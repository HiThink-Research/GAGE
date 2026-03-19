"""Runtime-scoped registry lifecycle primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence, TYPE_CHECKING
from uuid import uuid4
import weakref

from gage_eval.registry.entry import RegistryEntry

if TYPE_CHECKING:
    from gage_eval.registry.manager import RegistryManager


@dataclass(frozen=True, slots=True)
class DiscoveryPolicy:
    """Controls how discovery failures are handled during prepare."""

    mode: str = "warn"
    freeze_strict: bool = True

    @property
    def is_strict(self) -> bool:
        return str(self.mode).strip().lower() == "strict"


@dataclass(frozen=True, slots=True)
class DiscoveryFailureRecord:
    """Structured details for a discovery import failure."""

    kind: str
    package: str
    module: str
    exc_type: str
    exc_message: str
    hint: str = ""

    def render(self) -> str:
        base = f"{self.kind}:{self.module} failed with {self.exc_type}: {self.exc_message}"
        if self.hint:
            return f"{base} ({self.hint})"
        return base


class RegistryDiscoveryError(RuntimeError):
    """Raised when strict discovery encounters import failures."""

    def __init__(self, failures: Sequence[DiscoveryFailureRecord]) -> None:
        self.failures = tuple(failures)
        message = "; ".join(record.render() for record in self.failures[:3])
        if len(self.failures) > 3:
            message = f"{message}; +{len(self.failures) - 3} more"
        super().__init__(message or "registry discovery failed")


class RegistryRuntimeMutationError(RuntimeError):
    """Raised when runtime code attempts to mutate the global registry."""


@dataclass(frozen=True, slots=True)
class RegistryOverlayAsset:
    """Declarative overlay registration applied during prepare."""

    kind: str
    name: str
    obj: Any
    desc: str
    version: str = "v1"
    tags: tuple[str, ...] = ()
    impl: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


class RegistryLookup(Protocol):
    """Read-only registry lookup protocol shared by views and managers."""

    def describe_kind(self, kind: str) -> str: ...
    def kinds(self) -> Sequence[str]: ...
    def get(self, kind: str, name: str) -> Any: ...
    def entry(self, kind: str, name: str) -> RegistryEntry: ...
    def list(self, kind: str) -> Sequence[RegistryEntry]: ...
    def manifest(self) -> Dict[str, list[Dict[str, Any]]]: ...


class FrozenRegistryView:
    """Read-only snapshot used by a single runtime assembly."""

    def __init__(
        self,
        *,
        view_id: str,
        kind_desc: Mapping[str, str],
        entries: Mapping[str, Mapping[str, RegistryEntry]],
        objects: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.view_id = view_id
        self._kind_desc = MappingProxyType(dict(kind_desc))
        self._entries = {
            kind: MappingProxyType(dict(kind_entries))
            for kind, kind_entries in entries.items()
        }
        self._objects = {
            kind: MappingProxyType(dict(kind_objects))
            for kind, kind_objects in objects.items()
        }
        self._scoped_caches: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_manager(cls, manager: "RegistryManager", *, view_id: Optional[str] = None) -> "FrozenRegistryView":
        return cls(
            view_id=view_id or f"registry-view-{uuid4().hex}",
            kind_desc=manager._kind_desc_snapshot(),
            entries=manager._entries_snapshot(),
            objects=manager._objects_snapshot(),
        )

    def describe_kind(self, kind: str) -> str:
        return self._kind_desc.get(kind, "")

    def kinds(self) -> Sequence[str]:
        return tuple(self._kind_desc.keys())

    def get(self, kind: str, name: str) -> Any:
        try:
            return self._objects[kind][name]
        except KeyError as exc:
            raise KeyError(f"Unknown registry asset '{kind}:{name}'") from exc

    def entry(self, kind: str, name: str) -> RegistryEntry:
        try:
            return self._entries[kind][name]
        except KeyError as exc:
            raise KeyError(f"Unknown registry entry '{kind}:{name}'") from exc

    def list(self, kind: str) -> Sequence[RegistryEntry]:
        try:
            return tuple(self._entries[kind].values())
        except KeyError as exc:
            raise KeyError(f"Unknown registry kind '{kind}'") from exc

    def manifest(self) -> Dict[str, list[Dict[str, Any]]]:
        return {
            kind: [
                {
                    "kind": entry.kind,
                    "name": entry.name,
                    "impl": entry.impl,
                    "desc": entry.desc,
                    "version": entry.version,
                    "tags": tuple(entry.tags),
                    "extra": dict(entry.extra),
                }
                for entry in entries.values()
            ]
            for kind, entries in self._entries.items()
        }

    def get_scoped_cache(self, name: str) -> Dict[str, Any]:
        return self._scoped_caches.setdefault(name, {})

    def clear_scoped_cache(self) -> None:
        self._scoped_caches.clear()


class RegistryFacade:
    """Writable facade used only during prepare."""

    def __init__(self, manager: "RegistryManager") -> None:
        self._manager = manager

    @property
    def manager(self) -> "RegistryManager":
        return self._manager

    def freeze(self, *, view_id: Optional[str] = None) -> FrozenRegistryView:
        return self._manager.freeze(view_id=view_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class RegistryViewLease:
    """Lifecycle handle that releases runtime mutation guards and scoped caches."""

    def __init__(self, view: FrozenRegistryView, releaser: Callable[[str], None]) -> None:
        self.view = view
        self.view_id = view.view_id
        self._releaser = releaser
        self._closed = False
        self._finalizer = weakref.finalize(self, releaser, self.view_id)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._finalizer.alive:
            self._finalizer()


@dataclass(frozen=True, slots=True)
class RuntimeRegistryContext:
    """Run-scoped registry objects used during runtime assembly."""

    view: FrozenRegistryView
    lease: RegistryViewLease

    def close(self) -> None:
        self.lease.close()


class RegistryBootstrapCoordinator:
    """Builds a run-local registry view from the global baseline."""

    def __init__(self, registry: "RegistryManager") -> None:
        self._registry = registry

    def prepare_runtime_context(
        self,
        *,
        run_id: str,
        required_packages: Optional[Mapping[str, Iterable[str]]] = None,
        overlay_assets: Sequence[RegistryOverlayAsset] = (),
        policy: Optional[DiscoveryPolicy] = None,
    ) -> RuntimeRegistryContext:
        resolved_policy = policy or DiscoveryPolicy()
        working = self._registry.clone()
        facade = RegistryFacade(working)
        package_map = {
            kind: tuple(packages)
            for kind, packages in (required_packages or {}).items()
            if packages
        }

        with self._registry.route_to(working):
            for kind, packages in package_map.items():
                for package in packages:
                    self._registry.auto_discover(
                        kind,
                        package,
                        mode=resolved_policy.mode,
                        force=True,
                    )
            for overlay in overlay_assets:
                self._registry.register(
                    overlay.kind,
                    overlay.name,
                    overlay.obj,
                    desc=overlay.desc,
                    version=overlay.version,
                    tags=overlay.tags,
                    impl=overlay.impl,
                    **dict(overlay.extra),
                )

        view = facade.freeze(view_id=run_id)
        if resolved_policy.freeze_strict:
            self._registry.acquire_runtime_guard(view.view_id)

        def _release(view_id: str) -> None:
            view.clear_scoped_cache()
            if resolved_policy.freeze_strict:
                self._registry.release_runtime_guard(view_id)

        lease = RegistryViewLease(view, _release)
        return RuntimeRegistryContext(view=view, lease=lease)
