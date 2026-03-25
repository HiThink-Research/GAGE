"""Runtime-scoped registry lifecycle primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence, TYPE_CHECKING
from uuid import uuid4
import weakref

from gage_eval.registry.bootstrap_importer import BootstrapImporter, DiscoveryReport

from gage_eval.registry.entry import RegistryEntry

if TYPE_CHECKING:
    from gage_eval.registry.asset_planner import DiscoveryPlan
    from gage_eval.registry.manager import RegistryManager


@dataclass(frozen=True, slots=True)
class DiscoveryPolicy:
    """Controls how discovery failures are handled during prepare."""

    mode: str = "warn"
    strategy: str = "manifest"
    freeze_strict: bool = True
    dev_auto_refresh: bool = False

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
        object.__setattr__(self, "view_id", view_id)
        object.__setattr__(self, "_kind_desc", MappingProxyType(dict(kind_desc)))
        object.__setattr__(
            self,
            "_entries",
            {
                kind: MappingProxyType(dict(kind_entries))
                for kind, kind_entries in entries.items()
            },
        )
        object.__setattr__(
            self,
            "_objects",
            {
                kind: MappingProxyType(dict(kind_objects))
                for kind, kind_objects in objects.items()
            },
        )
        object.__setattr__(self, "_scoped_caches", {})
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False):
            raise RegistryRuntimeMutationError(
                f"runtime_discovery_forbidden: cannot mutate FrozenRegistryView attribute '{name}'"
            )
        object.__setattr__(self, name, value)

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

    def register(self, *args, **kwargs) -> None:  # pragma: no cover - defensive API guard
        raise RegistryRuntimeMutationError("runtime_discovery_forbidden: FrozenRegistryView is read-only")

    def auto_discover(self, *args, **kwargs) -> None:  # pragma: no cover - defensive API guard
        raise RegistryRuntimeMutationError("runtime_discovery_forbidden: FrozenRegistryView is read-only")

    def declare_kind(self, *args, **kwargs) -> None:  # pragma: no cover - defensive API guard
        raise RegistryRuntimeMutationError("runtime_discovery_forbidden: FrozenRegistryView is read-only")


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
    discovery_report: DiscoveryReport = field(default_factory=DiscoveryReport)

    def close(self) -> None:
        self.lease.close()


class RegistryBootstrapCoordinator:
    """Builds a run-local registry view from the global baseline."""

    def __init__(
        self,
        registry: "RegistryManager",
        *,
        importer_factory: Callable[[], BootstrapImporter] | None = None,
    ) -> None:
        self._registry = registry
        self._importer_factory = importer_factory or BootstrapImporter

    def prepare_runtime_context(
        self,
        *,
        run_id: str,
        discovery_plan: Optional["DiscoveryPlan"] = None,
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
            report = DiscoveryReport()
            if discovery_plan is not None:
                report = self._importer_factory().execute(discovery_plan, registry=working)
                if report.issues and resolved_policy.is_strict:
                    raise RegistryDiscoveryError(
                        (
                            DiscoveryFailureRecord(
                                kind=issue.kind,
                                package=issue.source or "<manifest>",
                                module=issue.module or issue.name,
                                exc_type=issue.code,
                                exc_message=issue.detail,
                                hint=f"source={issue.source}" if issue.source else "",
                            )
                            for issue in report.issues
                        )
                    )
            if resolved_policy.strategy in {"legacy", "hybrid"}:
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
        return RuntimeRegistryContext(view=view, lease=lease, discovery_report=report)
