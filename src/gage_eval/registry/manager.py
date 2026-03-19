"""Centralized registry manager for all extensible components."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import inspect
import pkgutil
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from gage_eval.registry.entry import RegistryEntry
from gage_eval.registry.runtime import (
    DiscoveryFailureRecord,
    RegistryDiscoveryError,
    RegistryRuntimeMutationError,
)


class RegistryManager:
    """In-memory registry with manifest export & auto-discovery support."""

    def __init__(self) -> None:
        self._kind_desc: Dict[str, str] = {}
        self._entries: Dict[str, MutableMapping[str, RegistryEntry]] = {}
        self._objects: Dict[str, MutableMapping[str, Any]] = {}
        self._auto_discovery: Dict[str, set[str]] = defaultdict(set)
        self._active_target: ContextVar[Optional["RegistryManager"]] = ContextVar(
            "registry_active_target",
            default=None,
        )
        self._runtime_guard_views: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Kind management
    # ------------------------------------------------------------------
    def declare_kind(self, kind: str, *, desc: str = "") -> None:
        """Declare a registry kind before registering assets."""

        target = self._dispatch_target()
        if target is not self:
            target.declare_kind(kind, desc=desc)
            return
        self._assert_mutation_allowed("declare_kind")
        self._declare_kind_local(kind, desc=desc)

    def _declare_kind_local(self, kind: str, *, desc: str = "") -> None:
        if kind in self._kind_desc:
            return
        self._kind_desc[kind] = desc
        self._entries[kind] = OrderedDict()
        self._objects[kind] = OrderedDict()

    def describe_kind(self, kind: str) -> str:
        target = self._dispatch_target()
        if target is not self:
            return target.describe_kind(kind)
        return self._kind_desc.get(kind, "")

    def kinds(self) -> List[str]:
        target = self._dispatch_target()
        if target is not self:
            return target.kinds()
        return list(self._kind_desc.keys())

    def _ensure_kind(self, kind: str) -> None:
        if kind not in self._kind_desc:
            raise KeyError(f"Unknown registry kind '{kind}'")

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------
    def asset(
        self,
        kind: str,
        name: str,
        *,
        desc: str,
        version: str = "v1",
        tags: Optional[Iterable[str]] = None,
        impl: Optional[str] = None,
        target: Any = None,
        **extra: Any,
    ) -> Callable[[Any], Any]:
        """Decorator used by asset implementations."""

        route_target = self._dispatch_target()
        if route_target is not self:
            return route_target.asset(
                kind,
                name,
                desc=desc,
                version=version,
                tags=tags,
                impl=impl,
                target=target,
                **extra,
            )
        self._assert_mutation_allowed("asset")
        self._ensure_kind(kind)
        if not desc:
            raise ValueError(f"Registry asset '{kind}:{name}' requires a non-empty desc")

        def decorator(obj: Any) -> Any:
            return self._register_local(
                kind=kind,
                name=name,
                desc=desc,
                version=version,
                tags=tags,
                impl=impl,
                target=obj,
                extra=extra,
            )

        return decorator(target) if target is not None else decorator

    def register(
        self,
        kind: str,
        name: str,
        obj: Any,
        *,
        desc: str,
        version: str = "v1",
        tags: Optional[Iterable[str]] = None,
        impl: Optional[str] = None,
        **extra: Any,
    ) -> Any:
        """Imperative registration helper."""

        target = self._dispatch_target()
        if target is not self:
            return target.register(
                kind,
                name,
                obj,
                desc=desc,
                version=version,
                tags=tags,
                impl=impl,
                **extra,
            )
        self._assert_mutation_allowed("register")
        return self._register_local(
            kind=kind,
            name=name,
            desc=desc,
            version=version,
            tags=tags,
            impl=impl,
            target=obj,
            extra=extra,
        )

    def _register_local(
        self,
        *,
        kind: str,
        name: str,
        desc: str,
        version: str,
        tags: Optional[Iterable[str]],
        impl: Optional[str],
        target: Any,
        extra: Mapping[str, Any],
    ) -> Any:
        self._ensure_kind(kind)
        impl_path = impl or self._resolve_impl_path(target)
        entry = RegistryEntry(
            kind=kind,
            name=name,
            impl=impl_path,
            desc=desc,
            version=version,
            tags=tuple(tags or ()),
            extra=dict(extra),
        )
        self._entries[kind][name] = entry
        self._objects[kind][name] = target
        return target

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------
    def get(self, kind: str, name: str) -> Any:
        target = self._dispatch_target()
        if target is not self:
            return target.get(kind, name)
        return self._get_local(kind, name)

    def _get_local(self, kind: str, name: str) -> Any:
        self._ensure_kind(kind)
        try:
            return self._objects[kind][name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown registry asset '{kind}:{name}'") from exc

    def entry(self, kind: str, name: str) -> RegistryEntry:
        target = self._dispatch_target()
        if target is not self:
            return target.entry(kind, name)
        return self._entry_local(kind, name)

    def _entry_local(self, kind: str, name: str) -> RegistryEntry:
        self._ensure_kind(kind)
        try:
            return self._entries[kind][name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown registry entry '{kind}:{name}'") from exc

    def list(self, kind: str) -> List[RegistryEntry]:
        target = self._dispatch_target()
        if target is not self:
            return target.list(kind)
        return self._list_local(kind)

    def _list_local(self, kind: str) -> List[RegistryEntry]:
        self._ensure_kind(kind)
        return list(self._entries[kind].values())

    def manifest(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return a manifest-like dictionary representation."""

        target = self._dispatch_target()
        if target is not self:
            return target.manifest()
        return {kind: [asdict(entry) for entry in entries.values()] for kind, entries in self._entries.items()}

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------
    def auto_discover(
        self,
        kind: str,
        package: str,
        *,
        mode: str = "warn",
        force: bool = False,
    ) -> tuple[DiscoveryFailureRecord, ...]:
        """Import every module under ``package`` to trigger registrations."""

        target = self._dispatch_target()
        if target is not self:
            return target.auto_discover(kind, package, mode=mode, force=force)
        self._assert_mutation_allowed("auto_discover")
        return self._auto_discover_local(kind, package, mode=mode, force=force)

    def _auto_discover_local(
        self,
        kind: str,
        package: str,
        *,
        mode: str = "warn",
        force: bool = False,
    ) -> tuple[DiscoveryFailureRecord, ...]:
        self._ensure_kind(kind)
        if not force and package in self._auto_discovery[kind]:
            return ()
        self._auto_discovery[kind].add(package)
        failures = self._walk_package(kind, package)
        if failures:
            if str(mode).strip().lower() == "strict":
                raise RegistryDiscoveryError(failures)
            for failure in failures:
                warnings.warn(
                    f"[registry] Failed to import {failure.module}: {failure.exc_message}",
                    RuntimeWarning,
                )
        return failures

    def discover_all(self, *, mode: str = "warn") -> tuple[DiscoveryFailureRecord, ...]:
        target = self._dispatch_target()
        if target is not self:
            return target.discover_all(mode=mode)
        self._assert_mutation_allowed("discover_all")
        failures: list[DiscoveryFailureRecord] = []
        for kind, package_names in self._auto_discovery.items():
            for package in package_names:
                failures.extend(self._walk_package(kind, package))
        if failures and str(mode).strip().lower() == "strict":
            raise RegistryDiscoveryError(tuple(failures))
        for failure in failures:
            warnings.warn(
                f"[registry] Failed to import {failure.module}: {failure.exc_message}",
                RuntimeWarning,
            )
        return tuple(failures)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def clone(self) -> "RegistryManager":
        cloned = RegistryManager()
        cloned._kind_desc = dict(self._kind_desc)
        cloned._entries = {
            kind: OrderedDict(entries)
            for kind, entries in self._entries.items()
        }
        cloned._objects = {
            kind: OrderedDict(objects)
            for kind, objects in self._objects.items()
        }
        cloned._auto_discovery = defaultdict(
            set,
            {kind: set(packages) for kind, packages in self._auto_discovery.items()},
        )
        return cloned

    def freeze(self, *, view_id: Optional[str] = None):
        from gage_eval.registry.runtime import FrozenRegistryView

        return FrozenRegistryView.from_manager(self, view_id=view_id)

    @contextmanager
    def route_to(self, target: "RegistryManager"):
        token = self._active_target.set(target)
        try:
            yield target
        finally:
            self._active_target.reset(token)

    def acquire_runtime_guard(self, view_id: str) -> None:
        self._runtime_guard_views[view_id] = self._runtime_guard_views.get(view_id, 0) + 1

    def release_runtime_guard(self, view_id: str) -> None:
        current = self._runtime_guard_views.get(view_id)
        if current is None:
            return
        if current <= 1:
            self._runtime_guard_views.pop(view_id, None)
            return
        self._runtime_guard_views[view_id] = current - 1

    def _kind_desc_snapshot(self) -> Dict[str, str]:
        return dict(self._kind_desc)

    def _entries_snapshot(self) -> Dict[str, Mapping[str, RegistryEntry]]:
        return {
            kind: OrderedDict(entries)
            for kind, entries in self._entries.items()
        }

    def _objects_snapshot(self) -> Dict[str, Mapping[str, Any]]:
        return {
            kind: OrderedDict(objects)
            for kind, objects in self._objects.items()
        }

    def _resolve_impl_path(self, target: Any) -> str:
        if inspect.isfunction(target) or inspect.isclass(target):
            module = target.__module__
            qualname = target.__qualname__
            return f"{module}:{qualname}"
        return repr(target)

    def _walk_package(
        self,
        kind: str,
        package_name: str,
    ) -> tuple[DiscoveryFailureRecord, ...]:
        failures: list[DiscoveryFailureRecord] = []
        try:
            module = importlib.import_module(package_name)
        except Exception as exc:  # pragma: no cover - import failure path
            return (self._build_failure_record(kind, package_name, package_name, exc),)
        package_path = getattr(module, "__path__", None)
        if not package_path:  # pragma: no cover - nothing to discover
            return ()
        prefix = module.__name__ + "."
        for module_info in pkgutil.walk_packages(package_path, prefix):
            try:
                importlib.import_module(module_info.name)
            except Exception as exc:  # pragma: no cover - optional dependency path
                recovered = False
                message = str(exc)
                # NOTE: Some modules may fail during auto-discovery with a
                # "partially initialized module" error (a common cause is a circular
                # import between `pipeline.steps.*` and `role_pool`). Attempt a
                # best-effort recovery before warning to reduce noise.
                if "partially initialized module 'gage_eval.role.role_pool'" in message:
                    try:
                        importlib.import_module("gage_eval.role.role_pool")
                        importlib.import_module(module_info.name)
                        recovered = True
                    except Exception:
                        recovered = False
                if recovered:
                    continue
                failures.append(
                    self._build_failure_record(
                        kind,
                        package_name,
                        module_info.name,
                        exc,
                    )
                )
        return tuple(failures)

    def _build_failure_record(
        self,
        kind: str,
        package_name: str,
        module_name: str,
        exc: Exception,
    ) -> DiscoveryFailureRecord:
        return DiscoveryFailureRecord(
            kind=kind or "<unknown>",
            package=package_name,
            module=module_name,
            exc_type=exc.__class__.__name__,
            exc_message=str(exc),
            hint=f"package={package_name}",
        )

    def _dispatch_target(self) -> "RegistryManager":
        target = self._active_target.get()
        return target if target is not None else self

    def _assert_mutation_allowed(self, operation: str) -> None:
        if self._runtime_guard_views:
            active_views = ", ".join(sorted(self._runtime_guard_views.keys()))
            raise RegistryRuntimeMutationError(
                f"Global registry mutation '{operation}' is not allowed while runtime views are active: {active_views}"
            )
