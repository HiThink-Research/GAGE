"""Centralized registry manager for all extensible components."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from gage_eval.registry.entry import RegistryEntry


class RegistryManager:
    """In-memory registry with manifest export & auto-discovery support."""

    def __init__(self) -> None:
        self._kind_desc: Dict[str, str] = {}
        self._entries: Dict[str, MutableMapping[str, RegistryEntry]] = {}
        self._objects: Dict[str, MutableMapping[str, Any]] = {}
        self._auto_discovery: Dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Kind management
    # ------------------------------------------------------------------
    def declare_kind(self, kind: str, *, desc: str = "") -> None:
        """Declare a registry kind before registering assets."""

        if kind in self._kind_desc:
            return
        self._kind_desc[kind] = desc
        self._entries[kind] = OrderedDict()
        self._objects[kind] = OrderedDict()

    def describe_kind(self, kind: str) -> str:
        return self._kind_desc.get(kind, "")

    def kinds(self) -> List[str]:
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

        self._ensure_kind(kind)
        if not desc:
            raise ValueError(f"Registry asset '{kind}:{name}' requires a non-empty desc")

        def decorator(obj: Any) -> Any:
            return self._register(
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

        return self._register(
            kind=kind,
            name=name,
            desc=desc,
            version=version,
            tags=tags,
            impl=impl,
            target=obj,
            extra=extra,
        )

    def _register(
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
        self._ensure_kind(kind)
        try:
            return self._objects[kind][name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown registry asset '{kind}:{name}'") from exc

    def entry(self, kind: str, name: str) -> RegistryEntry:
        self._ensure_kind(kind)
        try:
            return self._entries[kind][name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown registry entry '{kind}:{name}'") from exc

    def list(self, kind: str) -> List[RegistryEntry]:
        self._ensure_kind(kind)
        return list(self._entries[kind].values())

    def manifest(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return a manifest-like dictionary representation."""

        return {kind: [asdict(entry) for entry in entries.values()] for kind, entries in self._entries.items()}

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------
    def auto_discover(self, kind: str, package: str) -> None:
        """Import every module under ``package`` to trigger registrations."""

        self._ensure_kind(kind)
        if package in self._auto_discovery[kind]:
            return
        self._auto_discovery[kind].add(package)
        self._walk_package(package)

    def discover_all(self) -> None:
        for package_names in self._auto_discovery.values():
            for package in package_names:
                self._walk_package(package)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _resolve_impl_path(self, target: Any) -> str:
        if inspect.isfunction(target) or inspect.isclass(target):
            module = target.__module__
            qualname = target.__qualname__
            return f"{module}:{qualname}"
        return repr(target)

    def _walk_package(self, package_name: str) -> None:
        module = importlib.import_module(package_name)
        package_path = getattr(module, "__path__", None)
        if not package_path:  # pragma: no cover - nothing to discover
            return
        prefix = module.__name__ + "."
        for module_info in pkgutil.walk_packages(package_path, prefix):
            try:
                importlib.import_module(module_info.name)
            except Exception as exc:  # pragma: no cover - optional dependency path
                recovered = False
                message = str(exc)
                # 某些模块在 auto-discover 期间会命中“partially initialized”错误（典型场景：
                # pipeline.steps.* 与 role_pool 交叉导入）。在警告前尝试一次补救，以避免无谓噪音。
                if "partially initialized module 'gage_eval.role.role_pool'" in message:
                    try:
                        importlib.import_module("gage_eval.role.role_pool")
                        importlib.import_module(module_info.name)
                        recovered = True
                    except Exception:
                        recovered = False
                if recovered:
                    continue
                warnings.warn(
                    f"[registry] Failed to import {module_info.name}: {exc}",
                    RuntimeWarning,
                )
