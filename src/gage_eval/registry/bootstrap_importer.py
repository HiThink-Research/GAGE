"""Manifest-backed bootstrap importer."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING

from gage_eval.registry.discovery_manifest import (
    DiscoveryManifestEntry,
    DiscoveryManifestRepository,
    load_default_manifest_repository,
)
from gage_eval.registry.discovery_telemetry import telemetry

if TYPE_CHECKING:
    from gage_eval.registry.asset_planner import DiscoveryPlan, DiscoveryRequest
    from gage_eval.registry.manager import RegistryManager


@dataclass(frozen=True, slots=True)
class DiscoveryIssue:
    """Structured manifest/bootstrap issue."""

    code: str
    kind: str
    name: str
    detail: str
    source: str = ""
    module: str = ""
    load_phase: str = ""


@dataclass(frozen=True, slots=True)
class DiscoveryImport:
    """Successful bootstrap import record."""

    kind: str
    name: str
    module: str
    source: str = ""
    load_phase: str = ""


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    """Bootstrap discovery outcome."""

    imported: tuple[DiscoveryImport, ...] = ()
    issues: tuple[DiscoveryIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


class BootstrapImporter:
    """Imports runtime assets into a routed registry clone."""

    def __init__(self, repository: DiscoveryManifestRepository | None = None) -> None:
        self._repository = repository or load_default_manifest_repository()

    @property
    def repository(self) -> DiscoveryManifestRepository:
        return self._repository

    def execute(
        self,
        plan: "DiscoveryPlan",
        *,
        registry: "RegistryManager",
    ) -> DiscoveryReport:
        imports: list[DiscoveryImport] = []
        issues: list[DiscoveryIssue] = []
        imported_modules: set[str] = set()
        resolved_requests: list[tuple[int, DiscoveryManifestEntry, DiscoveryRequest]] = []

        for index, request in enumerate(plan.requests):
            entry = self._repository.resolve(request.kind, request.name)
            if entry is None:
                telemetry.record("manifest_missing", kind=request.kind)
                issues.append(
                    DiscoveryIssue(
                        code="manifest_missing",
                        kind=request.kind,
                        name=request.name,
                        detail=f"Manifest entry not found for {request.kind}:{request.name}",
                        source=request.source,
                        load_phase="unknown",
                    )
                )
                continue
            resolved_requests.append((index, entry, request))

        for _, entry, request in sorted(
            resolved_requests,
            key=lambda item: (_load_phase_sort_key(item[1].load_phase), item[0]),
        ):
            self._import_entry(entry, registry=registry, imported_modules=imported_modules, issues=issues, request=request)
            if _has_registry_entry(registry, request.kind, request.name):
                imports.append(
                    DiscoveryImport(
                        kind=request.kind,
                        name=request.name,
                        module=entry.module,
                        source=request.source,
                        load_phase=entry.load_phase,
                    )
                )
            else:
                issues.append(
                    DiscoveryIssue(
                        code="registration_missing",
                        kind=request.kind,
                        name=request.name,
                        detail=f"Imported module '{entry.module}' did not register '{request.kind}:{request.name}'",
                        source=request.source,
                        module=entry.module,
                        load_phase=entry.load_phase,
                    )
                )

        for kind in plan.eager_kinds:
            for entry in _sort_entries_by_phase(self._repository.entries_for_kind(kind)):
                module_name = entry.module
                if module_name in imported_modules:
                    continue
                try:
                    importlib.import_module(module_name)
                    imported_modules.add(module_name)
                    telemetry.record("eager_kind_import", kind=kind)
                    imports.append(
                        DiscoveryImport(
                            kind=entry.kind,
                            name=entry.name,
                            module=entry.module,
                            load_phase=entry.load_phase,
                        )
                    )
                except Exception as exc:
                    issues.append(
                        DiscoveryIssue(
                            code="module_import_failed",
                            kind=kind,
                            name=entry.name,
                            detail=str(exc),
                            module=module_name,
                            load_phase=entry.load_phase,
                        )
                    )

        return DiscoveryReport(imported=tuple(imports), issues=tuple(issues))

    def import_kind(
        self,
        kind: str,
        *,
        registry: "RegistryManager",
    ) -> DiscoveryReport:
        imports: list[DiscoveryImport] = []
        issues: list[DiscoveryIssue] = []
        for entry in _sort_entries_by_phase(self._repository.entries_for_kind(kind)):
            try:
                module = importlib.import_module(entry.module)
                if not _has_registry_entry(registry, entry.kind, entry.name):
                    module = importlib.reload(module)
                imports.append(
                    DiscoveryImport(
                        kind=entry.kind,
                        name=entry.name,
                        module=entry.module,
                        load_phase=entry.load_phase,
                    )
                )
                telemetry.record("kind_import", kind=kind)
            except Exception as exc:
                issues.append(
                    DiscoveryIssue(
                        code="module_import_failed",
                        kind=entry.kind,
                        name=entry.name,
                        detail=str(exc),
                        module=entry.module,
                        load_phase=entry.load_phase,
                    )
                )
        return DiscoveryReport(imported=tuple(imports), issues=tuple(issues))

    def import_asset(
        self,
        kind: str,
        name: str,
        *,
        registry: "RegistryManager",
        source: str = "",
    ) -> DiscoveryReport:
        entry = self._repository.resolve(kind, name)
        if entry is None:
            telemetry.record("manifest_missing", kind=kind)
            return DiscoveryReport(
                issues=(
                    DiscoveryIssue(
                        code="manifest_missing",
                        kind=kind,
                        name=name,
                        detail=f"Manifest entry not found for {kind}:{name}",
                        source=source,
                        load_phase="unknown",
                    ),
                )
            )
        issues: list[DiscoveryIssue] = []
        self._import_entry(
            entry,
            registry=registry,
            imported_modules=set(),
            issues=issues,
            request=None,
        )
        if issues:
            return DiscoveryReport(issues=tuple(issues))
        if not _has_registry_entry(registry, kind, name):
            return DiscoveryReport(
                issues=(
                    DiscoveryIssue(
                        code="registration_missing",
                        kind=kind,
                        name=name,
                        detail=f"Imported module '{entry.module}' did not register '{kind}:{name}'",
                        source=source,
                        module=entry.module,
                        load_phase=entry.load_phase,
                    ),
                )
            )
        return DiscoveryReport(
            imported=(
                DiscoveryImport(
                    kind=kind,
                    name=name,
                    module=entry.module,
                    source=source,
                    load_phase=entry.load_phase,
                ),
            )
        )

    def _import_entry(
        self,
        entry: DiscoveryManifestEntry,
        *,
        registry: "RegistryManager",
        imported_modules: set[str],
        issues: list[DiscoveryIssue],
        request: "DiscoveryRequest | None",
    ) -> None:
        module_name = entry.module
        if module_name in imported_modules:
            telemetry.record("manifest_hit", kind=entry.kind)
            return
        try:
            module = importlib.import_module(module_name)
            if request is not None and not _has_registry_entry(registry, request.kind, request.name):
                module = importlib.reload(module)
            elif request is None and not _has_registry_entry(registry, entry.kind, entry.name):
                module = importlib.reload(module)
            imported_modules.add(module_name)
            telemetry.record("manifest_hit", kind=entry.kind)
        except Exception as exc:
            telemetry.record("module_import_failed", kind=entry.kind)
            issues.append(
                DiscoveryIssue(
                    code="module_import_failed",
                    kind=entry.kind,
                    name=entry.name,
                    detail=str(exc),
                    source="" if request is None else request.source,
                    module=module_name,
                    load_phase=entry.load_phase,
                )
            )


def import_kind_from_manifest(
    kind: str,
    *,
    registry,
    repository: DiscoveryManifestRepository | None = None,
) -> DiscoveryReport:
    return BootstrapImporter(repository).import_kind(kind, registry=registry)


def import_asset_from_manifest(
    kind: str,
    name: str,
    *,
    registry,
    repository: DiscoveryManifestRepository | None = None,
    source: str = "",
) -> DiscoveryReport:
    return BootstrapImporter(repository).import_asset(kind, name, registry=registry, source=source)


def _has_registry_entry(registry: "RegistryManager", kind: str, name: str) -> bool:
    try:
        registry.entry(kind, name)
        return True
    except KeyError:
        return False


def _load_phase_sort_key(load_phase: str) -> tuple[int, str]:
    phase = str(load_phase or "").strip().lower()
    if phase == "baseline":
        return (0, phase)
    if phase == "prepare_only":
        return (1, phase)
    return (2, phase)


def _sort_entries_by_phase(
    entries: tuple[DiscoveryManifestEntry, ...],
) -> tuple[DiscoveryManifestEntry, ...]:
    return tuple(sorted(entries, key=lambda entry: (_load_phase_sort_key(entry.load_phase), entry.name, entry.module)))
