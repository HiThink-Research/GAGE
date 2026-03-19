"""Registry-backed pipeline step contracts shared by config and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from gage_eval.pipeline.steps.base import StepKind
from gage_eval.registry import RegistryEntry, registry


@dataclass(frozen=True, slots=True)
class StepContract:
    """Normalized contract describing a configured pipeline step."""

    step_type: str
    step_kind: StepKind
    requires_adapter: bool = False
    executor_name: Optional[str] = None
    allow_multiple: bool = True
    prerequisite_step_types: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StepSequenceIssue:
    """Structured validation issue for a configured step sequence."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class _StepDefaults:
    requires_adapter: bool = False
    executor_name: Optional[str] = None
    allow_multiple: bool = True
    prerequisite_step_types: tuple[str, ...] = ()


_STEP_DEFAULTS: Dict[str, _StepDefaults] = {
    "support": _StepDefaults(
        requires_adapter=True,
        executor_name="execute_support_step",
        allow_multiple=True,
    ),
    "inference": _StepDefaults(
        requires_adapter=True,
        executor_name="execute_inference",
        allow_multiple=False,
    ),
    "arena": _StepDefaults(
        requires_adapter=True,
        executor_name="execute_arena",
        allow_multiple=False,
    ),
    "judge": _StepDefaults(
        requires_adapter=True,
        executor_name="execute_judge",
        allow_multiple=False,
    ),
    "auto_eval": _StepDefaults(
        executor_name="execute_auto_eval",
        allow_multiple=False,
        prerequisite_step_types=("inference", "arena", "judge"),
    ),
    "report": _StepDefaults(
        executor_name=None,
        allow_multiple=False,
    ),
}


class StepContractCatalog:
    """Read-only mapping of step types to normalized contracts."""

    def __init__(self, contracts: Mapping[str, StepContract]) -> None:
        self._contracts = dict(contracts)

    def get(self, step_type: Optional[str]) -> Optional[StepContract]:
        if not step_type:
            return None
        return self._contracts.get(step_type)

    def require(self, step_type: Optional[str]) -> StepContract:
        contract = self.get(step_type)
        if contract is None:
            raise KeyError(f"Unknown pipeline step '{step_type}'")
        return contract

    def list(self) -> Sequence[StepContract]:
        return tuple(self._contracts.values())


def get_step_contract_catalog(registry_view=None) -> StepContractCatalog:
    if registry_view is None:
        return _load_step_contract_catalog()
    cache = registry_view.get_scoped_cache("step_contract_catalog")
    catalog = cache.get("catalog")
    if catalog is None:
        catalog = _build_catalog_from_entries(registry_view.list("pipeline_steps"))
        cache["catalog"] = catalog
    return catalog


def clear_step_contract_catalog_cache(registry_view=None) -> None:
    if registry_view is not None:
        registry_view.get_scoped_cache("step_contract_catalog").clear()
        return
    _load_step_contract_catalog.cache_clear()


@lru_cache(maxsize=1)
def _load_step_contract_catalog() -> StepContractCatalog:
    registry.auto_discover("pipeline_steps", "gage_eval.pipeline.steps", mode="warn")
    return _build_catalog_from_entries(registry.list("pipeline_steps"))


def _build_catalog_from_entries(entries: Sequence[RegistryEntry]) -> StepContractCatalog:
    contracts = {
        entry.name: _build_contract(entry)
        for entry in entries
    }
    return StepContractCatalog(contracts)


def get_step_type(step: Any) -> Optional[str]:
    if hasattr(step, "step_type"):
        return getattr(step, "step_type")
    if hasattr(step, "get"):
        return step.get("step") or step.get("step_type")
    return None


def get_step_adapter_id(step: Any) -> Optional[str]:
    if hasattr(step, "adapter_id"):
        return getattr(step, "adapter_id")
    if hasattr(step, "get"):
        value = step.get("adapter_id") or step.get("role_ref")
        return str(value) if value is not None else None
    return None


def collect_step_sequence_issues(
    steps: Sequence[Any],
    *,
    owner_label: str,
    adapter_ids: Optional[Iterable[str]] = None,
    registry_view=None,
) -> tuple[StepSequenceIssue, ...]:
    """Collect contract violations for an ordered step sequence."""

    catalog = get_step_contract_catalog(registry_view=registry_view)
    adapter_set = {str(adapter_id) for adapter_id in adapter_ids} if adapter_ids is not None else None
    seen_counts: Dict[str, int] = {}
    seen_prerequisites: set[str] = set()
    issues: list[StepSequenceIssue] = []

    for idx, step in enumerate(steps):
        step_type = get_step_type(step)
        location = f"{owner_label}[{idx}]"
        contract = catalog.get(step_type)
        if contract is None:
            issues.append(
                StepSequenceIssue(
                    code="unregistered_step",
                    message=f"{location} uses unregistered step '{step_type}'",
                )
            )
            continue

        seen_counts[step_type] = seen_counts.get(step_type, 0) + 1

        if contract.step_kind is not StepKind.SAMPLE:
            issues.append(
                StepSequenceIssue(
                    code="global_step",
                    message=f"{location} uses global step '{step_type}' which cannot run inside sample steps",
                )
            )
        elif contract.executor_name is None:
            issues.append(
                StepSequenceIssue(
                    code="missing_executor",
                    message=f"{location} step '{step_type}' has no sample executor mapping",
                )
            )

        adapter_id = get_step_adapter_id(step)
        if contract.requires_adapter and not adapter_id and step_type != "inference":
            issues.append(
                StepSequenceIssue(
                    code="missing_adapter",
                    message=f"{location} step '{step_type}' requires adapter_id",
                )
            )
        if adapter_set is not None and adapter_id and adapter_id not in adapter_set:
            issues.append(
                StepSequenceIssue(
                    code="unknown_adapter",
                    message=f"{location} references unknown role adapter '{adapter_id}'",
                )
            )
        if contract.prerequisite_step_types and not (
            seen_prerequisites & set(contract.prerequisite_step_types)
        ):
            required = "/".join(contract.prerequisite_step_types)
            issues.append(
                StepSequenceIssue(
                    code="missing_prerequisite",
                    message=f"{location} step '{step_type}' requires a preceding {required} step",
                )
            )
        if step_type in {"inference", "arena", "judge"}:
            seen_prerequisites.add(step_type)

    for step_type, count in seen_counts.items():
        contract = catalog.require(step_type)
        if count > 1 and not contract.allow_multiple:
            issues.append(
                StepSequenceIssue(
                    code="duplicate_singleton",
                    message=(
                        f"{owner_label} declares step '{step_type}' {count} times, "
                        "but only one occurrence is supported"
                    ),
                )
            )

    return tuple(issues)


def _build_contract(entry: RegistryEntry) -> StepContract:
    defaults = _STEP_DEFAULTS.get(entry.name, _StepDefaults())
    extra = dict(entry.extra)
    step_kind = _coerce_step_kind(extra.get("step_kind"), default=defaults, step_type=entry.name)
    return StepContract(
        step_type=entry.name,
        step_kind=step_kind,
        requires_adapter=_coerce_bool(
            extra.get("requires_adapter"),
            defaults.requires_adapter,
        ),
        executor_name=_coerce_optional_str(
            extra.get("executor_name", extra.get("executor")),
            defaults.executor_name,
        ),
        allow_multiple=_coerce_bool(
            extra.get("allow_multiple"),
            defaults.allow_multiple,
        ),
        prerequisite_step_types=_coerce_step_tuple(
            extra.get("prerequisite_step_types", extra.get("prerequisites")),
            defaults.prerequisite_step_types,
        ),
    )


def _coerce_step_kind(
    raw: Any,
    *,
    default: _StepDefaults,
    step_type: str,
) -> StepKind:
    if raw is None:
        if step_type == "report":
            return StepKind.GLOBAL
        return StepKind.SAMPLE
    if isinstance(raw, StepKind):
        return raw
    value = str(raw).strip().lower()
    if value == StepKind.SAMPLE.value:
        return StepKind.SAMPLE
    if value == StepKind.GLOBAL.value:
        return StepKind.GLOBAL
    raise ValueError(f"Unsupported step_kind '{raw}' for pipeline step '{step_type}'")


def _coerce_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _coerce_optional_str(raw: Any, default: Optional[str]) -> Optional[str]:
    if raw is None:
        return default
    value = str(raw).strip()
    return value or None


def _coerce_step_tuple(raw: Any, default: Iterable[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(default)
    if isinstance(raw, str):
        value = raw.strip()
        return (value,) if value else ()
    return tuple(str(item).strip() for item in raw if str(item).strip())
