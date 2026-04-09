from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import CompatMigrationShim


def resolve_compat_shim() -> CompatMigrationShim:
    """Resolve the AppWorld legacy hook migration shim."""

    return CompatMigrationShim(
        shim_id="appworld.hooks_to_runtime",
        legacy_source="sandbox.integrations.appworld.hooks",
        target_runtime_id="appworld_runtime",
        target_benchmark_kit_id="appworld",
        migration_mode="hook_to_runtime",
        warning_event="appworld_runtime_compat",
        removal_phase="phase_3",
    )
