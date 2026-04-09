from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import CompatMigrationShim


def resolve_compat_shim() -> CompatMigrationShim:
    """Resolve the Tau2 bootstrap migration shim."""

    return CompatMigrationShim(
        shim_id="tau2.bootstrap_to_runtime",
        legacy_source="role.context.tau2_bootstrap",
        target_runtime_id="tau2_runtime",
        target_benchmark_kit_id="tau2",
        migration_mode="support_to_runtime",
        warning_event="tau2_runtime_compat",
        removal_phase="phase_3",
    )
