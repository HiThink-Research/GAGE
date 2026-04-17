from __future__ import annotations

import importlib
from collections.abc import Generator
from pathlib import Path

import pytest

import gage_eval.config.smart_defaults.static_rules as static_smart_default_rules
from gage_eval.config.smart_defaults import SmartDefaultsError
from gage_eval.config.smart_defaults.profiles import select_smart_defaults_profile
from gage_eval.config.smart_defaults import registry as smart_defaults_registry
from gage_eval.config.smart_defaults.registry import apply_smart_defaults, registered_rules, smart_default
from gage_eval.config.smart_defaults.types import RuleContext


@pytest.fixture(autouse=True)
def restore_smart_defaults_registry() -> Generator[None, None, None]:
    original_rules = list(smart_defaults_registry._RULES)
    try:
        yield
    finally:
        smart_defaults_registry._RULES[:] = original_rules


def _rule_context(*, current_rule: str = "test_rule") -> RuleContext:
    ctx = RuleContext(source_path=Path("config.yaml"), cli_intent=None, scene="static")
    ctx.current_rule = current_rule
    return ctx


@pytest.mark.fast
@pytest.mark.parametrize(
    ("payload", "expected_scene"),
    [
        ({"kind": "PipelineConfig", "scene": "static"}, "static"),
        ({"kind": "PipelineConfig", "scene": "agent"}, "agent"),
        ({"kind": "PipelineConfig", "scene": "game"}, "game"),
        ({"kind": "PipelineConfig"}, "legacy"),
    ],
)
def test_select_smart_defaults_profile_by_scene(payload, expected_scene) -> None:
    profile = select_smart_defaults_profile(payload, Path("config.yaml"))

    assert profile.scene == expected_scene


@pytest.mark.fast
def test_select_smart_defaults_profile_rejects_unknown_scene() -> None:
    with pytest.raises(SmartDefaultsError, match="Unknown PipelineConfig scene 'arena'"):
        select_smart_defaults_profile({"kind": "PipelineConfig", "scene": "arena"}, Path("config.yaml"))


@pytest.mark.fast
def test_static_rule_registration_is_idempotent_across_module_reload() -> None:
    before_rules = registered_rules("static")
    before_rule_keys = tuple((rule.scene, rule.phase, rule.name) for rule in before_rules)

    importlib.reload(static_smart_default_rules)

    after_rules = registered_rules("static")
    after_rule_keys = tuple((rule.scene, rule.phase, rule.name) for rule in after_rules)
    after_rule_names = tuple(rule.name for rule in after_rules)

    assert after_rule_keys == before_rule_keys
    assert len(after_rules) == len(before_rules)
    assert len(after_rule_names) == len(set(after_rule_names))


@pytest.mark.fast
def test_duplicate_rule_registration_replaces_existing_rule_metadata_and_apply_function() -> None:
    @smart_default(
        scene="static",
        phase="contract",
        priority=10,
        name="replaceable_rule",
        description="Original description.",
    )
    def original_rule(payload: dict, ctx: RuleContext) -> None:
        payload["registered_value"] = "original"

    @smart_default(
        scene="static",
        phase="contract",
        priority=20,
        name="replaceable_rule",
        description="Replacement description.",
    )
    def replacement_rule(payload: dict, ctx: RuleContext) -> None:
        payload["registered_value"] = "replacement"

    matching_rules = [
        rule
        for rule in registered_rules("static")
        if (rule.scene, rule.phase, rule.name) == ("static", "contract", "replaceable_rule")
    ]

    assert len(matching_rules) == 1
    assert matching_rules[0].priority == 20
    assert matching_rules[0].description == "Replacement description."
    assert matching_rules[0].apply is replacement_rule

    payload: dict[str, str] = {}
    matching_rules[0].apply(payload, _rule_context())

    assert payload == {"registered_value": "replacement"}


@pytest.mark.fast
def test_non_pipeline_config_uses_legacy_profile() -> None:
    profile = select_smart_defaults_profile({"kind": "RunConfig", "scene": "static"}, Path("run.yaml"))

    assert profile.scene == "legacy"
    assert profile.rules == ()


@pytest.mark.fast
def test_rule_context_fill_preserves_explicit_values_without_trace() -> None:
    ctx = _rule_context()
    target = {"existing": "explicit"}

    ctx.fill(target, key="existing", value="default", path="existing")

    assert target == {"existing": "explicit"}
    assert ctx.traces == []


@pytest.mark.fast
def test_rule_context_migrate_moves_source_to_target_and_traces() -> None:
    ctx = _rule_context(current_rule="migrate_rule")
    source = {"legacy": "value", "other": "kept"}
    target: dict[str, str] = {}

    ctx.migrate(source, source_key="legacy", target=target, target_key="modern", path="modern")

    assert source == {"other": "kept"}
    assert target == {"modern": "value"}
    assert len(ctx.traces) == 1
    assert ctx.traces[0].rule == "migrate_rule"
    assert ctx.traces[0].action == "migrate"
    assert ctx.traces[0].path == "modern"


@pytest.mark.fast
def test_rule_context_migrate_preserves_explicit_target_and_removes_source() -> None:
    ctx = _rule_context()
    source = {"legacy": "default"}
    target = {"modern": "explicit"}

    ctx.migrate(source, source_key="legacy", target=target, target_key="modern", path="modern")

    assert source == {}
    assert target == {"modern": "explicit"}
    assert len(ctx.traces) == 1
    assert ctx.traces[0].action == "migrate"
    assert ctx.traces[0].path == "modern"


@pytest.mark.fast
def test_rule_context_replace_subtree_replaces_value_and_traces() -> None:
    ctx = _rule_context(current_rule="replace_rule")
    target = {"node": {"old": True}}

    ctx.replace_subtree(target, key="node", value={"new": True}, path="node")

    assert target == {"node": {"new": True}}
    assert len(ctx.traces) == 1
    assert ctx.traces[0].rule == "replace_rule"
    assert ctx.traces[0].action == "replace_subtree"
    assert ctx.traces[0].path == "node"


@pytest.mark.fast
def test_rule_context_fail_includes_provided_path_in_error_message() -> None:
    ctx = _rule_context()

    with pytest.raises(SmartDefaultsError, match="Cannot infer default at nested.value"):
        ctx.fail("Cannot infer default", path="nested.value")


@pytest.mark.fast
def test_registered_rule_accepts_description_payload_context_and_uses_caller_context() -> None:
    @smart_default(
        scene="static",
        phase="contract",
        priority=10,
        name="contract_rule",
        description="Exercise the public smart-default rule contract.",
    )
    def contract_rule(payload: dict, ctx: RuleContext) -> None:
        assert ctx.current_rule == "contract_rule"
        ctx.fill(payload, key="added", value=ctx.current_rule, path="added")

    raw = {"kind": "PipelineConfig", "scene": "static"}
    profile = select_smart_defaults_profile(raw, Path("config.yaml"))
    profile = type(profile)(scene=profile.scene, phases=("contract",), rules=profile.rules)
    ctx = RuleContext(source_path=Path("config.yaml"), cli_intent={"flag": True}, scene=profile.scene)

    expanded = apply_smart_defaults(raw, ctx, profile)

    assert raw == {"kind": "PipelineConfig", "scene": "static"}
    assert expanded["added"] == "contract_rule"
    assert ctx.source_path == Path("config.yaml")
    assert ctx.cli_intent == {"flag": True}
    assert ctx.current_rule == "contract_rule"
    assert ctx.traces[0].rule == "contract_rule"
    assert ctx.traces[0].action == "fill"
    assert ctx.traces[0].path == "added"


@pytest.mark.fast
def test_agent_and_game_profiles_are_empty_noop_even_with_registered_rules() -> None:
    @smart_default(
        scene="agent",
        phase="contract",
        priority=10,
        name="agent_rule",
        description="Should not run in Phase 0.",
    )
    def agent_rule(payload: dict, ctx: RuleContext) -> None:
        ctx.fill(payload, key="agent_added", value=True, path="agent_added")

    @smart_default(
        scene="game",
        phase="contract",
        priority=10,
        name="game_rule",
        description="Should not run in Phase 0.",
    )
    def game_rule(payload: dict, ctx: RuleContext) -> None:
        ctx.fill(payload, key="game_added", value=True, path="game_added")

    for scene in ("agent", "game"):
        raw = {"kind": "PipelineConfig", "scene": scene}
        profile = select_smart_defaults_profile(raw, Path(f"{scene}.yaml"))
        ctx = RuleContext(source_path=Path(f"{scene}.yaml"), cli_intent=None, scene=profile.scene)

        expanded = apply_smart_defaults(raw, ctx, profile)

        assert profile.phases == ()
        assert profile.rules == ()
        assert expanded == raw
        assert ctx.traces == []
        assert ctx.current_rule == "<unknown>"


@pytest.mark.fast
def test_legacy_no_scene_profile_is_empty_noop() -> None:
    raw = {"kind": "PipelineConfig", "metadata": {"name": "legacy"}}
    profile = select_smart_defaults_profile(raw, Path("legacy.yaml"))
    ctx = RuleContext(source_path=Path("legacy.yaml"), cli_intent=None, scene=profile.scene)

    expanded = apply_smart_defaults(raw, ctx, profile)

    assert profile.scene == "legacy"
    assert profile.phases == ()
    assert profile.rules == ()
    assert expanded == raw
    assert ctx.traces == []
