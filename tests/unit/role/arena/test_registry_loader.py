from __future__ import annotations

from uuid import uuid4

from gage_eval.registry import load_default_manifest_repository, registry
from gage_eval.role.arena.game_providers import (
    ArenaGameProvider,
    resolve_arena_game_provider,
)
from gage_eval.role.arena import registry_loader


def test_manifest_repository_exposes_split_arena_domains() -> None:
    repository = load_default_manifest_repository()

    arena_entry = repository.require("arena_impls", "gomoku_local_v1")
    parser_entry = repository.require("parser_impls", "grid_parser_v1")
    renderer_entry = repository.require("renderer_impls", "gomoku_board_v1")
    provider_entry = repository.require("arena_game_providers", "grid_board")

    assert arena_entry.module == "gage_eval.role.arena.games.gomoku.env"
    assert parser_entry.module.startswith("gage_eval.role.arena.parsers.")
    assert renderer_entry.module.startswith("gage_eval.role.arena.games.gomoku.")
    assert provider_entry.module == "gage_eval.role.arena.game_providers"


def test_import_all_arena_asset_modules_uses_manifest_entries() -> None:
    registry_loader.import_all_arena_asset_modules("arena_game_providers")

    assert registry.entry("arena_game_providers", "grid_board").name == "grid_board"
    assert registry.entry("arena_game_providers", "vizdoom").name == "vizdoom"


def test_resolve_arena_game_provider_uses_registry_assets() -> None:
    provider_name = f"test_provider_{uuid4().hex}"

    class _Provider(ArenaGameProvider):
        def matches(self, env_impl: str) -> bool:
            return env_impl == "ext_env_v1"

    registry.register(
        "arena_game_providers",
        provider_name,
        _Provider,
        desc="test arena provider",
    )

    provider = resolve_arena_game_provider("ext_env_v1")

    assert isinstance(provider, _Provider)
