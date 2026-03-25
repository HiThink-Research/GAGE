from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from gage_eval.registry import registry
from gage_eval.role.arena.game_providers import (
    ArenaGameProvider,
    resolve_arena_game_provider,
)
from gage_eval.role.arena import registry_loader


def test_asset_search_roots_split_arena_domains() -> None:
    root = Path(registry_loader.__file__).resolve().parent

    arena_roots = registry_loader._asset_search_roots(root=root, kind="arena_impls")  # noqa: SLF001
    parser_roots = registry_loader._asset_search_roots(root=root, kind="parser_impls")  # noqa: SLF001
    renderer_roots = registry_loader._asset_search_roots(root=root, kind="renderer_impls")  # noqa: SLF001
    provider_roots = registry_loader._asset_search_roots(root=root, kind="arena_game_providers")  # noqa: SLF001

    assert {path.relative_to(root).as_posix() for path in arena_roots} == {"games"}
    assert {path.relative_to(root).as_posix() for path in parser_roots} == {"games", "parsers"}
    assert {path.relative_to(root).as_posix() for path in renderer_roots} == {"games"}
    assert {path.relative_to(root).as_posix() for path in provider_roots} == {"game_providers.py"}


def test_iter_asset_source_paths_skips_test_directories(tmp_path: Path) -> None:
    games_dir = tmp_path / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    (games_dir / "env.py").write_text("x = 1\n", encoding="utf-8")
    (games_dir / "tests").mkdir(parents=True, exist_ok=True)
    (games_dir / "tests" / "ignored.py").write_text("x = 2\n", encoding="utf-8")

    paths = list(
        registry_loader._iter_asset_source_paths(  # noqa: SLF001
            root=tmp_path,
            kind="arena_impls",
        )
    )

    assert games_dir / "env.py" in paths
    assert games_dir / "tests" / "ignored.py" not in paths


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
