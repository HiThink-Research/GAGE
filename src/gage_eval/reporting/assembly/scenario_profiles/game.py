from __future__ import annotations

from typing import Any


class GameScenarioProfile:
    profile_name = "game"

    def build(self, index: Any) -> dict[str, Any]:
        illegal_games = 0
        illegal_total = 0
        move_count = 0
        replay_refs: list[str] = []
        game_kits: set[str] = set()
        for sample in getattr(index, "samples", []) or []:
            metadata = (sample.get("sample") or sample).get("metadata") or {}
            game_kit = ((metadata.get("game_arena") or {}).get("game_kit"))
            if not game_kit:
                dataset_id = (sample.get("sample") or sample).get("_dataset_id") or (sample.get("sample") or sample).get("dataset_id")
                if isinstance(dataset_id, str) and "gomoku" in dataset_id.lower():
                    game_kit = "gomoku"
            if game_kit:
                game_kits.add(str(game_kit))
            judge = sample.get("judge_output") or {}
            illegal = int(judge.get("illegal_move_count") or judge.get("illegal_action_count") or 0)
            if illegal:
                illegal_games += 1
                illegal_total += illegal
            move_count += int(judge.get("move_count") or 0)
            for artifact in sample.get("artifact_refs", []) or []:
                if "replay" in str(artifact.get("name") or artifact.get("path") or ""):
                    replay_refs.append(f"evidence://{artifact.get('path')}")
        return {
            "profile_version": "gage.scenario.game.v1",
            "game_kits": sorted(game_kits),
            "illegal_actions": {"games": illegal_games, "total": illegal_total},
            "move_count": move_count,
            "replay_refs": sorted(set(replay_refs)),
        }
