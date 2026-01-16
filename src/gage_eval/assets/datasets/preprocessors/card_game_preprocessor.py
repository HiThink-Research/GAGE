"""Card game dataset preprocessor."""

from __future__ import annotations

from typing import Any, Dict, List

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor


class CardGamePreprocessor(BasePreprocessor):
    """Standardize card game records into the Sample schema."""

    name = "card_game_preprocessor"

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Normalize card game records to the Sample schema.

        Args:
            record: Raw dataset record.
            **kwargs: Optional overrides for player metadata.

        Returns:
            Normalized sample dictionary.
        """

        sample = dict(record)
        metadata = dict(sample.get("metadata") or {})

        player_ids = metadata.get("player_ids") or kwargs.get("player_ids")
        if player_ids is None:
            player_ids = ["player_0", "player_1", "player_2"]
        if isinstance(player_ids, dict):
            player_ids = list(player_ids.values())
        player_ids = [str(player_id) for player_id in player_ids if player_id]

        player_names = metadata.get("player_names") or kwargs.get("player_names") or {}
        if isinstance(player_names, list):
            player_names = {
                player_ids[idx]: name for idx, name in enumerate(player_names) if idx < len(player_ids)
            }
        if not isinstance(player_names, dict):
            player_names = {}
        for player_id in player_ids:
            player_names.setdefault(player_id, player_id)

        start_player_id = metadata.get("start_player_id") or kwargs.get("start_player_id")
        if start_player_id not in player_ids:
            for player_id, name in player_names.items():
                if name == start_player_id:
                    start_player_id = player_id
                    break
        if start_player_id not in player_ids:
            start_player_id = player_ids[0] if player_ids else "player_0"

        metadata.update(
            {
                "player_ids": list(player_ids),
                "player_names": dict(player_names),
                "start_player_id": start_player_id,
            }
        )

        sample.setdefault("schema_version", "gage.v1")
        sample.setdefault("task_type", "agent")
        sample.setdefault("messages", [])
        sample.setdefault("choices", [])
        sample.setdefault("references", [])
        sample.setdefault("data_tag", {})
        sample.setdefault("eval_config", {})
        sample.setdefault("inputs", {})
        sample["metadata"] = metadata
        return sample


__all__ = ["CardGamePreprocessor"]
