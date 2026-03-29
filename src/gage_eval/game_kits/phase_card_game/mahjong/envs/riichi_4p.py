from __future__ import annotations

from typing import Any, Sequence

from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources
from gage_eval.role.arena.replay_paths import resolve_invocation_run_sample_ids
from gage_eval.role.arena.games.mahjong.env import MahjongArena
from gage_eval.role.arena.games.mahjong.formatters.mahjong import StandardMahjongFormatter
from gage_eval.role.arena.games.mahjong.parsers.mahjong import StandardMahjongParser
from gage_eval.role.arena.games.mahjong.renderers.mahjong import StandardMahjongRenderer

_ACTION_ID_TO_TEXT = {
    0: "B1",
    1: "C1",
    2: "D1",
    3: "East",
    4: "Red",
}
_ACTION_ID_TO_RAW = {
    0: "bamboo-1",
    1: "characters-1",
    2: "dots-1",
    3: "winds-east",
    4: "dragons-red",
}
_INITIAL_HANDS = {
    0: [
        "bamboo-1",
        "bamboo-2",
        "bamboo-3",
        "bamboo-4",
        "bamboo-5",
        "bamboo-9",
        "characters-3",
        "characters-4",
        "characters-5",
        "dots-2",
        "dots-3",
        "dots-4",
        "winds-south",
        "dragons-green",
    ],
    1: [
        "bamboo-2",
        "bamboo-3",
        "bamboo-4",
        "characters-2",
        "characters-3",
        "characters-4",
        "characters-5",
        "dots-5",
        "dots-6",
        "dots-7",
        "winds-north",
        "dragons-green",
        "dragons-white",
    ],
    2: [
        "bamboo-5",
        "bamboo-6",
        "bamboo-7",
        "characters-6",
        "characters-7",
        "characters-8",
        "dots-2",
        "dots-3",
        "dots-4",
        "dots-8",
        "dots-9",
        "winds-west",
        "dragons-white",
    ],
    3: [
        "bamboo-6",
        "bamboo-7",
        "bamboo-8",
        "characters-1",
        "characters-2",
        "characters-3",
        "dots-6",
        "dots-7",
        "dots-8",
        "winds-south",
        "winds-west",
        "winds-north",
        "dragons-green",
    ],
}
_TURN_DRAWS = {
    1: "characters-1",
    2: "dots-1",
    3: "winds-east",
    4: "dragons-red",
}

class _StubMahjongParser(StandardMahjongParser):
    def __init__(
        self,
        *,
        action_id_to_text: dict[int, str],
        action_id_to_raw: dict[int, str],
    ) -> None:
        self._action_id_to_text = dict(action_id_to_text)
        self._action_text_to_id = {
            text.lower(): action_id for action_id, text in action_id_to_text.items()
        }
        for action_id, raw_text in action_id_to_raw.items():
            self._action_text_to_id[str(raw_text).lower()] = int(action_id)


class _StubMahjongCore:
    num_players = 4

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active_player = 0
        self._terminal = False
        self._step_index = 0
        self._winner = None
        self._hands = {
            player_id: list(hand) for player_id, hand in _INITIAL_HANDS.items()
        }
        self._table: list[str] = []

    def step(self, action_id: int) -> None:
        if self._terminal:
            return
        if int(action_id) not in self.get_legal_actions(self._active_player):
            raise ValueError(f"Illegal action id {action_id}")

        actor = self._active_player
        raw_action = self.decode_action_raw(action_id)
        hand = list(self._hands[actor])
        if raw_action in hand:
            hand.remove(raw_action)
            self._hands[actor] = hand
        self._table.append(raw_action)
        self._step_index += 1
        if self._step_index >= 5:
            self._terminal = True
            self._winner = 0
            return
        self._active_player = self._step_index % self.num_players
        draw_tile = _TURN_DRAWS.get(self._step_index)
        if draw_tile is not None:
            self._hands[self._active_player].append(draw_tile)

    def get_active_player_id(self) -> int:
        return self._active_player

    def is_terminal(self) -> bool:
        return self._terminal

    def get_legal_actions(self, player_id: int | None = None) -> Sequence[int]:
        actor = self._active_player if player_id is None else int(player_id)
        if actor != self._active_player or self._terminal:
            return ()
        return (self._step_index,)

    def get_observation(self, player_id: int) -> dict[str, Any]:
        return {
            "current_hand": list(self._hands[int(player_id)]),
            "table": list(self._table),
        }

    def decode_action(self, action_id: int) -> str:
        return _ACTION_ID_TO_TEXT[int(action_id)]

    def decode_action_raw(self, action_id: int) -> str:
        return _ACTION_ID_TO_RAW[int(action_id)]

    def encode_action(self, action_text: str) -> int:
        normalized = str(action_text or "").strip().lower()
        for action_id, display in _ACTION_ID_TO_TEXT.items():
            if display.lower() == normalized:
                return action_id
        for action_id, raw in _ACTION_ID_TO_RAW.items():
            if raw.lower() == normalized:
                return action_id
        raise ValueError(f"Unknown action text: {action_text}")

    def get_payoffs(self) -> Sequence[float]:
        if not self._terminal or self._winner is None:
            return (0.0, 0.0, 0.0, 0.0)
        return tuple(1.0 if idx == self._winner else 0.0 for idx in range(self.num_players))

    def get_all_hands(self) -> dict[int, list[str]]:
        return {player_id: list(hand) for player_id, hand in self._hands.items()}


class Riichi4pEnvironment:
    def __init__(
        self,
        *,
        player_specs: Sequence[object],
        replay_output_dir: str | None,
        replay_filename: str | None,
        run_id: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        player_ids = [str(getattr(player, "player_id")) for player in player_specs]
        player_names = {
            str(getattr(player, "player_id")): str(getattr(player, "display_name"))
            for player in player_specs
        }
        self._adapter = MahjongArena(
            game_type="mahjong",
            core=_StubMahjongCore(),
            formatter=StandardMahjongFormatter(action_map=_ACTION_ID_TO_TEXT),
            parser=_StubMahjongParser(
                action_id_to_text=_ACTION_ID_TO_TEXT,
                action_id_to_raw=_ACTION_ID_TO_RAW,
            ),
            renderer=StandardMahjongRenderer(),
            player_ids=player_ids,
            player_names=player_names,
            replay_output_dir=str(replay_output_dir) if replay_output_dir else None,
            replay_filename=str(replay_filename or "mahjong_riichi_4p_replay.json"),
            run_id=run_id,
            sample_id=sample_id,
            replay_live=False,
            illegal_policy={"retry": 0, "on_fail": "loss"},
        )
        self._adapter._action_id_to_text = dict(_ACTION_ID_TO_TEXT)
        self._adapter._action_text_to_id = {
            text.lower(): action_id for action_id, text in _ACTION_ID_TO_TEXT.items()
        }
        self._adapter._action_id_to_raw = dict(_ACTION_ID_TO_RAW)

    @classmethod
    def from_runtime(cls, *, sample, resolved, resources, player_specs, invocation_context=None):
        defaults = {
            **dict(resolved.game_kit.defaults),
            **dict(resolved.env_spec.defaults),
            **dict(sample.runtime_overrides or {}),
        }
        run_id, sample_id = resolve_invocation_run_sample_ids(
            invocation_context=invocation_context,
            run_id=defaults.get("run_id"),
            sample_id=defaults.get("sample_id"),
        )
        environment = cls(
            player_specs=player_specs,
            replay_output_dir=defaults.get("replay_output_dir"),
            replay_filename=defaults.get("replay_filename"),
            run_id=run_id,
            sample_id=sample_id,
        )
        return attach_runtime_resources(environment, resources)

    def get_active_player(self) -> str:
        return self._adapter.get_active_player()

    def observe(self, player: str):
        return self._adapter.observe(player)

    def apply(self, action):
        return self._adapter.apply(action)

    def get_last_frame(self):
        return self._adapter.get_last_frame()

    def is_terminal(self) -> bool:
        return self._adapter.is_terminal()

    def build_result(self, *, result: str, reason: str | None):
        return self._adapter.build_result(result=result, reason=reason)


def build_riichi_4p_environment(*, sample, resolved, resources, player_specs, invocation_context=None) -> Any:
    return Riichi4pEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
        invocation_context=invocation_context,
    )
