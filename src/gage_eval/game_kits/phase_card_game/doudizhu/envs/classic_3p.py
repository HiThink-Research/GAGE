from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gage_eval.game_kits.phase_card_game.doudizhu.environment import GenericCardArena
from gage_eval.game_kits.phase_card_game.doudizhu.formatters.doudizhu import (
    DoudizhuFormatter,
)
from gage_eval.game_kits.phase_card_game.doudizhu.parsers.doudizhu import (
    DoudizhuMoveParser,
)
from gage_eval.game_kits.phase_card_game.doudizhu.renderers.doudizhu import (
    DoudizhuRenderer,
)
from gage_eval.role.arena.replay_paths import (
    resolve_invocation_run_sample_ids,
    resolve_replay_manifest_path,
)
from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult

_ACTION_ID_TO_TEXT = {
    0: "pass",
    1: "3",
    2: "4",
}
_ACTION_TEXT_TO_ID = {text: action_id for action_id, text in _ACTION_ID_TO_TEXT.items()}

class _StubDoudizhuCore:
    num_players = 3

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active_player = 0
        self._terminal = False
        self._step_index = 0
        self._winner = None
        self._hands = {
            0: "33456789TJQKABR22",
            1: "55667789TJQKA2BR3",
            2: "44778899TJQKA22BR",
        }
        self._played_cards = {0: "", 1: "", 2: ""}
        self._trace: list[dict[str, Any]] = []

    def step(self, action_id: int) -> None:
        if self._terminal:
            return
        expected_actions = self.get_legal_actions(self._active_player)
        if int(action_id) not in expected_actions:
            raise ValueError(f"Illegal action id {action_id}")

        action_text = self.decode_action(action_id)
        actor = self._active_player
        if action_text != "pass":
            hand = self._hands[actor]
            self._hands[actor] = hand.replace(action_text, "", 1)
            self._played_cards[actor] += action_text
        self._trace.append({"player": actor, "action": action_text})
        self._step_index += 1
        if self._step_index >= 4:
            self._terminal = True
            self._winner = 0
            return
        self._active_player = [1, 2, 0][self._step_index - 1]

    def get_active_player_id(self) -> int:
        return self._active_player

    def is_terminal(self) -> bool:
        return self._terminal

    def get_legal_actions(self, player_id: int | None = None) -> Sequence[int]:
        actor = self._active_player if player_id is None else int(player_id)
        if actor != self._active_player or self._terminal:
            return ()
        if self._step_index == 0:
            return (1,)
        if self._step_index in (1, 2):
            return (0,)
        if self._step_index == 3:
            return (2,)
        return ()

    def get_observation(self, player_id: int) -> dict[str, Any]:
        return {
            "landlord": 0,
            "played_cards": [self._played_cards[idx] for idx in range(self.num_players)],
            "num_cards_left": [len(self._hands[idx]) for idx in range(self.num_players)],
            "seen_cards": "".join(self._played_cards[idx] for idx in range(self.num_players)),
            "trace": list(self._trace),
            "self": int(player_id),
            "current_hand": self._hands[int(player_id)],
        }

    def decode_action(self, action_id: int) -> str:
        return _ACTION_ID_TO_TEXT[int(action_id)]

    def encode_action(self, action_text: str) -> int:
        normalized = str(action_text or "").strip().lower()
        if normalized not in _ACTION_TEXT_TO_ID:
            raise ValueError(f"Unknown action text: {action_text}")
        return _ACTION_TEXT_TO_ID[normalized]

    def get_payoffs(self) -> Sequence[float]:
        if not self._terminal or self._winner is None:
            return (0.0, 0.0, 0.0)
        return tuple(1.0 if idx == self._winner else 0.0 for idx in range(self.num_players))

    def get_all_hands(self) -> dict[str, str]:
        return {str(player_id): hand for player_id, hand in self._hands.items()}


class Classic3pEnvironment:
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
        player_id_map = {index: player_id for index, player_id in enumerate(player_ids)}
        self._replay_output_dir = str(replay_output_dir) if replay_output_dir else None
        self._replay_filename = str(replay_filename or "doudizhu_classic_3p_replay.json")
        self._run_id = str(run_id) if run_id else None
        self._sample_id = str(sample_id) if sample_id else None
        self._adapter = GenericCardArena(
            game_type="doudizhu",
            core=_StubDoudizhuCore(),
            formatter=DoudizhuFormatter(
                player_id_map=player_id_map,
                action_id_to_text=_ACTION_ID_TO_TEXT,
            ),
            parser=DoudizhuMoveParser(action_text_to_id=_ACTION_TEXT_TO_ID),
            renderer=DoudizhuRenderer(),
            player_ids=player_ids,
            illegal_action_policy="reject",
        )
        self._adapter.reset()
        self._final_result: GameResult | None = None

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

    def observe(self, player: str) -> ArenaObservation:
        observation = self._adapter.observe(player)
        public_state = dict(observation.get("public_state", {}))
        private_state = dict(observation.get("private_state", {}))
        legal_moves = [str(move) for move in observation.get("legal_moves", ())]
        board_text = "\n".join(
            [
                "Public State:",
                json.dumps(public_state, ensure_ascii=True),
                "",
                "Private State:",
                json.dumps(private_state, ensure_ascii=True),
                "",
                f"Legal Moves (preview): {', '.join(legal_moves) if legal_moves else 'none'}",
            ]
        )
        metadata = {
            "public_state": public_state,
            "private_state": private_state,
            "player_id": observation.get("player_id"),
            "active_player_id": observation.get("active_player_id"),
            "chat_log": list(observation.get("chat_log", ())),
        }
        return ArenaObservation(
            board_text=board_text,
            legal_moves=legal_moves,
            active_player=str(observation.get("active_player_id")),
            metadata=metadata,
            view={"text": board_text},
            legal_actions={"items": legal_moves},
            context={"mode": "turn"},
        )

    def apply(self, action: ArenaAction | Mapping[str, Any] | str) -> GameResult | None:
        payload = action
        if isinstance(action, ArenaAction):
            payload = {
                "player_id": action.player,
                "action": action.move,
                "raw": action.raw,
            }
        result = self._adapter.apply(payload)
        if result is None:
            return None
        return self._finalize_result(result_payload=result, result_label=None, reason=result.get("reason"))

    def get_last_frame(self):
        return self._adapter.get_last_frame()

    def is_terminal(self) -> bool:
        return self._adapter.is_terminal()

    def build_result(self, *, result: str, reason: str | None) -> GameResult:
        if self._final_result is not None:
            return self._final_result
        resolved_reason = str(reason or result or "completed")
        payload = self._adapter.build_result(reason=resolved_reason)
        return self._finalize_result(
            result_payload=payload,
            result_label=result,
            reason=resolved_reason,
        )

    def _finalize_result(
        self,
        *,
        result_payload: Mapping[str, Any],
        result_label: str | None,
        reason: str | None,
    ) -> GameResult:
        if self._final_result is not None:
            return self._final_result

        winner = result_payload.get("winner")
        final_board = str(result_payload.get("final_board_html", ""))
        replay_path = self._write_replay(final_board)
        resolved_result = str(result_label or "").strip()
        if not resolved_result or resolved_result == "completed":
            resolved_result = "win" if winner else "draw"

        self._final_result = GameResult(
            winner=str(winner) if winner is not None else None,
            result=resolved_result,
            reason=str(reason or result_payload.get("reason") or "completed"),
            move_count=len(list(result_payload.get("move_log", ()))),
            illegal_move_count=0,
            final_board=final_board,
            move_log=tuple(dict(move) for move in result_payload.get("move_log", ())),
            rule_profile="doudizhu",
            replay_path=replay_path,
        )
        return self._final_result

    def _write_replay(self, serialized_replay: str) -> str:
        output_path = resolve_replay_manifest_path(
            run_id=self._run_id,
            sample_id=self._sample_id,
            output_dir=self._replay_output_dir,
        )
        if output_path is None:
            raise RuntimeError("doudizhu_replay_output_path_unresolved")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(serialized_replay)
        except json.JSONDecodeError:
            payload = {"raw_replay": serialized_replay}
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return str(output_path)


def build_classic_3p_environment(*, sample, resolved, resources, player_specs, invocation_context=None) -> Any:
    return Classic3pEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
        invocation_context=invocation_context,
    )
