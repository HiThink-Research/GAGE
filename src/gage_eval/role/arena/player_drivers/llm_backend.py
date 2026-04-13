from __future__ import annotations

import json
from typing import Any

from gage_eval.role.arena.core.errors import PlayerExecutionUnavailableError
from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.players import BaseBoundPlayer, PlayerBindingSpec
from gage_eval.role.arena.player_drivers.base import PlayerDriver
from gage_eval.role.arena.types import ArenaAction, ArenaObservation
from gage_eval.utils.messages import stringify_message_content


class LLMBackendBoundPlayer(BaseBoundPlayer):
    def __init__(
        self,
        *,
        backend: Any,
        backend_id: str,
        sample_payload: dict[str, Any],
        max_retries: int = 0,
        fallback_policy: str = "first_legal",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._backend = backend
        self._backend_id = backend_id
        self._sample_payload = dict(sample_payload)
        self._max_retries = max(0, int(max_retries))
        self._fallback_policy = str(fallback_policy or "first_legal").strip().lower()

    def next_action(self, observation: ArenaObservation) -> ArenaAction:
        messages = self._build_messages(observation)
        last_text = ""
        for _ in range(self._max_retries + 1):
            result = self._backend.invoke(
                {
                    "sample": self._sample_payload,
                    "messages": messages,
                    "player_id": self.player_id,
                    "legal_moves": list(observation.legal_actions_items),
                }
            )
            last_text = _extract_text(result)
            move = _select_legal_move(last_text, observation.legal_actions_items)
            if move is not None:
                return ArenaAction(
                    player=self.player_id,
                    move=move,
                    raw=last_text,
                    metadata={
                        "driver_id": self.metadata.get("driver_id"),
                        "backend_id": self._backend_id,
                        "player_type": "llm",
                    },
                )
        fallback_move = _resolve_fallback_move(
            observation=observation,
            fallback_policy=self._fallback_policy,
        )
        if fallback_move is not None:
            return ArenaAction(
                player=self.player_id,
                move=fallback_move,
                raw=last_text,
                metadata={
                    "driver_id": self.metadata.get("driver_id"),
                    "backend_id": self._backend_id,
                    "player_type": "llm",
                    "fallback": self._fallback_policy,
                },
            )
        raise ValueError(
            f"LLM player '{self.player_id}' could not produce a legal move from backend '{self._backend_id}'"
        )

    def _build_messages(self, observation: ArenaObservation) -> list[dict[str, Any]]:
        base_messages = list(self._sample_payload.get("messages") or [])
        prompt = _format_observation_prompt(observation)
        return [*base_messages, {"role": "user", "content": prompt}]


class LLMBackendDriver(PlayerDriver):
    def bind(
        self,
        spec: PlayerBindingSpec,
        *,
        invocation: GameArenaInvocationContext | None = None,
    ) -> LLMBackendBoundPlayer:
        if spec.backend_id is None:
            raise ValueError(f"LLM player '{spec.player_id}' requires backend_id")
        if invocation is None or invocation.role_manager is None:
            raise PlayerExecutionUnavailableError(
                f"LLM player '{spec.player_id}' requires a runtime role_manager to resolve backend '{spec.backend_id}'"
            )
        get_backend = getattr(invocation.role_manager, "get_backend", None)
        if not callable(get_backend):
            raise PlayerExecutionUnavailableError(
                f"Runtime role_manager cannot resolve backend '{spec.backend_id}'"
            )
        backend = get_backend(spec.backend_id)
        if backend is None:
            raise PlayerExecutionUnavailableError(
                f"Backend '{spec.backend_id}' is not registered in the runtime"
            )
        params = self.resolve_params(spec)
        return LLMBackendBoundPlayer(
            player_id=spec.player_id,
            display_name=spec.player_id,
            seat=spec.seat,
            player_kind=spec.player_kind,
            backend=backend,
            backend_id=spec.backend_id,
            sample_payload=dict(invocation.sample_payload),
            max_retries=int(params.get("max_retries", 0)),
            fallback_policy=str(params.get("fallback_policy", "first_legal")),
            metadata={
                "driver_id": self.driver_id,
                "seat": spec.seat,
                "backend_id": spec.backend_id,
            },
        )


def _format_observation_prompt(observation: ArenaObservation) -> str:
    legal_moves = ", ".join(str(item) for item in observation.legal_actions_items) or "none"
    return (
        f"Active player: {observation.active_player}\n"
        f"Board:\n{observation.view_text}\n"
        f"Legal moves: {legal_moves}\n"
        "Return exactly one legal move."
    )


def _extract_text(output: Any) -> str:
    if isinstance(output, dict):
        if isinstance(output.get("answer"), str):
            return output["answer"]
        if isinstance(output.get("text"), str):
            return output["text"]
        message = output.get("message")
        if isinstance(message, dict):
            return stringify_message_content(message.get("content"))
        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                return stringify_message_content(last.get("content"))
    return "" if output is None else str(output)


def _select_legal_move(raw_text: str, legal_moves) -> str | None:
    legal = [str(move).strip() for move in legal_moves if str(move).strip()]
    if not legal:
        stripped = str(raw_text).strip()
        return stripped or None
    candidates: list[str] = []
    stripped = str(raw_text).strip()
    if stripped:
        candidates.append(stripped)
        candidates.extend(line.strip() for line in stripped.splitlines() if line.strip())
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("action", "move", "answer", "text"):
                value = payload.get(key)
                if value is not None:
                    candidates.append(str(value).strip())
    normalized = {move.lower(): move for move in legal}
    for candidate in candidates:
        token = candidate.strip().strip("`").strip("\"").strip("'")
        resolved = normalized.get(token.lower())
        if resolved is not None:
            return resolved
    return None


def _resolve_fallback_move(
    *,
    observation: ArenaObservation,
    fallback_policy: str,
) -> str | None:
    if fallback_policy != "first_legal":
        return None
    legal_moves = list(observation.legal_actions_items)
    if not legal_moves:
        return None
    return str(legal_moves[0])
