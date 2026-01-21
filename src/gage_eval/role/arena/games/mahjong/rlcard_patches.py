"""Local RLCard Mahjong patches used by GAGE."""

from __future__ import annotations

from typing import Any, Optional

from rlcard.games.mahjong.judger import MahjongJudger
from rlcard.games.mahjong.round import MahjongRound


class PatchedMahjongJudger(MahjongJudger):
    """Mahjong judger that allows using a pair from 3/4-of-a-kind tiles."""

    def judge_hu(self, player: Any) -> tuple[bool, int]:  # type: ignore[override]
        """Judge whether the player has a winning hand under simplified rules.

        Args:
            player: RLCard Mahjong player instance.

        Returns:
            Tuple of (is_win, max_set_count).
        """

        # STEP 1: Keep the original simplified rule: 4 sets + 1 pair.
        set_count = len(player.pile)
        if set_count >= 4:
            return True, set_count

        hand = [card.get_str() for card in player.hand]
        count_dict = {card: hand.count(card) for card in hand}
        used = []
        maximum = 0

        # STEP 2: Try every possible pair candidate (allowing counts >= 2).
        for each in count_dict:
            if each in used:
                continue
            if count_dict[each] < 2:
                continue
            tmp_hand = hand.copy()
            for _ in range(2):
                tmp_hand.pop(tmp_hand.index(each))
            tmp_set_count, _set = self.cal_set(tmp_hand)
            used.extend(_set)
            maximum = max(maximum, tmp_set_count + set_count)
            if tmp_set_count + set_count >= 4:
                return True, maximum

        return False, maximum


class PatchedMahjongRound(MahjongRound):
    """Mahjong round that checks chow when no pong/gong is available."""

    def proceed_round(self, players: list[Any], action: Any) -> None:  # type: ignore[override]
        """Advance the round with chow fallback after a discard.

        Args:
            players: List of Mahjong players.
            action: The action taken by the current player.
        """

        # STEP 1: Handle stand and special actions.
        if action == "stand":
            valid_act, player, cards = self.judger.judge_chow(
                self.dealer,
                players,
                self.last_player,
            )
            if valid_act:
                self.valid_act = valid_act
                self.last_cards = cards
                self.last_player = self.current_player
                self.current_player = player.player_id
            else:
                self.last_player = self.current_player
                self.current_player = (self.player_before_act + 1) % 4
                self.dealer.deal_cards(players[self.current_player], 1)
                self.valid_act = False
            return

        if action == "gong":
            players[self.current_player].gong(self.dealer, self.last_cards)
            self.last_player = self.current_player
            self.valid_act = False
            return

        if action == "pong":
            players[self.current_player].pong(self.dealer, self.last_cards)
            self.last_player = self.current_player
            self.valid_act = False
            return

        if action == "chow":
            players[self.current_player].chow(self.dealer, self.last_cards)
            self.last_player = self.current_player
            self.valid_act = False
            return

        # STEP 2: Handle normal discard flow with chow fallback.
        players[self.current_player].play_card(self.dealer, action)
        self.player_before_act = self.current_player
        self.last_player = self.current_player
        valid_act, player, cards = self.judger.judge_pong_gong(
            self.dealer,
            players,
            self.last_player,
        )
        if valid_act:
            self.valid_act = valid_act
            self.last_cards = cards
            self.last_player = self.current_player
            self.current_player = player.player_id
            return

        chow_act, chow_player, chow_cards = self.judger.judge_chow(
            self.dealer,
            players,
            self.last_player,
        )
        if chow_act:
            self.valid_act = chow_act
            self.last_cards = chow_cards
            self.last_player = self.current_player
            self.current_player = chow_player.player_id
            return

        self.last_player = self.current_player
        self.current_player = (self.current_player + 1) % 4
        self.dealer.deal_cards(players[self.current_player], 1)
        self.valid_act = False


def patch_rlcard_game(game: Optional[Any]) -> None:
    """Patch an RLCard Mahjong game instance with local fixes.

    Args:
        game: RLCard MahjongGame instance.
    """

    if not game or getattr(game, "_gage_patched", False):
        return

    judger = getattr(game, "judger", None)
    if judger is not None and not isinstance(judger, PatchedMahjongJudger):
        game.judger = PatchedMahjongJudger(getattr(game, "np_random", None))

    round_obj = getattr(game, "round", None)
    if round_obj is not None and not isinstance(round_obj, PatchedMahjongRound):
        new_round = PatchedMahjongRound(
            game.judger,
            getattr(game, "dealer", None),
            getattr(game, "num_players", 4),
            getattr(game, "np_random", None),
        )
        new_round.__dict__.update(round_obj.__dict__)
        new_round.judger = game.judger
        new_round.dealer = getattr(game, "dealer", None)
        new_round.num_players = getattr(game, "num_players", 4)
        new_round.np_random = getattr(game, "np_random", None)
        game.round = new_round

    game._gage_patched = True


__all__ = ["PatchedMahjongJudger", "PatchedMahjongRound", "patch_rlcard_game"]
