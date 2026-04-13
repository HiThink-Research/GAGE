from __future__ import annotations

from pathlib import Path

from gage_eval.evaluation.cache import EvalCache
from gage_eval.reporting.summary_generators.gomoku import GomokuSummaryGenerator


def test_gomoku_summary_ignores_non_gomoku_arena_records(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="gomoku-summary-non-gomoku")
    cache.write_sample(
        "doudizhu-1",
        {
            "sample": {
                "id": "doudizhu-1",
                "metadata": {
                    "game_arena": {
                        "game_kit": "doudizhu",
                    }
                },
            },
            "model_output": {
                "sample": {"game_kit": "doudizhu", "env": "classic_3p"},
                "winner": "landlord",
                "result": "win",
                "move_count": 4,
            },
        },
    )

    assert GomokuSummaryGenerator().generate(cache) is None


def test_gomoku_summary_detects_explicit_gomoku_records(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="gomoku-summary-gomoku")
    cache.write_sample(
        "gomoku-1",
        {
            "sample": {
                "id": "gomoku-1",
                "metadata": {
                    "player_ids": ["Black", "White"],
                    "game_arena": {
                        "game_kit": "gomoku",
                    },
                },
            },
            "model_output": {
                "sample": {"game_kit": "gomoku", "env": "gomoku_standard"},
                "winner": "Black",
                "result": "win",
                "move_count": 5,
                "illegal_move_count": 0,
            },
        },
    )

    summary = GomokuSummaryGenerator().generate(cache)

    assert summary is not None
    assert summary["gomoku_summary"]["overall"]["total"] == 1
    assert summary["gomoku_summary"]["wins"] == {"Black": 1, "White": 0}
