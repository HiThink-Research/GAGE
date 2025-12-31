from gage_eval.assets.datasets.preprocessors.builtin import GridGamePreprocessor
from gage_eval.registry import registry


def test_grid_game_preprocessor_fills_defaults():
    preprocessor = GridGamePreprocessor()
    record = {
        "id": "gomoku_1",
        "metadata": {"board_size": 9},
    }

    sample = dict(record)
    preprocessor.transform(sample, dataset_id="gomoku", dataset_metadata={"path": "tests"})

    assert sample["metadata"]["board_size"] == 9
    assert sample["metadata"]["win_len"] == 5
    assert sample["metadata"]["start_player_id"] == "Black"
    assert sample["metadata"]["player_ids"] == ["Black", "White"]
    assert isinstance(sample.get("messages"), list)
    assert isinstance(sample.get("choices"), list)


def test_grid_game_preprocessor_registry() -> None:
    pre_cls = registry.get("dataset_preprocessors", "grid_game_preprocessor")

    assert pre_cls is GridGamePreprocessor
