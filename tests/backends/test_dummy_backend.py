from gage_eval.role.model.backends.dummy_backend import DummyBackend


def test_dummy_backend_random_seed_reproducible():
    config = {
        "responses": ["A", "B", "C"],
        "random": True,
        "seed": 123,
    }
    backend_a = DummyBackend(config)
    outputs_a = [backend_a.generate({})["answer"] for _ in range(5)]

    backend_b = DummyBackend(config)
    outputs_b = [backend_b.generate({})["answer"] for _ in range(5)]

    assert outputs_a == outputs_b
