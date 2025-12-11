import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        rid = kwargs.get("request_id")
        return {"outputs": [{"text": f"resp-{rid}"}]}


def make_backend(config=None):
    orig_load = LegacyVLLMBackend.load_model
    LegacyVLLMBackend.load_model = lambda self, cfg: DummyModel()
    try:
        backend = LegacyVLLMBackend(config or {"model_path": "repo"})
    finally:
        LegacyVLLMBackend.load_model = orig_load
    return backend


class LegacyVLLMBackendOutputTypeTests(unittest.TestCase):
    def test_sampling_params_loss(self):
        backend = make_backend()
        prepared = backend.prepare_inputs({"model_path": "repo", "output_type": "loss"})
        sp = prepared["sampling_params"]
        self.assertEqual(getattr(sp, "prompt_logprobs"), 1)
        self.assertEqual(getattr(sp, "max_tokens"), 1)
        self.assertEqual(getattr(sp, "temperature"), 0)

    def test_sampling_params_next_token_prob(self):
        backend = make_backend()
        prepared = backend.prepare_inputs(
            {
                "model_path": "repo",
                "output_type": "next_token_prob",
                "sampling_params": {"top_logprobs_num": 7},
            }
        )
        sp = prepared["sampling_params"]
        self.assertEqual(getattr(sp, "max_tokens"), 1)
        self.assertEqual(getattr(sp, "logprobs"), 7)

    def test_sample_n_request_ids(self):
        backend = make_backend()
        prepared = backend.prepare_inputs({"model_path": "repo", "output_type": "text", "sample_n": 3, "request_id": "req"})
        out = backend.generate(prepared)
        self.assertEqual(out["_sample_n"], 3)
        # DummyModel.generate called 3 times with derived request ids
        calls = backend.model.calls
        self.assertEqual(len(calls), 3)
        expected_ids = {"req_0", "req_1", "req_2"}
        self.assertEqual({c["request_id"] for c in calls}, expected_ids)
        # Answers should follow the derived ids
        self.assertEqual(out["answer"], ["resp-req_0", "resp-req_1", "resp-req_2"])


if __name__ == "__main__":
    unittest.main()
