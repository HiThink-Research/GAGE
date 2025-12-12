import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vlm_transformers_backend import VLMTransformersBackend


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def dim(self):
        return 2

    def unsqueeze(self, dim):
        return self

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            row, col = idx
            rows = self.data if isinstance(row, slice) else [self.data[row]]
            sliced = [r[col] for r in rows]
            return FakeTensor(sliced)
        val = self.data[idx]
        return FakeTensor(val) if isinstance(val, list) else val

    def tolist(self):
        return self.data


class DummyModel:
    def generate(self, **kwargs):
        class R:
            sequences = kwargs["input_ids"]

        return R()


def make_backend(config=None):
    orig_load = VLMTransformersBackend.load_model
    VLMTransformersBackend.load_model = lambda self, cfg: DummyModel()
    try:
        cfg = {
            "model_path": "repo",
            "model_name_or_path": "repo",
            "processor_name_or_path": "repo",
            "use_chat_template_vlm": "never",
        }
        if config:
            cfg.update(config)
        backend = VLMTransformersBackend(cfg)
        backend.processor = type("P", (), {"tokenizer": type("T", (), {"pad_token_id": 0, "eos_token_id": 0})()})()
        backend.model = DummyModel()
        backend.device = "cpu"
        backend._torch = __import__("types")
        backend._torch.inference_mode = lambda: __import__("contextlib").nullcontext()
        backend._torch.tensor = lambda x, device=None: x
        backend._torch.ones_like = lambda x, device=None: x
        backend._prepare_model_inputs = lambda *args, **kwargs: ({"input_ids": FakeTensor([[1, 2]])}, 0, 0)
        backend._base_generation_kwargs = {}
        backend._default_stop_sequences = []
        backend.processor = type(
            "P",
            (),
            {
                "tokenizer": type("T", (), {"pad_token_id": 0, "eos_token_id": 0})(),
                "batch_decode": lambda self, tokens, skip_special_tokens=True: ["decoded"],
            },
        )()
    finally:
        VLMTransformersBackend.load_model = orig_load
    return backend


class VLMTransformersBackendConflictTests(unittest.TestCase):
    def test_conflict_raises(self):
        backend = make_backend({"model_path": "repo", "processor_name_or_path": "backend_tok"})
        payload = {"_tokenizer_path": "dataset_tok", "messages": [{"role": "user", "content": "hi"}]}
        with self.assertRaises(ValueError):
            backend.generate(payload)

    def test_dataset_only(self):
        backend = make_backend({"model_path": "repo", "processor_name_or_path": "repo"})
        payload = {"_tokenizer_path": "repo", "messages": [{"role": "user", "content": "hi"}]}
        out = backend.generate(payload)
        self.assertIn("answer", out)


if __name__ == "__main__":
    unittest.main()
