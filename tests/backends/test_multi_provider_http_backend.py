import asyncio
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import gage_eval.role.model.backends.multi_provider_http_backend as mp_backend
from gage_eval.role.model.backends import wrap_backend
from gage_eval.role.model.backends.multi_provider_http_backend import MultiProviderHTTPBackend


class FakeAsyncClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "pong"}}]}


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _name):
        return cls()


class MultiProviderHTTPBackendTests(unittest.TestCase):
    def test_multi_provider_uses_defaults_and_async(self):
        fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)
        with mock.patch.object(mp_backend, "AsyncInferenceClient", FakeAsyncClient), mock.patch.dict(
            sys.modules, {"transformers": fake_transformers}
        ):
            backend = MultiProviderHTTPBackend({"provider": "together", "model_name": "stub-model"})
            wrapped = wrap_backend(backend)

            result = asyncio.run(wrapped.ainvoke({"messages": [{"role": "user", "content": "hi"}]}))

            self.assertEqual(result["answer"], "pong")
            # http_retry_params must include attempts/interval for HttpRetry proxy
            self.assertGreater(backend.http_retry_params["attempts"], 0)
            # config default sets parallel_calls_count to 10
            self.assertEqual(backend.parallel_calls, 10)


if __name__ == "__main__":
    unittest.main()
