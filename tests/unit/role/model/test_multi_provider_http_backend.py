import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import gage_eval.role.model.backends.multi_provider_http_backend as mp_backend
from gage_eval.role.model.backends import wrap_backend
from gage_eval.role.model.backends.multi_provider_http_backend import MultiProviderHTTPBackend


class FakeAsyncClient:
    def __init__(self, failures_before_success: int = 0, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self._remaining_failures = failures_before_success
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("temporary provider failure")
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
            self.assertEqual(backend.http_retry_mode, "native")
            self.assertEqual(
                backend.http_retry_params,
                {"max_retries": 5, "base_sleep": 3.0, "multiplier": 2.0},
            )
            # config default sets parallel_calls_count to 10
            self.assertEqual(backend.parallel_calls, 10)

    def test_multi_provider_maps_legacy_retry_keys_without_changing_behavior(self):
        fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)
        client_holder = {}

        def build_client(**kwargs):
            client = FakeAsyncClient(failures_before_success=1, **kwargs)
            client_holder["client"] = client
            return client

        with mock.patch.object(mp_backend, "AsyncInferenceClient", build_client), mock.patch.dict(
            sys.modules, {"transformers": fake_transformers}
        ):
            backend = MultiProviderHTTPBackend(
                {
                    "provider": "together",
                    "model_name": "stub-model",
                    "http_retry_params": {"attempts": 2, "interval": 0.0},
                }
            )
            wrapped = wrap_backend(backend)

            result = asyncio.run(wrapped.ainvoke({"messages": [{"role": "user", "content": "hi"}]}))

        self.assertEqual(result["answer"], "pong")
        self.assertEqual(len(client_holder["client"].calls), 2)
        self.assertEqual(
            backend.http_retry_params,
            {"max_retries": 1, "base_sleep": 0.0, "multiplier": 2.0},
        )

    def test_wrapped_backend_does_not_apply_http_retry_proxy_on_top_of_native_retry(self):
        fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)
        client_holder = {}

        def build_client(**kwargs):
            client = FakeAsyncClient(failures_before_success=2, **kwargs)
            client_holder["client"] = client
            return client

        with mock.patch.object(mp_backend, "AsyncInferenceClient", build_client), mock.patch.dict(
            sys.modules, {"transformers": fake_transformers}
        ):
            backend = MultiProviderHTTPBackend(
                {
                    "provider": "together",
                    "model_name": "stub-model",
                    "http_retry_params": {"max_retries": 2, "base_sleep": 0.0, "multiplier": 1.0},
                }
            )
            wrapped = wrap_backend(backend)

            result = asyncio.run(wrapped.ainvoke({"messages": [{"role": "user", "content": "hi"}]}))

        self.assertEqual(result["answer"], "pong")
        self.assertEqual(len(client_holder["client"].calls), 3)
        self.assertEqual(
            backend.http_retry_params,
            {"max_retries": 2, "base_sleep": 0.0, "multiplier": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
