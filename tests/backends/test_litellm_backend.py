import sys
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend


class _FakeLitellm(types.SimpleNamespace):
    def __init__(self, *, raise_error: bool = False):
        super().__init__()
        self.calls = []
        self.raise_error = raise_error
        self.drop_params = False
        self.verbose = False
        self.api_key = None
        self.api_base = None
        self.headers = None

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_error:
            raise RuntimeError("litellm failure")
        return {"choices": [{"message": {"content": "pong-lite"}}]}

    def supports_reasoning(self, _model):
        return False


class _FakeResponse:
    def __init__(self, data, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            from requests import HTTPError

            raise HTTPError(f"status={self.status_code}")

    def json(self):
        return self._data


class LiteLLMBackendTests(unittest.TestCase):
    def test_litellm_backend_merges_sampling_and_headers(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16, "stop": ["END"]},
                    "extra_headers": {"X-Test": "1"},
                    "timeout": 5.0,
                }
            )
            result = backend.generate(
                {
                    "messages": [{"role": "user", "content": "ping"}],
                    "sampling_params": {"temperature": 0.1, "max_new_tokens": 8},
                }
            )

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["temperature"], 0.1)
        self.assertEqual(call["max_tokens"], 8)
        self.assertEqual(call["stop"], ["END"])
        self.assertEqual(call["headers"]["X-Test"], "1")

    def test_kimi_fallback_on_litellm_error(self):
        fake_litellm = _FakeLitellm(raise_error=True)
        calls = []

        def _fake_post(url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return _FakeResponse({"choices": [{"message": {"content": "pong-kimi"}}]})

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}), mock.patch("requests.post", _fake_post):
            backend = LiteLLMBackend(
                {
                    "model": "moonshot-v1-8k",
                    "provider": "kimi",
                    "api_key": "kimi-key",
                    "retry_sleep": 0.01,
                    "max_retries": 2,
                }
            )
            result = backend.generate({"messages": [{"role": "user", "content": "hello"}], "sampling_params": {"max_new_tokens": 5}})

        self.assertEqual(result["answer"], "pong-kimi")
        self.assertTrue(calls, "Kimi HTTP fallback should be used when liteLLM fails")
        self.assertIn("moonshot", calls[0]["url"])
        self.assertEqual(calls[0]["json"]["max_tokens"], 5)
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer kimi-key")


if __name__ == "__main__":
    unittest.main()
