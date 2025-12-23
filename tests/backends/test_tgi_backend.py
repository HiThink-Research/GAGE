import asyncio
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.tgi_backend import TGIBackend, _extract_tgi_text


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, payload=None):
        self.payload = payload or {"choices": [{"text": "pong"}]}
        self.post_calls = []
        self.closed = False

    def post(self, url, json=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(self.payload)

    def close(self):
        self.closed = True


class TGIBackendTests(unittest.TestCase):
    def test_generate_merges_defaults_and_overrides(self):
        fake_session = FakeSession()
        with mock.patch("gage_eval.role.model.backends.tgi_backend.requests.Session", lambda: fake_session):
            backend = TGIBackend(
                {
                    "host": "0.0.0.0",
                    "port": 9000,
                    "max_new_tokens": 4,
                    "temperature": 0.5,
                    "stop": ["</s>"],
                    "repetition_penalty": 1.05,
                }
            )

            result = asyncio.run(
                backend.ainvoke({"prompt": "ping", "sampling_params": {"top_p": 0.9, "temperature": 0.1}})
            )

        self.assertEqual(result["answer"], "pong")
        self.assertEqual(fake_session.post_calls[0]["url"], "http://0.0.0.0:9000/generate")
        params = fake_session.post_calls[0]["json"]["parameters"]
        self.assertEqual(params["max_new_tokens"], 4)
        # sampling_params.temperature 覆盖默认 temperature
        self.assertEqual(params["temperature"], 0.1)
        self.assertEqual(params["top_p"], 0.9)
        self.assertEqual(params["repetition_penalty"], 1.05)
        self.assertEqual(params["stop_sequences"], ["</s>"])

    def test_extract_tgi_text_handles_various_shapes(self):
        self.assertEqual(_extract_tgi_text({"generated_text": "a"}), "a")
        self.assertEqual(_extract_tgi_text({"choices": [{"text": "b"}]}), "b")
        self.assertEqual(_extract_tgi_text([{"generated_text": "c"}]), "c")
        self.assertEqual(_extract_tgi_text(["d"]), "d")
        self.assertEqual(_extract_tgi_text("raw"), "raw")


if __name__ == "__main__":
    unittest.main()
