import asyncio
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.sglang_backend import SGLangBackend


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.post_calls = []
        self.closed = False

    def post(self, url, json=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        payload = self.responses.pop(0) if self.responses else {}
        return FakeResponse(payload)

    def close(self):
        self.closed = True


class SGLangBackendTests(unittest.TestCase):
    def test_builds_prompt_request_and_extracts_text(self):
        fake_session = FakeSession([{"outputs": [{"text": "hello"}]}])
        with mock.patch("gage_eval.role.model.backends.sglang_backend.requests.Session", lambda: fake_session):
            backend = SGLangBackend(
                {
                    "host": "0.0.0.0",
                    "port": 31000,
                    "max_new_tokens": 16,
                    "temperature": 0.3,
                    "stop": ["<eos>"],
                }
            )
            result = asyncio.run(
                backend.ainvoke({"prompt": "ping", "sampling_params": {"top_p": 0.8, "presence_penalty": 0.1}})
            )

        self.assertEqual(result["answer"], "hello")
        call = fake_session.post_calls[0]
        self.assertEqual(call["url"], "http://0.0.0.0:31000/generate")
        self.assertEqual(call["json"]["prompt"], "ping")
        # stop 字段被放在请求顶层，sampling_params 中保留非空参数
        self.assertEqual(call["json"]["stop"], ["<eos>"])
        self.assertEqual(call["json"]["sampling_params"]["max_new_tokens"], 16)
        self.assertEqual(call["json"]["sampling_params"]["temperature"], 0.3)
        self.assertEqual(call["json"]["sampling_params"]["top_p"], 0.8)
        self.assertEqual(call["json"]["sampling_params"]["presence_penalty"], 0.1)

    def test_extracts_top_logprobs_with_logprob_ids(self):
        top_logprob_payload = {
            "meta_info": {"output_top_logprobs": [[(-0.1, 42, "foo"), (-0.2, 7, "bar")]]}
        }
        fake_session = FakeSession([top_logprob_payload])
        with mock.patch("gage_eval.role.model.backends.sglang_backend.requests.Session", lambda: fake_session):
            backend = SGLangBackend({"base_url": "http://sglang:30000", "timeout": 30})
            result = asyncio.run(
                backend.ainvoke(
                    {
                        "prompt": "topk",
                        "output_type": "next_token_prob",
                        "logprob_token_ids": [42, 7],
                        "sampling_params": {"max_new_tokens": 1},
                    }
                )
            )

        self.assertEqual(result["top_logprobs"], [[42, -0.1], [7, -0.2]])
        call = fake_session.post_calls[0]
        self.assertEqual(call["json"]["logprob_token_ids"], [42, 7])
        # stop 未配置时不应出现在请求里
        self.assertNotIn("stop", call["json"])


if __name__ == "__main__":
    unittest.main()
