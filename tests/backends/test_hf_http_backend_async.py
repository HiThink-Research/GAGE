import asyncio
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.hf_http_backend import HFInferenceEndpointBackend, HFServerlessBackend


class FakeAsyncClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    async def text_generation(self, prompt, wait_for_model=True, extra_headers=None, **params):
        self.calls.append(
            {"prompt": prompt, "wait_for_model": wait_for_model, "headers": extra_headers, "params": params}
        )
        return {"generated_text": f"{prompt}-async"}


class FakeInferenceClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def text_generation(self, prompt, wait_for_model=True, extra_headers=None, **params):
        self.calls.append(
            {"prompt": prompt, "wait_for_model": wait_for_model, "headers": extra_headers, "params": params}
        )
        return {"generated_text": f"{prompt}-sync"}


class FakeEndpoint:
    def __init__(self, client=None, async_client=None):
        self.client = client or FakeInferenceClient()
        self.async_client = async_client or FakeAsyncClient()
        self.status = "running"
        self.name = "demo-endpoint"

    def wait(self, *_args, **_kwargs):
        return self

    def delete(self):
        return None

    def update(self, **_kwargs):
        return None


def _fake_hub(**attrs):
    return types.SimpleNamespace(**attrs)


class HFHTTPBackendAsyncTests(unittest.TestCase):
    def test_hf_serverless_async_path(self):
        fake_hub = _fake_hub(AsyncInferenceClient=FakeAsyncClient, InferenceClient=FakeInferenceClient)
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            backend = HFServerlessBackend({"model_name": "stub-model", "enable_async": True, "async_max_concurrency": 1})
            result = asyncio.run(backend.ainvoke({"prompt": "ping"}))

        self.assertEqual(result["answer"], "ping-async")
        self.assertTrue(backend._async_client.calls, "Async client should capture calls for observability")

    def test_hf_inference_endpoint_async_path(self):
        endpoint = FakeEndpoint()
        fake_hub = _fake_hub(
            InferenceEndpoint=FakeEndpoint,
            InferenceEndpointError=Exception,
            HfHubHTTPError=Exception,
            create_inference_endpoint=lambda **_kwargs: endpoint,
            get_inference_endpoint=lambda **_kwargs: endpoint,
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}), mock.patch.object(
            HFInferenceEndpointBackend, "_ensure_endpoint", return_value=endpoint
        ):
            backend = HFInferenceEndpointBackend(
                {
                    "endpoint_name": "demo",
                    "model_name": "stub",
                    "huggingface_token": "token",
                    "enable_async": True,
                    "async_max_concurrency": 2,
                }
            )
            result = asyncio.run(backend.ainvoke({"prompt": "pong"}))

        self.assertEqual(result["answer"], "pong-async")
        self.assertTrue(endpoint.async_client.calls, "Async endpoint client should be invoked")


if __name__ == "__main__":
    unittest.main()
