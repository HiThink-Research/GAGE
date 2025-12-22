import os
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.hf_http_backend import HFInferenceEndpointBackend


class _RecordingEndpoint:
    def __init__(self, *, fail_once_with=None, status="starting"):
        self.client = types.SimpleNamespace()
        self.async_client = None
        self.status = status
        self.name = "demo-endpoint"
        self.wait_calls = []
        self.update_calls = []
        self._fail_once_with = fail_once_with

    def wait(self, timeout=None, refresh_every=None):
        self.wait_calls.append({"timeout": timeout, "refresh": refresh_every})
        if self._fail_once_with:
            exc = self._fail_once_with
            self._fail_once_with = None
            raise exc
        self.status = "running"
        return self

    def update(self, **kwargs):
        self.update_calls.append(kwargs)

    def delete(self):
        return None


class HFInferenceEndpointBackendTests(unittest.TestCase):
    def test_requires_explicit_token(self):
        fake_hub = types.SimpleNamespace(
            InferenceEndpoint=object,
            InferenceEndpointError=Exception,
            HfHubHTTPError=Exception,
            create_inference_endpoint=lambda **_kwargs: None,
            get_inference_endpoint=lambda **_kwargs: None,
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}), mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                HFInferenceEndpointBackend({"model_name": "stub-model"})

    def test_creates_endpoint_and_merges_env(self):
        class FakeHTTPError(Exception):
            pass

        class FakeEndpointError(Exception):
            pass

        endpoint = _RecordingEndpoint()
        create_kwargs = {}

        def _create_inference_endpoint(**kwargs):
            create_kwargs.update(kwargs)
            return endpoint

        def _get_inference_endpoint(**_kwargs):
            raise FakeHTTPError("not found")

        fake_hub = types.SimpleNamespace(
            InferenceEndpoint=_RecordingEndpoint,
            InferenceEndpointError=FakeEndpointError,
            HfHubHTTPError=FakeHTTPError,
            create_inference_endpoint=_create_inference_endpoint,
            get_inference_endpoint=_get_inference_endpoint,
        )

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            backend = HFInferenceEndpointBackend(
                {
                    "endpoint_name": "demo-endpoint",
                    "model_name": "stub-model",
                    "huggingface_token": "token",
                    "instance_type": "nvidia-a10g",
                    "instance_size": "x1",
                    "vendor": "aws",
                    "region": "us-east-1",
                    "dtype": "float16",
                    "env_vars": {"EXTRA": "1"},
                }
            )

        self.assertIs(backend._endpoint, endpoint)
        self.assertFalse(backend.config.reuse_existing, "auto-created endpoint should flip reuse_existing to False")
        env_vars = create_kwargs["custom_image"]["env"]
        self.assertEqual(env_vars["DTYPE"], "float16")
        self.assertEqual(env_vars["HF_INFERENCE_PRECISION"], "float16")
        self.assertEqual(env_vars["EXTRA"], "1")
        self.assertEqual(create_kwargs["vendor"], "aws")
        self.assertEqual(create_kwargs["region"], "us-east-1")

    def test_scales_up_when_startup_fails(self):
        class FakeHTTPError(Exception):
            pass

        class FakeEndpointError(Exception):
            pass

        endpoint = _RecordingEndpoint(fail_once_with=FakeEndpointError("boot failed"))

        fake_hub = types.SimpleNamespace(
            InferenceEndpoint=_RecordingEndpoint,
            InferenceEndpointError=FakeEndpointError,
            HfHubHTTPError=FakeHTTPError,
            create_inference_endpoint=lambda **_kwargs: endpoint,
            get_inference_endpoint=lambda **_kwargs: None,
        )

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            backend = HFInferenceEndpointBackend(
                {
                    "endpoint_name": "demo",
                    "model_name": "stub-model",
                    "huggingface_token": "token",
                    "instance_type": "nvidia-a10g",
                    "instance_size": "x1",
                    "wait_timeout": 60,
                    "poll_interval": 5,
                }
            )

        self.assertIs(backend._endpoint, endpoint)
        self.assertTrue(endpoint.update_calls, "Endpoint should be rescaled after start failure")
        self.assertEqual(endpoint.update_calls[0]["instance_type"], "nvidia-t4")
        self.assertEqual(endpoint.update_calls[0]["instance_size"], "x4")
        self.assertEqual(endpoint.status, "running")

    def test_reuses_existing_endpoint_on_conflict(self):
        class FakeHTTPError(Exception):
            pass

        class FakeEndpointError(Exception):
            pass

        existing = _RecordingEndpoint(status="starting")

        def _create_inference_endpoint(**_kwargs):
            raise FakeHTTPError("Conflict for url")

        def _get_inference_endpoint(**_kwargs):
            return existing

        fake_hub = types.SimpleNamespace(
            InferenceEndpoint=_RecordingEndpoint,
            InferenceEndpointError=FakeEndpointError,
            HfHubHTTPError=FakeHTTPError,
            create_inference_endpoint=_create_inference_endpoint,
            get_inference_endpoint=_get_inference_endpoint,
        )

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            backend = HFInferenceEndpointBackend(
                {
                    "endpoint_name": "conflict-demo",
                    "model_name": "stub-model",
                    "huggingface_token": "token",
                    "instance_type": "nvidia-a10g",
                    "instance_size": "x1",
                }
            )

        self.assertIs(backend._endpoint, existing)
        self.assertTrue(existing.wait_calls, "Existing endpoint should still be waited on until running")
        self.assertTrue(backend.config.reuse_existing)
        self.assertEqual(existing.status, "running")


if __name__ == "__main__":
    unittest.main()
