from __future__ import annotations

import concurrent.futures
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.http_backend import HTTPGenerationBackend


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class RecordingSession:
    def __init__(self, payload=None):
        self.payload = payload or {"generated_text": "pong"}
        self.post_calls = []
        self.closed = False

    def post(self, url, json=None, headers=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(self.payload)

    def close(self):
        self.closed = True


class ParallelCallState:
    def __init__(self) -> None:
        self.entered = 0
        self.lock = threading.Lock()
        self.overlap = threading.Event()


class CoordinatedSession:
    def __init__(self, state: ParallelCallState, payload=None):
        self.state = state
        self.payload = payload or {"generated_text": "pong"}
        self.closed = False
        self.post_calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        with self.state.lock:
            self.state.entered += 1
            if self.state.entered == 2:
                self.state.overlap.set()
        if not self.state.overlap.wait(timeout=1):
            raise AssertionError("HTTP requests were serialized")
        return FakeResponse(self.payload)

    def close(self):
        self.closed = True


class CoordinatedSessionFactory:
    def __init__(self, state: ParallelCallState, payload=None):
        self.state = state
        self.payload = payload or {"generated_text": "pong"}
        self.sessions = []

    def __call__(self):
        session = CoordinatedSession(self.state, self.payload)
        self.sessions.append(session)
        return session


class HTTPGenerationBackendTests(unittest.TestCase):
    def test_generate_merges_request_fields_and_sampling_params(self):
        fake_session = RecordingSession({"results": [{"text": "hello"}]})
        with mock.patch("gage_eval.role.model.backends.http_backend.requests.Session", lambda: fake_session):
            backend = HTTPGenerationBackend(
                {
                    "base_url": "http://backend",
                    "endpoint": "/custom-generate",
                    "timeout": 15,
                    "headers": {"Authorization": "Bearer stub"},
                    "request_fields": {"stream": False},
                }
            )
            result = backend.generate(
                {
                    "prompt": "ping",
                    "sampling_params": {"max_new_tokens": 32, "temperature": 0.2, "top_p": 0.9},
                }
            )
            backend.close()

        self.assertEqual(result["answer"], "hello")
        self.assertEqual(fake_session.post_calls[0]["url"], "http://backend/custom-generate")
        self.assertEqual(fake_session.post_calls[0]["headers"], {"Authorization": "Bearer stub"})
        self.assertEqual(fake_session.post_calls[0]["timeout"], 15)
        self.assertEqual(
            fake_session.post_calls[0]["json"],
            {
                "inputs": "ping",
                "parameters": {"max_new_tokens": 32, "temperature": 0.2, "top_p": 0.9},
                "stream": False,
            },
        )
        self.assertTrue(fake_session.closed)

    def test_generate_allows_parallel_requests_on_one_backend_instance(self):
        state = ParallelCallState()
        session_factory = CoordinatedSessionFactory(state)
        with mock.patch("gage_eval.role.model.backends.http_backend.requests.Session", session_factory):
            backend = HTTPGenerationBackend({"base_url": "http://backend"})
            start_barrier = threading.Barrier(2)

            def _invoke(prompt: str) -> str:
                start_barrier.wait()
                return backend.generate({"prompt": prompt})["answer"]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(_invoke, "first"), executor.submit(_invoke, "second")]
                results = [future.result(timeout=2) for future in futures]

            backend.close()

        self.assertEqual(results, ["pong", "pong"])
        self.assertTrue(state.overlap.is_set())
        self.assertEqual(len(session_factory.sessions), 2)
        self.assertTrue(all(session.closed for session in session_factory.sessions))


if __name__ == "__main__":
    unittest.main()
