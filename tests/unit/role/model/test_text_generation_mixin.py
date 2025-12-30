from types import SimpleNamespace

from gage_eval.role.model.runtime.text_generation_mixin import TextGenerationMixin


class _DummyAdapter(TextGenerationMixin):
    def render_prompt(self, payload):
        return SimpleNamespace(messages=None, prompt=None, metadata=None)

    def _compose_sampling_params(self, *, sample_params, runtime_params, legacy_params):
        return {}

    def _runtime_namespace(self, payload):
        return None


def test_prepare_backend_request_prefers_payload_messages():
    adapter = _DummyAdapter()
    payload_messages = [{"role": "user", "content": "payload"}]
    sample = {"messages": [{"role": "system", "content": "sample"}]}

    request = adapter.prepare_backend_request({"sample": sample, "messages": payload_messages})

    assert request["messages"] == payload_messages


def test_prepare_backend_request_falls_back_to_sample_messages():
    adapter = _DummyAdapter()
    sample_messages = [{"role": "system", "content": "sample"}]
    sample = {"messages": sample_messages}

    request = adapter.prepare_backend_request({"sample": sample})

    assert request["messages"] == sample_messages
