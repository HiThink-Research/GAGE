from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor


class _StripRolePre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        sample = dict(record)
        sample.setdefault("messages", [{"role": "system", "content": [{"type": "text", "text": "meta"}]}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        sample.setdefault("choices", [])
        sample.setdefault("inputs", {"prompt": "hi"})
        return sample


def test_roles_to_remove_and_on_error_raise():
    pre = _StripRolePre(roles_to_remove=("system",), on_error="raise")
    sample = {
        "id": "s1",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "meta"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
    }
    _ = pre.transform(sample)
    msgs = sample.get("messages", [])
    assert all(msg.get("role") != "system" for msg in msgs)
    assert sample["inputs"]["prompt"] == "hi"
