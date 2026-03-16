from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch


def _ensure_package(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def _install_metric_stubs() -> type:
    if "openai" not in sys.modules:
        openai_module = types.ModuleType("openai")

        class OpenAI:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        openai_module.OpenAI = OpenAI
        sys.modules["openai"] = openai_module

    if "loguru" not in sys.modules:
        loguru_module = types.ModuleType("loguru")
        loguru_module.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
        sys.modules["loguru"] = loguru_module

    registry_module = types.ModuleType("gage_eval.registry")

    class _Registry:
        def asset(self, *args: object, **kwargs: object):
            def decorator(obj: object) -> object:
                return obj

            return decorator

    registry_module.registry = _Registry()

    metrics_base_module = types.ModuleType("gage_eval.metrics.base")

    @dataclass(frozen=True)
    class MetricContext:
        sample_id: str
        sample: dict[str, object]
        model_output: dict[str, object]
        judge_output: dict[str, object]
        args: dict[str, object]
        trace: object

    @dataclass(frozen=True)
    class MetricResult:
        sample_id: str
        values: dict[str, float]
        metadata: dict[str, object] = field(default_factory=dict)
        explanation: str | None = None

    class SimpleMetric:
        value_key = "score"

        def __init__(self, spec: object) -> None:
            self.spec = spec
            self.args = dict(getattr(spec, "params", {}) or {})
            self.setup()

        def setup(self) -> None:
            return None

    metrics_base_module.MetricContext = MetricContext
    metrics_base_module.MetricResult = MetricResult
    metrics_base_module.SimpleMetric = SimpleMetric

    metrics_utils_module = types.ModuleType("gage_eval.metrics.utils")

    def extract_field(context: MetricContext, descriptor: str, default: object = None) -> object:
        if descriptor == "sample":
            return context.sample
        return default

    def get_first_reference(sample_dict: dict[str, object]) -> str | None:
        references = sample_dict.get("references") or []
        return references[0] if references else None

    def get_text_content_of_first_predict_result(sample_dict: dict[str, object]) -> str | None:
        return sample_dict["predict_result"][0]["message"]["content"][0]["text"]

    metrics_utils_module.extract_field = extract_field
    metrics_utils_module.get_first_reference = get_first_reference
    metrics_utils_module.get_text_content_of_first_predict_result = get_text_content_of_first_predict_result

    _ensure_package("gage_eval")
    _ensure_package("gage_eval.metrics")
    _ensure_package("gage_eval.metrics.builtin")
    _ensure_package("gage_eval.metrics.builtin.hle")
    sys.modules["gage_eval.registry"] = registry_module
    sys.modules["gage_eval.metrics.base"] = metrics_base_module
    sys.modules["gage_eval.metrics.utils"] = metrics_utils_module
    return MetricContext


def _load_module(name: str, relative_path: str) -> types.ModuleType:
    module_path = pathlib.Path(__file__).resolve().parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_deepseek_module() -> tuple[type, types.ModuleType]:
    metric_context_cls = _install_metric_stubs()
    deepseek_module = _load_module(
        "gage_eval.metrics.builtin.hle.hle_deepseek",
        "src/gage_eval/metrics/builtin/hle/hle_deepseek.py",
    )
    return metric_context_cls, deepseek_module


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **_: object) -> object:
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


class HLEDeepSeekMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric_context_cls, cls.deepseek_module = _load_deepseek_module()
        cls.metric_cls = cls.deepseek_module.HLEAccuracyDeepSeekMetric

    def test_setup_uses_deepseek_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_API_BASE": "https://api.deepseek.com",
                "OPENAI_API_KEY": "openai-key",
            },
            clear=False,
        ):
            spec = SimpleNamespace(metric_id="hle_acc", params={})
            metric = self.metric_cls(spec)
        self.assertEqual(metric._api_key, "deepseek-key")
        self.assertEqual(metric._api_base, "https://api.deepseek.com")
        self.assertEqual(metric._judge_model, "deepseek-chat")

    def test_metric_does_not_fall_back_to_openai_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
            },
            clear=True,
        ):
            spec = SimpleNamespace(metric_id="hle_acc", params={})
            metric = self.metric_cls(spec)
        self.assertIsNone(metric._api_key)

    def test_extract_answer_parses_json_completion(self) -> None:
        spec = SimpleNamespace(
            metric_id="hle_acc",
            params={
                "judge_model": "deepseek-chat",
                "api_key": "judge-key",
                "api_base": "https://api.deepseek.com",
            },
        )
        metric = self.metric_cls(spec)
        fake_response = (
            '{"extracted_final_answer":"42","reasoning":"Matches the reference.",'
            '"correct":"yes","confidence":88}'
        )
        with patch.object(metric, "_get_client", return_value=_FakeClient(fake_response)):
            result = metric.extract_answer("What is 6*7?", "42", "ANSWER: 42")
        self.assertEqual(result["model_answer"], "42")
        self.assertEqual(result["correct"], "yes")
        self.assertEqual(result["confidence"], 88)

    def test_extract_answer_parses_plain_text_completion(self) -> None:
        spec = SimpleNamespace(
            metric_id="hle_acc",
            params={
                "judge_model": "deepseek-chat",
                "api_key": "judge-key",
                "api_base": "https://api.deepseek.com",
            },
        )
        metric = self.metric_cls(spec)
        fake_response = (
            "Explanation: The response matches the reference answer.\n\n"
            "Answer: True\n"
            "Confidence: 85%"
        )
        with patch.object(metric, "_get_client", return_value=_FakeClient(fake_response)):
            result = metric.extract_answer("Question?", "True", "Model response")
        self.assertEqual(result["model_answer"], "True")
        self.assertEqual(result["correct"], "yes")
        self.assertEqual(result["confidence"], 85)

    def test_extract_answer_parses_prompt_field_plain_text_completion(self) -> None:
        spec = SimpleNamespace(
            metric_id="hle_acc",
            params={
                "judge_model": "deepseek-chat",
                "api_key": "judge-key",
                "api_base": "https://api.deepseek.com",
            },
        )
        metric = self.metric_cls(spec)
        fake_response = (
            "extracted_final_answer: 42\n"
            "reasoning: Matches the provided correct answer.\n"
            "correct: yes\n"
            "confidence: 91"
        )
        with patch.object(metric, "_get_client", return_value=_FakeClient(fake_response)):
            result = metric.extract_answer("What is 6*7?", "42", "The answer is 42")
        self.assertEqual(result["model_answer"], "42")
        self.assertEqual(result["correct"], "yes")
        self.assertEqual(result["confidence"], 91)

    def test_compute_prefers_cleaned_prediction(self) -> None:
        spec = SimpleNamespace(metric_id="hle_acc", params={})
        metric = self.metric_cls(spec)
        with patch.object(
            metric,
            "extract_answer",
            return_value={
                "correct_answer": "42",
                "model_answer": "42",
                "reasoning": "Matches.",
                "correct": "yes",
                "confidence": 100,
            },
        ) as mocked_extract:
            context = self.metric_context_cls(
                sample_id="demo",
                sample={
                    "predict_result": [{"message": {"content": [{"text": "wrong"}]}}],
                    "references": ["42"],
                    "metadata": {"question": "What is 6*7?"},
                    "support_outputs": [{"answer": "42"}],
                },
                model_output={},
                judge_output={},
                args={},
                trace=None,
            )
            result = metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata["prediction_cleaned"], "42")
        self.assertEqual(mocked_extract.call_args.args[2], "42")


if __name__ == "__main__":
    unittest.main()
