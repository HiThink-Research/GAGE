"""HLE accuracy metric (OpenAI default judge config)."""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger
from openai import OpenAI

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.judge.defaults import OpenAIJudgePromptDefaults
from gage_eval.metrics.utils import (
    extract_field,
    get_first_reference,
    get_text_content_of_first_predict_result,
)
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "hle_accuracy_openai_default",
    desc="HLE accuracy (default judge config)",
    tags=("HLE",),
    default_aggregation="mean",
)
class HLEAccuracyOpenAIDefaultMetric(OpenAIJudgePromptDefaults, SimpleMetric):
    """Compute HLE accuracy using the shared judge defaults."""

    value_key = "acc"

    def setup(self) -> None:
        """Initialize judge configuration for OpenAI evaluation."""

        self._judge_model = str(
            self.args.get("judge_model")
            or os.environ.get("HLE_JUDGE_MODEL")
            or "gpt-5.1"
        )
        self._api_key = self.args.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self._api_base = self.args.get("api_base") or os.environ.get("OPENAI_BASE_URL")
        self._client: Optional[OpenAI] = None
        self._client_error: Optional[str] = None

    def _get_client(self) -> Optional[OpenAI]:
        if self._client is not None:
            return self._client
        if self._client_error is not None:
            return None
        if not self._api_key:
            self._client_error = "OPENAI_API_KEY not set"
            logger.warning("HLE OpenAI judge disabled: {}", self._client_error)
            return None
        client_kwargs = {"api_key": self._api_key, "timeout": 300.0, "max_retries": 1}
        if self._api_base:
            client_kwargs["base_url"] = self._api_base
        try:
            self._client = OpenAI(**client_kwargs)
        except Exception as exc:
            self._client_error = str(exc)
            logger.warning("HLE OpenAI judge client init failed: {}", exc)
            return None
        return self._client

    def extract_answer(
        self, question: str, correct_answer: str, response: str
    ) -> Optional[dict[str, Any]]:
        """Call the OpenAI judge and return the parsed answer payload."""

        client = self._get_client()
        if client is None:
            return None
        prompt = self.build_judge_prompt(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )
        try:
            result = client.beta.chat.completions.parse(
                model=self._judge_model,
                max_completion_tokens=4096,  # overkill for judge
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format=self.response_model,
            )
            content = result.choices[0].message.parsed
            return {
                "correct_answer": correct_answer,
                "model_answer": content.extracted_final_answer,
                "reasoning": content.reasoning,
                "correct": content.correct,
                "confidence": content.confidence,
            }
        except Exception as e:  # very, very rare
            logger.warning(f"Error:{e}")
            return None

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample, prediction, and ground truth
        sample_dict = extract_field(context, "sample")
        answer = get_first_reference(sample_dict)
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)
        question = sample_dict["metadata"]["question"]

        # STEP 2: prefer cleaned answers from support outputs
        support_outputs = sample_dict.get("support_outputs") or []
        cleaned_prediction = None
        if isinstance(support_outputs, list) and support_outputs:
            last_output = support_outputs[-1]
            if isinstance(last_output, dict):
                cleaned_prediction = last_output.get("answer")
        prediction = cleaned_prediction or prediction_raw or ""

        # STEP 3: call openai judge
        openai_judge = self.extract_answer(question, answer, prediction)

        # STEP 4: compute score
        metadata = {
            "prediction": prediction,
            "prediction_raw": prediction_raw,
            "references": answer,
        }
        if cleaned_prediction is not None:
            metadata["prediction_cleaned"] = cleaned_prediction
        if not openai_judge:
            metadata["judge_error"] = self._client_error or "openai_judge_failed"
            return MetricResult(
                sample_id=context.sample_id,
                values={self.value_key: 0.0},
                metadata=metadata,
            )
        correct = openai_judge.get("correct") == "yes"
        score = float(correct)
        metadata.update(
            {
                "judge_model": self._judge_model,
                "judge_answer": openai_judge.get("model_answer"),
                "judge_correct": openai_judge.get("correct"),
                "judge_confidence": openai_judge.get("confidence"),
            }
        )
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)


__all__ = ["HLEAccuracyOpenAIDefaultMetric"]


if __name__ == "__main__":
    from gage_eval.config.pipeline_config import MetricSpec

    spec = MetricSpec(metric_id="test", implementation="fake_acc")
    hle_metric = HLEAccuracyOpenAIDefaultMetric(spec=spec)
