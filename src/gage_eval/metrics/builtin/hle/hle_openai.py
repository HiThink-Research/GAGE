"""HLE accuracy metric using a DeepSeek judge model."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from loguru import logger
from openai import OpenAI

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import (
    extract_field,
    get_first_reference,
    get_text_content_of_first_predict_result,
)
from gage_eval.registry import registry

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0 and 100 from [response]. Put 100 if there is no confidence score available.

Return exactly one JSON object with these keys:
{{
  "extracted_final_answer": "string",
  "reasoning": "string",
  "correct": "yes" or "no",
  "confidence": 0
}}
Do not wrap the JSON in markdown fences."""


@registry.asset(
    "metrics",
    "hle_accuracy_openai",
    desc="HLE accuracy (DeepSeek judge)",
    tags=("HLE",),
    default_aggregation="mean",
)
class HLEAccuracyDeepSeekMetric(SimpleMetric):
    """Compute HLE accuracy using a DeepSeek judge model."""

    value_key = "acc"

    def setup(self) -> None:
        """Initialize judge configuration for DeepSeek evaluation."""

        self._judge_model = str(
            self.args.get("judge_model")
            or os.environ.get("HLE_JUDGE_MODEL")
            or "deepseek-chat"
        )
        self._api_key = (
            self.args.get("api_key")
            or os.environ.get("HLE_JUDGE_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
        )
        self._api_base = (
            self.args.get("api_base")
            or os.environ.get("HLE_JUDGE_API_BASE")
            or os.environ.get("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com"
        )
        self._client: Optional[OpenAI] = None
        self._client_error: Optional[str] = None

    def _get_client(self) -> Optional[OpenAI]:
        """Return a cached OpenAI-compatible client for DeepSeek."""

        if self._client is not None:
            return self._client
        if self._client_error is not None:
            return None
        if not self._api_key:
            self._client_error = "DEEPSEEK_API_KEY not set"
            logger.warning("HLE DeepSeek judge disabled: {}", self._client_error)
            return None
        client_kwargs = {
            "api_key": self._api_key,
            "base_url": self._api_base,
            "timeout": 300.0,
            "max_retries": 1,
        }
        try:
            self._client = OpenAI(**client_kwargs)
        except Exception as exc:
            self._client_error = str(exc)
            logger.warning("HLE DeepSeek judge client init failed: {}", exc)
            return None
        return self._client

    def _extract_message_text(self, response: Any) -> str:
        """Extract plain text from a chat completion response."""

        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif hasattr(item, "text"):
                    chunks.append(str(getattr(item, "text", "")))
            return "".join(chunks)
        return str(content or "")

    def _extract_json_payload(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from the response text."""

        candidate = raw_text.strip()
        if candidate.startswith("```"):
            parts = candidate.split("```")
            if len(parts) >= 3:
                candidate = parts[1]
                if "\n" in candidate:
                    candidate = candidate.split("\n", 1)[1]
                candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            return parsed if isinstance(parsed, dict) else None
        return None

    def _normalize_answer_for_compare(self, value: Any) -> str:
        """Normalize answer text before exact-match comparison."""

        normalized = str(value).strip().strip("`'\"")
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.rstrip(" .")
        return normalized.casefold()

    def _determine_correct(self, extracted_answer: str, correct_answer: str) -> str:
        """Determine correctness by comparing the extracted answer to the reference."""

        extracted = self._normalize_answer_for_compare(extracted_answer)
        reference = self._normalize_answer_for_compare(correct_answer)
        return "yes" if extracted and extracted == reference else "no"

    def _extract_text_payload(self, raw_text: str, correct_answer: str) -> Optional[Dict[str, Any]]:
        """Extract a structured payload from plain-text judge output."""

        explanation_match = re.search(
            r"(?is)explanation\s*:\s*(.*?)(?:\n\s*(?:answer|final answer)\s*:|$)",
            raw_text,
        )
        answer_match = re.search(
            r"(?is)(?:answer|final answer)\s*:\s*(.*?)(?:\n\s*confidence\s*:|$)",
            raw_text,
        )
        if answer_match is None:
            return None
        confidence_match = re.search(
            r"(?i)confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            raw_text,
        )
        extracted_answer = answer_match.group(1).strip()
        reasoning = (
            explanation_match.group(1).strip()
            if explanation_match is not None
            else raw_text.strip()
        )
        confidence = confidence_match.group(1) if confidence_match is not None else 100
        return {
            "extracted_final_answer": extracted_answer,
            "reasoning": reasoning,
            "correct": self._determine_correct(extracted_answer, correct_answer),
            "confidence": self._normalize_confidence(confidence),
        }

    def _normalize_correct(self, value: Any) -> str:
        """Normalize the correctness field."""

        return "yes" if str(value).strip().lower() == "yes" else "no"

    def _normalize_confidence(self, value: Any) -> int:
        """Normalize the confidence field into an integer percentage."""

        try:
            confidence = int(float(value))
        except (TypeError, ValueError):
            confidence = 100
        return max(0, min(100, confidence))

    def extract_answer(self, question: str, correct_answer: str, response: str) -> Optional[Dict[str, Any]]:
        """Call the DeepSeek judge and return the parsed answer payload."""

        client = self._get_client()
        if client is None:
            return None
        prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
        try:
            result = client.chat.completions.create(
                model=self._judge_model,
                max_tokens=4096,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            content = self._extract_message_text(result)
            payload = self._extract_json_payload(content)
            if payload is None:
                payload = self._extract_text_payload(content, correct_answer)
            if payload is None:
                raise ValueError("judge response is not valid JSON")
            return {
                "correct_answer": correct_answer,
                "model_answer": str(payload.get("extracted_final_answer", "None")),
                "reasoning": str(payload.get("reasoning", "")),
                "correct": self._normalize_correct(payload.get("correct")),
                "confidence": self._normalize_confidence(payload.get("confidence")),
            }
        except Exception as exc:
            logger.warning("HLE DeepSeek judge failed: {}", exc)
            return None

    def compute(self, context: MetricContext) -> MetricResult:
        """Compute the per-sample HLE accuracy score."""

        # STEP 1: Extract sample, prediction, and ground truth.
        sample_dict = extract_field(context, "sample")
        answer = get_first_reference(sample_dict)
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)
        question = sample_dict["metadata"]["question"]

        # STEP 2: Prefer cleaned answers from support outputs.
        support_outputs = sample_dict.get("support_outputs") or []
        cleaned_prediction = None
        if isinstance(support_outputs, list) and support_outputs:
            last_output = support_outputs[-1]
            if isinstance(last_output, dict):
                cleaned_prediction = last_output.get("answer")
        prediction = cleaned_prediction or prediction_raw or ""

        # STEP 3: Call the DeepSeek judge.
        deepseek_judge = self.extract_answer(question, answer, prediction)

        # STEP 4: Convert the judge result into the metric payload.
        metadata = {
            "prediction": prediction,
            "prediction_raw": prediction_raw,
            "references": answer,
        }
        if cleaned_prediction is not None:
            metadata["prediction_cleaned"] = cleaned_prediction
        if not deepseek_judge:
            metadata["judge_error"] = self._client_error or "deepseek_judge_failed"
            return MetricResult(
                sample_id=context.sample_id,
                values={self.value_key: 0.0},
                metadata=metadata,
            )
        correct = deepseek_judge.get("correct") == "yes"
        score = float(correct)
        metadata.update(
            {
                "judge_model": self._judge_model,
                "judge_answer": deepseek_judge.get("model_answer"),
                "judge_correct": deepseek_judge.get("correct"),
                "judge_confidence": deepseek_judge.get("confidence"),
            }
        )
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)


__all__ = ["HLEAccuracyDeepSeekMetric"]
