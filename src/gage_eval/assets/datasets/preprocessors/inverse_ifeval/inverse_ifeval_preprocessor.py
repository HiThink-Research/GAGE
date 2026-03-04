"""Inverse IFEval dataset preprocessor."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Sequence

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Message,
    MessageContent,
    Sample,
)


class InverseIFEvalPreprocessor(BasePreprocessor):
    """Converts Inverse IFEval records into the standard Sample schema.

    The preprocessor is intentionally tolerant to minor schema variations so that
    the same implementation can work across hub revisions.
    """

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        schema_version: str = SCHEMA_VERSION,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Sample:
        """Converts a raw Inverse IFEval row to a Sample.

        Args:
            record: Raw row from the dataset loader.
            schema_version: Sample schema version.
            system_prompt: Optional system instruction prepended to messages.
            **kwargs: Reserved for forward compatibility.

        Returns:
            A normalized Sample object for downstream inference and metrics.
        """

        # STEP 1: Resolve primary fields with conservative fallback order.
        prompt_text = self._resolve_prompt(record)
        sample_id = self._resolve_sample_id(record, prompt_text)
        references = self._resolve_references(record)
        response_reference = self._resolve_response_reference(record, references)

        constraints = self._as_list(record.get("constraints"))
        instruction_ids = self._as_list(record.get("instruction_id_list") or record.get("instruction_ids"))
        instruction_kwargs = self._as_dict(record.get("kwargs") or record.get("instruction_kwargs"))
        raw_schema_fragment = self._resolve_schema_fragment(record)

        # STEP 2: Build canonical chat messages.
        messages: list[Message] = []
        if system_prompt:
            messages.append(
                Message(
                    role="system",
                    content=[MessageContent(type="text", text=system_prompt.strip())],
                )
            )
        messages.append(
            Message(
                role="user",
                content=[MessageContent(type="text", text=prompt_text)],
            )
        )

        # STEP 3: Build metadata payload used by downstream metrics/judge prompts.
        metadata: Dict[str, Any] = {
            "prompt_text": prompt_text,
            "response_reference": response_reference,
            "constraints": constraints,
            "instruction_id_list": instruction_ids,
            "kwargs": instruction_kwargs,
            "raw_schema_fragment": raw_schema_fragment,
        }
        for key in (
            "task_id",
            "subset",
            "category",
            "source",
            "difficulty",
            "split",
            "sample_id",
            "id",
            "language",
            "judge_prompt_template",
            "judge_system_prompt",
        ):
            if key in record and record.get(key) is not None:
                metadata[key] = record.get(key)

        label = references[0] if references else None
        return Sample(
            id=sample_id,
            schema_version=schema_version,
            messages=messages,
            references=references,
            label=label,
            metadata=metadata,
        )

    def _resolve_prompt(self, record: Dict[str, Any]) -> str:
        """Resolves prompt text from known IFEval field aliases."""

        for key in ("prompt", "question", "input", "instruction", "text"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _resolve_sample_id(self, record: Dict[str, Any], prompt_text: str) -> str:
        """Builds a stable sample id from explicit ids or content hash."""

        explicit_id = record.get("id") or record.get("sample_id")
        if explicit_id is not None and str(explicit_id).strip():
            return str(explicit_id).strip()

        hash_payload = {
            "prompt": prompt_text,
            "constraints": record.get("constraints"),
            "instruction_id_list": record.get("instruction_id_list") or record.get("instruction_ids"),
            "kwargs": record.get("kwargs") or record.get("instruction_kwargs"),
        }
        encoded = json.dumps(hash_payload, sort_keys=True, ensure_ascii=True, default=str)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"inverse_ifeval_{digest[:16]}"

    def _resolve_references(self, record: Dict[str, Any]) -> list[str]:
        """Extracts references/targets when present; otherwise returns empty list."""

        for key in ("references", "reference", "targets", "target", "answers", "answer", "label"):
            if key not in record:
                continue
            values = self._as_list(record.get(key))
            cleaned = [str(item).strip() for item in values if str(item).strip()]
            if cleaned:
                return cleaned
        response_reference = record.get("response_reference")
        if isinstance(response_reference, str) and response_reference.strip():
            return [response_reference.strip()]
        return []

    def _resolve_response_reference(self, record: Dict[str, Any], references: Sequence[str]) -> str:
        """Resolves canonical response reference text for judge prompts."""

        value = record.get("response_reference")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if references:
            first = str(references[0]).strip()
            if first:
                return first
        return ""

    def _resolve_schema_fragment(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts an optional schema-like fragment for debugging/traceability."""

        for key in ("schema", "schema_fragment", "rule_schema", "constraints_schema"):
            value = record.get(key)
            if isinstance(value, dict):
                return dict(value)
        return {}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Converts unknown inputs into a list with best-effort normalization."""

        if value is None:
            return []
        if isinstance(value, dict):
            return [dict(value)]
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        if isinstance(value, Iterable):
            return list(value)
        return [value]

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        """Returns a shallow dict for kwargs-like values."""

        if isinstance(value, dict):
            return dict(value)
        return {}


__all__ = ["InverseIFEvalPreprocessor"]
