"""Rule-based metric for Inverse IFEval instruction compliance."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "inverse_ifeval_pass_rate",
    desc="Inverse IFEval rule-based pass-rate metric",
    tags=("instruction_following", "ifeval", "inverse"),
    default_aggregation="mean",
)
class InverseIFEvalPassRateMetric(SimpleMetric):
    """Computes pass rate from explicit constraints or instruction identifiers."""

    value_key = "pass_rate"

    def compute(self, context: MetricContext) -> MetricResult:
        """Evaluates the model answer against instruction/constraint rules.

        Args:
            context: Runtime metric context carrying sample and model outputs.

        Returns:
            MetricResult with `pass_rate` and `passed` values and diagnostics.
        """

        # STEP 1: Resolve normalized answer text and rule payload.
        answer = self._normalize_text(self._resolve_answer(context))
        sample = dict(context.sample or {})
        metadata = dict(sample.get("metadata") or {})

        constraints = self._as_list(metadata.get("constraints"))
        instruction_ids = self._as_list(metadata.get("instruction_id_list") or metadata.get("instruction_ids"))
        instruction_kwargs = self._as_dict(metadata.get("kwargs") or metadata.get("instruction_kwargs"))

        # STEP 2: Build evaluable rule list (constraints take precedence).
        if constraints:
            rules = constraints
            rule_origin = "constraints"
        else:
            rules = self._materialize_rules_from_instruction_ids(instruction_ids, instruction_kwargs)
            rule_origin = "instruction_id_list"

        # STEP 3: Evaluate each rule with strict handling for unsupported ones.
        failed_rule_ids: list[str] = []
        unsupported_rule_ids: list[str] = []
        passed_rules = 0
        total_rules = len(rules)

        for idx, raw_rule in enumerate(rules):
            rule = self._normalize_rule(raw_rule, idx)
            rule_id = rule["rule_id"]
            verdict = self._evaluate_rule(answer, rule)
            if verdict == "pass":
                passed_rules += 1
                continue
            if verdict == "unsupported":
                unsupported_rule_ids.append(rule_id)
            failed_rule_ids.append(rule_id)

        pass_rate = float(passed_rules / total_rules) if total_rules > 0 else 0.0
        passed = 1.0 if total_rules > 0 and passed_rules == total_rules else 0.0

        metric_metadata = {
            "rule_origin": rule_origin,
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rule_ids": failed_rule_ids,
            "unsupported_rule_ids": unsupported_rule_ids,
        }
        return MetricResult(
            sample_id=context.sample_id,
            values={"pass_rate": pass_rate, "passed": passed},
            metadata=metric_metadata,
        )

    def _resolve_answer(self, context: MetricContext) -> str:
        """Finds answer text from model_output and sample-level fallbacks."""

        model_output = dict(context.model_output or {})
        answer = model_output.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer

        sample = dict(context.sample or {})
        predict_result = sample.get("predict_result")
        if isinstance(predict_result, list) and predict_result:
            first = predict_result[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, list) and content:
                        fragment = content[0]
                        if isinstance(fragment, Mapping):
                            text = fragment.get("text")
                            if isinstance(text, str):
                                return text
        return ""

    def _normalize_rule(self, raw_rule: Any, index: int) -> Dict[str, Any]:
        """Converts free-form rule payloads into a canonical structure."""

        if isinstance(raw_rule, str):
            return {
                "rule_id": f"rule_{index}",
                "type": "must_contain",
                "value": raw_rule,
            }

        if not isinstance(raw_rule, dict):
            return {
                "rule_id": f"rule_{index}",
                "type": "unsupported",
            }

        rule = dict(raw_rule)
        rule.setdefault("rule_id", str(rule.get("id") or rule.get("instruction_id") or f"rule_{index}"))
        rule_type = (
            rule.get("type")
            or rule.get("op")
            or rule.get("constraint_type")
            or rule.get("name")
        )
        rule["type"] = str(rule_type).strip().lower() if rule_type else "unsupported"
        return rule

    def _evaluate_rule(self, answer: str, rule: Dict[str, Any]) -> str:
        """Evaluates a single normalized rule against the answer.

        Returns:
            "pass", "fail", or "unsupported".
        """

        rule_type = str(rule.get("type") or "").lower()

        if rule_type in {"must_contain", "contains", "include"}:
            needle = self._normalize_text(rule.get("value") or rule.get("needle") or "")
            if not needle:
                return "fail"
            return "pass" if needle in answer else "fail"

        if rule_type in {"must_not_contain", "not_contains", "exclude"}:
            needle = self._normalize_text(rule.get("value") or rule.get("needle") or "")
            if not needle:
                return "fail"
            return "pass" if needle not in answer else "fail"

        if rule_type in {"regex_match", "match_regex", "pattern"}:
            pattern = rule.get("pattern") or rule.get("value")
            if not isinstance(pattern, str) or not pattern:
                return "fail"
            flags = re.IGNORECASE if not self._case_sensitive() else 0
            return "pass" if re.search(pattern, answer, flags=flags) else "fail"

        if rule_type in {"max_length", "length_max"}:
            threshold = self._to_int(rule.get("value") or rule.get("max"))
            if threshold is None:
                return "fail"
            return "pass" if len(answer) <= threshold else "fail"

        if rule_type in {"min_length", "length_min"}:
            threshold = self._to_int(rule.get("value") or rule.get("min"))
            if threshold is None:
                return "fail"
            return "pass" if len(answer) >= threshold else "fail"

        if rule_type in {"equals", "exact_match"}:
            target = self._normalize_text(rule.get("value") or rule.get("target") or "")
            if not target:
                return "fail"
            return "pass" if answer == target else "fail"

        return "unsupported"

    def _materialize_rules_from_instruction_ids(
        self,
        instruction_ids: Sequence[Any],
        instruction_kwargs: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Builds canonical rules from instruction ids and kwargs payload."""

        rules: list[Dict[str, Any]] = []
        for idx, raw_id in enumerate(instruction_ids):
            if raw_id is None:
                continue
            instruction_id = str(raw_id).strip().lower()
            if not instruction_id:
                continue

            # Keep handlers intentionally small and explicit to stay diagnosable.
            if instruction_id in {"contains", "must_contain"}:
                needle = instruction_kwargs.get(instruction_id) or instruction_kwargs.get("contains")
                rules.append({"rule_id": f"inst_{idx}_{instruction_id}", "type": "must_contain", "value": needle})
            elif instruction_id in {"not_contains", "must_not_contain"}:
                needle = instruction_kwargs.get(instruction_id) or instruction_kwargs.get("not_contains")
                rules.append({"rule_id": f"inst_{idx}_{instruction_id}", "type": "must_not_contain", "value": needle})
            elif instruction_id in {"regex", "regex_match"}:
                pattern = instruction_kwargs.get(instruction_id) or instruction_kwargs.get("regex")
                rules.append({"rule_id": f"inst_{idx}_{instruction_id}", "type": "regex_match", "pattern": pattern})
            elif instruction_id in {"max_length", "min_length", "exact_match", "equals"}:
                value = instruction_kwargs.get(instruction_id)
                rules.append({"rule_id": f"inst_{idx}_{instruction_id}", "type": instruction_id, "value": value})
            else:
                rules.append({"rule_id": f"inst_{idx}_{instruction_id}", "type": "unsupported"})
        return rules

    def _normalize_text(self, value: Any) -> str:
        """Normalizes text deterministically with configurable case policy."""

        text = "" if value is None else str(value)
        if bool(self.args.get("strip_whitespace", True)):
            text = text.strip()
        if bool(self.args.get("collapse_whitespace", True)):
            text = re.sub(r"\s+", " ", text)
        if not self._case_sensitive():
            text = text.lower()
        return text

    def _case_sensitive(self) -> bool:
        """Returns case-sensitivity preference."""

        return bool(self.args.get("case_sensitive", False))

    @staticmethod
    def _to_int(value: Any) -> int | None:
        """Converts values to int when possible."""

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Normalizes values to list for rule payload extraction."""

        if value is None:
            return []
        if isinstance(value, dict):
            return [dict(value)]
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, Iterable):
            return list(value)
        return [value]

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        """Normalizes kwargs-like payloads to dictionary."""

        if isinstance(value, dict):
            return dict(value)
        return {}


@registry.asset(
    "metrics",
    "inverse_ifeval_judge_pass_rate",
    desc="Inverse IFEval pass-rate metric using judge model outputs",
    tags=("instruction_following", "ifeval", "inverse", "judge"),
    default_aggregation="mean",
)
class InverseIFEvalJudgePassRateMetric(SimpleMetric):
    """Computes pass rate from judge model verdicts."""

    value_key = "pass_rate"

    def compute(self, context: MetricContext) -> MetricResult:
        """Converts judge output payload into a stable binary pass signal.

        Args:
            context: Runtime metric context carrying judge outputs.

        Returns:
            MetricResult with normalized pass-rate values and parsing metadata.
        """

        # STEP 1: Resolve threshold/fallback and normalize judge output payload.
        threshold = float(self.args.get("threshold", 0.5))
        fallback = float(self.args.get("fallback", 0.0))
        judge_output = dict(context.judge_output or {})

        # STEP 2: Parse score signal from judge payload using resilient fallbacks.
        score, source, raw_value = self._extract_score(judge_output, fallback=fallback)
        score = max(0.0, min(1.0, float(score)))
        passed = 1.0 if score >= threshold else 0.0

        # STEP 3: Emit metric values and diagnostics for traceability.
        return MetricResult(
            sample_id=context.sample_id,
            values={"pass_rate": passed, "passed": passed},
            metadata={
                "judge_score": score,
                "judge_threshold": threshold,
                "judge_source": source,
                "judge_raw_value": raw_value,
            },
        )

    def _extract_score(self, judge_output: Mapping[str, Any], *, fallback: float) -> tuple[float, str, Any]:
        """Extracts a normalized score from judge output mappings."""

        if not judge_output:
            return fallback, "fallback.empty_judge_output", None

        for key in ("score", "pass_rate", "passed", "pass", "result", "verdict", "label", "answer"):
            if key not in judge_output:
                continue
            parsed = self._coerce_score(judge_output.get(key))
            if parsed is not None:
                return parsed, f"judge_output.{key}", judge_output.get(key)

        return fallback, "fallback.unrecognized_judge_output", dict(judge_output)

    def _coerce_score(self, value: Any) -> float | None:
        """Coerces heterogeneous judge values into a [0, 1] score."""

        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return self._normalize_numeric_score(float(value))
        if isinstance(value, Mapping):
            for key in ("score", "pass_rate", "passed", "pass", "result", "verdict", "label", "answer"):
                if key in value:
                    parsed = self._coerce_score(value.get(key))
                    if parsed is not None:
                        return parsed
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed_json = json.loads(text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed_json = None
                if isinstance(parsed_json, Mapping):
                    parsed = self._coerce_score(parsed_json)
                    if parsed is not None:
                        return parsed

            normalized = self._normalize_token(text)
            if normalized in {
                "1",
                "true",
                "yes",
                "y",
                "pass",
                "passed",
                "correct",
                "compliant",
                "符合",
                "通过",
                "是",
            }:
                return 1.0
            if normalized in {
                "0",
                "false",
                "no",
                "n",
                "fail",
                "failed",
                "incorrect",
                "non_compliant",
                "不符合",
                "未通过",
                "否",
            }:
                return 0.0

            number = self._extract_first_number(text)
            if number is not None:
                return self._normalize_numeric_score(number)

        return None

    @staticmethod
    def _normalize_token(text: str) -> str:
        """Normalizes token-like verdict strings."""

        token = text.strip().lower()
        token = token.replace("-", "_")
        token = re.sub(r"\s+", "_", token)
        return token

    @staticmethod
    def _extract_first_number(text: str) -> float | None:
        """Extracts the first numeric token from a string."""

        match = re.search(r"[-+]?\d*\.?\d+", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _normalize_numeric_score(value: float) -> float:
        """Normalizes raw numeric scores into [0, 1]."""

        if value < 0:
            return 0.0
        if value <= 1:
            return value
        if value <= 100:
            return value / 100.0
        return 1.0


__all__ = ["InverseIFEvalPassRateMetric", "InverseIFEvalJudgePassRateMetric"]
