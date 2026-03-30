"""Tests for gage_eval.role.model.reasoning utilities."""

from __future__ import annotations

import types
import unittest

from gage_eval.role.model.reasoning import (
    extract_reasoning_content,
    resolve_thinking_kwargs,
    strip_thinking_tags,
)


class TestExtractReasoningContent(unittest.TestCase):
    """Tests for extract_reasoning_content()."""

    def test_returns_none_for_none_message(self) -> None:
        self.assertIsNone(extract_reasoning_content(None))

    def test_extracts_from_dict(self) -> None:
        msg = {"content": "42", "reasoning_content": "Let me think..."}
        self.assertEqual(extract_reasoning_content(msg), "Let me think...")

    def test_returns_none_when_missing_from_dict(self) -> None:
        msg = {"content": "42"}
        self.assertIsNone(extract_reasoning_content(msg))

    def test_extracts_from_object_attribute(self) -> None:
        msg = types.SimpleNamespace(content="42", reasoning_content="Chain of thought")
        self.assertEqual(extract_reasoning_content(msg), "Chain of thought")

    def test_returns_none_when_attribute_missing(self) -> None:
        msg = types.SimpleNamespace(content="42")
        self.assertIsNone(extract_reasoning_content(msg))

    def test_converts_non_string_to_string(self) -> None:
        msg = {"content": "42", "reasoning_content": 123}
        self.assertEqual(extract_reasoning_content(msg), "123")


class TestStripThinkingTags(unittest.TestCase):
    """Tests for strip_thinking_tags()."""

    def test_no_tags_returns_original(self) -> None:
        text = "The answer is 42."
        clean, thinking = strip_thinking_tags(text)
        self.assertEqual(clean, "The answer is 42.")
        self.assertIsNone(thinking)

    def test_empty_string(self) -> None:
        clean, thinking = strip_thinking_tags("")
        self.assertEqual(clean, "")
        self.assertIsNone(thinking)

    def test_single_think_tag(self) -> None:
        text = "<think>Let me reason step by step.</think>The answer is 42."
        clean, thinking = strip_thinking_tags(text)
        self.assertEqual(clean, "The answer is 42.")
        self.assertEqual(thinking, "Let me reason step by step.")

    def test_multiple_think_tags(self) -> None:
        text = "<think>Step 1</think>Part 1 <think>Step 2</think>Part 2"
        clean, thinking = strip_thinking_tags(text)
        self.assertEqual(clean, "Part 1 Part 2")
        self.assertIn("Step 1", thinking)
        self.assertIn("Step 2", thinking)

    def test_multiline_thinking(self) -> None:
        text = "<think>\nLine 1\nLine 2\n</think>\nFinal answer."
        clean, thinking = strip_thinking_tags(text)
        self.assertEqual(clean, "Final answer.")
        self.assertIn("Line 1", thinking)
        self.assertIn("Line 2", thinking)


class TestResolveThinkingKwargs(unittest.TestCase):
    """Tests for resolve_thinking_kwargs()."""

    def test_none_mode_returns_empty_dict(self) -> None:
        result = resolve_thinking_kwargs(None)
        self.assertEqual(result, {})

    def test_disabled_mode(self) -> None:
        result = resolve_thinking_kwargs("disabled")
        self.assertFalse(result["enable_thinking"])

    def test_enabled_mode(self) -> None:
        result = resolve_thinking_kwargs("enabled")
        self.assertTrue(result["enable_thinking"])

    def test_case_insensitive(self) -> None:
        for mode in ("Disabled", "DISABLED", "off", "OFF", "False", "no"):
            result = resolve_thinking_kwargs(mode)
            self.assertFalse(result["enable_thinking"], f"Failed for mode={mode}")

        for mode in ("Enabled", "ENABLED", "on", "ON", "True", "yes"):
            result = resolve_thinking_kwargs(mode)
            self.assertTrue(result["enable_thinking"], f"Failed for mode={mode}")

    def test_extra_kwargs_override(self) -> None:
        # Even if thinking_mode is "disabled", explicit enable_thinking=True wins
        result = resolve_thinking_kwargs("disabled", {"enable_thinking": True})
        self.assertTrue(result["enable_thinking"])

    def test_extra_kwargs_merge(self) -> None:
        result = resolve_thinking_kwargs("disabled", {"reasoning_effort": "high"})
        self.assertFalse(result["enable_thinking"])
        self.assertEqual(result["reasoning_effort"], "high")

    def test_none_mode_with_extra_kwargs(self) -> None:
        result = resolve_thinking_kwargs(None, {"reasoning_effort": "low"})
        self.assertEqual(result, {"reasoning_effort": "low"})


if __name__ == "__main__":
    unittest.main()
