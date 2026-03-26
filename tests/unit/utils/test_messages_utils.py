import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.utils.messages import normalize_messages_for_template, stringify_message_content


class MessageUtilsTests(unittest.TestCase):
    def test_stringify_message_content_flattens_fragments(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image"},
            {"type": "text", "text": "world"},
            42,
            None,
        ]

        result = stringify_message_content(content)

        self.assertEqual(result, "hello <image> world 42")

    def test_stringify_message_content_respects_placeholders_and_skip_flags(self):
        content = [
            {"type": "text", "text": "foo"},
            {"type": "image_url", "url": "http://example.com"},
            "inline",
        ]

        without_images = stringify_message_content(content, image_placeholder=None)
        skip_non_text = stringify_message_content(content, coerce_non_text_fragments=False)

        self.assertEqual(without_images, "foo inline")
        self.assertEqual(skip_non_text, "foo <image>")

    def test_normalize_messages_for_template_returns_copies(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image"}]},
            {"role": "assistant", "content": None},
        ]

        normalized = normalize_messages_for_template(messages)

        self.assertIsNot(normalized[0], messages[0])
        self.assertEqual(normalized[0]["content"], "hi <image>")
        self.assertEqual(normalized[1]["content"], "")
        self.assertIsInstance(messages[0]["content"], list)


if __name__ == "__main__":
    unittest.main()
