import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.utils.chat_templates import (
    DEFAULT_TEXT_CHAT_TEMPLATE,
    DEFAULT_VLM_CHAT_TEMPLATE,
    get_fallback_template,
)


class ChatTemplatesTests(unittest.TestCase):
    def test_default_constants(self):
        self.assertIn("messages", DEFAULT_TEXT_CHAT_TEMPLATE)
        self.assertIn("assistant", DEFAULT_TEXT_CHAT_TEMPLATE)
        self.assertIn("messages", DEFAULT_VLM_CHAT_TEMPLATE)

    def test_get_fallback_template_text(self):
        tpl = get_fallback_template("text")
        self.assertEqual(tpl, DEFAULT_TEXT_CHAT_TEMPLATE)

    def test_get_fallback_template_vlm(self):
        tpl = get_fallback_template("vlm")
        self.assertEqual(tpl, DEFAULT_VLM_CHAT_TEMPLATE)

    def test_get_fallback_template_unknown(self):
        tpl = get_fallback_template("unknown")  # type: ignore[arg-type]
        self.assertEqual(tpl, DEFAULT_TEXT_CHAT_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
