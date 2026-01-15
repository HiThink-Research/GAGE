import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.choice import extract_single_choice_letter


class ChoiceTests(unittest.TestCase):
    def test_choice(self):
        ss = "answer is \boxed{A}"
        ch = extract_single_choice_letter(ss)
        self.assertEqual(ch, "A")

        ss = "<answer> B"
        ch = extract_single_choice_letter(ss)
        self.assertEqual(ch, "B")

        ss = "A B C"
        ch = extract_single_choice_letter(ss)
        self.assertEqual(ch, "C")

        ss = "The Choice is D"
        ch = extract_single_choice_letter(ss)
        self.assertEqual(ch, "D")                  

if __name__ == "__main__":
    unittest.main()
