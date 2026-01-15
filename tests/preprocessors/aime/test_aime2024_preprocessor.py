import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.aime.aime2024 import AIME2024Preprocessor
from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class AIME2024PreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        ID = '2024-II-4'
        Problem = """Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
\[\log_2\left({x \over yz}\right) = {1 \over 2}\]
\[\log_2\left({y \over xz}\right) = {1 \over 3}\]
\[\log_2\left({z \over xy}\right) = {1 \over 4}\]
Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$."""
        Solution = """Denote $\log_2(x) = a$, $\log_2(y) = b$, and $\log_2(z) = c$.

Then, we have:
$a-b-c = \frac{1}{2}$,
$-a+b-c = \frac{1}{3}$,
$-a-b+c = \frac{1}{4}$.

Now, we can solve to get $a = \frac{-7}{24}, b = \frac{-9}{24}, c = \frac{-5}{12}$.
Plugging these values in, we obtain $|4a + 3b + 2c| = \frac{25}{8} \implies \boxed{033}$."""
        Answer = '33'
        sample = {
            "ID": ID,
            "Problem": Problem,
            "Solution": Solution,
            "Answer": Answer
        }
        pre = AIME2024Preprocessor()
        
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIsNotNone(ret.metadata)
        self.assertIn(Problem, ret.messages[0].content[0].text)
        self.assertEqual(ret.id, ID)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)
        self.assertEqual(ret.label, Answer)