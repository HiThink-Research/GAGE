import sys
import threading
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.loaders.loader_utils import TokenizerManager


class DummyLoader:
    def __init__(self):
        self.count = 0

    def load(self):
        self.count += 1
        return f"tok-{self.count}"


class TokenizerManagerTests(unittest.TestCase):
    def test_lru_reuse(self):
        mgr = TokenizerManager(max_size=2)
        loader = DummyLoader()
        tok1 = mgr.get("a", {}, loader.load)
        tok2 = mgr.get("a", {}, loader.load)
        self.assertEqual(tok1, tok2)
        self.assertEqual(loader.count, 1)

    def test_lru_evict(self):
        mgr = TokenizerManager(max_size=2)
        loader = DummyLoader()
        mgr.get("a", {}, loader.load)
        mgr.get("b", {}, loader.load)
        mgr.get("c", {}, loader.load)  # should evict oldest
        mgr.get("a", {}, loader.load)  # reload a
        self.assertEqual(loader.count, 4)

    def test_concurrent_get(self):
        mgr = TokenizerManager(max_size=4)
        loader = DummyLoader()
        results = []

        def worker():
            results.append(mgr.get("a", {}, loader.load))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(loader.count, 1)
        self.assertTrue(all(r == "tok-1" for r in results))


if __name__ == "__main__":
    unittest.main()
