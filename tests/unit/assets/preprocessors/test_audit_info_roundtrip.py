import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.sample import sample_from_dict, sample_to_dict


class AuditInfoRoundtripTests(unittest.TestCase):
    def test_audit_roundtrip(self):
        raw = {
            "id": "a1",
            "_dataset_id": "d1",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "audit_info": {
                "task_id": "t1",
                "version_id": "v1",
                "check_user": "u",
                "review_time": "now",
            },
        }
        sample = sample_from_dict(raw)
        back = sample_to_dict(sample)
        self.assertEqual(back["id"], "a1")


if __name__ == "__main__":
    unittest.main()
