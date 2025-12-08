import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.utils.multimodal import embed_local_message_images


class DocToVisualContentRootTests(unittest.TestCase):
    def test_content_root_arg_expanded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            img_path.write_bytes(b"test")
            sample = {
                "messages": [
                    {
                        "content": [
                            {"type": "image_url", "image_url": {"url": "img.png"}}
                        ]
                    }
                ]
            }
            embed_local_message_images(sample, content_field="messages.0.content", content_root=tmpdir)
            url = sample["messages"][0]["content"][0]["image_url"]["url"]
            self.assertTrue(url.startswith("data:"))
            self.assertEqual(sample["metadata"]["image_root"], str(Path(tmpdir).resolve()))


if __name__ == "__main__":
    unittest.main()
