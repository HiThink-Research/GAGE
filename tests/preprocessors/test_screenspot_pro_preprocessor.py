import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.screespot_pro import ScreenSpotProPreprocessor
from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass
from PIL import Image
import random
import numpy as np

def generate_random_image_v2(width, height):
    # 生成 [height, width, 3] 的随机 uint8 数组
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    return img

class ScreenSpotProPreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        """Test basic ScreenSpot-Pro sample processing."""
        label_idx = 0  # android_studio_mac
        expected_class = "android_studio_mac"
        decoded_image = generate_random_image_v2(32, 32)
        sample = {
            "decoded_image": decoded_image,
            "label": label_idx,
        }
        pre = ScreenSpotProPreprocessor()

        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertEqual(len(ret.messages), 1)
        self.assertEqual(ret.messages[0].role, "user")
        self.assertEqual(len(ret.messages[0].content), 2)  # text + image
        self.assertIn("I want to identify a UI element that best matches my instruction", ret.messages[0].content[0].text)
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)
        self.assertEqual(ret.references[0], expected_class)
        self.assertEqual(ret.label, expected_class)
        self.assertEqual(ret.metadata["label_index"], label_idx)
        self.assertEqual(ret.metadata["class_name"], expected_class)

    def test_to_sample_with_system_prompt(self):
        """Test ScreenSpot-Pro preprocessor with system prompt."""
        label_idx = 1  # autocad_windows
        expected_class = "autocad_windows"
        decoded_image = generate_random_image_v2(32, 32)
        sample = {
            "image": decoded_image,
            "label": label_idx,
        }
        pre = ScreenSpotProPreprocessor()

        ret = pre.to_sample(sample, system_prompt="You are a helpful assistant.")
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertEqual(len(ret.messages), 2)
        self.assertEqual(ret.messages[0].role, "system")
        self.assertEqual(ret.messages[1].role, "user")
        self.assertEqual(ret.references[0], expected_class)
        self.assertEqual(ret.label, expected_class)

    def test_screenspot_pro_all_classes(self):
        """Test ScreenSpot-Pro preprocessor handles all class labels."""
        pre = ScreenSpotProPreprocessor()
        decoded_image = generate_random_image_v2(32, 32)

        expected_classes = [
            "android_studio_mac", "autocad_windows", "blender_windows", "davinci_resolve_macos",
            "eviews_windows", "excel_macos", "fl_studio_windows", "illustrator_windows",
            "inventor_windows", "linux_ubuntu", "macos_sonoma", "matlab_windows",
            "origin_windows", "photoshop_windows", "powerpoint_windows", "premiere_windows",
            "pycharm_macos", "quartus_windows", "solidworks_windows", "stata_windows",
            "unreal_engine_windows", "vivado_windows", "visual_studio_code_macos",
            "vmware_fusion_macos", "windows_11", "word_macos"
        ]

        for label_idx in range(len(expected_classes)):
            record = {
                "decoded_image": decoded_image,
                "label": label_idx,
            }
            sample = pre.to_sample(record)
            self.assertEqual(sample.references[0], expected_classes[label_idx])
            self.assertEqual(sample.label, expected_classes[label_idx])
            self.assertEqual(sample.metadata["label_index"], label_idx)
            self.assertEqual(sample.metadata["class_name"], expected_classes[label_idx])

    def test_screenspot_pro_invalid_label(self):
        """Test ScreenSpot-Pro preprocessor handles invalid label indices."""
        pre = ScreenSpotProPreprocessor()
        decoded_image = generate_random_image_v2(32, 32)

        # Test negative label
        record = {
            "image": decoded_image,
            "label": -1,
        }
        with self.assertRaises(ValueError):
            pre.to_sample(record)

        # Test label too large
        record = {
            "image": decoded_image,
            "label": 26,  # Only 0-25 are valid
        }
        with self.assertRaises(ValueError):
            pre.to_sample(record)

    def test_screenspot_pro_no_image(self):
        """Test ScreenSpot-Pro preprocessor without image."""
        pre = ScreenSpotProPreprocessor()
        record = {
            "label": 0,
        }

        sample = pre.to_sample(record)

        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        self.assertEqual(len(sample.messages), 1)
        self.assertEqual(len(sample.messages[0].content), 1)  # Only text
        self.assertEqual(sample.messages[0].content[0].type, "text")
        self.assertEqual(sample.references[0], "android_studio_mac")
        self.assertEqual(sample.label, "android_studio_mac")

    def test_screenspot_pro_missing_label(self):
        """Test ScreenSpot-Pro preprocessor handles missing label."""
        pre = ScreenSpotProPreprocessor()
        record = {
            "image": generate_random_image_v2(32, 32),
        }

        with self.assertRaises(ValueError):
            pre.to_sample(record)

    def test_screenspot_pro_custom_id(self):
        """Test ScreenSpot-Pro preprocessor with custom sample ID."""
        pre = ScreenSpotProPreprocessor()
        decoded_image = generate_random_image_v2(32, 32)
        custom_id = "test_sample_001"

        record = {
            "id": custom_id,
            "image": decoded_image,
            "label": 0,
        }

        sample = pre.to_sample(record)
        self.assertEqual(sample.id, custom_id)


if __name__ == "__main__":
    unittest.main()