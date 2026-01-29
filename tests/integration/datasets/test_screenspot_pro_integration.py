"""Integration tests for ScreenSpot-Pro dataset loading and preprocessing."""

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.registry import registry


class ScreenSpotProIntegrationTests(unittest.TestCase):
    """Integration tests for ScreenSpot-Pro dataset end-to-end flow."""

    def test_registry_has_screenspot_pro_components(self):
        """Test that ScreenSpot-Pro components are registered in the registry."""
        # Check bundle registration
        bundle_class = registry.get("bundles", "screenspot_pro")
        self.assertIsNotNone(bundle_class)
        self.assertEqual(bundle_class.__name__, "ScreenSpotProBundleProvider")

        # Check preprocessor registration
        preprocessor_class = registry.get("dataset_preprocessors", "screenspot_pro")
        self.assertIsNotNone(preprocessor_class)
        self.assertEqual(preprocessor_class.__name__, "ScreenSpotProPreprocessorProvider")

    def test_screenspot_pro_preprocessor_basic_functionality(self):
        """Test basic ScreenSpot-Pro preprocessor functionality."""
        preprocessor_class = registry.get("dataset_preprocessors", "screenspot_pro")
        preprocessor = preprocessor_class()

        # Test with minimal data
        from PIL import Image
        import numpy as np

        # Create a small test image
        data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        test_image = Image.fromarray(data, 'RGB')

        record = {
            "decoded_image": test_image,
            "label": 0,  # android_studio_mac
        }

        sample = preprocessor.to_sample(record)

        # Verify basic properties
        self.assertIsNotNone(sample)
        self.assertEqual(sample.schema_version, "0.0.1")
        self.assertIsNotNone(sample.id)
        self.assertEqual(len(sample.messages), 1)
        self.assertEqual(sample.messages[0].role, "user")
        self.assertEqual(len(sample.messages[0].content), 2)  # text + image
        self.assertEqual(sample.references[0], "android_studio_mac")
        self.assertEqual(sample.label, "android_studio_mac")
        self.assertEqual(sample.metadata["label_index"], 0)
        self.assertEqual(sample.metadata["class_name"], "android_studio_mac")


if __name__ == "__main__":
    unittest.main()