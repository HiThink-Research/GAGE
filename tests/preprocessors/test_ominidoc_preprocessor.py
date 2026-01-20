import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.omnidoc_preprocessor import OmniDocPreprocessor
from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class OmniDocPreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        sample_id='omninidoc-123'
        prompt = "You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:\n\n        1. Text Processing:\n        - Accurately recognize all text content in the PDF image without guessing or inferring.\n        - Convert the recognized text into Markdown format.\n        - Maintain the original document structure, including headings, paragraphs, lists, etc.\n\n        2. Mathematical Formula Processing:\n        - Convert all mathematical formulas to LaTeX format.\n        - Enclose inline formulas with \\( \\). For example: This is an inline formula \\( E = mc^2 \\)\n        - Enclose block formulas with \\\\[ \\\\]. For example: \\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]\n\n        3. Table Processing:\n        - Convert tables to HTML format.\n        - Wrap the entire table with <table> and </table>.\n\n        4. Figure Handling:\n        - Ignore figures content in the PDF image. Do not attempt to describe or convert images.\n\n        5. Output Format:\n        - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.\n        - For complex layouts, try to maintain the original document's structure and format as closely as possible.\n\n        Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments."
        image='PPT_esea-app101_page_003.png'
        _dataset_id='omnidocbench_val'
        _dataset_metadata={'path': '/mnt/sdb1/ywt/OmniDocBench-main/OmniDocBench1_5/omnidocbench15_gage_mini_r.jsonl'}
        sample = {
            "id": sample_id,
            "image": image,
            'prompt': prompt,
            "_dataset_id": _dataset_id,
            "_dataset_metadata": _dataset_metadata,
        }
        pre = OmniDocPreprocessor()

        ret = pre.to_sample(sample, question_field="prompt", content_field='image', content_root='/mnt/sdb1/ywt/OmniDocBench-main/OmniDocBench1_5/images')

        self.assertIsNotNone(ret)
        self.assertIn(prompt, ret['messages'][0]['content'][0]['text'])
        self.assertIsNotNone(ret['messages'][0]['content'][1]['image_url'])
        self.assertIsNotNone(ret['id'])
        self.assertIsNotNone(ret['image'])
        self.assertIsNotNone(ret['_dataset_id'])