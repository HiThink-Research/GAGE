"""Mathvista Resource Providers"""

from __future__ import annotations

import random
from typing import Any, Dict

from gage_eval.assets.datasets.bundles.base import BaseBundle
from gage_eval.assets.datasets.utils.reader import read_json

import os

class MathVistaBundle(BaseBundle):
    """ Provide MathVista related resources """

    caption_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data/texts/captions_bard.json'
    )
    
    ocr_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data/texts/ocrs_easyocr.json'
    )
    def __init__(self,  **kwargs: Any):
        super().__init__(**kwargs)        
        self.caption_data = {}
        self.ocr_data = {}

    def load(self) -> None:
        if os.path.exists(self.caption_data_path):
            self.caption_data = read_json(self.caption_data_path)["texts"]
        
        if os.path.exists(self.ocr_data_path):
            self.ocr_data = read_json(self.ocr_data_path)["texts"]

    def provide(self, sample: Dict[str, Any], **kwargs: Any) -> Any:  # pragma: no cover - abstract
        sample_dict = dict(sample)   
        pid = sample_dict.get("pid")
        sample_dict['caption'] = self.caption_data.get(pid)
        sample_dict['ocr'] = self.ocr_data.get(pid)       
        return sample_dict

if __name__ == '__main__':
    t = MathVistaBundle()
    t.load()
    t.provide({})