import os
import sys

#os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

sys.path.append(os.path.realpath(os.path.dirname(os.path.abspath(__file__))))
from adaptor_base import set_device, EngineAdaptorBase

set_device()

import argparse
import asyncio
import ujson as json
import numpy as np
#import faiss

import librosa
from faster_whisper import WhisperModel

OUTPUT_TYPES = ['text']


class AsrEngineAdaptor(EngineAdaptorBase):

    async def predict_sample(self,inputs: str ) -> str:
        model = self.model
        segments, info = model.transcribe(inputs["multi_modal_data"]["audio"][0], beam_size=5, word_timestamps=False,without_timestamps=True,  condition_on_previous_text=False)  # for finetune whisper
        result = "".join([segment.text for segment in segments])
        lang = info.language  
        print(result) 
        return result     

    def load_model(self):
        print(" model loading...")
        
        try:
            model = WhisperModel(args.model, device="cuda", compute_type="int8_float16") #"int8_float16") #"float16")
            self.model = model
        except Exception as e:
            print("load model failed:", str(e))
            return None
        
        print(" load model success.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model file path")
#    parser.add_argument("--documents", type=str, required=True, help="Document vector file path")
    parser.add_argument("--server_addr", type=str, required=True, help="Data server address")
    parser.add_argument("--server_port", type=int, required=True, help="Data server port")
    parser.add_argument("--output_type", type=str, default='text', choices=OUTPUT_TYPES, help="Output type")
#    parser.add_argument("--search_num", type=int, default=20, help="Max number of retieved documents")
    args, unknown_args = parser.parse_known_args()

    asyncio.run(AsrEngineAdaptor(args).run_predict_until_complete())
