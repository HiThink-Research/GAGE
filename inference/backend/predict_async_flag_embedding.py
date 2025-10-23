import os
import sys

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

sys.path.append(os.path.realpath(os.path.dirname(os.path.abspath(__file__))))
from adaptor_base import set_device, EngineAdaptorBase

set_device()

import argparse
import asyncio

from FlagEmbedding import BGEM3FlagModel
from collections import deque
from dataclasses import dataclass, field

OUTPUT_TYPES = ['embedding']


@dataclass
class Sample:
    query: str
    output: list[float] = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


class FlagEmbeddingEngineAdaptor(EngineAdaptorBase):

    def __init__(self, *args, batch_size=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.sample_queue: deque[Sample] = deque()

    async def add_sample_until_full(self):
        await super().add_sample_until_full()
        self.encode_batch()

    def should_stop_adding_sample(self, n_added):
        return len(self.sample_queue) >= self.batch_size

    async def predict_sample(self, inputs: dict) -> list[float]:
        """
        单条样本推理，计算文本对应的embedding向量

        Args:
            inputs (`list[int] | dict`):
                包含输入模型的文本（query或content字段）
        """
        s = Sample(query=inputs['query'] if 'query' in inputs else inputs['content'])
        self.sample_queue.append(s)
        await s.done_event.wait()
        return s.output

    def encode_batch(self):
        """
        batch样本推理
        """
        batch: list[Sample] = []
        contents: list[str] = []
        while self.sample_queue and len(batch) < self.batch_size:
            s = self.sample_queue.popleft()
            batch.append(s)
            contents.append(s.query)
        if not batch:
            return
        embeddings = self.model.encode(contents, batch_size=self.batch_size, max_length=512)["dense_vecs"]
        embeddings = embeddings.tolist()
        for s, e in zip(batch, embeddings):
            s.output = e
            s.done_event.set()

    def load_model(self):
        """加载模型"""
        self.model = BGEM3FlagModel(self.args.model, use_fp16=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model file path")
    parser.add_argument("--server_addr", type=str, required=True, help="Data server address")
    parser.add_argument("--server_port", type=int, required=True, help="Data server port")
    parser.add_argument("--output_type", type=str, default='embedding', choices=OUTPUT_TYPES, help="Output type")
    parser.add_argument("--max_length", type=int, default=None, help="Max number of tokens (input and output)")
    parser.add_argument("--dtype", type=str, default='auto', help="dtype of the loaded model")
    parser.add_argument("--low_vram", action='store_true', help="Lower gpu memory usage")
    args, unknown_args = parser.parse_known_args()

    asyncio.run(FlagEmbeddingEngineAdaptor(args).run_predict_until_complete())
