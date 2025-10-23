import os
import sys

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

sys.path.append(os.path.realpath(os.path.dirname(os.path.abspath(__file__))))
from adaptor_base import set_device, EngineAdaptorBase

set_device()

import argparse
import asyncio
import ujson as json
import numpy as np
import faiss

OUTPUT_TYPES = ['retrieval']


class FaissEngineAdaptor(EngineAdaptorBase):

    async def predict_sample(self, inputs: dict) -> list[str]:
        """
        单条样本推理，检索与query相近的docs

        Args:
            inputs (`list[int] | dict`):
                包含query的embedding（predict_result字段）
        """
        qvec = inputs['predict_result']
        qm = np.array([qvec]).astype(np.float32)
        distances, ids = self.index.search(qm, self.args.search_num)
        distances, ids = distances[0], ids[0]
        uids = [self.uids[i] for i in ids]
        return uids

    def load_model(self):
        print("加载向量")
        self.uids = []
        vector_list = []
        for line in open(self.args.documents):
            d = json.loads(line)
            vector_list.append(np.array(d['predict_result']).astype(np.float32))
            self.uids.append(d['uid'])
        vector_matrix = np.array(vector_list).astype(np.float32)

        print("构建索引")
        res = faiss.StandardGpuResources()
        res.setTempMemory(30 * 1024 * 1024 * 1024)
        index = faiss.IndexFlatIP(vector_matrix.shape[1])
        self.index = faiss.index_cpu_to_gpu(res, 0, index)
        self.index.add(vector_matrix)
        print("加载完成")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=str, required=True, help="Document vector file path")
    parser.add_argument("--server_addr", type=str, required=True, help="Data server address")
    parser.add_argument("--server_port", type=int, required=True, help="Data server port")
    parser.add_argument("--output_type", type=str, default='retrieval', choices=OUTPUT_TYPES, help="Output type")
    parser.add_argument("--search_num", type=int, default=20, help="Max number of retieved documents")
    args, unknown_args = parser.parse_known_args()

    asyncio.run(FaissEngineAdaptor(args).run_predict_until_complete())
