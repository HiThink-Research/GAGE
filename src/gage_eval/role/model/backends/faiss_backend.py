"""Faiss-based retrieval backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "faiss",
    desc="Faiss vector retrieval backend",
    tags=("retrieval", "local"),
    modalities=("embedding",),
)
class FaissRetrievalBackend(EngineBackend):
    """Loads document embeddings and performs nearest-neighbour search."""

    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss-gpu/faiss-cpu is required for FaissRetrievalBackend") from exc

        documents_path = config.get("documents_path")
        if not documents_path:
            raise ValueError("FaissRetrievalBackend requires 'documents_path'")
        documents = Path(documents_path)
        if not documents.exists():
            raise FileNotFoundError(f"Faiss document file not found: {documents}")

        uids: List[str] = []
        vectors: List[List[float]] = []
        with documents.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                vectors.append(record["predict_result"])
                uids.append(record["uid"])

        vector_matrix = np.array(vectors, dtype=np.float32)

        index = faiss.IndexFlatIP(vector_matrix.shape[1])
        if config.get("use_gpu", True) and faiss.get_num_gpus() > 0:
            resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(resources, 0, index)
        else:
            self.index = index
        self.index.add(vector_matrix)
        self.uids = uids
        self.search_num = config.get("search_num", 20)

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        vector = (
            payload.get("query_vector")
            or payload.get("inputs")
            or payload.get("sample", {}).get("predict_result")
        )
        if not vector:
            raise ValueError("FaissRetrievalBackend expects 'query_vector' or sample['predict_result']")
        if not isinstance(vector, Sequence):
            raise TypeError("Query vector must be a sequence of floats")

        query = np.array([vector], dtype=np.float32)
        distances, indices = self.index.search(query, self.search_num)
        hits = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.uids):
                continue
            hits.append({"uid": self.uids[idx], "score": float(score)})
        return {"retrieval": hits}
