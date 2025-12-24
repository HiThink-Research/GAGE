"""FlagEmbedding backend for dense embeddings."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "flag_embedding",
    desc="FlagEmbedding text embedding backend",
    tags=("embedding", "local"),
    modalities=("text",),
)
class FlagEmbeddingBackend(EngineBackend):
    """Thin wrapper around ``FlagEmbedding.BGEM3FlagModel``."""

    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("FlagEmbedding must be installed for FlagEmbeddingBackend") from exc

        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("FlagEmbeddingBackend requires 'model_name'")
        self.model = BGEM3FlagModel(model_name, use_fp16=config.get("use_fp16", True))
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 32)

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample = payload.get("sample", {})
        text = payload.get("text") or sample.get("text") or sample.get("prompt") or sample.get("query")
        if not text:
            raise ValueError("FlagEmbeddingBackend expects text under payload['text'] or sample")
        embeddings = self.model.encode(
            [text],
            batch_size=1,
            max_length=self.max_length,
        )["dense_vecs"][0]
        return {"embedding": embeddings.tolist()}
