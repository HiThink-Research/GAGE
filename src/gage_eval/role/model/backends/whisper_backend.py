"""faster-whisper backend for ASR tasks."""

from __future__ import annotations

from typing import Any, Dict, List

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "whisper_asr",
    desc="faster-whisper ASR backend",
    tags=("asr", "local"),
    modalities=("audio",),
)
class WhisperASRBackend(EngineBackend):
    """Simple wrapper around ``faster_whisper.WhisperModel``."""

    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper must be installed for WhisperASRBackend") from exc

        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("WhisperASRBackend requires 'model_path'")

        device = config.get("device", "cuda")
        compute_type = config.get("compute_type", "int8_float16")
        self.transcribe_kwargs = {
            "beam_size": 5,
            "word_timestamps": False,
            "without_timestamps": True,
            "condition_on_previous_text": False,
        }
        self.transcribe_kwargs.update(config.get("transcribe_kwargs", {}))
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        audio_paths = self._extract_audio_paths(payload)
        if not audio_paths:
            raise ValueError("WhisperASRBackend expects audio paths under sample['inputs']['multi_modal_data']['audio']")
        text_fragments: List[str] = []
        lang = None
        for path in audio_paths:
            segments, info = self.model.transcribe(path, **self.transcribe_kwargs)
            text_fragments.append("".join(segment.text for segment in segments))
            lang = info.language or lang
        return {"answer": "".join(text_fragments), "language": lang}

    @staticmethod
    def _extract_audio_paths(payload: Dict[str, Any]) -> List[str]:
        sample = payload.get("sample", {})
        inputs = payload.get("inputs") or sample.get("inputs") or {}
        mm_data = inputs.get("multi_modal_data") or sample.get("multi_modal_data") or {}
        audio = mm_data.get("audio") or inputs.get("audio")
        if isinstance(audio, list):
            return audio
        if isinstance(audio, str):
            return [audio]
        return []
