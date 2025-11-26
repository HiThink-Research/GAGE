"""Dataset preprocessor abstractions."""

from __future__ import annotations

from typing import Any, Dict


class DatasetPreprocessor:
    """Base class for structured preprocessors."""

    name = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def transform(self, sample: Dict[str, Any], **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError
