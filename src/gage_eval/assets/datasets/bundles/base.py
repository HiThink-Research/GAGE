"""Dataset Resource Provider abstractions (template method)."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Optional, Sequence

from loguru import logger

class BaseBundle:
    """Base class for structured resource providers."""

    name = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def load(self) -> None:
        raise NotImplementedError

    def provider(self, sample: Dict[str, Any], **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError
