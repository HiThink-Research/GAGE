"""Base scheduler contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping


class Scheduler(ABC):
    binding_id: str
    family: str
    defaults: dict[str, object]

    def __init__(
        self,
        *,
        binding_id: str,
        family: str | None = None,
        defaults: Mapping[str, object] | None = None,
    ) -> None:
        self.binding_id = binding_id
        self.family = family or self.family
        self.defaults = dict(defaults or {})

    @abstractmethod
    def run(self, session) -> None:
        raise NotImplementedError
