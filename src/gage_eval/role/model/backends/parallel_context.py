"""Parallel context provider stubs."""

from __future__ import annotations

from typing import Dict


class ParallelContextProvider:
    def build(self, node_plan: Dict[str, int]) -> Dict[str, int]:
        return node_plan
