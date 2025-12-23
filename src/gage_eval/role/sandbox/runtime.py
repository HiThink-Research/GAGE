"""Sandbox runtime adapter stub."""

from __future__ import annotations

from typing import Any


class SandboxRuntimeAdapter:
    def __init__(self, profile: dict) -> None:
        self.profile = profile

    def borrow(self) -> Any:
        return {"sandbox": self.profile.get("profile_id", "default")}

    def return_(self, handle: Any) -> None:
        _ = handle
