from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SwebenchKitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: str | None = None

