from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AppWorldKitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

