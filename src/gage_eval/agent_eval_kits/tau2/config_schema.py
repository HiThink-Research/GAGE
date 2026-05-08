from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Tau2KitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str | None = None
    user_simulator: dict[str, Any] | None = None
    data_dir: str | None = None
    max_steps: int | None = Field(default=None, ge=1)
    max_errors: int | None = Field(default=None, ge=0)
    respond_tool_name: str | None = None

    @field_validator("user_simulator", mode="before")
    @classmethod
    def _normalize_user_simulator(cls, value: Any) -> dict[str, Any] | None:
        return normalize_user_simulator_config(value)


def normalize_user_simulator_config(value: Any) -> dict[str, Any] | None:
    """Normalize Tau2 user simulator config into tau2-bench constructor args."""

    if value in (None, ""):
        return None
    if isinstance(value, str):
        return {"model": value, "model_args": {}}
    if not isinstance(value, dict):
        raise ValueError("user_simulator must be a string or object")

    payload = dict(value)
    simulator_type = str(payload.pop("type", "") or "").strip()
    model = payload.pop("model", None) or payload.pop("llm", None)
    model_args = payload.pop("model_args", None) or payload.pop("llm_args", None) or {}
    if not isinstance(model_args, dict):
        raise ValueError("user_simulator.model_args must be an object")
    normalized_args = dict(model_args)

    base_url = payload.pop("base_url", None) or payload.pop("api_base", None)
    if base_url is not None:
        normalized_args["api_base"] = str(base_url)
    api_key = payload.pop("api_key", None)
    if api_key is not None:
        normalized_args["api_key"] = str(api_key)
    normalized_args.update(payload)
    if simulator_type in {"openai_http", "openai-compatible", "vllm"} and not model:
        raise ValueError("user_simulator.model is required")
    return {"model": str(model) if model is not None else None, "model_args": normalized_args}
