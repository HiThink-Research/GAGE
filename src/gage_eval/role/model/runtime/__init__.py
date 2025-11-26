"""Runtime mixins shared by ModelRoleAdapter implementations."""

from gage_eval.role.model.runtime.async_generation import AsyncGenerationMixin
from gage_eval.role.model.runtime.http_retry import HttpRetryMixin
from gage_eval.role.model.runtime.text_generation import TextGenerationMixin

__all__ = [
    "AsyncGenerationMixin",
    "HttpRetryMixin",
    "TextGenerationMixin",
]
