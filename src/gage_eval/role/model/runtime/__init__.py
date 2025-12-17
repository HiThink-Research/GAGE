"""Runtime mixins shared by ModelRoleAdapter implementations."""

from gage_eval.role.model.runtime.async_generation_mixin import AsyncGenerationMixin
from gage_eval.role.model.runtime.chat_template_mixin import (
    BackendCapabilities,
    ChatTemplateMixin,
    ChatTemplatePolicy,
)
from gage_eval.role.model.runtime.http_retry_mixin import HttpRetryMixin
from gage_eval.role.model.runtime.text_generation_mixin import TextGenerationMixin

__all__ = [
    "AsyncGenerationMixin",
    "HttpRetryMixin",
    "TextGenerationMixin",
    "ChatTemplateMixin",
    "ChatTemplatePolicy",
    "BackendCapabilities",
]
