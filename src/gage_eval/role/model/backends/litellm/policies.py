"""LiteLLM request policies."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal
from urllib.parse import urlsplit

from loguru import logger

from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile

ThinkingMode = Literal["enabled", "disabled", "auto", "unknown"]
ThinkingCapability = Literal["auto", "supported", "unsupported"]
UnsupportedAction = Literal["warn_and_continue", "fail_fast", "ignore"]


@dataclass(frozen=True)
class ThinkingResolution:
    """Resolved thinking request policy."""

    mode: ThinkingMode
    kwargs_patch: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThinkingControlPolicy:
    """Map GAGE thinking controls to provider-specific LiteLLM kwargs."""

    def __init__(
        self,
        *,
        service_profile: VLLMServiceProfile | None = None,
        capability: Any | None = None,
        on_unsupported: UnsupportedAction | None = None,
        on_mismatch: UnsupportedAction | None = None,
        record_effective_state: bool | None = None,
    ) -> None:
        self.service_profile = service_profile
        self.capability = self._normalize_capability(capability)
        self.on_unsupported: UnsupportedAction = on_unsupported or "warn_and_continue"
        # Reserved for the response-normalization phase: request building only
        # resolves outbound kwargs and does not compare returned effective state.
        self.on_mismatch: UnsupportedAction = on_mismatch or "warn_and_continue"
        self.record_effective_state = bool(record_effective_state)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        service_profile: VLLMServiceProfile | None = None,
    ) -> "ThinkingControlPolicy":
        """Build a policy from the optional LiteLLM thinking policy config."""

        if config is None:
            return cls(service_profile=service_profile)
        get = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
        return cls(
            service_profile=service_profile,
            capability=get("capability"),
            on_unsupported=get("on_unsupported"),
            on_mismatch=get("on_mismatch"),
            record_effective_state=get("record_effective_state", None),
        )

    def resolve(
        self,
        *,
        model: str,
        provider: str | None = None,
        custom_llm_provider: str | None = None,
        api_base: str | None = None,
        thinking_config: Dict[str, Any] | None = None,
    ) -> ThinkingResolution:
        """Resolve thinking mode into LiteLLM kwargs and request metadata."""

        thinking_config = dict(thinking_config or {})
        mode = self._resolve_mode(thinking_config)
        include_profile_parser_target = self.capability != "supported"
        is_vllm_target = self._is_vllm_openai_compatible_target(
            model=model,
            provider=provider,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            include_profile_parser=include_profile_parser_target,
        )
        metadata: Dict[str, Any] = {
            "thinking_mode": mode,
            "thinking_capability": self.capability,
            "thinking_target": "vllm_openai_compatible" if is_vllm_target else "generic",
        }
        if self.service_profile and self.service_profile.reasoning_parser:
            metadata["reasoning_parser"] = self.service_profile.reasoning_parser

        kwargs_patch: Dict[str, Any] = {}
        reasoning_effort = thinking_config.get("reasoning_effort")

        if mode == "enabled" and self.capability == "unsupported":
            self._handle_unsupported_enabled_capability(model)
            if reasoning_effort:
                kwargs_patch["reasoning_effort"] = reasoning_effort
            return ThinkingResolution(mode=mode, kwargs_patch=kwargs_patch, metadata=metadata)

        if (
            mode == "enabled"
            and self.capability == "auto"
            and is_vllm_target
            and not metadata.get("reasoning_parser")
        ):
            self._handle_unsupported_enabled_vllm_target(model)

        if mode in {"enabled", "disabled"} and is_vllm_target:
            kwargs_patch = {
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": mode == "enabled",
                    }
                }
            }

        if reasoning_effort:
            kwargs_patch["reasoning_effort"] = reasoning_effort

        return ThinkingResolution(mode=mode, kwargs_patch=kwargs_patch, metadata=metadata)

    @staticmethod
    def _normalize_capability(capability: Any | None) -> ThinkingCapability:
        if capability is None:
            return "auto"
        if capability is True:
            return "supported"
        if capability is False:
            return "unsupported"
        normalized = str(capability).strip().lower().replace("-", "_")
        aliases: dict[str, ThinkingCapability] = {
            "": "auto",
            "auto": "auto",
            "automatic": "auto",
            "default": "auto",
            "none": "auto",
            "supported": "supported",
            "support": "supported",
            "supports": "supported",
            "true": "supported",
            "yes": "supported",
            "enabled": "supported",
            "unsupported": "unsupported",
            "not_supported": "unsupported",
            "false": "unsupported",
            "no": "unsupported",
            "disabled": "unsupported",
        }
        return aliases.get(normalized, "auto")

    @staticmethod
    def _resolve_mode(thinking_config: Dict[str, Any]) -> ThinkingMode:
        raw_mode = thinking_config.get("thinking_mode")
        chat_template_kwargs = thinking_config.get("chat_template_kwargs")
        if raw_mode is None and isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
            enable_thinking = chat_template_kwargs.get("enable_thinking")
            if enable_thinking is True:
                return "enabled"
            if enable_thinking is False:
                return "disabled"
        if raw_mode is None and "enable_thinking" in thinking_config:
            enable_thinking = thinking_config.get("enable_thinking")
            if enable_thinking is True:
                return "enabled"
            if enable_thinking is False:
                return "disabled"
        if raw_mode is None:
            return "auto"
        normalized = str(raw_mode).strip().lower()
        if normalized in {"enabled", "enable", "true", "on"}:
            return "enabled"
        if normalized in {"disabled", "disable", "false", "off"}:
            return "disabled"
        if normalized in {"auto", "default", "none", ""}:
            return "auto"
        return "unknown"

    def _is_vllm_openai_compatible_target(
        self,
        *,
        model: str,
        provider: str | None,
        custom_llm_provider: str | None,
        api_base: str | None,
        include_profile_parser: bool = True,
    ) -> bool:
        model_lower = (model or "").lower()
        providers = {
            (provider or "").lower().replace("-", "_"),
            (custom_llm_provider or "").lower().replace("-", "_"),
        }
        official_openai_providers = {"openai", "azure", "azure_openai"}
        if model_lower.startswith("hosted_vllm/"):
            return True
        if providers & {"hosted_vllm", "vllm", "openai_compatible"}:
            return True
        if self._is_official_openai_or_azure_endpoint(api_base):
            return False
        if not (api_base or "").strip() and providers & official_openai_providers:
            return False
        if include_profile_parser and self.service_profile and self.service_profile.reasoning_parser:
            return True
        return False

    @staticmethod
    def _is_official_openai_or_azure_endpoint(api_base: str | None) -> bool:
        if not api_base:
            return False
        parsed = urlsplit(api_base)
        if not parsed.netloc:
            parsed = urlsplit(f"//{api_base}")
        host = (parsed.hostname or "").lower().rstrip(".")
        path = (parsed.path or "").lower()
        if host == "api.openai.com":
            return True
        if host.endswith(".openai.azure.com"):
            return True
        if host.endswith(".azure.com") and ("openai" in host or path.startswith("/openai")):
            return True
        return False

    def _handle_unsupported_enabled_capability(self, model: str) -> None:
        message = (
            f"thinking enabled for model {model!r}, but thinking capability is unsupported "
            "(error_type=unsupported_capability)"
        )
        if self.on_unsupported == "fail_fast":
            raise ValueError(message)
        if self.on_unsupported == "warn_and_continue":
            logger.warning(message)

    def _handle_unsupported_enabled_vllm_target(self, model: str) -> None:
        message = (
            f"thinking enabled for vLLM/OpenAI-compatible model {model!r}, "
            "but vllm.reasoning_parser is not configured (error_type=unsupported_capability)"
        )
        if self.on_unsupported == "fail_fast":
            raise ValueError(message)
        if self.on_unsupported == "warn_and_continue":
            logger.warning(message)


class ToolCallPolicy:
    """Format tool-call requests and normalize tool-call responses."""

    def __init__(self, config: Any | None = None, *, supports_function_calling: bool | None = None) -> None:
        self.config = config
        self.supports_function_calling = supports_function_calling
        self.enabled = bool(self._get_config_value("enabled", False))
        self.tool_choice = self._get_config_value("tool_choice")
        self.parallel_tool_calls = self._get_config_value("parallel_tool_calls")
        self.auto_tool_choice = bool(self._get_config_value("auto_tool_choice", False))
        self.require_server_parser = bool(self._get_config_value("require_server_parser", False))
        self.expected_server_flags = self._get_config_value("expected_server_flags") or {}
        self.expected_tool_parser = self._expected_server_flag("tool_call_parser") or self._get_config_value(
            "expected_tool_parser"
        )

    @classmethod
    def from_config(cls, config: Any | None) -> "ToolCallPolicy":
        """Builds a policy from optional LiteLLM tool-calling config."""

        return cls(config)

    def apply(
        self,
        kwargs: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        supports_function_calling: bool | None = None,
        is_vllm_tool_target: bool = False,
    ) -> None:
        """Applies tool-call kwargs to a LiteLLM request payload.

        Args:
            kwargs: Mutable LiteLLM kwargs under construction.
            inputs: Normalized backend inputs containing optional tools.

        Raises:
            ValueError: If tools are requested but required vLLM parser metadata
                is missing from the backend configuration.
        """

        formatted_tools = self.format_tools(inputs.get("tools"))
        if not formatted_tools:
            return

        tool_choice = self._resolve_tool_choice(inputs)
        parallel_tool_calls = self._resolve_parallel_tool_calls(inputs)
        auto_tools = self._is_auto_tool_choice(tool_choice)
        require_auto_server_flags = auto_tools and (
            is_vllm_tool_target or self.require_server_parser or self.auto_tool_choice
        )
        self._validate_server_parser(auto_tools=require_auto_server_flags)
        self._validate_function_calling_support(
            supports_function_calling,
            is_vllm_tool_target=is_vllm_tool_target,
        )
        kwargs["tools"] = formatted_tools

        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls

    def format_tools(self, tools: Any) -> list[Dict[str, Any]]:
        """Formats simplified and OpenAI tool schemas for LiteLLM."""

        if not tools:
            return []
        if isinstance(tools, dict):
            tools = [tools]
        if not isinstance(tools, list):
            return []

        formatted: list[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                formatted.append(
                    {
                        "type": "function",
                        "function": self._format_function_schema(tool["function"]),
                    }
                )
                continue
            if "name" in tool and "parameters" in tool:
                formatted.append(
                    {
                        "type": "function",
                        "function": self._format_function_schema(tool),
                    }
                )
                continue
            formatted.append(copy.deepcopy(tool))
        return formatted

    @classmethod
    def normalize_tool_calls(cls, tool_calls: Any) -> list[Dict[str, Any]]:
        """Normalizes LiteLLM/OpenAI tool calls into GAGE result shape."""

        if not tool_calls:
            return []
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if not isinstance(tool_calls, list):
            return []

        normalized: list[Dict[str, Any]] = []
        for tool_call in tool_calls:
            function = cls._field(tool_call, "function") or {}
            raw_arguments = cls._field(function, "arguments")
            if raw_arguments is None:
                raw_arguments = cls._field(tool_call, "arguments")
            arguments, raw_arguments_text, error = cls._parse_arguments(raw_arguments)

            item: Dict[str, Any] = {
                "id": cls._field(tool_call, "id"),
                "type": cls._field(tool_call, "type") or "function",
                "name": cls._field(function, "name") or cls._field(tool_call, "name"),
                "arguments": arguments,
                "raw_arguments": raw_arguments_text,
            }
            if error:
                item["error"] = error
            normalized.append(item)
        return normalized

    @classmethod
    def merge_tool_call_delta(cls, state: Dict[Any, Dict[str, Any]], tool_call_delta: Any) -> None:
        """Merges one streaming tool-call delta into mutable aggregation state."""

        index = cls._field(tool_call_delta, "index")
        tool_call_id = cls._field(tool_call_delta, "id")
        key = index if index is not None else tool_call_id
        if key is None:
            key = len(state)

        entry = state.setdefault(
            key,
            {
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if tool_call_id is not None:
            entry["id"] = tool_call_id
        tool_type = cls._field(tool_call_delta, "type")
        if tool_type is not None:
            entry["type"] = tool_type

        function_delta = cls._field(tool_call_delta, "function") or {}
        function_state = entry.setdefault("function", {"name": "", "arguments": ""})
        name_delta = cls._field(function_delta, "name")
        if name_delta is not None:
            function_state["name"] = f"{function_state.get('name') or ''}{name_delta}"
        arguments_delta = cls._field(function_delta, "arguments")
        if arguments_delta is not None:
            function_state["arguments"] = f"{function_state.get('arguments') or ''}{arguments_delta}"

    def _validate_function_calling_support(
        self,
        supports_function_calling: bool | None,
        *,
        is_vllm_tool_target: bool,
    ) -> None:
        resolved_support = self.supports_function_calling if supports_function_calling is None else supports_function_calling
        if resolved_support is False:
            if is_vllm_tool_target and self.has_declared_vllm_tool_support():
                return
            raise ValueError(
                "tool_calling_not_supported: model/provider does not support function calling "
                "(error_type=unsupported_capability)"
            )

    def _validate_server_parser(self, *, auto_tools: bool) -> None:
        if self.require_server_parser and not self.expected_tool_parser:
            raise ValueError(
                "tool_parser_missing: tool_calling.require_server_parser=true requires "
                "tool_calling.expected_server_flags.tool_call_parser or tool_calling.expected_tool_parser"
            )
        if auto_tools:
            if self._expected_server_flag("enable_auto_tool_choice") is not True:
                raise ValueError(
                    "tool_parser_missing: auto tool choice requires "
                    "tool_calling.expected_server_flags.enable_auto_tool_choice=true"
                )
            if not self.expected_tool_parser:
                raise ValueError(
                    "tool_parser_missing: auto tool choice requires "
                    "tool_calling.expected_server_flags.tool_call_parser or tool_calling.expected_tool_parser"
                )

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            return default
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _expected_server_flag(self, key: str, default: Any = None) -> Any:
        flags = self.expected_server_flags
        if isinstance(flags, dict):
            return flags.get(key, default)
        return getattr(flags, key, default)

    def has_declared_vllm_tool_support(self) -> bool:
        """Returns whether config declares vLLM tool parser/server flags."""

        return bool(self.expected_tool_parser) or self._expected_server_flag("enable_auto_tool_choice") is not None

    def _resolve_tool_choice(self, inputs: Dict[str, Any]) -> Any:
        if inputs.get("tool_choice") is not None:
            return inputs["tool_choice"]
        if self.tool_choice is not None:
            return copy.deepcopy(self.tool_choice)
        if self.auto_tool_choice:
            return "auto"
        return None

    def _resolve_parallel_tool_calls(self, inputs: Dict[str, Any]) -> Any:
        if inputs.get("parallel_tool_calls") is not None:
            return inputs["parallel_tool_calls"]
        return self.parallel_tool_calls

    @staticmethod
    def _is_auto_tool_choice(tool_choice: Any) -> bool:
        return isinstance(tool_choice, str) and tool_choice.strip().lower() == "auto"

    @staticmethod
    def _format_function_schema(function_schema: Dict[str, Any]) -> Dict[str, Any]:
        formatted = copy.deepcopy(function_schema)
        formatted["name"] = formatted.get("name")
        formatted["description"] = formatted.get("description", "")
        formatted["parameters"] = formatted.get("parameters") or {"type": "object", "properties": {}}
        return formatted

    @classmethod
    def _parse_arguments(cls, raw_arguments: Any) -> tuple[Any, str, str | None]:
        if raw_arguments is None:
            return {}, "", None
        if isinstance(raw_arguments, str):
            if raw_arguments.strip() == "":
                return {}, raw_arguments, None
            try:
                return json.loads(raw_arguments), raw_arguments, None
            except json.JSONDecodeError:
                return None, raw_arguments, "invalid_tool_arguments"

        jsonable = cls._to_jsonable(raw_arguments)
        try:
            raw_arguments_text = json.dumps(jsonable, ensure_ascii=True)
        except TypeError:
            raw_arguments_text = str(jsonable)
        return jsonable, raw_arguments_text, None

    @staticmethod
    def _field(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @classmethod
    def _to_jsonable(cls, obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {key: cls._to_jsonable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._to_jsonable(value) for value in obj]
        if hasattr(obj, "model_dump"):
            return cls._to_jsonable(obj.model_dump())
        if hasattr(obj, "__dict__"):
            return {key: cls._to_jsonable(value) for key, value in vars(obj).items() if not key.startswith("_")}
        return str(obj)
