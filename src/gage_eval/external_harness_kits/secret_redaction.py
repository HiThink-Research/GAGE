"""Secret redaction helpers for external harness artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

REDACTED = "<redacted>"
OMITTED = "<omitted>"

_SECRET_VALUE_PREFIXES = ("sk-", "ak-", "pk-")
_SECRET_VALUE_PREFIX_PATTERN = "|".join(re.escape(prefix) for prefix in _SECRET_VALUE_PREFIXES)
_CREDENTIAL_LITERAL_RE = re.compile(
    r"(?P<prefix>\b(?:api[_-]?key|token|secret|password|authorization)\b\s*[:=]\s*)(?P<quote>[\"']?)(?P<value>(?:Bearer|Basic|ApiKey|Token)\s+[^\"'\s,;]+|[^\"'\s,;]+)(?P=quote)",
    re.IGNORECASE,
)
_INVOCATION_ALLOWLIST = (
    "job_name",
    "jobs_dir",
    "job_config_path",
    "job_config",
    "launcher_mode",
    "launcher_argv",
    "environ",
    "workdir",
    "expected_total_trials",
    "metadata",
)


@dataclass(frozen=True, init=False)
class SecretRedactionContext:
    """Resolved secret values and keys used to sanitize launcher artifacts."""

    secret_values: frozenset[str]
    secret_keys: frozenset[str]

    def __init__(
        self,
        secret_values: Iterable[str] | None = None,
        secret_keys: Iterable[str] | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "secret_values",
            frozenset(
                str(value)
                for value in (secret_values or ())
                if _is_redactable_secret_value(str(value))
            ),
        )
        object.__setattr__(
            self,
            "secret_keys",
            frozenset(str(key) for key in (secret_keys or ()) if str(key)),
        )

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None) -> "SecretRedactionContext":
        if not environ:
            return cls()
        secret_keys = {str(key) for key in environ if is_secret_key(str(key))}
        secret_values = {
            str(value)
            for key, value in environ.items()
            if is_secret_key(str(key)) and _is_redactable_secret_value(str(value))
        }
        return cls(secret_values=secret_values, secret_keys=secret_keys)

    def merge(self, other: "SecretRedactionContext") -> "SecretRedactionContext":
        return SecretRedactionContext(
            secret_values=self.secret_values | other.secret_values,
            secret_keys=self.secret_keys | other.secret_keys,
        )

    def is_secret_key(self, key: str) -> bool:
        return key in self.secret_keys or is_secret_key(key)


def redact_for_artifact(
    value: Any,
    *,
    context: SecretRedactionContext | None = None,
) -> Any:
    """Recursively redact values before writing an external harness artifact."""

    context = context or SecretRedactionContext()
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if context.is_secret_key(key_text):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_for_artifact(item, context=context)
        return redacted
    if isinstance(value, list):
        return [redact_for_artifact(item, context=context) for item in value]
    if isinstance(value, tuple):
        return [redact_for_artifact(item, context=context) for item in value]
    if isinstance(value, set):
        return [redact_for_artifact(item, context=context) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return redact_text(str(value), context=context)
    if isinstance(value, str):
        return redact_text(value, context=context)
    return value


def redact_text(
    text: str,
    *,
    context: SecretRedactionContext | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Redact resolved secret values and common credential literals from text."""

    active_context = context or SecretRedactionContext()
    if environ is not None:
        active_context = active_context.merge(SecretRedactionContext.from_environ(environ))
    else:
        active_context = active_context.merge(SecretRedactionContext.from_environ(os.environ))
    redacted = text
    for secret in sorted(active_context.secret_values, key=len, reverse=True):
        redacted = redacted.replace(secret, REDACTED)
    redacted = _CREDENTIAL_LITERAL_RE.sub(_redact_credential_literal, redacted)
    redacted = re.sub(rf"\b(?:{_SECRET_VALUE_PREFIX_PATTERN})[A-Za-z0-9._-]+", REDACTED, redacted)
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", f"Bearer {REDACTED}", redacted)
    redacted = re.sub(r"\$\{[^}]+\}", REDACTED, redacted)
    return redacted


def contains_secret_like_text(
    text: str,
    *,
    context: SecretRedactionContext | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    return redact_text(text, context=context, environ=environ) != text


def to_invocation_artifact(
    invocation: Mapping[str, Any] | Any,
    *,
    context: SecretRedactionContext | None = None,
) -> dict[str, Any]:
    """Build the allowlisted, redacted invocation artifact payload."""

    payload = _mapping_from_object(invocation)
    environ = _mapping(payload.get("environ"))
    active_context = (context or SecretRedactionContext()).merge(
        SecretRedactionContext.from_environ({str(key): str(value) for key, value in environ.items()})
    )
    artifact: dict[str, Any] = {}
    for key in _INVOCATION_ALLOWLIST:
        if key not in payload:
            continue
        value = payload[key]
        if key == "environ":
            artifact[key] = _redacted_environ(environ, context=active_context)
        elif key in {"jobs_dir", "job_config_path", "workdir"}:
            artifact[key] = redact_text(str(value), context=active_context)
        else:
            artifact[key] = redact_for_artifact(value, context=active_context)
    return artifact


def is_secret_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if "api_key" in normalized or "apikey" in normalized:
        return True
    if normalized in {"authorization", "auth_token", "bearer_token", "token"}:
        return True
    if "per_token" in normalized:
        return False
    parts = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
    if any(part in parts for part in ("secret", "password", "credential", "authorization")):
        return True
    return "token" in parts or normalized.endswith("_token")


def _redacted_environ(
    environ: Mapping[str, Any],
    *,
    context: SecretRedactionContext,
) -> dict[str, dict[str, Any]]:
    artifact: dict[str, dict[str, Any]] = {}
    for key, value in sorted(environ.items()):
        key_text = str(key)
        value_text = str(value)
        is_secret = context.is_secret_key(key_text) or redact_text(value_text, context=context) != value_text
        artifact[key_text] = {
            "is_secret": is_secret,
            "value": REDACTED if is_secret else OMITTED,
            "value_sha256": _sha256(value_text),
        }
    return artifact


def _redact_credential_literal(match: re.Match[str]) -> str:
    prefix = match.group("prefix")
    quote = match.group("quote")
    value = match.group("value")
    if " " in value:
        scheme = value.split(" ", 1)[0]
        return f"{prefix}{quote}{scheme} {REDACTED}{quote}"
    return f"{prefix}{quote}{REDACTED}{quote}"


def _normalize_key(key: str) -> str:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return camel_split.lower().replace("-", "_")


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _mapping_from_object(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_redactable_secret_value(value: str) -> bool:
    if len(value) < 4:
        return False
    return value.upper() not in {"EMPTY", "NONE", "NULL", "TRUE", "FALSE"}


__all__ = [
    "OMITTED",
    "REDACTED",
    "SecretRedactionContext",
    "contains_secret_like_text",
    "is_secret_key",
    "redact_for_artifact",
    "redact_text",
    "to_invocation_artifact",
]
