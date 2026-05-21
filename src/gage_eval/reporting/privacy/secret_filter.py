from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Pattern

from gage_eval.reporting.contracts import (
    RedactionFinding,
    RedactionResult,
    SecretPattern,
)


@dataclass(frozen=True)
class _CompiledSecretPattern:
    pattern: SecretPattern
    regex: Pattern[str]


class SecretFilter:
    """Detects and redacts secrets from report-visible values."""

    def __init__(self, patterns: list[SecretPattern] | None = None) -> None:
        """Initializes the filter with default and optional extra patterns."""
        self._patterns = [
            _CompiledSecretPattern(pattern, re.compile(pattern.regex, re.IGNORECASE))
            for pattern in [*DEFAULT_PATTERNS, *(patterns or [])]
        ]

    def detect(self, text: str) -> list[RedactionFinding]:
        """Returns secret findings in text without exposing matched values."""
        findings: list[RedactionFinding] = []
        for compiled in self._patterns:
            for match in compiled.regex.finditer(text):
                findings.append(
                    RedactionFinding(
                        kind=compiled.pattern.kind,
                        start=match.start(),
                        end=match.end(),
                        pattern_name=compiled.pattern.name,
                    )
                )
        findings.sort(key=lambda item: (item.start, item.end, item.kind))
        return findings

    def redact(self, value: Any) -> RedactionResult:
        """Recursively redacts strings, lists, and dictionaries."""
        cloned_value = copy.deepcopy(value)
        redacted_value, findings = self._redact_value(cloned_value, key_hint=None)
        return RedactionResult(
            value=redacted_value,
            findings=findings,
            redacted=bool(findings),
        )

    def assert_safe(self, value: Any) -> None:
        """Raises ValueError if the value contains report-forbidden secrets."""
        result = self.redact(value)
        if result.findings:
            kinds = ", ".join(sorted({finding.kind for finding in result.findings}))
            raise ValueError(f"Unsafe report-visible value contains secret kinds: {kinds}")

    def _redact_value(
        self, value: Any, key_hint: str | None
    ) -> tuple[Any, list[RedactionFinding]]:
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            findings: list[RedactionFinding] = []
            for key, child in value.items():
                child_value, child_findings = self._redact_value(
                    child, key_hint=str(key).lower()
                )
                redacted[key] = child_value
                findings.extend(child_findings)
            return redacted, findings
        if isinstance(value, list):
            redacted_items: list[Any] = []
            findings = []
            for item in value:
                child_value, child_findings = self._redact_value(item, key_hint=None)
                redacted_items.append(child_value)
                findings.extend(child_findings)
            return redacted_items, findings
        if isinstance(value, str):
            if _is_redaction_placeholder(value):
                return value, []
            if key_hint:
                forced_kind = _kind_for_key(key_hint)
                if forced_kind is not None and value:
                    return (
                        f"<redacted:{forced_kind}>",
                        [
                            RedactionFinding(
                                kind=forced_kind,
                                start=0,
                                end=len(value),
                                pattern_name=f"{forced_kind}_field",
                            )
                        ],
                    )
            return self._redact_text(value)
        return value, []

    def _redact_text(self, text: str) -> tuple[str, list[RedactionFinding]]:
        protected, placeholders = _protect_redaction_placeholders(text)
        findings = self.detect(protected)
        redacted = protected
        for compiled in self._patterns:
            redacted = compiled.regex.sub(compiled.pattern.placeholder, redacted)
        for token, placeholder in placeholders.items():
            redacted = redacted.replace(token, placeholder)
        return redacted, findings


def _kind_for_key(key: str) -> str | None:
    normalized = key.lower().replace("-", "_")
    if normalized in {"authorization", "proxy_authorization"}:
        return "auth"
    if normalized in {"password", "passwd"} or normalized.endswith("_password"):
        return "secret"
    if normalized in {"secret", "client_secret"} or normalized.endswith("_secret"):
        return "secret"
    if normalized in {"cookie", "set_cookie", "session_id"}:
        return "session"
    if normalized in {"api_key", "access_token", "refresh_token", "id_token"}:
        return "token"
    return None


_REDACTION_PLACEHOLDER_RE = re.compile(r"<redacted:[^>]+>")


def _is_redaction_placeholder(text: str) -> bool:
    return bool(_REDACTION_PLACEHOLDER_RE.fullmatch(text))


def _protect_redaction_placeholders(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"__GAGE_REDACTION_PLACEHOLDER_{len(placeholders)}__"
        placeholders[token] = match.group(0)
        return token

    return _REDACTION_PLACEHOLDER_RE.sub(replace, text), placeholders


DEFAULT_PATTERNS = [
    SecretPattern(
        kind="auth",
        name="authorization_header",
        regex=r"\bAuthorization\s*:\s*(?:Bearer|Basic)\s+[^\s,;]+",
    ),
    SecretPattern(kind="auth", name="bearer_token", regex=r"\bBearer\s+[^\s,;]+"),
    SecretPattern(kind="auth", name="basic_auth", regex=r"\bBasic\s+[^\s,;]+"),
    SecretPattern(
        kind="token",
        name="openai_key",
        regex=r"\bsk-[A-Za-z0-9_\-]{8,}\b",
    ),
    SecretPattern(
        kind="token",
        name="token_field",
        regex=r"[\"']?\b(?:api_key|access_token|refresh_token|id_token)\b[\"']?\s*[:=]\s*[\"']?[^\"'\s,;}]+[\"']?",
    ),
    SecretPattern(
        kind="secret",
        name="password_field",
        regex=r"[\"']?\b(?:password|passwd|client_secret|[A-Za-z0-9_]+_password|[A-Za-z0-9_]+_secret)\b[\"']?\s*[:=]\s*[\"']?[^\"'\s,;}]+[\"']?",
    ),
    SecretPattern(
        kind="session",
        name="cookie_header",
        regex=r"\b(?:Cookie|Set-Cookie)\s*:\s*[^\n\r]+",
    ),
    SecretPattern(
        kind="session",
        name="session_field",
        regex=r"[\"']?\bsession_id\b[\"']?\s*[:=]\s*[\"']?[^\"'\s,;}]+[\"']?",
    ),
    SecretPattern(
        kind="email",
        name="email",
        regex=r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    ),
    SecretPattern(
        kind="private_url",
        name="localhost_url_with_query",
        regex=r"https?://(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2})[^\s]*\?[^ \n\r]*",
    ),
]
