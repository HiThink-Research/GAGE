from __future__ import annotations


def normalize_backend_mode(value: object, *, default: str = "real") -> str:
    normalized = str(value or default).strip().lower()
    if not normalized:
        return default
    if normalized in {"dummy", "stub"}:
        return "dummy"
    return normalized


__all__ = ["normalize_backend_mode"]
