from __future__ import annotations

import inspect
from typing import Any, Callable

from gage_eval.agent_runtime.clients.contracts import ClientSurface
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.registry.utils import ensure_async, run_sync


class StructuredClientSurfaceAdapter:
    """Adapts a structured client object to the runtime client contract."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._setup = getattr(client, "setup", None)
        self._run = getattr(client, "run", None)
        self._arun = getattr(client, "arun", None)
        if not callable(self._run) and not callable(self._arun):
            raise TypeError("structured client requires run/arun")

    def setup(
        self,
        environment: dict[str, Any],
        session: AgentRuntimeSession,
    ) -> dict[str, Any] | None:
        """Prepare the client against one benchmark-owned environment."""

        if not callable(self._setup):
            return None
        return _call_client_method(
            self._setup,
            kwargs={"environment": environment, "session": session},
            args=(environment, session),
        )

    def run(
        self,
        request: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the client via a synchronous compatibility path."""

        if callable(self._run):
            return _normalize_result(
                _call_client_method(
                    self._run,
                    kwargs={"request": request, "environment": environment},
                    args=(request, environment),
                )
            )
        return run_sync(self.arun(request, environment))

    async def arun(
        self,
        request: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the client via the preferred async path."""

        if callable(self._arun):
            result = _call_client_method(
                self._arun,
                kwargs={"request": request, "environment": environment},
                args=(request, environment),
            )
            if inspect.isawaitable(result):
                result = await result
            return _normalize_result(result)
        if not callable(self._run):
            raise TypeError("structured client requires run/arun")
        result = await ensure_async(_call_client_method)(
            self._run,
            kwargs={"request": request, "environment": environment},
            args=(request, environment),
        )
        return _normalize_result(result)


class LegacyInvokeClientSurface:
    """Wraps legacy invoke/ainvoke clients behind the runtime client contract."""

    def __init__(self, client: Any) -> None:
        self._client = client
        invoker = getattr(client, "ainvoke", None) or getattr(client, "invoke", None)
        if not callable(invoker):
            raise TypeError("legacy installed client requires invoke/ainvoke")
        self._invoker = ensure_async(invoker)
        self._session: AgentRuntimeSession | None = None

    def setup(
        self,
        environment: dict[str, Any],
        session: AgentRuntimeSession,
    ) -> dict[str, Any] | None:
        """Store the active session for legacy payload projection."""

        self._session = session
        return None

    def run(
        self,
        request: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the wrapped legacy client synchronously."""

        return run_sync(self.arun(request, environment))

    async def arun(
        self,
        request: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the wrapped legacy client asynchronously."""

        payload = dict(request or {})
        payload.setdefault("request", dict(request or {}))
        payload.setdefault("environment", dict(environment or {}))
        if self._session is not None:
            payload.setdefault("session", self._session)
        sandbox_provider = environment.get("sandbox_provider")
        if sandbox_provider is not None:
            payload.setdefault("sandbox_provider", sandbox_provider)
        result = await self._invoker(payload)
        return _normalize_result(result)


def build_client_surface(client: Any) -> ClientSurface:
    """Build the canonical runtime client surface for installed-client execution."""

    if isinstance(client, ClientSurface):
        return client
    if callable(getattr(client, "run", None)) or callable(getattr(client, "arun", None)):
        return StructuredClientSurfaceAdapter(client)
    return LegacyInvokeClientSurface(client)


def instantiate_builtin_client(client_id: str) -> Any:
    """Instantiates one builtin installed client implementation.

    Args:
        client_id: Stable runtime client identifier.

    Returns:
        One builtin client instance.

    Raises:
        KeyError: If the client id is not registered.
    """

    normalized = str(client_id or "").strip().lower()
    if normalized == "codex":
        from gage_eval.agent_runtime.clients.codex import CodexClient

        return CodexClient()
    raise KeyError(f"Unknown installed client '{client_id}'")


def resolve_installed_client(
    *,
    client_id: str | None,
    client_override: Any = None,
) -> ClientSurface:
    """Resolves one installed client surface from override or builtin id.

    Args:
        client_id: Runtime client identifier from the compiled plan.
        client_override: Optional explicit client object supplied by the caller.

    Returns:
        The normalized installed-client surface.

    Raises:
        ValueError: If no client surface can be resolved.
    """

    if client_override is not None:
        return build_client_surface(client_override)
    if not client_id:
        raise ValueError("installed client resolution requires client_id")
    try:
        return build_client_surface(instantiate_builtin_client(client_id))
    except KeyError as exc:
        raise ValueError(f"Unknown installed client '{client_id}'") from exc


def _call_client_method(
    target: Callable[..., Any],
    *,
    kwargs: dict[str, Any],
    args: tuple[Any, ...],
) -> Any:
    """Call one client method using the most specific supported signature."""

    if _supports_keyword_bind(target, kwargs):
        return target(**kwargs)
    return target(*args)


def _supports_keyword_bind(target: Callable[..., Any], kwargs: dict[str, Any]) -> bool:
    """Return whether the callable accepts the keyword-based client contract."""

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind_partial(**kwargs)
    except TypeError:
        return False
    return True


def _normalize_result(result: Any) -> dict[str, Any]:
    """Normalize client outputs into a predictable dict payload."""

    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    serializer = getattr(result, "to_dict", None)
    if callable(serializer):
        normalized = serializer()
        if isinstance(normalized, dict):
            return normalized
    return {"result": result}
