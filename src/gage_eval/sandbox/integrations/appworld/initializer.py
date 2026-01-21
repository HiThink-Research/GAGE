"""AppWorld server initializer wrapper."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class AppWorldInitializer:
    """Manage AppWorld background servers with a minimal wrapper."""

    def __init__(
        self,
        *,
        env_endpoint: Optional[str] = None,
        apis_endpoint: Optional[str] = None,
        mcp_endpoint: Optional[str] = None,
        start_servers: bool = False,
        show_server_logs: bool = False,
        update_defaults: bool = True,
        initializer_factory: Optional[Callable[..., Any]] = None,
        **config: Any,
    ) -> None:
        self._config = dict(config)
        if env_endpoint and "remote_environment_url" not in self._config:
            self._config["remote_environment_url"] = env_endpoint
        if apis_endpoint and "remote_apis_url" not in self._config:
            self._config["remote_apis_url"] = apis_endpoint
        if mcp_endpoint and "remote_mcp_url" not in self._config:
            self._config["remote_mcp_url"] = mcp_endpoint
        self._start_servers = start_servers
        self._show_server_logs = show_server_logs
        self._update_defaults = update_defaults
        self._initializer_factory = initializer_factory or _default_initializer_factory
        self._initializer: Optional[Any] = None

    @property
    def parallelizable_across(self) -> Optional[str]:
        if self._initializer:
            return getattr(self._initializer, "parallelizable_across", None)
        return None

    @property
    def config(self) -> Dict[str, Any]:
        if self._initializer and hasattr(self._initializer, "config"):
            return dict(self._initializer.config)
        return dict(self._config)

    @property
    def configs(self) -> list[Dict[str, Any]]:
        if self._initializer and hasattr(self._initializer, "configs"):
            return list(self._initializer.configs)
        return [dict(self._config)]

    def start(self) -> AppWorldInitializer:
        """Start AppWorld servers if configured."""

        if self._initializer is None:
            self._initializer = self._initializer_factory(
                update_defaults=self._update_defaults,
                start_servers=self._start_servers,
                show_server_logs=self._show_server_logs,
                **self._config,
            )
        if hasattr(self._initializer, "start"):
            self._initializer.start()
        return self

    def stop(self) -> AppWorldInitializer:
        """Stop AppWorld servers if running."""

        if self._initializer and hasattr(self._initializer, "stop"):
            self._initializer.stop()
        return self

    def __enter__(self) -> AppWorldInitializer:
        return self.start()

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: Any | None) -> None:
        self.stop()


def _default_initializer_factory(**config: Any) -> Any:
    from appworld.environment import AppWorld

    return AppWorld.initializer(**config)
