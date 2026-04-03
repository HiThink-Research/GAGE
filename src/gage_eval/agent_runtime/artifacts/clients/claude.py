"""Claude client driver shell."""

from __future__ import annotations


class ClaudeClient:
    """Claude client shell. Not yet implemented."""

    def setup(self, environment, session) -> None:  # pragma: no cover - shell
        """Prepare the environment."""
        raise NotImplementedError("ClaudeClient.setup not yet implemented")

    def run(self, request, environment):  # pragma: no cover - shell
        """Run the client."""
        raise NotImplementedError("ClaudeClient.run not yet implemented")

    def cleanup(self, environment, session) -> None:  # pragma: no cover - shell
        """Release environment resources."""
        raise NotImplementedError("ClaudeClient.cleanup not yet implemented")

