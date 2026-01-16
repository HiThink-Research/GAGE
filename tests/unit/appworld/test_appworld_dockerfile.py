from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.io
def test_appworld_dockerfile_contains_data_install() -> None:
    dockerfile = Path(__file__).resolve().parents[3] / "docker" / "appworld" / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")

    assert "appworld install" in content
    assert 'ENTRYPOINT ["/usr/local/bin/appworld", "serve"]' in content
    assert 'CMD ["multiple"' in content
    assert "output-type structured_data_only" in content
    assert "FROM --platform=$APPWORLD_PLATFORM ghcr.io/stonybrooknlp/appworld:latest" in content
