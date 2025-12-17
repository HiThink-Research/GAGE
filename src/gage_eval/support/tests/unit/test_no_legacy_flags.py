from __future__ import annotations

from typer.testing import CliRunner

from gage_eval.support.main import app


runner = CliRunner()


def test_cli_help_has_no_legacy_flags() -> None:
    result = runner.invoke(app, ["implement", "--help"])
    assert result.exit_code == 0
    legacy_tokens = [
        "design_spec",
        "status.success",
        "code_backup",
        "changes.md",
        "~/.config",
    ]
    for tok in legacy_tokens:
        assert tok not in result.stdout

