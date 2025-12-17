from __future__ import annotations

from typer.testing import CliRunner

from gage_eval.support.main import app


runner = CliRunner()


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("check", "inspect", "design", "implement", "status"):
        assert name in result.stdout


def test_timeout_flag_exposed_on_agent_commands() -> None:
    design_help = runner.invoke(app, ["design", "--help"])
    implement_help = runner.invoke(app, ["implement", "--help"])
    assert "--timeout" in design_help.stdout
    assert "--timeout" in implement_help.stdout
