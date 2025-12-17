from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger


app = typer.Typer(
    add_completion=False,
    help="gage-eval Bench-Support CLI: inspect -> design -> implement benchmarks.",
)


@app.command()
def check(config: Optional[Path] = typer.Option(None, "--config", help="Path to project-level support config.")) -> None:
    """Check local environment, config, and agent availability."""

    # Lazy import to keep CLI boot lightweight.
    from .config import load_config
    from .agent_bridge import ping_agent

    cfg = load_config(config_path=config)
    ok = ping_agent(cfg)
    if not ok:
        raise typer.Exit(code=1)
    logger.info("Bench-Support check passed.")


@app.command()
def inspect(
    dataset: str = typer.Argument(..., help="HF hub id or local path."),
    subset: Optional[str] = typer.Option(None, help="HF subset."),
    split: Optional[str] = typer.Option(None, help="HF split."),
    max_samples: int = typer.Option(5, "--max-samples", help="Max samples to snapshot."),
    local_path: Optional[Path] = typer.Option(None, "--local-path", help="Prefer local dataset path."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to project-level support config."),
) -> None:
    """Inspect a dataset and write meta/sample/schema snapshots."""

    from .config import load_config
    from .inspector import inspect_dataset

    cfg = load_config(config_path=config)
    inspect_dataset(
        dataset_name=dataset,
        subset=subset,
        split=split,
        max_samples=max_samples,
        local_path=local_path,
        cfg=cfg,
    )


@app.command()
def design(
    dataset: str = typer.Argument(..., help="Dataset slug or hub id."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing design.md."),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Override agent timeout seconds."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to project-level support config."),
) -> None:
    """Generate design.md (single source) for a dataset."""

    from .config import load_config
    from .pipeline import run_design

    cfg = load_config(config_path=config)
    if timeout is not None:
        cfg.agent.timeout = timeout
    run_design(dataset, cfg=cfg, force=force)


@app.command()
def implement(
    dataset: str = typer.Argument(..., help="Dataset slug."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only print planned changes."),
    force: bool = typer.Option(False, "--force", help="Apply writes (override default dry-run) and skip git guard."),
    skip_tests: bool = typer.Option(False, "--skip-tests", help="Do not execute tests."),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Override agent timeout seconds."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to project-level support config."),
) -> None:
    """Implement benchmark assets from design.md."""

    from .config import load_config
    from .pipeline import run_implement

    cfg = load_config(config_path=config)
    if timeout is not None:
        cfg.agent.timeout = timeout
    run_implement(dataset, cfg=cfg, dry_run=dry_run, force=force, skip_tests=skip_tests)


@app.command()
def status(
    dataset: Optional[str] = typer.Argument(None, help="Dataset slug."),
    next_step: bool = typer.Option(False, "--next", help="Print next recommended step."),
    auto: bool = typer.Option(False, "--auto", help="Auto-run next step interactively."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to project-level support config."),
) -> None:
    """Show current status of datasets under workspace_root."""

    from .config import load_config
    from .utils import show_status

    cfg = load_config(config_path=config)
    show_status(dataset, cfg=cfg, next_step=next_step, auto=auto)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
