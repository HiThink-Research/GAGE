#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools" / "appworld" / "tool_docs_probe.py"),
    run_name="__main__",
)
