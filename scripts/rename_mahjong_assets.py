#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "assets" / "mahjong" / "rename_tiles.py"),
    run_name="__main__",
)
