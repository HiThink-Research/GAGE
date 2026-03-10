#!/usr/bin/env python3
"""Compatibility wrapper for PettingZoo ws_rgb replay.

This script is kept only for backward compatibility with existing commands.
Core replay logic lives in ``gage_eval.tools.ws_rgb_replay``.
"""

from __future__ import annotations

from gage_eval.tools.ws_rgb_replay import main


if __name__ == "__main__":
    main()
