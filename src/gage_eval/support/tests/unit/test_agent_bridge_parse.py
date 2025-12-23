from __future__ import annotations

from pathlib import Path

from gage_eval.support.agent_bridge import parse_file_blocks


def test_parse_file_blocks_filters_noise() -> None:
    text = """Here is the code...
### FILE: a.txt
hello
world
### END
some chatter
### FILE: dir/b.py
print("ok")
### END
"""
    blocks = parse_file_blocks(text)
    assert blocks == [
        (Path("a.txt"), "hello\nworld\n"),
        (Path("dir/b.py"), 'print("ok")\n'),
    ]

