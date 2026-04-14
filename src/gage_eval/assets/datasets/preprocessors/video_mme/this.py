"""Prompt builder for Video-MME."""

from __future__ import annotations

from typing import List, Optional


def create_prompt(
    question: str,
    options: Optional[List[str]] = None,
    include_subtitles: bool = False,
    subtitles: Optional[str] = None,
) -> str:
    """Build the evaluation prompt for Video-MME.

    The prompt format follows the official Video-MME evaluation session:
    https://github.com/BradyFU/Video-MME
    """
    lines: List[str] = []
    if include_subtitles and subtitles:
        lines.append("This video's subtitles are listed below:")
        lines.append(str(subtitles).strip())

    lines.append(
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    lines.append(str(question).strip())

    if options:
        for i, opt in enumerate(options):
            opt_str = str(opt).strip()
            # If the option already contains a letter prefix (e.g. "A. Apples."), keep it as-is.
            if opt_str and opt_str[0].isalpha() and opt_str[1] in (".", ")", " "):
                lines.append(opt_str)
            else:
                lines.append(f"({chr(ord('A') + i)}) {opt_str}")

    lines.append("The best answer is:")
    return "\n".join(lines)
