from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

def construct_inner_messages(user_prompt, ans):
    sample = {}
    messages: List[Dict[str, Any]] = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ],
        }
    )
    sample["messages"] = messages
    sample["choices"] = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": ans,
                    }
                ],
            },
        }

    ]
    return sample