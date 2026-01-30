
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
    sample_from_dict
)

def normalize_messages(dict_messages):
    ret = []
    for message in dict_messages:
        role = message.get("role") or "user"
        content = message.get("content") or ""
        ret.append(
            Message(
                role=role,
                content=[
                    MessageContent(
                        type="text",
                        text=content
                    )
                ]
            )
        )
    return ret