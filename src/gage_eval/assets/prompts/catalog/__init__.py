"""Built-in prompt assets exposed by the registry."""

from gage_eval.assets.prompts.catalog.general import register_general_prompts
from gage_eval.assets.prompts.catalog.support import register_support_prompts

register_general_prompts()
register_support_prompts()
