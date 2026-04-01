"""AMO-Bench dataset preprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
)

# Template for user prompt with instruction suffix
_USER_PROMPT_TEMPLATE = """{prompt}

After solving the above problem, please output your final answer in the following format:
### The final answer is: $\\boxed{{<your answer>}}$
Example:
### The final answer is: $\\boxed{{123}}$
The final answer should be given as precisely as possible (using LaTeX symbols such as \\sqrt, \\frac, \\pi, etc.). If
the final answer involves a decimal approximation, it must be accurate to at least four decimal places."""


def _append_try_list(record: Dict[str, Any]) -> Dict[str, Any]:
    """Append try_list for variable-type questions based on question_id.
    
    This is ported from the original AMO-Bench utils.py.
    """
    info = dict(record)
    question_id = info.get("question_id")
    
    if question_id == 5:
        assert info.get("answer_type") == "variable"
        try_list = ["n=1", "n=2", "n=3", "n=4", "n=5", "n=6", "n=7", "n=8", "n=9", "n=10",
                    "n=11", "n=12", "n=13", "n=14", "n=15", "n=16", "n=17", "n=18", "n=19", "n=20"]
        info["try_list"] = try_list
    elif question_id == 37:
        assert info.get("answer_type") == "variable"
        try_list = ["a=2,b=3,c=4", "a=3,b=4,c=5", "a=4,b=5,c=6", "a=5,b=6,c=7", "a=6,b=7,c=8",
                    "a=7,b=8,c=9", "a=8,b=9,c=10", "a=9,b=10,c=11", "a=10,b=11,c=12",
                    "a=11,b=12,c=13", "a=12,b=13,c=14", "a=13,b=14,c=15", "a=14,b=15,c=16",
                    "a=15,b=16,c=17", "a=16,b=17,c=18", "a=17,b=18,c=19", "a=18,b=19,c=20"]
        info["try_list"] = try_list
    
    return info


class AMOBenchPreprocessor(BasePreprocessor):
    """Preprocesses AMO-Bench records into the Sample schema.
    
    AMO-Bench is an Advanced Mathematical reasoning benchmark with Olympiad level 
    or even higher difficulty, comprising 50 human-crafted problems.
    
    Field mappings:
        question_id => sample_id
        prompt => messages
        solution => meta_info
        try_list => meta_info (auto-generated for variable type)
        answer => reference
        answer_type => meta_info
    """

    def to_sample(
        self, 
        record: Dict[str, Any], 
        schema_version: str = SCHEMA_VERSION, 
        **kwargs: Any
    ) -> Sample:
        """Converts a raw AMO-Bench record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            schema_version: Schema version string.
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)
        
        # Append try_list for variable-type questions
        sample = _append_try_list(sample)
        
        # Extract fields from record
        sample_id = str(sample.get("question_id", ""))
        prompt = sample.get("prompt", "")
        solution = sample.get("solution", "")
        answer = sample.get("answer", "")
        answer_type = sample.get("answer_type", "")
        try_list = sample.get("try_list", [])
        
        # Format prompt with the required suffix
        formatted_prompt = _USER_PROMPT_TEMPLATE.format(prompt=prompt)
        
        # Build references and label
        references = [answer]
        label = answer
        
        # Build metadata with all relevant info
        metadata = {
            "solution": solution,
            "answer_type": answer_type,
            "try_list": try_list,
            "original_prompt": prompt,
        }
        
        # Build messages
        message_content = MessageContent(type="text", text=formatted_prompt)
        message = Message(role="user", content=[message_content])
        
        ret_sample = Sample(
            id=sample_id,
            schema_version=schema_version,
            messages=[message],
            metadata=metadata,
            references=references,
            label=label,
        )
        return ret_sample


__all__ = ["AMOBenchPreprocessor", "_append_try_list"]
