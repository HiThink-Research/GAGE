from abc import ABC, abstractmethod
import re

class RegexFilter(ABC):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
        ignore_case: bool = False
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        if ignore_case:
            self.regex = re.compile(regex_pattern, re.IGNORECASE)
        else:
            self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resp: str, doc: str = None) -> str:
        def filter_set(inst):
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        if match:
                            match = match[0]
                        else:
                            match = self.fallback
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        filtered_resps = filter_set([resp])
        return filtered_resps[0]

if __name__ == '__main__':
    pattern = r'ANSWER[：:]\s*(.*)'
    input_1 = 'ANSWER: 3'
    rf = RegexFilter(regex_pattern=pattern, group_select=-1)
    ss = rf.apply(input_1)
    print("ss:", ss)

    input_1 = 'ANSWER: 3\ANSWER: 4\nANSWER: 6'
    rf = RegexFilter(regex_pattern=pattern, group_select=-1)
    ss = rf.apply(input_1)
    print("ss:", ss)