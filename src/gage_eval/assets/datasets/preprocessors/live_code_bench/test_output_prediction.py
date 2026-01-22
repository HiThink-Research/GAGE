import json
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from gage_eval.assets.datasets.preprocessors.live_code_bench.lm_styles import LanguageModelStore, LMStyle


@dataclass
class Test:
    input: str
    output: str
    testtype: str


@dataclass
class TestOutputPredictionProblem:
    question_title: str
    question_content: str
    question_id: str
    contest_id: str
    contest_date: datetime
    difficulty: str
    test: list[Test]
    starter_code: str
    function_name: str
    test_id: int

    def __post_init__(self):
        self.test = [Test(**t) for t in json.loads(self.test)]  # type: ignore

    def insert_output(self, output_list: list[str], pred_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "difficulty": self.difficulty,
            "output_list": output_list,
            "pred_list": pred_list,
            "test_id": self.test_id,
            "function_name": self.function_name,
            "starter_code": self.starter_code,
        }

    def insert_output_evaluation(
        self, output_list: list[str], code_list: list[str], graded_list: list[bool]
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        return output

    def get_evaluation_sample(self) -> dict:
        return {
            "input": self.question_content,
            "output": self.test[0].output,
        }

class PromptConstants:
    SYSTEM_MESSAGE_CHAT_GENERIC = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function.\
 The user has written a input for the testcase.\
 You will calculate the output of the testcase and\
 write the whole assertion statement in the markdown code block with the correct output."

    SYSTEM_MESSAGE_COMPLETION_GENERIC = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function."

    SYSTEM_MESSAGE_INST_CLLAMA = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function.\
 The user has written a input for the testcase.\
 You will calculate the output of the testcase and \
 write out the complete assertion statement between [PYTHON] and [/PYTHON] tags."

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entired fixed program within code delimiters only for once., for example: 
```python 
# YOUR CODE HERE
```"""

    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_MESSAGE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


def truncate_io(io):
    if len(str(io)) > 1000:
        io = str(io)[:1000] + "...."
        print(io)
    return io


def format_testcase_func_name_input(function_name, testcase):
    """
    use the form of "assert func_name(input) == "
    """
    # TODO should there be a space after the == ?
    input_str = ", ".join(testcase.split("\n"))
    return f"assert {function_name}({input_str}) == # TODO"


def parse_function_name_from_starter_code(starter_code):
    """
    starter_code : str
    """
    import ast

    tree = ast.parse(starter_code)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert fn is None
            fn = node.name
    return fn


def get_generic_question_template_test_completion(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"Problem:\n{question.question_content}"
    prompt += f"Function:\n```\n{question.starter_code}\n```\n"

    # parse function name from starter_code
    func_name = parse_function_name_from_starter_code(question.starter_code)
    prompt += "Please complete the following test case:\n\n"
    prompt += (
        f"```\n{format_testcase_func_name_input(func_name, testcase_input)}\n```\n"
    )

    return prompt


def get_cllama_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"### Question\n"
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"### Answer\n"
    return prompt


def get_deepseekcode_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"### Instruction: {PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n\n"
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"### Response:\n\n"
    return prompt


def get_magicoder_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    # prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt = f"Question:\n"
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"@@ Response \n"
    return prompt


def get_mixtral_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = get_generic_question_template_test_completion(question, testcase_input)
    return prompt


def get_wizard_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"""### Instruction: {PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"""
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"### Response:\n"
    return prompt


def get_phind_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"\n\n### Assistant"
    return prompt

def get_qwen_question_template_answer(question: TestOutputPredictionProblem, testcase_input: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "abacusai/Dracarys-72B-Instruct", padding_side="left", use_fast=False
    )

    prompt = f"""### Instruction: {PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"""
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"### Response:\n"

    messages = [
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
    return prompt

def format_prompt_test_output(
    question: TestOutputPredictionProblem, LanguageModelStyle: LMStyle
) -> str:
    testcase_input = question.test[0].input
    if LanguageModelStyle == LMStyle.OpenAIChat:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_test_completion(
                    question, testcase_input
                ),
            },
        ]
        return chat_messages

    if LanguageModelStyle == LMStyle.CodeQwenInstruct:
        chat_messages = [
            {
                "role": "user",
                "content": PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC + '\n\n' + get_generic_question_template_test_completion(
                    question, testcase_input
                ),
            },
        ]
        return chat_messages

    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_test_completion(
                    question, testcase_input
                ),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"
        prompt += (
            f"{get_generic_question_template_test_completion(question, testcase_input)}"
        )
        chat_messages = [
            {
                "role": "user",
                "content": prompt
            },
        ]
        return chat_messages

    elif LanguageModelStyle == LMStyle.StarCoderInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"
        prompt += (
            f"{get_generic_question_template_test_completion(question, testcase_input)}"
        )
        chat_messages = [
            {
                "role": "user",
                "content": prompt
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = (
            f"{get_deepseekcode_question_template_answer(question, testcase_input)}"
        )
        chat_messages = [
            {
                "role": "user",
                "content": prompt
            },
        ]
        return chat_messages
    
    elif LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
        prompt = f"[INST] <<SYS>>\n{PromptConstants.SYSTEM_MESSAGE_INST_CLLAMA}\n<</SYS>>\n\n"
        prompt += (
            f"{get_cllama_question_template_answer(question, testcase_input)}\n[/INST]"
        )
        chat_messages = [
            {
                "role": "user",
                "content": prompt
            },
        ]
        return chat_messages
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )
