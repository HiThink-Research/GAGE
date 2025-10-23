from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from ..data import Role
from ..extras.constants import CHOICES


if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class EvalTemplate:
    system: str
    choice: str
    answer: str
    prefix: str

    def parse_example(self, example: Dict[str, str]) -> Tuple[str, str]:
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example and example[ch]!='']
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]
    
    def parse_selections(self, example: Dict[str, str]) -> Tuple[str, str]:
        # candidates = [self.choice.format(content=example[ch]) for ch in CHOICES if ch in example and example[ch]!='']
        # result = []
        # for c in candidates:
        #     result.append({
        #         "question":example["question"],
        #         "answer":example['answer'],
        #         "choice":c
        #     })
        import ast
        target_scores = example['target_scores']
        target_scores = ast.literal_eval(target_scores)
        candidates = [key for key in target_scores]
        result = []
        for index, c in enumerate(candidates):
            result.append({
                "question":example["question"],
                "answer":example['answer'],
                "choice":c
            })
        return result

    def format_example(
        self, target_data: Dict[str, str], support_set: "Dataset", subject_name: str, task="default",eval_type='next_word_proboablity'
    ) -> List[Dict[str, str]]:
        # print(target_data)
        messages = []
        for k in range(len(support_set)):
            prompt, response = self.parse_example(support_set[k])
            messages.append({"role": Role.USER, "content": prompt})
            messages.append({"role": Role.ASSISTANT, "content": response+'\n'})

        if eval_type in ['loss']:
            if task == 'boolq':
                target_data['question'] = target_data['passage'] + '\nquestion:' + target_data['question']
            result = self.parse_selections(target_data)
            for i,r in enumerate(result):
                messages.append({"role": Role.USER, "content": r['question']})
                messages.append({"role": Role.ASSISTANT, "content": r["choice"]})
                # messages.append({"role": Role.ASSISTANT, "content": r['answer']})
                # concate the template
                messages[i]["content"] = self.system.format(subject=subject_name) + messages[i]["content"]
        else:
            prompt, response = self.parse_example(target_data)
            messages.append({"role": Role.USER, "content": prompt})
            messages.append({"role": Role.ASSISTANT, "content": response})
            # concate the template
            messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages


eval_templates: Dict[str, "EvalTemplate"] = {}


def register_eval_template(name: str, system: str, choice: str, answer: str, prefix: str) -> None:
    eval_templates[name] = EvalTemplate(system=system, choice=choice, answer=answer, prefix=prefix)


def get_eval_template(name: str) -> "EvalTemplate":
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


register_eval_template(
    name="default",
    system="The following are multiple choice questions (with answers).\n\n",
    # system="",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    # answer="",
    prefix="",
)


register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix="\n",
)

register_eval_template(
    name="mbpp",
    # system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    system="You are an expert Python programmer, and here is your task: ",
    choice="\n{choice}. {content}",
    # answer="\nAnswer: ",
    answer="",
    prefix=" ",
)

register_eval_template(
    name="ant_fineval",
    system="",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix="",
)

register_eval_template(
    name="none",
    system="",
    choice="\n{choice}. {content}",
    answer="\n",
    prefix="",
)

register_eval_template(
    name="selection",
    system="",
    choice=" {content}",
    answer="",
    prefix="",
)

register_eval_template(
    name="gsm8k",
    system="Q:",
    choice="",
    answer="\nA:\n",
    prefix="",
)