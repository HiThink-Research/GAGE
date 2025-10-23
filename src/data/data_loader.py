import json
import os
import datasets
from typing import Any, Dict, List, Optional

class Config(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

    
class data_engine(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    def __init__(self, task, task_path, **kwargs):
        super().__init__(**kwargs)
        self.task_path = task_path
        

    def _info(self):
        # TODO(hellaswag): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=self.dataset_name,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    # These are the features of your dataset like images, labels ...
                    "target_scores":datasets.Value("string"),
                    "question":datasets.Value("string"),
                    "A":datasets.Value("string"),
                    "B":datasets.Value("string"),
                    "C":datasets.Value("string"),
                    "D":datasets.Value("string"),
                    "E":datasets.Value("string"),
                    "answer": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": self.task_path},
            )
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        print("loading from {}".format(filepath))
        with open(filepath, encoding="utf-8") as f:
            data = [json.loads(l) for l in f]
            for id_, row in enumerate(data):
                question = row['query']
                choices_text = row['choices'][0]['message']['content'][0]['text']
                target_scores = json.loads(choices_text)
                # 判断是选项超过了6
                if len(target_scores) > 5:
                    continue
                processed_row = {
                    "question": question,
                    "target_scores": str(target_scores),
                    "A": "",
                    "B": "",
                    "C": "",
                    "D": "",
                    "E": ""
                }
                choice_labels = ["A", "B", "C", "D", "E"]
                for i, (option, score) in enumerate(target_scores.items()):
                    if i < len(choice_labels):
                        processed_row[choice_labels[i]] = option
                        if score == 1:
                            processed_row["answer"] = choice_labels[i]
                
                yield id_, processed_row

