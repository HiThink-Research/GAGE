# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import inspect
import json
import os
from typing import Any, Dict, List, Optional
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from datasets import load_dataset,builder,DownloadMode
from tqdm import tqdm, trange
from transformers.utils import cached_file
from loguru import logger

from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model_and_tokenizer, RemoteModel
from .template import get_eval_template
from torch.nn import CrossEntropyLoss
from ..extras.misc import count_parameters,get_current_device
from .execution import execution, write_jsonl
from ..data import get_template_and_fix_tokenizer,data_engine
loss_func = CrossEntropyLoss(reduction='none')


class Evaluator:
    """Class representing very high temperatures"""
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, finetuning_args)
        self.tokenizer.padding_side = "left"  # avoid overflow issue in batched inference for llama2
        # self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [
            self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in CHOICES
        ]
    # 对服务器发请求获取loss和next_token_prob
    def batch_inference_remote_model(self, batch_input: List[Dict[str, torch.Tensor]]) -> List[str]:
        choice_probs = self.model.run_batch(
            input_ids=[d['input_ids'] for d in batch_input],
            output_type='next_token_prob',
            next_token_ids=self.choice_inputs
        )
        
        out = []
        for c in choice_probs:
            tmp = []
            for choice_token in self.choice_inputs:
                tmp.append(c[str(choice_token)])
            out.append(tmp)
        return out

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        batch_input.to(get_current_device())
        # if self.model_args.data_parallel:
        #     self.model.to(get_current_device())
        logits = self.model(**batch_input).logits
        
        lengths = [logits.size(1)]*logits.size(0)
        # lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        # print(lengths)
        # print(logits.shape)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        # if self.eval_args.task in ["ant_fineval",'arc']:
        decode_text = self.tokenizer.batch_decode(batch_input['input_ids'],skip_special_tokens=True)[0] #TODO:检查单个batch有几个选项可能会有风险
        if "\nE" in decode_text:
            self.choice_inputs = [self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in ["A", "B", "C", "D","E"]]
        elif "\nC" not in decode_text:
            self.choice_inputs = [self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]
        elif "\nD" not in decode_text:
            self.choice_inputs = [self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in ["A", "B","C"]]
        
        
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        # logger.debug(choice_probs)
        
        return choice_probs.tolist()
    
    @torch.inference_mode()
    def batch_ppl(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        batch_input.to(get_current_device())
        if self.model_args.data_parallel:
            self.model.to(get_current_device())
        logger.debug(self.tokenizer.decode(batch_input["input_ids"][0], skip_special_tokens=True))
        logits = self.model(**batch_input).logits
        logits = logits[:, :-1]
        logits_t = logits.transpose(1, 2)
        input_t = batch_input["input_ids"][:, 1:]
        # print(logits_t,input_t)
        loss = loss_func(logits_t, input_t) * batch_input["attention_mask"][:, :-1]
        return loss

    def batch_selection_remote_model(self, batch_input: List[List[Dict[str, torch.Tensor]]]) -> List[str]:
        input_ids_list = [d['input_ids'] for ds in batch_input for d in ds]

        losses = self.model.run_batch(input_ids=input_ids_list, output_type='loss')
        batch_losses = []
        i = 0
        for choice_input in batch_input:
            batch_losses.append([])
            for _ in choice_input:
                batch_losses[-1].append(losses[i])
                i += 1
        # import pdb; pdb.set_trace()
        return batch_losses

    @torch.inference_mode()
    def batch_selection(self, batch_input, offset) -> List[str]:
        # batch_input.to(get_current_device())
        batch_input_list = [batch.to(get_current_device()) for batch in batch_input]

        if self.model_args.data_parallel:
            self.model.to(get_current_device())
        
        batch_choice_loss = []
        for index in range(len(batch_input_list)):
            logits = self.model(**batch_input_list[index]).logits
            logits = logits[:, :-1]
            logits_t = logits.transpose(1, 2)
            input_t = batch_input_list[index]["input_ids"][:, 1:]
            loss = loss_func(logits_t, input_t) * batch_input_list[index]["attention_mask"][:, :-1]
            lengths = torch.sum(batch_input_list[index]["attention_mask"][:, 1:], dim=-1)
            
            # choice_loss = [torch.mean(loss[i, offset[index][i]+logits.size(1)-lengths[i]:]).item() for i in range(len(offset[index]))]
            choice_loss = [torch.sum(loss[i]).item()/lengths[i] for i in range(len(lengths))]
            
            batch_choice_loss.append(choice_loss)
            
        return batch_choice_loss

    @torch.inference_mode()
    def batch_generate(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        batch_input.to(get_current_device())
        response = self.model.generate(
            **batch_input,
            max_new_tokens = 2048,
            temperature=0.15,
            do_sample=True
        )
        decoded_texts = [self.tokenizer.decode(output[len(batch_input["input_ids"][i]):], skip_special_tokens=True) for i,output in enumerate(response)]
        
        return decoded_texts

    def eval_logits(self) -> None:
        print(self.eval_args.task_path, self.eval_args.task)
        # building the data_engine

        DE = data_engine(dataset_name=self.eval_args.task,task=self.eval_args.task,task_path=self.eval_args.task_path)

        category_corrects = {}
        category_corrects["Average"] = np.array([], dtype="bool")
        results = []

        if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
            kwargs = {"trust_remote_code": True}
        else:
            kwargs = {}
        DE.download_and_prepare()
        dataset = DE.as_dataset()

        inputs, outputs, labels,offset = [], [], [], []
        all_preds = []
        for i in trange(len(dataset[self.data_args.split]), desc="Evaluating {}".format(self.eval_args.task), position=1, leave=False):
            # if i >= 100:
            #     break
            # take k-shot example
            if(self.eval_args.n_shot != 0):
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
            else:
                support_set = ()
            messages = self.eval_template.format_example(
                target_data=dataset[self.data_args.split][i],
                support_set=support_set,
                subject_name=self.eval_args.task,
                task=self.eval_args.task,
                eval_type=self.eval_args.eval_type
            )
            
            # print(messages)

            if self.eval_args.n_shot==0 and len(messages) > 2:
                in_temp = []
                offset_temp = []
                for idx in range(0,(int(len(messages)/2))):
                    input_ids, answer_ids = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages[idx*2:(idx+1)*2])
                    input_ids = input_ids + self.tokenizer.encode(" ") + answer_ids
                    in_temp.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                inputs.append(in_temp)
                labels.append(str(int(len(messages)/2)-1))
            else:
                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])
        for i in trange(
            0, len(inputs), self.eval_args.batch_size, desc="Predicting {}".format(self.eval_args.task), position=1, leave=False
        ):
            # import pdb; pdb.set_trace()
            if self.eval_args.eval_type in ['loss']:
                if isinstance(self.model, RemoteModel):
                    batch_input = inputs[i : i + self.eval_args.batch_size]
                    preds = self.batch_selection_remote_model(batch_input)
                else:
                    batch_input = []
                    batch_in_temp = inputs[i : i + self.eval_args.batch_size]
                    for batch_in in batch_in_temp:
                        batch_in = self.tokenizer.pad(batch_in, return_attention_mask=True, return_tensors="pt")
                        batch_input.append(batch_in)
                    preds = self.batch_selection(batch_input,offset[i : i + self.eval_args.batch_size])
                
                # print(preds)
                for j in range(0,len(preds)):
                    tmp_preds = preds[j]
                    if isinstance(self.model, RemoteModel):
                        all_preds.append(tmp_preds)
                    else:
                        all_preds.append([t.item() for t in tmp_preds])
                
            else:
                if isinstance(self.model, RemoteModel):
                    batch_input = inputs[i : i + self.eval_args.batch_size]
                    preds = self.batch_inference_remote_model(batch_input)
                    all_preds += preds
                else:
                    batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                    )
                    preds = self.batch_inference(batch_input)
                    all_preds += preds
                # import pdb; pdb.set_trace()
                
        self._save_results(category_corrects, all_preds)

    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: list) -> None:
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=True)
            with open(self.eval_args.task_path,'r') as f:
                input_data = [json.loads(l) for l in f]
                tmp_data = []
                for data in input_data:
                    choices_text = data['choices'][0]['message']['content'][0]['text']
                    target_scores = json.loads(choices_text)
                    if len(target_scores) < 6: 
                        tmp_data.append(data)

            assert len(tmp_data) == len(results), f"Length mismatch: tmp_data has {len(tmp_data)} elements, but results has {len(results)} elements"
            predict_out = []
            for i,d in enumerate(tmp_data):
                d['predict_result'] = results[i]
                predict_out.append(d)
            with open(os.path.join(self.eval_args.save_dir, "{}.jsonl".format(self.eval_args.task)), "w", encoding="utf-8", newline="\n") as f:
                for o in predict_out:
                    f.write(json.dumps(o, ensure_ascii=False)+'\n')
            # with open(os.path.join(self.eval_args.save_dir, "{}.log".format(self.eval_args.task)), "w", encoding="utf-8", newline="\n") as f:
            #     json.dump(out, f, indent=2,ensure_ascii=False)
        logger.info("save to {}".format(self.eval_args.save_dir))

if __name__ == "__main__":
    evaluator = Evaluator()
    predictions = ["def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) "]
    references = [["assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))", "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))", "assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))"]]
    print(evaluator._eval_mbpp(predictions, references))
