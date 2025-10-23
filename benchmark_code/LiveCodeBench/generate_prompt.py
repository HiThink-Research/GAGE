import os
import json
import random
import string
import time
import requests
import re
import tiktoken
import openai

from eval_test_prediction import preprocess_data as prep_test_pred
from eval_execution import preprocess_data as prep_exe
from eval_code_generation import preprocess_data as prep_gen

def main():



    # test_prediction pre-process
    save_dir_t_p="../../eval_result/LiveCodeBench/test_prediction_prompt"
    os.makedirs(save_dir_t_p, exist_ok=True)
    prep_test_pred(
        input_path="../../../test_pred",input_file="livecodebench_test_pred.jsonl", save_dir=save_dir_t_p, task="test_prediction"
        )
    print("Saving prompt to", save_dir_t_p)
    print("Prompt generated successfully(test_pred task))!")



    # execution pre-process
    save_dir_e="../../eval_result/LiveCodeBench/execution_prompt"
    os.makedirs(save_dir_e, exist_ok=True)
    print("getting into prep_exe function....")

    prep_exe(
        input_path="../../../execution/data",input_file="livecodebench_execution.jsonl", save_dir=save_dir_e, task="execution"
        )
    try:
        
        print("Saving prompt to", save_dir_e)
        print("Prompt generated successfully(execution))!")
    except Exception as e:
        print("Exception found:", e)

    # code generation pre-process
    save_dir_gen="../../eval_result/LiveCodeBench/generation_prompt"
    os.makedirs(save_dir_gen, exist_ok=True)
    print("getting into prep_gen function....")

    prep_gen(
        input_path="../../../generation/generation_lite",input_file="livecodebench_generation_lite.jsonl", save_dir=save_dir_gen, task="generation"
        )
    try:
        
        print("Saving prompt to", save_dir_gen)
        print("Prompt generated successfully(generation))!")
    except Exception as e:
        print("Exception found:", e)

if __name__ == "__main__":
    print("Generating prompt for LiveCodeBench test prediction, execution & generation...")
    main()