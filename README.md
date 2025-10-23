# GAGE Architecture
![WorkFlow](docs/static/Eval_wrokflow.png)

### 🌐 Why **GAGE**

**GAGE (General AI Gauge Engine)** is a unified, extensible evaluation framework designed for large language, multimodal models, audio models and diffusion models. Compared with other evaluation engine, GAGE focuses on *scalability, flexibility,* and *real-world agent evaluation*.

**Key Advantages**

1. 🧩 **Extensible Design** – Modular config, dataset, and adapter system for rapid integration of new tasks and models.
2. ⚙️ **Multi-Engine Inference** – Supports vLLM, SGLang, and HF backends with distributed multi-GPU execution.
3. 🧠 **Agent Sandbox Evaluation** – Built-in sandbox (FinQuery, GUI, Search) for realistic agent performance testing.
4. 🤖 **LLM-as-a-Judge** – Native support for external APIs (GPT-4o, DeepSeek, Gemini, Claude) for automatic judgment.
5. 📊 **Unified Scoring Framework** – Customizable metrics (accuracy, F1, reasoning consistency) across all tasks.


## Support testsets
14 NLP testsets，2 Finance testsets，9 audio testsets，5 GUI testsets，18 multimodal testsets，see config folder for more information

## Directory structure
```
llm-eval
├── README.md
├── __init__.py
├── benchmark_code        # All scoring (evaluation) functions are located in this folder
├── config                # Custom sample configuration files can be found here
├── docs                  # Automatically generated API documentation (built with Sphinx)
├── inference             # Contains all inference engine–related code
├── post_eval.py          # Script for launching evaluation after inference is completed
├── requirements.txt
├── run.py                # Used together with run_pipeline.py
├── run_pipeline.py       # The main entry point for running the entire evaluation pipeline
├── scripts               # Example shell scripts (e.g., run.sh)
├── tools                 # Utility functions and wrappers (e.g., HTTP requests)
├── statistic.py          # Script for aggregating and uploading final evaluation statistics
├── testsets              # All non-business test sets are located here
└── utils                 # Common utility functions

```

## Example

### Example config.yaml

```yaml
# where to save your predict result
save_dir: /mnt/workspace/inference

# Subsample（optional）
# subsample: 0.001
# seed: 1235

# infra params（optional）
# backend: hf
# temperature: 0.6
# preprocess: preprocess_no_think
# max_length: 32768
# max_new_tokens: 4096
# load_type: last
# engine_args: "--kv-cache-dtype fp8_e5m2  --quantization awq_marlin"
# tensor_parallel: 2
# judge_tensor_parallel: 4
# judge_max_length: 65536
# judge_max_new_tokens: 4096

tasks:
    金融股票涨跌评估数据集_0-0-1:
        compare_func:
            path: benchmark_code/BizFinBench/eval_stock_prediction.py
        data_path: /mnt/data/llm_benchmark/folder_1475/test_version_0.0.1/TESTSET__金融股票预测评估数据集__0-0-12.jsonl
        type: text
    MATH (LLM as judge):
        type: text
        data_path: /sft/data/TESTSET/TESTSET__开源测试集-math__1-0-2.jsonl
        judge:
            preprocess: utils.judge.data_preprocess
            method: gpt-4o #support gpt-4o，deepseek，gemini，claude
        compare_func:
            path: utils/eval_math500.py
```

### optional params list

| Name                  |     Type    |    Default    | Description                                                                                                                                                                                             |
| --------------------- | :---------: | :-----------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| subsample             | float / int |      None     | Test set downsampling. If set to a float between 0 and 1, samples proportionally. If set to an int > 1, samples by count.                                                                               |
| seed                  |     int     |      None     | Random seed for test set sampling, used to ensure reproducible samples.                                                                                                                                 |
| backend               |     str     |      vllm     | Model inference backend. Supported options: vllm / sglang / hf.                                                                                                                                         |
| preprocess            |     str     |   preprocess  | Preprocessing script for test samples.                                                                                                                                                                  |
| prompt_type           |     str     | chat_template | Prompt template type; defaults to the model’s built-in `chat_template`.                                                                                                                                 |
| chat_template_kwargs  |     str     |      None     | Keyword arguments passed to `tokenizer.apply_chat_template`, e.g., `"enable_thinking=False"`.                                                                                                           |
| temperature           |    float    |       0       | Sampling temperature during generation; defaults to greedy decoding.                                                                                                                                    |
| max_length            |     int     |      None     | Maximum model sequence length (input + output); defaults to model configuration.                                                                                                                        |
| max_new_tokens        |     int     |     32768     | Maximum number of tokens the model can generate in the output.                                                                                                                                          |
| load_type             |     str     |      last     | When the model directory contains multiple checkpoints, automatically load the last one (`last`) or the best-performing one on the validation set (`best`, requires training with the Swift framework). |
| engine_args           |     str     |      None     | Arguments passed directly to the inference engine (vllm / sglang).                                                                                                                                      |
| tensor_parallel       |     int     |       1       | Tensor parallelism degree (number of GPUs) used for model evaluation.                                                                                                                                   |
| judge_tensor_parallel |     int     |       1       | Tensor parallelism degree (number of GPUs) used for the judge model.                                                                                                                                    |
| judge_max_length      |     int     |      None     | Maximum sequence length (input + output) for the judge model; defaults to model configuration.                                                                                                          |
| judge_max_new_tokens  |     int     |      2048     | Maximum number of tokens the judge model can generate in the output.                                                                                                                                    |


### Quick Start evaluate local model

```sh
python run_pipeline.py \
    --config unit_test.yaml \
    --model_path /mnt/data/llm/models/chat/Qwen3-0.6B \
```

### Quick Start: Evaluate a Local Model (Disable Thinking Mode)

When evaluating hybrid reasoning models such as **Qwen3**, you can disable the thinking mode by adding
`chat_template_kwargs: enable_thinking=False`
to the configuration file.

### Quick Start: Evaluate a Local Model (Enable GPT-OSS Thinking Mode)

When evaluating **gpt-oss** series models, you can control the reasoning depth by adding
`chat_template_kwargs: reasoning_effort="high"`
to the configuration file.
Supported options include `low`, `medium`, and `high` (default: `medium`).


### Quick Start: Evaluate External APIs

```sh
export EXTERNAL_API=gpt-4o  # Supports gpt-4o, deepseek, gemini, claude, and custom additions.  
                           # If using official APIs such as OpenAI, set this field to 'chatgpt'.
export API_KEY=API_KEY     # Required if using official APIs like OpenAI.
export MODEL_NAME=gpt-4o

python run_pipeline.py \
    --config unit_test.yaml \
```

### Quick Start: Evaluate External APIs and Use External Models as Judges

You can specify an external API as the **judge model** directly in the YAML file:

```yaml
test:
    type: text
    data_path: /sft/data/TESTSET/TESTSET__OpenSource-Math__1-0-2.jsonl
    judge:
        preprocess: utils.judge.data_preprocess
        method: gpt-4o  # Specify which external model acts as the judge.  
                        # Supports gpt-4o, deepseek, gemini, claude, and custom additions.
    compare_func:
        path: utils/eval_math500.py
```


### post eval
如果已经完成模型推理，评估环节出错，可以手动执行如下代码获取评估结果

```sh
python post_eval.py --eval_func benchmark_code/Multiple_Choice_QA/eval_multi_choice.py --input_path xxx.jsonl --output_path xxx.log
```