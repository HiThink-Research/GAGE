# Benchmark

## Config

### bizfinbench.v2
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/biz_fin_bench_v2/bizfinbench_v2.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id bizfinbench_v2
```  

### MRCR v2
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/mrcr/openai_mrcr.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id mrcr
```

### Global PIQA
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/global_piqa/global_piqa_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id global_piqa
```  

### LiveCodeBench
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/live_code_bench/live_code_bench_test.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id live_code_bench_test
```

### GPQA-diamond

```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/gpqa_diamond_vllm_async_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id gpqa_diamond
```

### MathVista
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/mathvista_vllm_async_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id mathvista_chat
```

### AIME 2024
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/aime2024_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id aime2024
```

### AIME 2025
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/aime2025_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id aime2025
```


### AIME 2025
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/aime2025_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id aime2025
```


### MMLU-Pro
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/mmlu_pro/mmlu_pro_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id mmlu_pro_chat
```

### HLE
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/hle_chat_openai.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id hle
```

### MATH500
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/math500_vllm_async_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id math500
```

### MME
```bash
python GAGE_dev/run.py \
  --config GAGE_dev/config/custom/mme_vllm_async_chat.yaml \
  --output-dir ./gage_runs/final_test \
  --run-id mme
```

### MMAU-Pro

MMAU-Pro is a comprehensive benchmark designed to evaluate **audio intelligence in multimodal models**.  
It covers speech, environmental sounds, music and their combinations, spanning **49 distinct perceptual and reasoning skills** such as acoustic source characterization, acoustic scene reasoning, temporal and quantitative reasoning, and procedural reasoning ([dataset card](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro)).

The dataset contains **5,305 expert-annotated audio–question–answer triplets**, with audio clips collected from diverse real-world scenarios (including long-form and multi-audio cases).

#### 1. Pre-Execution

Before running the evaluation, prepare a local JSONL version of the test split and the corresponding audio files.  
Each JSONL record should follow the same field schema as the Hugging Face dataset (e.g. `id`, `audio_path`, `question`, `answer`, `choices`, `length_type`, `perceptual_skills`, `reasoning_skills`, `category`, `transcription`, etc.).

#### 2. Execution Command

Use the following command to initiate the benchmark process:

```bash
python zyw_bench/run.py \
  --config zyw_bench/config/custom/mmau_pro/mmau_pro_audio.yaml \
  --output-dir ./gage_runs_mmau_pro/final_test \
  --run-id mmau_pro
```

#### 3. Detailed Configuration

| Parameter | Description | Supported Values |
| --- | --- | --- |
| **`path`** | Path to the local MMAU-Pro JSONL file (schema aligned with `gamma-lab-umd/MMAU-Pro`). | *Valid JSONL file path* |
| **`audio_path_root`** | Root directory where the referenced audio files (e.g. `data/*.wav`) are stored. | *Valid directory path* |
| **`audio_index`** | Index of the audio segment to be used when multiple audio paths are provided per sample. | *Non-negative integer (default `0`)* |
| **`model_path` / `tokenizer_path`** | Local path to the audio-capable model and tokenizer (e.g. Qwen2-Audio family) used by the vLLM backend. | *Model directory containing `config.json` and tokenizer files* |

