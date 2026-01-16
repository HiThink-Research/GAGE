# Benchmark

## 启动方式

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