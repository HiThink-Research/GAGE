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
