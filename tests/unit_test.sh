# export EXTERNAL_API=chatgpt #支持gpt-4o，deepseek，gemini，claude并支持手动添加，如果使用openai等官方API，该字段为chatgpt
# export API_KEY=API_KEY #如果使用openai等官方API，需要填写该字段
# export MODEL_NAME=chatgpt

# --config yaml files {unit_test_nlp.yaml, unit_test_mm.yaml, unit_test_external.yaml} 

python ../run_pipeline.py \
    --config unit_test.yaml \
    --model_path /data/qwen3-0_6B \
    --max_length 8000 \
    --remote_model_port 16666 \
    --prompt_type chat_template \