import sys
import argparse
import subprocess
import os
import time
import yaml

cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)

def run_command(cmd):
    if isinstance(cmd, list):
        cmd = ' '.join(map(str, cmd))
    process = subprocess.Popen(
        cmd,
        shell=True,
    )
    while True:
        # 检查进程是否结束
        if process.poll() is not None:
            break
        time.sleep(1)

    # 获取命令的返回码
    rc = process.poll()
    return rc


if __name__ == "__main__":
    env = os.environ.copy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="File path of config yaml")
    parser.add_argument("--model_path", type=str, default='model_path', help="model_path")
    parser.add_argument("--max_length", type=str, help="model_path")
    parser.add_argument("--max_new_tokens", type=str, default='32768', help="model_path")
    parser.add_argument("--temperature", type=float, help="0.0 means greedy decoding")
    parser.add_argument("--remote_model_port", type=str, default="1666", help="remote_model_port")
    parser.add_argument("--prompt_type", type=str, default="chat_template", help="prompt_type")
    parser.add_argument("--tensor_parallel", type=str, default="1", help="tensor_parallel")
    parser.add_argument("--model_name", type=str, required=False, default="", help="model name that is evaluated")
    parser.add_argument("--judge_model_path", type=str, required=False, default="", help="judge_model_path")
    parser.add_argument("--judge_model_port", type=str, required=False, default="", help="judge_model_port")
    parser.add_argument("--judge_tensor_parallel", type=str, required=False, help="judge_tensor_parallel")
    parser.add_argument("--backend", type=str, default='vllm', choices=['vllm', 'hf', 'sglang','asr'], help="inference backend")
    parser.add_argument("--load_type", type=str, default='last', choices=['last', 'best'], help="Load the latest model or the best performing model on the validation set")
    parser.add_argument("--subsample", type=float, help="proportion (0.0 - 1.0) or number (>= 1) of samples used")
    parser.add_argument("--seed", type=int, help="seed used for subsampling")
    args, unknown = parser.parse_known_args()

    with open(args.config) as F:
        config = yaml.safe_load(F)

    # tgi environment
    if config.get('backend', "") == 'tgi':
        for key in config:
            if key.startswith("tgi_"):
                upper_key = key.upper()
                os.environ[upper_key] = str(config.get(key, ''))

    #set env variable
    os.environ['REMOTE_MODEL_URL'] = "http://127.0.0.1:{}/model".format(args.remote_model_port)
    os.environ['MODEL_PATH'] = args.model_path
    os.environ['REMOTE_MODEL_PORT'] = args.remote_model_port
    os.environ['JUDGE_MODEL_PORT'] = args.judge_model_port
    
    #check is use external api
    external_api = os.environ.get('EXTERNAL_API')
    if not external_api:
        #model server
        model_server = [
            "python", os.path.join(cur_path, "inference", "predict_multi_gpu.py"),
            "--model", args.model_path,
            "--server_port", args.remote_model_port,
            "--prompt", config.get('prompt_type') or args.prompt_type,
            "--tensor_parallel", config.get('tensor_parallel') or args.tensor_parallel,
            "--backend", config.get('backend') or args.backend,
            "--preprocess", config.get('preprocess') or "preprocess",
            "--max_new_tokens", config.get('max_new_tokens') or args.max_new_tokens,
            "--load_type", config.get('load_type') or args.load_type,
            "--run_forever",
            "&"
        ]
        for k in ['max_length', 'temperature', 'stop', 'subsample', 'seed', 'chat_template_kwargs']:
            if v := (config.get(k) or getattr(args, k, None)):
                model_server.insert(-1, f'--{k}')
                model_server.insert(-1, str(v))
        if v := config.get('engine_args'):
            model_server[-1:-1] = v.split()
        run_command(model_server)

    #检测是否有judge模型
    print(args.judge_model_path)
    if args.judge_model_path:
        print("start judge")
        judge_server = [
            "python", os.path.join(cur_path, "inference", "predict_multi_gpu.py"),
            "--model", args.judge_model_path,
            "--server_port", args.judge_model_port,
            "--tensor_parallel", args.judge_tensor_parallel,
            "--prompt", "chat_template",
            "--preprocess", "preprocess",
            "--max_new_tokens", args.max_new_tokens,
            "--run_forever",
            "--manual_start",
            "&"
        ]
        run_command(judge_server)

    post_cmd = "python {} --config {}".format(os.path.join(cur_path, "run.py"), args.config)
    returncode = run_command(post_cmd)
    if returncode != 0:
        sys.exit(returncode)
