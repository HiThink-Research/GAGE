import os
import sys

sys.path.append(os.path.realpath(os.path.dirname(os.path.abspath(__file__))))
from adaptor_base import set_device, EngineAdaptorBase

set_device()

import argparse
import asyncio
import subprocess
import time
import ujson as json
import requests
import aiohttp
from typing import List, Dict, Any, Union

from sglang.utils import terminate_process  # Reusing utility from SGLang
import socket
from contextlib import closing
from transformers import AutoTokenizer
OUTPUT_TYPES = ['text', 'next_token_prob']


class TGIEngineAdaptor(EngineAdaptorBase):

    async def run_predict_until_complete(self) -> None:
        try:
            # self.load_model()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as self.session:
                await super().run_predict_until_complete()
        except Exception as e:
            print(f"Error encountered: {e}")
            self.terminate_server()
            raise
        finally:
            self.terminate_server()

    def terminate_server(self) -> None:
        """Terminate TGI server process"""
        if hasattr(self, 'server_proc') and self.server_proc:
            terminate_process(self.server_proc)
            self.server_proc = None

    def should_stop_adding_sample(self, n_added: int) -> bool:
        """Determine if we should stop adding new samples to the queue"""
        # TGI handles its own batching, so we can use a larger queue
        return n_added >= 8 or len(self.idx2task) >= 64

    def prepare_inputs(self, r: dict) -> Dict[str, Any]:
        otype = r.get('output_type', self.args.output_type)
        params = r.get('generation_params', {})
        sample_n = params.get('n', self.args.sample_n)
        
        # Common parameters for TGI
        tgi_params = {
            "max_new_tokens": params.get('max_new_tokens', self.args.max_new_tokens),
            "temperature": params.get('temperature', self.args.temperature),
            "top_p": params.get('top_p', self.args.top_p),
            "repetition_penalty": params.get('repetition_penalty', self.args.repetition_penalty),
            "do_sample": params.get('temperature', self.args.temperature) > 0 or 
                         params.get('top_p', self.args.top_p) < 1.0,
            "stop_sequences": params.get('stop', self.args.stop) or [],
            "return_full_text": False,
        }

        # Special handling for different output types
        if otype == 'next_token_prob':
            tgi_params.update({
                "max_new_tokens": 1,
                "details": True,
                "top_n_tokens": params.get('top_logprobs_num', 5)
            })

        return {
            'inputs': r['inputs'],
            'params': tgi_params,
            'request_id': r['idx'],
            'otype': otype,
            'sample_n': sample_n
        }

    async def predict_sample(
        self,
        inputs: Union[List[int], str],
        params: Dict[str, Any],
        request_id: str,
        otype: str,
        sample_n: int = 1
    ) -> Dict[str, Any]:
        """
        Single sample prediction for TGI
        
        Args:
            inputs: Input text (string) or token IDs (list)
            params: TGI generation parameters
            request_id: Unique request ID
            otype: Output type
            sample_n: Number of samples to generate
        """
        # TGI requires string input, convert if token IDs are provided
        if isinstance(inputs, list):
            # In practice, you would decode token IDs to string
            # For simplicity, we assume inputs is already a string
            inputs = self.tokenizer.decode(inputs)
            print("转换后的输入:", inputs)
        
        # Prepare request data
        request_data = {
            "inputs": inputs,
            "parameters": params
        }
        
        # Handle multiple samples
        if sample_n > 1:
            request_data["parameters"]["best_of"] = sample_n

        url = f"http://127.0.0.1:{self.port}/generate"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    url, 
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(f"TGI error {response.status}: {text}")
                    
                    result = await response.json()
                    return result
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def convert_final_output(
        self,
        inputs: Dict,
        final_output: Any,
        otype: str
    ) -> Any:
        """Convert TGI output to desired format"""
        if otype == 'next_token_prob':
            try:
                # Extract next token probabilities from TGI response
                token_info = final_output['details']['prefill'][-1]
                return [
                    [tok['id'], tok['logprob']] 
                    for tok in token_info['tokens']
                ]
            except (KeyError, IndexError):
                return []
        
        # For text output, return generated text

        return self.tokenizer.encode(final_output['generated_text'])

    def load_model(self) -> None:
        """Load model by starting TGI server"""
        args = self.args
        tgi_port = self.random_port()
        cmd = [
            "text-generation-launcher",
            "--model-id", args.model,
            "--port", str(tgi_port),
            "--num-shard", str(self.get_gpu_count()),
            "--dtype", args.dtype,
            "--max-input-length", str(args.max_input_length),
            "--max-total-tokens", str(args.max_total_tokens),
            "--max-batch-prefill-tokens", str(args.max_batch_prefill_tokens),
            "--max-batch-total-tokens", str(args.max_batch_total_tokens),
        ]
        
        # Add quantization if specified
        if args.quantize:
            cmd.extend(["--quantize", args.quantize])
        
        # Add additional arguments
        # cmd.extend(self.additional_args)
        
        # Print final command
        print("Starting TGI server with command:")
        print(" ".join(cmd))
        
        # Start server process
        self.server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for server to start
        timeout = 600  # 10 minutes
        start_time = time.time()
        server_ready = False
        
        # Print server output while waiting
        def monitor_output():
            for line in iter(self.server_proc.stdout.readline, ''):
                print("[TGI]", line.strip())
                if "Connected" in line or "Ready" in line:
                    nonlocal server_ready
                    server_ready = True
        
        # Start output monitoring thread
        import threading
        monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        monitor_thread.start()
        
        # Wait for server readiness
        while not server_ready:
            if time.time() - start_time > timeout:
                self.terminate_server()
                raise TimeoutError("TGI server startup timed out")
            
            # Check if process died
            if self.server_proc.poll() is not None:
                self.terminate_server()
                raise RuntimeError("TGI server failed to start")
            
            time.sleep(1)
        
        print("TGI server is ready")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.port = tgi_port

    def random_port(self) -> int:
        """Generate a random port for TGI server"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGI Inference Adapter")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID or path")
    parser.add_argument("--server_addr", type=str, required=True, help="Data server address")
    parser.add_argument("--server_port", type=int, required=True, help="Data server port")
    
    # TGI specific arguments
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"], 
                        help="Model data type")
    parser.add_argument("--quantize", type=str, default=None, 
                        choices=["bitsandbytes", "bitsandbytes-nf4", "gptq"],
                        help="Quantization method")
    
    # Generation parameters
    parser.add_argument("--output_type", type=str, default='text', choices=OUTPUT_TYPES, help="Output type")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--max_input_length", type=int, default=2048, help="Max input tokens")
    parser.add_argument("--max_total_tokens", type=int, default=4096, help="Max total tokens (input + output)")
    parser.add_argument("--sample_n", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--stop", type=str, default=None, help="Stop sequences (comma separated)")
    
    # Batching parameters
    parser.add_argument("--max_batch_prefill_tokens", type=int, default=4096, 
                        help="Max tokens for prefill in batch")
    parser.add_argument("--max_batch_total_tokens", type=int, default=16384, 
                        help="Max total tokens in batch")
    
    args, engine_args = parser.parse_known_args()
    
    # Preprocess stop sequences
    if args.stop:
        args.stop = [s.strip() for s in args.stop.split(",") if s.strip()]
    
    asyncio.run(TGIEngineAdaptor(args, engine_args).run_predict_until_complete())