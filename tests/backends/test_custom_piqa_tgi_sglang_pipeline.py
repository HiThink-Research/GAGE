import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from textwrap import dedent
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNPY = ROOT / "run.py"
FIXTURE = ROOT / "tests" / "fixtures" / "piqa_mini.jsonl"


def _serve(handler_cls, port_queue):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    host, port = server.server_address
    server.timeout = 0.5
    port_queue.append(port)
    try:
        while getattr(server, "_keep_running", True):
            server.handle_request()
    finally:
        server.server_close()


class _TGIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            _ = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = {"generated_text": "A"}
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, *_args, **_kwargs):
        return


class _SGLangHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            _ = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = {"outputs": [{"text": "A"}]}
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, *_args, **_kwargs):
        return


def _start_server(handler_cls):
    port_queue = []
    thread = threading.Thread(target=_serve, args=(handler_cls, port_queue))
    thread.daemon = True
    
    # We need to set the flag on the thread object or similar, but the server running loop
    # inside _serve needs access. 
    # To keep it simple, we wrap the server object creation or pass a control object.
    # Actually, HTTPServer doesn't straightforwardly support "stop serving" from outside 
    # unless using `shutdown`, which blocks.
    # Let's simplify: Just run serve_forever in a thread and daemonize it. 
    # We don't need clean shutdown for tests, just daemon thread killing on exit.
    
    try:
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
    except PermissionError as exc:
        raise unittest.SkipTest("local HTTPServer bind not permitted in this environment") from exc
    port = server.server_address[1]
    
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    return server, port

# Note: _run_pipeline stays same

def _write_piqa_backend_config(output_path: Path, *, backend_type: str, base_url: str) -> Path:
    extra_backend = ""
    if backend_type == "sglang":
        extra_backend = "\n              presence_penalty: 0.0"
    config_text = dedent(
        f"""
        api_version: gage/v1alpha1
        kind: PipelineConfig

        metadata:
          name: piqa_{backend_type}_unit
          description: "PIQA mini + {backend_type} backend test config"

        custom:
          steps:
            - step: inference
            - step: auto_eval

        prompts:
          - prompt_id: piqa_infer_prompt
            renderer: jinja
            params:
              system_prompt: >
                你是一位擅长常识推理的助手，请阅读题目并在最后只输出正确选项对应的大写字母（A 或 B）。
              instruction: >
                请判断哪个答案更合理，并在最后一行仅输出一个大写字母（A 或 B）。
            template: |
              {{{{ system_prompt }}}}

              题目：
              {{{{ sample.goal }}}}

              选项：
              {{% for label, text in (sample.metadata.option_map or {{}}).items() %}}
              {{{{ label }}}}. {{{{ text }}}}
              {{% endfor %}}

              {{{{ instruction }}}}

        datasets:
          - dataset_id: piqa_validation_mini
            loader: jsonl
            params:
              path: {FIXTURE}
              preprocess: piqa_struct_only
              streaming: false

        backends:
          - backend_id: {backend_type}_backend_unit
            type: {backend_type}
            config:
              base_url: {base_url}
              timeout: 30
              max_new_tokens: 16
              temperature: 0.0
              top_p: 1.0{extra_backend}

        role_adapters:
          - adapter_id: dut_{backend_type}
            role_type: dut_model
            backend_id: {backend_type}_backend_unit
            prompt_id: piqa_infer_prompt
            capabilities:
              - chat_completion

        metrics:
          - metric_id: piqa_acc
            implementation: multi_choice_accuracy

        tasks:
          - task_id: piqa_validation_eval
            dataset_id: piqa_validation_mini
            max_samples: 1
            concurrency: 1
            reporting:
              sinks:
                - type: console
        """
    ).strip() + "\n"
    output_path.write_text(config_text, encoding="utf-8")
    return output_path


def _run_pipeline(config_path: Path, extra_env: dict | None, output_dir: Path):
    env = os.environ.copy()
    env.update(extra_env or {})
    cmd = [
        sys.executable,
        str(RUNPY),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        "1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if result.returncode != 0:
        raise AssertionError(f"run.py failed: {result.stderr}")
    # NOTE: run.py creates a run_id subdirectory under GAGE_EVAL_SAVE_DIR.
    # Internal helper directories such as .sandbox_leases may also appear there.
    run_dirs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert run_dirs, f"no run_dir created under {output_dir}"
    summary_paths = [path / "summary.json" for path in run_dirs if (path / "summary.json").exists()]
    assert summary_paths, "summary.json should be generated"
    summary = json.loads(summary_paths[0].read_text())
    metrics = summary.get("metrics") or []
    assert metrics, "metrics should not be empty"
    return summary


class CustomPiqaPipelineTests(unittest.TestCase):
    def test_piqa_tgi_pipeline_with_custom_config(self):
        server, port = _start_server(_TGIHandler)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "tgi_run"
                config_path = _write_piqa_backend_config(
                    Path(tmpdir) / "piqa_tgi_unit.yaml",
                    backend_type="tgi",
                    base_url=f"http://127.0.0.1:{port}",
                )
                summary = _run_pipeline(
                    config_path,
                    None,
                    out_dir,
                )
                self.assertIn("metrics", summary)
        finally:
            server.shutdown() # Blocks until request handler returns

    def test_piqa_sglang_pipeline_with_custom_config(self):
        server, port = _start_server(_SGLangHandler)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "sglang_run"
                config_path = _write_piqa_backend_config(
                    Path(tmpdir) / "piqa_sglang_unit.yaml",
                    backend_type="sglang",
                    base_url=f"http://127.0.0.1:{port}",
                )
                summary = _run_pipeline(
                    config_path,
                    None,
                    out_dir,
                )
                self.assertIn("metrics", summary)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
