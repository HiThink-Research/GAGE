import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
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
    
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    return server, port

# Note: _run_pipeline stays same

def _run_pipeline(config_path: Path, extra_env: dict, output_dir: Path):
    env = os.environ.copy()
    env.update(extra_env)
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
    run_dirs = list(output_dir.iterdir())
    assert run_dirs, f"no run_dir created under {output_dir}"
    summary_path = run_dirs[0] / "summary.json"
    assert summary_path.exists(), "summary.json should be generated"
    summary = json.loads(summary_path.read_text())
    metrics = summary.get("metrics") or []
    assert metrics, "metrics should not be empty"
    return summary


class CustomPiqaPipelineTests(unittest.TestCase):
    def test_piqa_tgi_pipeline_with_custom_config(self):
        server, port = _start_server(_TGIHandler)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "tgi_run"
                summary = _run_pipeline(
                    ROOT / "config" / "custom" / "piqa_tgi.yaml",
                    {
                        "PIQA_MINI_PATH": str(FIXTURE),
                        "TGI_BASE_URL": f"http://127.0.0.1:{port}",
                    },
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
                summary = _run_pipeline(
                    ROOT / "config" / "custom" / "piqa_sglang.yaml",
                    {
                        "PIQA_MINI_PATH": str(FIXTURE),
                        "SGLANG_BASE_URL": f"http://127.0.0.1:{port}",
                    },
                    out_dir,
                )
                self.assertIn("metrics", summary)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
