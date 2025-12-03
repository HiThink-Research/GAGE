import json
import os
import subprocess
import sys
import tempfile
import multiprocessing as mp
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNPY = ROOT / "run.py"
FIXTURE = ROOT / "tests" / "fixtures" / "piqa_mini.jsonl"


def _serve(handler_cls, port_queue):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    host, port = server.server_address
    port_queue.put(port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


class _TGIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = {"generated_text": "A"}
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, *_args, **_kwargs):
        return


class _SGLangHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = {"outputs": [{"text": "A"}]}
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, *_args, **_kwargs):
        return


def _start_server(handler_cls):
    queue = mp.Queue()
    proc = mp.Process(target=_serve, args=(handler_cls, queue))
    proc.start()
    port = queue.get()
    return proc, port


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
    # run.py 会在 GAGE_EVAL_SAVE_DIR 下创建 run_id 子目录
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
        proc, port = _start_server(_TGIHandler)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "tgi_run"
                summary = _run_pipeline(
                    ROOT / "config" / "custom" / "piqa_tgi_unittest.yaml",
                    {
                        "PIQA_MINI_PATH": str(FIXTURE),
                        "TGI_BASE_URL": f"http://127.0.0.1:{port}",
                    },
                    out_dir,
                )
                self.assertIn("metrics", summary)
        finally:
            proc.terminate()
            proc.join(5)

    def test_piqa_sglang_pipeline_with_custom_config(self):
        proc, port = _start_server(_SGLangHandler)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "sglang_run"
                summary = _run_pipeline(
                    ROOT / "config" / "custom" / "piqa_sglang_unittest.yaml",
                    {
                        "PIQA_MINI_PATH": str(FIXTURE),
                        "SGLANG_BASE_URL": f"http://127.0.0.1:{port}",
                    },
                    out_dir,
                )
                self.assertIn("metrics", summary)
        finally:
            proc.terminate()
            proc.join(5)


if __name__ == "__main__":
    unittest.main()
