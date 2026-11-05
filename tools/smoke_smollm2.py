"""Bounded CLI and loopback-API checks; no third-party Python dependencies."""
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build/llama-cpu/bin/Release"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
common = ["--model", str(ROOT / "models/SmolLM2-135M-Instruct-f16.gguf"),
          "--gpu-layers", "0", "--threads", "8", "--threads-batch", "8",
          "--ctx-size", "2048", "--batch-size", "128", "--ubatch-size", "128"]
prompt = "In one sentence, name the capital of France."
start = time.perf_counter()
with (OUT / "smollm2-cli.stdout.txt").open("wb") as stdout, (OUT / "smollm2-cli.stderr.txt").open("wb") as stderr:
    cli = subprocess.run([str(BIN / "llama-cli.exe"), *common, "--single-turn",
                          "--simple-io", "--prompt", prompt, "--predict", "64",
                          "--temp", "0", "--seed", "42"],
                         stdout=stdout, stderr=stderr, timeout=120)
cli_wall = time.perf_counter() - start
if cli.returncode:
    raise RuntimeError(f"CLI failed: {cli.returncode}")
summary = {"prompt": prompt, "cli_exit_code": cli.returncode, "cli_wall_seconds": cli_wall}
with (OUT / "smollm2-api.stdout.txt").open("wb") as stdout, (OUT / "smollm2-api.stderr.txt").open("wb") as stderr:
    server = subprocess.Popen([str(BIN / "llama-server.exe"), *common,
                               "--host", "127.0.0.1", "--port", "18085"], stdout=stdout, stderr=stderr)
    try:
        deadline = time.monotonic() + 60
        while True:
            if server.poll() is not None:
                raise RuntimeError("API server exited during startup")
            try:
                with urllib.request.urlopen("http://127.0.0.1:18085/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() > deadline:
                    raise RuntimeError("API startup timed out")
                time.sleep(0.2)
        request = urllib.request.Request("http://127.0.0.1:18085/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": prompt}],
                             "temperature": 0, "seed": 42, "max_tokens": 64}).encode(),
            headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        summary.update(api_request_seconds=time.perf_counter() - start,
                       api_response=result)
        if not result["choices"][0]["message"]["content"].strip():
            raise RuntimeError("API returned an empty answer")
    finally:
        server.terminate()
        server.wait(timeout=30)
(OUT / "smollm2-smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
