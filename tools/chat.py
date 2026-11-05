"""Send an experiment prompt to Serve-SmolLM2.cmd using the Python standard library."""
import argparse
import json
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("prompt")
parser.add_argument("--max-tokens", type=int, default=128)
args = parser.parse_args()
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps({
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens, "temperature": 0,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
