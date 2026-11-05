"""Install the pinned SmolLM2 FP16 GGUF with SHA-256 verification."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
REVISION = "09816acd5d99df7be770d85ea30822623dab342c"
FILENAME = "SmolLM2-135M-Instruct-f16.gguf"
SHA256 = "f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"
SIZE = 270885952
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}?download=true"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    target = ROOT / "models" / FILENAME
    target.parent.mkdir(exist_ok=True)
    if not target.exists() or target.stat().st_size != SIZE or digest(target) != SHA256:
        temporary = target.with_suffix(".part")
        # Bounded ranges avoid a stalled full-file CDN response on this connection.
        chunk_size = 4 * 1024 * 1024

        def fetch(start):
            end = min(start + chunk_size, SIZE) - 1
            request = urllib.request.Request(
                URL + f"&range={start}", headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as source:
                expected = f"bytes {start}-{end}/{SIZE}"
                if source.status != 206 or source.headers.get("Content-Range") != expected:
                    raise RuntimeError("Unexpected download range response")
                data = source.read(end - start + 2)
            if len(data) != end - start + 1:
                raise RuntimeError("Incomplete download range")
            return data

        with temporary.open("wb") as output, ThreadPoolExecutor(max_workers=4) as pool:
            for group in range(0, SIZE, 4 * chunk_size):
                for data in pool.map(fetch, range(group, min(group + 4 * chunk_size, SIZE), chunk_size)):
                    output.write(data)
                    print(f"Downloaded {output.tell()}/{SIZE} bytes", flush=True)
        if temporary.stat().st_size != SIZE or digest(temporary) != SHA256:
            raise RuntimeError("Downloaded model failed size/SHA-256 verification")
        temporary.replace(target)
    metadata = {
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "original_revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "format": "FP16 GGUF", "filename": FILENAME, "size_bytes": SIZE,
        "sha256": SHA256, "gguf_repository": REPO, "gguf_revision": REVISION,
        "url": URL, "license": "Apache-2.0",
    }
    (target.parent / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Installed and verified:", target, flush=True)


if __name__ == "__main__":
    main()
