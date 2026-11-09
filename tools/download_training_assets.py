"""Pin and verify public SmolLM2 checkpoints and instruction-training datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "training-assets"
MANIFEST = ASSETS / "manifest.json"
RECEIPT = ASSETS / "download-receipt.json"


def read_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def make_manifest():
    snapshots = []
    repo = "HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints"
    refs = read_json(f"https://huggingface.co/api/models/{repo}/refs")
    branches = sorted((r for r in refs["branches"] if r["name"].startswith("step-")),
                      key=lambda r: int(r["name"].split("-")[1]))
    requests = [("models", repo, r["targetCommit"], "checkpoints/" + r["name"]) for r in branches]
    requests += [("datasets", "HuggingFaceTB/smol-smoltalk", "main", "datasets/smol-smoltalk"),
                 ("datasets", "HuggingFaceH4/ultrafeedback_binarized", "main", "datasets/ultrafeedback_binarized")]
    for kind, repo, revision, local in requests:
        info = read_json(f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true")
        snapshots.append({"kind": kind, "repo": repo, "revision": info["sha"], "directory": local,
                          "files": [f for f in info["siblings"] if f["rfilename"] != ".gitattributes"]})
    ASSETS.mkdir(exist_ok=True)
    result = {"snapshots": snapshots,
              "scope": "All published step branches and both full instruction-training dataset repositories. Not the full pretraining corpus; not optimizer/RNG state."}
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    return result


def hashes(path):
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            sha256.update(chunk)
            git_blob.update(chunk)
    return sha256.hexdigest(), git_blob.hexdigest()


def verified(path, entry):
    if not path.exists() or path.stat().st_size != entry["size"]:
        return None
    sha256, blob = hashes(path)
    valid = sha256 == entry["lfs"]["sha256"] if "lfs" in entry else blob == entry["blobId"]
    return sha256 if valid else None


def download(snapshot, entry):
    path = ASSETS / "data" / snapshot["directory"] / entry["rfilename"]
    if not path.resolve().is_relative_to((ASSETS / "data").resolve()):
        raise RuntimeError("Invalid manifest destination")
    sha256 = verified(path, entry)
    if sha256:
        return {"path": str(path.relative_to(ROOT)), "size_bytes": entry["size"], "sha256": sha256}
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "datasets/" if snapshot["kind"] == "datasets" else ""
    url = (f"https://huggingface.co/{prefix}{snapshot['repo']}/resolve/"
           f"{snapshot['revision']}/{urllib.parse.quote(entry['rfilename'])}?download=true")
    temporary = path.with_name(path.name + ".part")
    size = entry["size"]
    if "lfs" in entry:
        chunk_size = 8 * 1024 * 1024

        def fetch(start):
            end = min(start + chunk_size, size) - 1
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url + f"&range={start}",
                        headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise RuntimeError("Unexpected range response")
                        data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Truncated range")
                    return data
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)

        with temporary.open("wb") as output, ThreadPoolExecutor(max_workers=8) as pool:
            for group in range(0, size, 8 * chunk_size):
                for data in pool.map(fetch, range(group, min(group + 8 * chunk_size, size), chunk_size)):
                    output.write(data)
                print(f"{snapshot['directory']}/{entry['rfilename']}: {output.tell()}/{size}", flush=True)
    else:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    sha256 = verified(temporary, entry)
    if not sha256:
        raise RuntimeError(f"Checksum mismatch: {path}")
    temporary.replace(path)
    return {"path": str(path.relative_to(ROOT)), "size_bytes": size, "sha256": sha256}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else make_manifest()
    total = sum(f["size"] for s in manifest["snapshots"] for f in s["files"])
    print(f"Pinned download set: {total} bytes", flush=True)
    if args.plan_only:
        return
    missing = sum(f["size"] for s in manifest["snapshots"] for f in s["files"]
                  if not (ASSETS / "data" / s["directory"] / f["rfilename"]).exists())
    if shutil.disk_usage(ROOT).free < missing + 2 * 1024**3:
        raise RuntimeError("Insufficient disk space with 2 GiB reserve")
    receipt = {"complete": False, "verified_files": []}
    try:
        for snapshot in manifest["snapshots"]:
            for entry in snapshot["files"]:
                receipt["verified_files"].append(download(snapshot, entry))
                RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["complete"] = True
    finally:
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print("All selected assets downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
