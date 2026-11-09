"""Capture 1,000 complete source-prefix records with bounded remote range reads."""
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import pyarrow.parquet as pq
import zstandard

ROOT = Path(__file__).resolve().parents[1] / "training-assets"
LIMIT = 1000
BYTE_LIMIT = 25 * 1024 * 1024
NETWORK_LIMIT = 256 * 1024 * 1024


class RemoteFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid remote seek")
        return self.position

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if self.transferred + count > NETWORK_LIMIT:
            raise RuntimeError("256 MiB remote-read bound reached")
        if count == 0:
            return b""
        start, end = self.position, self.position + count - 1
        for attempt in range(3):
            request = urllib.request.Request(self.url + f"?download=true&range={start}-{end}",
                headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                        raise RuntimeError("Unexpected source range response")
                    data = response.read(count + 1)
                if len(data) != count:
                    raise RuntimeError("Truncated source read")
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        self.position += count
        self.transferred += count
        return data


def main():
    output = ROOT / "data/pretraining-samples"
    output.mkdir(parents=True, exist_ok=True)
    catalogue = json.loads((ROOT / "pretraining-sources.json").read_text())
    receipts = []
    for source in catalogue:
        repo, revision = source["repo"], source["revision"]
        if source["gated"]:
            receipts.append({"repo": repo, "status": "not_downloaded_requires_user_access_approval"})
            continue
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true", timeout=60) as response:
            info = json.load(response)
        extension = ".parquet" if "fineweb" in repo else ".jsonl.zst"
        entry = next(f for f in info["siblings"] if f["rfilename"].endswith(extension))
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(entry['rfilename'])}"
        remote = RemoteFile(url, entry["size"])
        name = repo.split("/")[-1]
        target = output / (name + ".jsonl")
        count, byte_count = 0, 0
        with remote, target.open("wb") as destination:
            if extension == ".parquet":
                parquet = pq.ParquetFile(remote)
                def records():
                    for batch in parquet.iter_batches(batch_size=1000):
                        yield from batch.to_pylist()
                iterator = records()
            else:
                stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(remote), encoding="utf-8")
                def records():
                    while line := stream.readline(BYTE_LIMIT + 1):
                        if len(line.encode()) > BYTE_LIMIT:
                            raise RuntimeError("Record exceeds sample byte bound")
                        yield json.loads(line)
                iterator = records()
            for row in iterator:
                if not isinstance(row.get("text"), str) or not row["text"]:
                    raise RuntimeError("Expected complete nonempty text")
                data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                if count >= LIMIT or byte_count + len(data) > BYTE_LIMIT:
                    break
                destination.write(data)
                count += 1
                byte_count += len(data)
                if count == LIMIT:
                    break
        receipts.append({"repo": repo, "revision": revision, "source_file": entry["rfilename"],
                         "source_file_size": entry["size"], "published_source_sha256": entry.get("lfs", {}).get("sha256"),
                         "path": str(target.relative_to(ROOT)), "records": count, "bytes": byte_count,
                         "remote_bytes_read": remote.transferred,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "selection": "First complete records of first source shard; at most 1000 records/25 MiB. Not random or representative.",
                         "limits": "Public-source sample, not verified as the exact filtered documents used in SmolLM2. Partial range reads cannot verify the full source-file digest; local sample digest recorded."})
        (ROOT / "sample-receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(name, count, "records", byte_count, "bytes", flush=True)
    (ROOT / "sample-receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
