"""Inspect local checkpoint headers, dataset metadata and bounded sample records."""
import hashlib
import json
import math
from pathlib import Path
import struct

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1] / "training-assets"
receipt = json.loads((ROOT / "download-receipt.json").read_text())
if not receipt["complete"]:
    raise RuntimeError("Download is incomplete")
summary = {"checkpoint_headers": [], "parquet_files": [], "samples": []}
for path in sorted((ROOT / "data/checkpoints").glob("*/model.safetensors")):
    with path.open("rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise RuntimeError("Unexpected safetensors header size")
        header = json.loads(source.read(length))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    payload = path.stat().st_size - 8 - length
    intervals = []
    for tensor in tensors:
        start, end = tensor["data_offsets"]
        if not 0 <= start <= end <= payload:
            raise RuntimeError("Tensor offset outside checkpoint")
        intervals.append((start, end))
    intervals.sort()
    if intervals[0][0] != 0 or intervals[-1][1] != payload or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
        raise RuntimeError("Checkpoint tensor payload is not contiguous")
    summary["checkpoint_headers"].append({"step": path.parent.name, "tensors": len(tensors),
        "stored_parameters": sum(math.prod(t["shape"]) for t in tensors),
        "dtypes": sorted({t["dtype"] for t in tensors})})
for path in sorted((ROOT / "data/datasets").rglob("*.parquet")):
    parquet = pq.ParquetFile(path)
    first = next(parquet.iter_batches(batch_size=1), None)
    if first is None:
        raise RuntimeError("Empty dataset shard")
    summary["parquet_files"].append({"file": str(path.relative_to(ROOT)),
        "rows": parquet.metadata.num_rows, "columns": parquet.schema_arrow.names})
for sample in json.loads((ROOT / "sample-receipt.json").read_text()):
    if "path" not in sample:
        continue
    path = ROOT / sample["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
        raise RuntimeError("Sample hash mismatch")
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if len(rows) != sample["records"] or any(not r.get("text") for r in rows):
        raise RuntimeError("Sample count/text mismatch")
    summary["samples"].append({"repo": sample["repo"], "records": len(rows)})
if len(summary["checkpoint_headers"]) != 8:
    raise RuntimeError("Expected all eight published checkpoints")
summary["verified_download_bytes"] = sum(f["size_bytes"] for f in receipt["verified_files"])
summary["verified_download_files"] = len(receipt["verified_files"])
(ROOT / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
