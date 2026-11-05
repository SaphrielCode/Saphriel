"""Generate tiny deterministic GGUF fixtures, not trained models."""
from array import array
from pathlib import Path
import random
import struct

out = Path(__file__).resolve().parents[1] / "build/cache-fixtures"
out.mkdir(parents=True, exist_ok=True)


def string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


for arch in ("llama", "olmo2"):
    rng = random.Random(42)
    metadata = {"general.architecture": arch, "tokenizer.ggml.model": "none",
                f"{arch}.vocab_size": 16, f"{arch}.context_length": 64,
                f"{arch}.embedding_length": 32, f"{arch}.block_count": 4,
                f"{arch}.feed_forward_length": 64, f"{arch}.attention.head_count": 4,
                f"{arch}.attention.head_count_kv": 2, f"{arch}.rope.dimension_count": 8,
                f"{arch}.attention.layer_norm_rms_epsilon": 1e-5}
    if arch == "olmo2":
        metadata[f"{arch}.attention.sliding_window"] = 16
    tensors = [("token_embd.weight", (32, 16)), ("output_norm.weight", (32,)), ("output.weight", (32, 16))]
    for i in range(4):
        for name, dims in (("attn_q", (32, 32)), ("attn_k", (32, 16)), ("attn_v", (32, 16)),
                           ("attn_output", (32, 32)), ("ffn_gate", (32, 64)),
                           ("ffn_up", (32, 64)), ("ffn_down", (64, 32))):
            tensors.append((f"blk.{i}.{name}.weight", dims))
        norms = (("attn_norm", 32), ("ffn_norm", 32)) if arch == "llama" else (
            ("attn_q_norm", 32), ("attn_k_norm", 16), ("post_attention_norm", 32), ("post_ffw_norm", 32))
        for name, size in norms:
            tensors.append((f"blk.{i}.{name}.weight", (size,)))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        kind = 8 if isinstance(value, str) else 6 if isinstance(value, float) else 4
        header += string(key) + struct.pack("<I", kind)
        header += string(value) if kind == 8 else struct.pack("<f" if kind == 6 else "<I", value)
    data = bytearray()
    for name, dims in tensors:
        while len(data) % 32:
            data.append(0)
        header += string(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        header += struct.pack("<IQ", 0, len(data))
        count = 1
        for dim in dims:
            count *= dim
        values = array("f", [1.0] * count if len(dims) == 1 else [rng.uniform(-0.1, 0.1) for _ in range(count)])
        data.extend(values.tobytes())
    header += bytes((-len(header)) % 32)
    path = out / f"tiny-{arch}.gguf"
    path.write_bytes(header + data)
    print(path)
