"""Extract the existing GGUF chat template and guard an absent tools variable."""
import json
from pathlib import Path
import struct

root = Path(__file__).resolve().parents[1]
model = json.loads((root / "models/model.json").read_text(encoding="utf-8-sig"))
formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
with (root / "models" / model["filename"]).open("rb") as f:
    def unpack(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

    def text():
        return f.read(unpack("Q")).decode("utf-8")

    def value(kind, retain=False):
        if kind == 8:
            length = unpack("Q")
            if retain:
                return f.read(length).decode("utf-8")
            f.seek(length, 1)
        elif kind == 9:
            subtype, count = unpack("I"), unpack("Q")
            for _ in range(count):
                value(subtype)
        else:
            size = struct.calcsize("<" + formats[kind])
            f.seek(size, 1)

    if f.read(4) != b"GGUF" or unpack("I") != 3:
        raise RuntimeError("Expected GGUF v3")
    tensors, count = unpack("Q"), unpack("Q")
    template = None
    for _ in range(count):
        name, kind = text(), unpack("I")
        found = value(kind, name == "tokenizer.chat_template")
        if name == "tokenizer.chat_template":
            template = found
    if not isinstance(template, str):
        raise RuntimeError("GGUF chat template not found")
guard = "{% if tools is not defined %}{% set tools = none %}{% endif %}"
target = root / "templates/olmo3.1-chat.jinja"
target.parent.mkdir(exist_ok=True)
target.write_text(guard + template, encoding="utf-8")
assert target.read_text(encoding="utf-8")[len(guard):] == template
print("Preserved original GGUF template and added missing-tools guard:", target)
