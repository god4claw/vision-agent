"""Extract the PP-OCR recognition character dictionary from the ONNX model.

The PP-OCRv6 rec model embeds its char list in ONNX metadata under the key
'character' (newline-separated). The Go native OCR engine needs this as a file.

Usage:
    python scripts/extract_dict.py
Writes assets/ppocr_keys.txt (one character per line, in model index order).
"""
import os

import onnxruntime as ort

REC_MODEL = os.path.join(
    os.path.dirname(ort.__file__), "..", "rapidocr", "models",
    "PP-OCRv6_rec_small.onnx",
)
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "ppocr_keys.txt")

sess = ort.InferenceSession(REC_MODEL, providers=["CPUExecutionProvider"])
chars = sess.get_modelmeta().custom_metadata_map["character"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(chars)

n = len([ln for ln in chars.split("\n")])
print(f"wrote {OUT} ({n} entries, {len(chars)} bytes)")
