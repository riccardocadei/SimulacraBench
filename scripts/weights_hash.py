"""Content hash of a safetensors file (sorted tensor names, dtypes, shapes, raw bytes; header layout ignored), so a
rebuilt fp16 checkpoint can be checked against the shipped one even if the safetensors writer lays out bytes differently.
Usage: python scripts/weights_hash.py <file.safetensors>"""
import hashlib
import sys

from safetensors.torch import load_file

h = hashlib.sha256()
for k, v in sorted(load_file(sys.argv[1]).items()):
    h.update(f"{k}|{v.dtype}|{tuple(v.shape)}|".encode())
    h.update(v.contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes())
print(h.hexdigest())
