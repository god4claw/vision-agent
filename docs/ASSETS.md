# Native OCR assets (`assets/`)

The fully native, no-Python transport (`-transport native`) runs the PP-OCRv6
detection + recognition ONNX models in-process via `onnxruntime_go` (CGo). The
required binaries are **large and gitignored** (`/assets/` in `.gitignore`), so
they are not committed. This document explains how to reproduce them locally.

## What lives under `assets/`

```
assets/onnxruntime.dll                 onnxruntime shared library
assets/onnxruntime_providers_shared.dll
assets/models/det.onnx                 PP-OCRv6 text detection
assets/models/rec.onnx                 PP-OCRv6 text recognition
assets/models/cls.onnx                 PP-OCRv6 angle classification
assets/ppocr_keys.txt                  recognition char dictionary (CTC classes)
```

These default paths are configurable via the agent flags `-ort-dll`,
`-det-model`, `-rec-model`, `-rec-keys` (see the README "Run fully native"
section).

## How to obtain them

All of the assets come from the same Python packages the perception sidecar
already uses (`perception/requirements.txt`):

```
rapidocr>=2.0.0
onnxruntime>=1.17.0
```

Install them into a virtual environment:

```powershell
pip install -r perception/requirements.txt
```

### ONNX models (`det` / `rec` / `cls`)

`rapidocr` ships the PP-OCRv6 ONNX models inside its package, under
`<site-packages>/rapidocr/models/`. Copy the detection, recognition, and angle
classification models into `assets/models/` as `det.onnx`, `rec.onnx`, and
`cls.onnx` respectively. The exact source filenames (e.g.
`PP-OCRv6_rec_small.onnx`) are the ones referenced by
[`scripts/extract_dict.py`](../scripts/extract_dict.py).

### onnxruntime shared library

`onnxruntime.dll` (and `onnxruntime_providers_shared.dll`) ship inside the
`onnxruntime` Python package, under `<site-packages>/onnxruntime/capi/`. Copy
them into `assets/`. On non-Windows hosts the equivalent shared library is
`libonnxruntime.so` / `.dylib`; point `-ort-dll` at it.

### Recognition dictionary (`ppocr_keys.txt`)

The PP-OCRv6 rec model embeds its character list in ONNX metadata under the key
`character`. Regenerate the dictionary file directly from the model with:

```powershell
python scripts/extract_dict.py
```

This reads the rec model from the installed `rapidocr` package and writes
`assets/ppocr_keys.txt` (one entry per line, in model index order, LF-newline).

## Notes

- The native transport requires a CGo toolchain (`CGO_ENABLED=1`, Mingw-w64
  gcc on Windows) at build time.
- Containerized builds disable CGo and use the HTTP/gRPC sidecar instead, so the
  Docker images do **not** include these assets (see `Dockerfile` and
  `docker-compose.yml`).
- For GPU execution providers (DirectML / CUDA) you may need a provider-specific
  onnxruntime build; see the README "Decision policy" example using
  `assets/dml/onnxruntime.dll`.
