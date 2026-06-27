#!/usr/bin/env python3
"""Stage 2 perception sidecar (real OCR via RapidOCR / PP-OCR ONNX models).

An HTTP/JSON service that the Go agent calls each tick. It runs real OCR
(PP-OCR models on ONNX Runtime via RapidOCR) and maps detections to the
shared {text, boxes} schema. If the engine cannot initialize, it falls back
to a deterministic stub so the agent loop never hard-breaks.

Stage 3: export to ONNX consumed natively inside the Go binary; gRPC IPC.

Endpoints:
  GET  /health -> {"status": "ok", "ocr": "rapidocr"|"stub", "err": "..."}
  POST /ocr    -> {"text": "...", "boxes": [{x,y,w,h,text,conf}, ...]}
                  request body: {"image_b64": "<base64 png>"}
"""
import base64
import http.server
import json
import socketserver

HOST = "127.0.0.1"
PORT = 8089

# Try to bring up the real OCR engine (PP-OCR models on ONNX Runtime).
# If anything fails (missing models, offline first-run, import error) we fall
# back to the deterministic stub so the agent loop never hard-breaks.
try:
    import cv2
    import numpy as np
    from rapidocr import RapidOCR

    _engine = RapidOCR()
    OCR_BACKEND = "rapidocr"
    _OCR_ERR = ""
except Exception as exc:  # noqa: BLE001
    _engine = None
    OCR_BACKEND = "stub"
    _OCR_ERR = str(exc)


def _stub() -> dict:
    return {
        "text": "ATTACK target hp 80",
        "boxes": [
            {"x": 10, "y": 20, "w": 60, "h": 14, "text": "ATTACK", "conf": 0.95},
            {"x": 120, "y": 20, "w": 40, "h": 14, "text": "hp 80", "conf": 0.90},
        ],
    }


def _run_ocr(image_bytes: bytes) -> dict:
    if _engine is None or not image_bytes:
        return _stub()
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return _stub()
        res = _engine(img)
    except Exception:  # noqa: BLE001
        return _stub()

    boxes = []
    texts = []
    out_boxes = getattr(res, "boxes", None)
    out_txts = getattr(res, "txts", None)
    out_scores = getattr(res, "scores", None)
    if out_boxes is not None and out_txts is not None:
        scores = out_scores if out_scores is not None else [0.0] * len(out_txts)
        for box, txt, score in zip(out_boxes, out_txts, scores):
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
            boxes.append({
                "x": x, "y": y, "w": w, "h": h,
                "text": str(txt), "conf": float(score),
            })
            texts.append(str(txt))
    return {"text": " ".join(texts), "boxes": boxes}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "ocr": OCR_BACKEND, "err": _OCR_ERR})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/ocr":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
            img_b64 = payload.get("image_b64", "")
            image_bytes = base64.b64decode(img_b64) if img_b64 else b""
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"bad request: {exc}"})
            return
        self._send(200, _run_ocr(image_bytes))

    def log_message(self, *_args):  # silence per-request logging
        return


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer((HOST, PORT), Handler) as httpd:
        print(f"perception sidecar listening on http://{HOST}:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nperception sidecar stopped")
