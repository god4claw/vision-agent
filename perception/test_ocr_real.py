#!/usr/bin/env python3
"""Real ONNX-OCR test for the perception sidecar.

This exercises the SAME OCR path the sidecar serves (`server._run_ocr`),
which decodes a PNG, runs RapidOCR (PP-OCR ONNX models on ONNX Runtime) and
returns the shared {text, boxes} schema. A clear test image is rendered
in-memory with Pillow so the test is self-contained (no fixtures, no network).

It is guarded: if RapidOCR / Pillow are unavailable, or the engine could not
initialize (offline first-run, missing models), the test skips cleanly so it
never hard-fails.

Run from the `perception/` directory:
    cd perception
    python -m unittest test_ocr_real
or via discovery:
    python -m unittest discover -s perception -p "test_*.py"
"""
import io
import unittest

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception as exc:  # noqa: BLE001
    _PIL_OK = False
    _PIL_ERR = str(exc)

import server

EXPECTED_TEXT = "START GAME 123"


def _render_png(text: str) -> bytes:
    """Render `text` in a large legible font on a white background -> PNG bytes."""
    img = Image.new("RGB", (640, 160), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            font = ImageFont.truetype(name, 64)
            break
        except Exception:  # noqa: BLE001
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((30, 45), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class RealOCRTest(unittest.TestCase):
    def test_run_ocr_reads_rendered_text(self):
        if not _PIL_OK:
            self.skipTest(f"Pillow unavailable: {_PIL_ERR}")
        if server.OCR_BACKEND != "rapidocr" or server._engine is None:
            self.skipTest(f"real OCR engine unavailable: {server._OCR_ERR}")

        png = _render_png(EXPECTED_TEXT)
        result = server._run_ocr(png)

        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("boxes", result)

        # Guard against silently hitting the deterministic stub fallback.
        stub_text = server._stub()["text"]
        self.assertNotEqual(
            result["text"], stub_text,
            "got the stub fallback instead of a real OCR result",
        )

        recognized = result["text"].upper()
        # Real models make occasional minor errors; assert strong substrings.
        self.assertTrue(
            "START" in recognized or "GAME" in recognized,
            f"expected START/GAME in recognized text, got: {result['text']!r}",
        )
        self.assertGreater(
            len(result["boxes"]), 0, "expected at least one detected box"
        )
        # Surface the recognized text when run with -v.
        print(f"\n[real-ocr] recognized: {result['text']!r}")


if __name__ == "__main__":
    unittest.main()
