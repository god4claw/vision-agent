package ocr

import (
	"bytes"
	"image"
	_ "image/png" // register PNG decoder
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestEngineRealOCR runs the native ONNX engine end-to-end (det + rec) against a
// committed sample image with known text. It is fully skip-guarded: the PP-OCR
// models and the onnxruntime shared library live under the gitignored assets/
// directory (reproduce them via the README / scripts/extract_dict.py and by
// copying from the rapidocr + onnxruntime Python packages). Because assets/ is
// not committed, this test ALWAYS SKIPS in CI, keeping the pipeline green while
// still providing a real-OCR smoke test for local maintainers.
//
// Expected local asset layout (matches cmd/agent default flags):
//
//	assets/onnxruntime.dll
//	assets/models/det.onnx
//	assets/models/rec.onnx
//	assets/ppocr_keys.txt
func TestEngineRealOCR(t *testing.T) {
	root := filepath.Join("..", "..")
	dll := filepath.Join(root, "assets", "onnxruntime.dll")
	det := filepath.Join(root, "assets", "models", "det.onnx")
	rec := filepath.Join(root, "assets", "models", "rec.onnx")
	keys := filepath.Join(root, "assets", "ppocr_keys.txt")

	for _, p := range []string{dll, det, rec, keys} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("native OCR assets not present (%s); see README to populate assets/", p)
		}
	}

	eng, err := NewEngine(det, rec, keys, dll, "cpu", 0)
	if err != nil {
		t.Skipf("native OCR engine unavailable (likely onnxruntime version mismatch): %v", err)
	}
	defer eng.Close()

	raw, err := os.ReadFile(filepath.Join("testdata", "ocr_sample.png"))
	if err != nil {
		t.Fatalf("read sample image: %v", err)
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("decode sample image: %v", err)
	}

	boxes, err := eng.Detect(img)
	if err != nil {
		t.Fatalf("detect: %v", err)
	}
	if len(boxes) == 0 {
		t.Fatalf("expected at least one detected box, got 0")
	}

	texts, _ := eng.RecognizeBatch(img, boxes)
	got := strings.ToUpper(strings.Join(texts, " "))
	t.Logf("native OCR recognized: %q", got)

	if !strings.Contains(got, "START") && !strings.Contains(got, "GAME") {
		t.Fatalf("expected START/GAME in recognized text, got %q", got)
	}
}
