package perception

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"testing"
)

func makeFramePNG(t *testing.T, w, h int, fill color.RGBA, patch image.Rectangle, patchCol color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := fill
			if patch.Dx() > 0 && image.Pt(x, y).In(patch) {
				c = patchCol
			}
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode: %v", err)
	}
	return buf.Bytes()
}

// DoD: attention skips identical frames and only re-OCRs on a changed region.
func TestAttentionSkipPartialFull(t *testing.T) {
	ctx := context.Background()
	inner := &countingPerceiver{res: Result{
		Text:  "x",
		Boxes: []Box{{X: 1, Y: 1, W: 2, H: 2, Text: "x"}},
	}}
	ap := NewAttentionPerceiver(inner, 16, 2.0)

	black := color.RGBA{A: 255}
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	frame := makeFramePNG(t, 64, 64, black, image.Rectangle{}, black)

	// 1) First frame -> full OCR.
	if _, err := ap.Perceive(ctx, frame); err != nil {
		t.Fatalf("perceive1: %v", err)
	}
	// 2) Identical frame -> skipped (inner not called again).
	if _, err := ap.Perceive(ctx, frame); err != nil {
		t.Fatalf("perceive2: %v", err)
	}
	if inner.calls != 1 {
		t.Fatalf("identical frame should be skipped; inner calls=%d want 1", inner.calls)
	}
	// 3) Changed region -> partial OCR (inner called on the crop).
	changed := makeFramePNG(t, 64, 64, black, image.Rect(0, 0, 16, 16), white)
	if _, err := ap.Perceive(ctx, changed); err != nil {
		t.Fatalf("perceive3: %v", err)
	}
	if inner.calls != 2 {
		t.Fatalf("changed frame should re-OCR; inner calls=%d want 2", inner.calls)
	}

	full, partial, skipped := ap.Stats()
	if full != 1 || skipped != 1 || partial != 1 {
		t.Fatalf("want full=1 skipped=1 partial=1, got full=%d skipped=%d partial=%d", full, skipped, partial)
	}
}
