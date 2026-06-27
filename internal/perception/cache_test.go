package perception

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"testing"
)

type countingPerceiver struct {
	calls int
	res   Result
}

func (c *countingPerceiver) Perceive(_ context.Context, _ []byte) (Result, error) {
	c.calls++
	return c.res, nil
}

func solidPNG(t *testing.T, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode: %v", err)
	}
	return buf.Bytes()
}

func TestCachingPerceiverSkipsUnchangedFrame(t *testing.T) {
	ctx := context.Background()
	inner := &countingPerceiver{res: Result{Text: "x"}}
	cp := NewCachingPerceiver(inner, 1.5)

	black := solidPNG(t, color.RGBA{A: 255})

	// First frame: miss -> inner called.
	if _, err := cp.Perceive(ctx, black); err != nil {
		t.Fatalf("perceive1: %v", err)
	}
	// Identical frame: hit -> inner NOT called again.
	if _, err := cp.Perceive(ctx, black); err != nil {
		t.Fatalf("perceive2: %v", err)
	}
	if inner.calls != 1 {
		t.Fatalf("want inner called once for identical frames, got %d", inner.calls)
	}
	hits, misses := cp.Stats()
	if hits != 1 || misses != 1 {
		t.Fatalf("want hits=1 misses=1, got hits=%d misses=%d", hits, misses)
	}

	// Very different frame: miss -> inner called again.
	white := solidPNG(t, color.RGBA{R: 255, G: 255, B: 255, A: 255})
	if _, err := cp.Perceive(ctx, white); err != nil {
		t.Fatalf("perceive3: %v", err)
	}
	if inner.calls != 2 {
		t.Fatalf("want inner called twice after a changed frame, got %d", inner.calls)
	}
}
