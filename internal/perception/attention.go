package perception

import (
	"bytes"
	"context"
	"image"
	"image/draw"
	"image/png"
	"math"
	"sync"
)

// AttentionPerceiver is a "look only where it changed" decorator over any
// Perceiver. It splits each frame into a tile grid, finds which tiles changed
// since the previous frame, runs OCR only on the bounding region of the changed
// tiles, and merges those fresh boxes with the cached boxes from the unchanged
// part of the screen.
//
// This is transport-agnostic: the inner perceiver is fed a cropped PNG, so it
// works with the native engine and with the HTTP/gRPC sidecars alike.
type AttentionPerceiver struct {
	inner  Perceiver
	tile   int
	thresh float64

	mu       sync.Mutex
	lastSig  []float64
	gw, gh   int
	lastW    int
	lastH    int
	lastRes  Result
	hasLast  bool
	full     int
	partial  int
	skipped  int
}

func NewAttentionPerceiver(inner Perceiver, tile int, thresh float64) *AttentionPerceiver {
	if tile <= 0 {
		tile = 32
	}
	return &AttentionPerceiver{inner: inner, tile: tile, thresh: thresh}
}

// Stats reports how many frames were full OCR, partial (diff-ROI), or skipped.
func (a *AttentionPerceiver) Stats() (full, partial, skipped int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.full, a.partial, a.skipped
}

func (a *AttentionPerceiver) Perceive(ctx context.Context, framePNG []byte) (Result, error) {
	img, err := png.Decode(bytes.NewReader(framePNG))
	if err != nil {
		// Can't diff; fall back to inner on the raw frame.
		return a.inner.Perceive(ctx, framePNG)
	}
	rgba := toRGBANRGBA(img)
	b := rgba.Bounds()
	w, h := b.Dx(), b.Dy()
	sig, gw, gh := tileSignature(rgba, a.tile)

	a.mu.Lock()
	reuseValid := a.hasLast && w == a.lastW && h == a.lastH && gw == a.gw && gh == a.gh
	var lastRes Result
	var lastSig []float64
	if reuseValid {
		lastRes = a.lastRes
		lastSig = a.lastSig
	}
	a.mu.Unlock()

	if !reuseValid {
		// First frame or resolution change: full OCR.
		res, err := a.inner.Perceive(ctx, framePNG)
		if err != nil {
			return Result{}, err
		}
		a.store(sig, gw, gh, w, h, res)
		a.bump(&a.full)
		return res, nil
	}

	// Find the bounding rect of changed tiles.
	minTx, minTy := gw, gh
	maxTx, maxTy := -1, -1
	for ty := 0; ty < gh; ty++ {
		for tx := 0; tx < gw; tx++ {
			i := ty*gw + tx
			if math.Abs(sig[i]-lastSig[i]) > a.thresh {
				if tx < minTx {
					minTx = tx
				}
				if tx > maxTx {
					maxTx = tx
				}
				if ty < minTy {
					minTy = ty
				}
				if ty > maxTy {
					maxTy = ty
				}
			}
		}
	}

	if maxTx < 0 {
		// Nothing changed: reuse the entire previous result.
		a.store(sig, gw, gh, w, h, lastRes)
		a.bump(&a.skipped)
		return lastRes, nil
	}

	// Dirty rect in pixels (clamped to the frame).
	dx := minTx * a.tile
	dy := minTy * a.tile
	dx2 := min((maxTx+1)*a.tile, w)
	dy2 := min((maxTy+1)*a.tile, h)
	dw, dh := dx2-dx, dy2-dy

	// Crop the dirty region and OCR only that.
	crop := image.NewRGBA(image.Rect(0, 0, dw, dh))
	draw.Draw(crop, crop.Bounds(), rgba, image.Pt(b.Min.X+dx, b.Min.Y+dy), draw.Src)
	var buf bytes.Buffer
	if err := png.Encode(&buf, crop); err != nil {
		return Result{}, err
	}
	sub, err := a.inner.Perceive(ctx, buf.Bytes())
	if err != nil {
		return Result{}, err
	}

	// Offset new boxes into full-frame coordinates.
	newBoxes := make([]Box, 0, len(sub.Boxes))
	for _, bx := range sub.Boxes {
		bx.X += dx
		bx.Y += dy
		newBoxes = append(newBoxes, bx)
	}

	// Keep previous boxes that lie entirely outside the dirty rect.
	merged := make([]Box, 0, len(lastRes.Boxes)+len(newBoxes))
	for _, ob := range lastRes.Boxes {
		if !boxIntersectsRect(ob, dx, dy, dx2, dy2) {
			merged = append(merged, ob)
		}
	}
	merged = append(merged, newBoxes...)

	res := Result{Boxes: merged, Text: joinBoxText(merged)}
	a.store(sig, gw, gh, w, h, res)
	a.bump(&a.partial)
	return res, nil
}

func (a *AttentionPerceiver) store(sig []float64, gw, gh, w, h int, res Result) {
	a.mu.Lock()
	a.lastSig = sig
	a.gw, a.gh = gw, gh
	a.lastW, a.lastH = w, h
	a.lastRes = res
	a.hasLast = true
	a.mu.Unlock()
}

func (a *AttentionPerceiver) bump(p *int) {
	a.mu.Lock()
	*p++
	a.mu.Unlock()
}

// tileSignature returns the mean grayscale value of each tile in row-major
// order, plus the grid dimensions.
func tileSignature(img *image.RGBA, tile int) (sig []float64, gw, gh int) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	gw = (w + tile - 1) / tile
	gh = (h + tile - 1) / tile
	sig = make([]float64, gw*gh)
	counts := make([]int, gw*gh)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(b.Min.X+x, b.Min.Y+y)
			gray := 0.299*float64(img.Pix[i]) + 0.587*float64(img.Pix[i+1]) + 0.114*float64(img.Pix[i+2])
			gi := (y/tile)*gw + (x / tile)
			sig[gi] += gray
			counts[gi]++
		}
	}
	for i := range sig {
		if counts[i] > 0 {
			sig[i] /= float64(counts[i])
		}
	}
	return sig, gw, gh
}

func boxIntersectsRect(b Box, x0, y0, x1, y1 int) bool {
	return b.X < x1 && b.X+b.W > x0 && b.Y < y1 && b.Y+b.H > y0
}

func joinBoxText(boxes []Box) string {
	var sb bytes.Buffer
	for _, b := range boxes {
		if b.Text != "" {
			sb.WriteString(b.Text)
			sb.WriteByte(' ')
		}
	}
	return string(bytes.TrimSpace(sb.Bytes()))
}

// toRGBANRGBA returns an *image.RGBA copy of img (or img itself if already one).
func toRGBANRGBA(img image.Image) *image.RGBA {
	if r, ok := img.(*image.RGBA); ok {
		return r
	}
	b := img.Bounds()
	dst := image.NewRGBA(b)
	draw.Draw(dst, b, img, b.Min, draw.Src)
	return dst
}
