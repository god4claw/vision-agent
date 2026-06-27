package perception

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/png"
	"math"
	"net/http"
	"sync"
	"time"
)

// Box is a single detected element with its location on the screen matrix.
type Box struct {
	X    int     `json:"x"`
	Y    int     `json:"y"`
	W    int     `json:"w"`
	H    int     `json:"h"`
	Text string  `json:"text"`
	Conf float64 `json:"conf"`
}

// Result is the structured output of the perception layer for one frame.
type Result struct {
	Text  string `json:"text"`
	Boxes []Box  `json:"boxes"`
}

// Perceiver turns a raw frame into structured understanding.
type Perceiver interface {
	Perceive(ctx context.Context, framePNG []byte) (Result, error)
}

// HTTPPerceiver talks to the Python perception sidecar over HTTP/JSON.
type HTTPPerceiver struct {
	URL    string
	client *http.Client
}

func NewHTTPPerceiver(url string) *HTTPPerceiver {
	return &HTTPPerceiver{URL: url, client: &http.Client{Timeout: 5 * time.Second}}
}

// Health checks that the sidecar is reachable and ready.
func (p *HTTPPerceiver) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.URL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("perception health status %d", resp.StatusCode)
	}
	return nil
}

func (p *HTTPPerceiver) Perceive(ctx context.Context, framePNG []byte) (Result, error) {
	payload, err := json.Marshal(map[string]string{
		"image_b64": base64.StdEncoding.EncodeToString(framePNG),
	})
	if err != nil {
		return Result{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.URL+"/ocr", bytes.NewReader(payload))
	if err != nil {
		return Result{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return Result{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Result{}, fmt.Errorf("perception ocr status %d", resp.StatusCode)
	}
	var r Result
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return Result{}, err
	}
	return r, nil
}

// FakePerceiver is a deterministic perceiver for tests (no network).
type FakePerceiver struct {
	Fixed Result
	Err   error
}

func (f FakePerceiver) Perceive(_ context.Context, _ []byte) (Result, error) {
	return f.Fixed, f.Err
}

// CachingPerceiver wraps another Perceiver and skips the (expensive) OCR call
// when the frame is essentially unchanged from the previous one. This is the
// biggest latency win for a non-terminating loop on a mostly-static screen.
type CachingPerceiver struct {
	inner     Perceiver
	threshold float64

	mu      sync.Mutex
	lastSig []float64
	lastRes Result
	hasLast bool
	hits    int
	misses  int
}

// NewCachingPerceiver wraps inner. threshold is the max mean absolute
// difference (0..255 per cell of a 16x16 grayscale signature) under which two
// frames are considered "the same". Larger threshold = more aggressive reuse.
func NewCachingPerceiver(inner Perceiver, threshold float64) *CachingPerceiver {
	return &CachingPerceiver{inner: inner, threshold: threshold}
}

func (c *CachingPerceiver) Perceive(ctx context.Context, framePNG []byte) (Result, error) {
	sig, sigErr := frameSignature(framePNG)

	c.mu.Lock()
	if sigErr == nil && c.hasLast && meanAbsDiff(sig, c.lastSig) <= c.threshold {
		res := c.lastRes
		c.hits++
		c.mu.Unlock()
		return res, nil
	}
	c.mu.Unlock()

	res, err := c.inner.Perceive(ctx, framePNG)
	if err != nil {
		return Result{}, err
	}

	c.mu.Lock()
	c.misses++
	if sigErr == nil {
		c.lastSig = sig
		c.lastRes = res
		c.hasLast = true
	}
	c.mu.Unlock()
	return res, nil
}

// Stats returns cache hits and misses so far.
func (c *CachingPerceiver) Stats() (hits, misses int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses
}

const sigGrid = 16

// frameSignature decodes a PNG frame and reduces it to a 16x16 grayscale
// signature for cheap whole-frame change detection.
func frameSignature(framePNG []byte) ([]float64, error) {
	img, err := png.Decode(bytes.NewReader(framePNG))
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	sig := make([]float64, sigGrid*sigGrid)
	if w == 0 || h == 0 {
		return sig, nil
	}
	for gy := 0; gy < sigGrid; gy++ {
		for gx := 0; gx < sigGrid; gx++ {
			sx := b.Min.X + gx*w/sigGrid
			sy := b.Min.Y + gy*h/sigGrid
			r, g, bl, _ := img.At(sx, sy).RGBA()
			// RGBA returns 16-bit channels; scale to 0..255.
			gray := (0.299*float64(r) + 0.587*float64(g) + 0.114*float64(bl)) / 257.0
			sig[gy*sigGrid+gx] = gray
		}
	}
	return sig, nil
}

func meanAbsDiff(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return math.MaxFloat64
	}
	var sum float64
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum / float64(len(a))
}
