package ocr

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Collector is the data-collection half of the Stage 3 flywheel: it persists
// low-confidence recognition crops plus metadata so they can later be reviewed,
// labeled and fed into an offline fine-tune.
type Collector struct {
	dir    string
	thresh float64

	mu    sync.Mutex
	jsonl *os.File
	n     int
}

type sampleRecord struct {
	Time string  `json:"time"`
	File string  `json:"file"`
	Text string  `json:"text"`
	Conf float64 `json:"conf"`
	X    int     `json:"x"`
	Y    int     `json:"y"`
	W    int     `json:"w"`
	H    int     `json:"h"`
}

// NewCollector opens (creating if needed) dir and its samples.jsonl manifest.
func NewCollector(dir string, thresh float64) (*Collector, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(dir, "samples.jsonl"),
		os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &Collector{dir: dir, thresh: thresh, jsonl: f}, nil
}

func (c *Collector) Close() error {
	if c.jsonl != nil {
		return c.jsonl.Close()
	}
	return nil
}

// Add persists the crop + record when conf is below the collection threshold.
func (c *Collector) Add(crop image.Image, box Box, text string, conf float64) {
	if conf >= c.thresh {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	name := fmt.Sprintf("crop_%d_%04d.png", time.Now().UnixNano(), c.n)
	c.n++
	f, err := os.Create(filepath.Join(c.dir, name))
	if err != nil {
		return
	}
	_ = png.Encode(f, crop)
	_ = f.Close()

	rec := sampleRecord{
		Time: time.Now().Format(time.RFC3339Nano),
		File: name, Text: text, Conf: conf,
		X: box.X, Y: box.Y, W: box.W, H: box.H,
	}
	if b, err := json.Marshal(rec); err == nil {
		_, _ = c.jsonl.Write(append(b, '\n'))
	}
}

// CropImage extracts the (deskewed) box region as an image for dataset storage.
func (e *Engine) CropImage(img image.Image, b Box) *image.RGBA {
	rgba := toRGBA(img)
	rw := cropWidth(b)
	c := b.Corners
	if math.Hypot(c[1].X-c[0].X, c[1].Y-c[0].Y) < 1 {
		c = [4]pt{
			{float64(b.X), float64(b.Y)},
			{float64(b.X + b.W), float64(b.Y)},
			{float64(b.X + b.W), float64(b.Y + b.H)},
			{float64(b.X), float64(b.Y + b.H)},
		}
	}
	ux := (c[1].X - c[0].X) / float64(rw)
	uy := (c[1].Y - c[0].Y) / float64(rw)
	vx := (c[3].X - c[0].X) / float64(recHeight)
	vy := (c[3].Y - c[0].Y) / float64(recHeight)
	dst := image.NewRGBA(image.Rect(0, 0, rw, recHeight))
	for y := 0; y < recHeight; y++ {
		for x := 0; x < rw; x++ {
			fx := c[0].X + float64(x)*ux + float64(y)*vx
			fy := c[0].Y + float64(x)*uy + float64(y)*vy
			r, g, bl := sampleBilinear(rgba, fx, fy)
			o := dst.PixOffset(x, y)
			dst.Pix[o] = uint8(r * 255)
			dst.Pix[o+1] = uint8(g * 255)
			dst.Pix[o+2] = uint8(bl * 255)
			dst.Pix[o+3] = 255
		}
	}
	return dst
}
