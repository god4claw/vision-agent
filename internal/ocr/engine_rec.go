package ocr

import (
	"image"
	"math"
	"sort"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	recHeight   = 48
	recMinWidth = 16
	recMaxWidth = 1600
	recBatch    = 8
)

// recBuckets are the fixed widths the rec input is padded up to when running on
// a GPU provider, so the tensor shape only ever takes a handful of values.
var recBuckets = []int{128, 256, 512, 1024, recMaxWidth}

// bucketShapes reports whether to use fixed-shape (bucketed width + fixed batch)
// rec batches. Enabled for GPU providers, where dynamic shapes force the
// execution provider to re-optimize the graph on every call; CPU keeps tight
// per-batch widths to avoid the wasted compute of over-padding.
func (e *Engine) bucketShapes() bool {
	switch e.provider {
	case "directml", "dml", "cuda":
		return true
	}
	return false
}

// recPadWidth rounds w up to the nearest bucket (capped at recMaxWidth) on GPU
// providers; on CPU it returns w unchanged.
func (e *Engine) recPadWidth(w int) int {
	if !e.bucketShapes() {
		return w
	}
	for _, b := range recBuckets {
		if w <= b {
			return b
		}
	}
	return recMaxWidth
}

// Recognize is a single-box convenience wrapper over RecognizeBatch.
func (e *Engine) Recognize(img image.Image, b Box) (string, float64) {
	ts, cs := e.RecognizeBatch(img, []Box{b})
	return ts[0], cs[0]
}

// cropWidth is the recognition input width for a box at the fixed height 48,
// using the rotated-rect side lengths (falling back to the bbox).
func cropWidth(b Box) int {
	w := math.Hypot(b.Corners[1].X-b.Corners[0].X, b.Corners[1].Y-b.Corners[0].Y)
	h := math.Hypot(b.Corners[3].X-b.Corners[0].X, b.Corners[3].Y-b.Corners[0].Y)
	if w <= 0 || h <= 0 {
		if b.W <= 0 || b.H <= 0 {
			return recMinWidth
		}
		w, h = float64(b.W), float64(b.H)
	}
	rw := int(math.Round(recHeight * w / h))
	if rw < recMinWidth {
		rw = recMinWidth
	}
	if rw > recMaxWidth {
		rw = recMaxWidth
	}
	return rw
}

// writeCrop samples the (possibly rotated) box region from rgba along its own
// axes via bilinear interpolation and writes the normalized CHW crop into dst
// (laid out [B,3,48,maxW]) at batch index bi, right-zero-padded to maxW.
func writeCrop(dst []float32, bi, maxW int, rgba *image.RGBA, b Box) {
	rw := cropWidth(b)
	c := b.Corners
	if math.Hypot(c[1].X-c[0].X, c[1].Y-c[0].Y) < 1 {
		// Corners unset: derive an axis-aligned rect from the bbox.
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
	plane := recHeight * maxW
	base := bi * 3 * plane
	for y := 0; y < recHeight; y++ {
		for x := 0; x < rw; x++ {
			fx := c[0].X + float64(x)*ux + float64(y)*vx
			fy := c[0].Y + float64(x)*uy + float64(y)*vy
			r, g, bl := sampleBilinear(rgba, fx, fy)
			o := y*maxW + x
			dst[base+o] = (r - 0.5) / 0.5
			dst[base+plane+o] = (g - 0.5) / 0.5
			dst[base+2*plane+o] = (bl - 0.5) / 0.5
		}
	}
}

// sampleBilinear returns the bilinearly-interpolated RGB (each in [0,1]) at the
// fractional coordinate (fx,fy), clamped to the image bounds.
func sampleBilinear(rgba *image.RGBA, fx, fy float64) (r, g, b float32) {
	bnds := rgba.Bounds()
	if fx < float64(bnds.Min.X) {
		fx = float64(bnds.Min.X)
	}
	if fy < float64(bnds.Min.Y) {
		fy = float64(bnds.Min.Y)
	}
	if mx := float64(bnds.Max.X - 1); fx > mx {
		fx = mx
	}
	if my := float64(bnds.Max.Y - 1); fy > my {
		fy = my
	}
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	x1, y1 := x0+1, y0+1
	if x1 > bnds.Max.X-1 {
		x1 = bnds.Max.X - 1
	}
	if y1 > bnds.Max.Y-1 {
		y1 = bnds.Max.Y - 1
	}
	dx := float32(fx - float64(x0))
	dy := float32(fy - float64(y0))
	get := func(x, y int) (float32, float32, float32) {
		i := rgba.PixOffset(x, y)
		return float32(rgba.Pix[i]) / 255, float32(rgba.Pix[i+1]) / 255, float32(rgba.Pix[i+2]) / 255
	}
	lerp := func(a, b, t float32) float32 { return a + (b-a)*t }
	r00, g00, b00 := get(x0, y0)
	r10, g10, b10 := get(x1, y0)
	r01, g01, b01 := get(x0, y1)
	r11, g11, b11 := get(x1, y1)
	r = lerp(lerp(r00, r10, dx), lerp(r01, r11, dx), dy)
	g = lerp(lerp(g00, g10, dx), lerp(g01, g11, dx), dy)
	b = lerp(lerp(b00, b10, dx), lerp(b01, b11, dx), dy)
	return
}

// RecognizeBatch recognizes many boxes efficiently: it sorts by width, groups
// them into batches padded to the batch max width, and runs one rec inference
// per batch instead of per box.
func (e *Engine) RecognizeBatch(img image.Image, boxes []Box) ([]string, []float64) {
	texts := make([]string, len(boxes))
	confs := make([]float64, len(boxes))
	if len(boxes) == 0 {
		return texts, confs
	}
	rgba := toRGBA(img)

	order := make([]int, len(boxes))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, c int) bool {
		return cropWidth(boxes[order[a]]) < cropWidth(boxes[order[c]])
	})

	for s := 0; s < len(order); s += recBatch {
		end := s + recBatch
		if end > len(order) {
			end = len(order)
		}
		chunk := order[s:end]
		maxW := 0
		for _, idx := range chunk {
			if w := cropWidth(boxes[idx]); w > maxW {
				maxW = w
			}
		}
		// Snap the width up to a fixed bucket and (on GPU) pad the batch to a
		// fixed size, so the rec input shape is stable across batches/frames
		// and the execution provider compiles the graph once per shape.
		maxW = e.recPadWidth(maxW)
		bsz := len(chunk)
		allocB := bsz
		if e.bucketShapes() {
			allocB = recBatch
		}
		data := make([]float32, allocB*3*recHeight*maxW) // zero == pad value
		for bi, idx := range chunk {
			writeCrop(data, bi, maxW, rgba, boxes[idx])
		}

		in, err := ort.NewTensor(ort.NewShape(int64(allocB), 3, int64(recHeight), int64(maxW)), data)
		if err != nil {
			continue
		}
		outputs := []ort.Value{nil}
		e.mu.Lock()
		err = e.rec.Run([]ort.Value{in}, outputs)
		e.mu.Unlock()
		in.Destroy()
		if err != nil {
			continue
		}
		out, ok := outputs[0].(*ort.Tensor[float32])
		if !ok {
			continue
		}
		shape := out.GetShape() // [B, T, C]
		if len(shape) < 3 {
			out.Destroy()
			continue
		}
		classes := int(shape[len(shape)-1])
		steps := int(shape[len(shape)-2])
		all := out.GetData()
		stride := steps * classes
		for bi, idx := range chunk {
			row := all[bi*stride : (bi+1)*stride]
			texts[idx], confs[idx] = e.ctcGreedyDecode(row, steps, classes)
		}
		out.Destroy()
	}
	return texts, confs
}

// ctcGreedyDecode performs greedy CTC decoding: argmax per timestep, collapse
// repeats, drop the blank (index 0), map indices to characters.
func (e *Engine) ctcGreedyDecode(probs []float32, steps, classes int) (string, float64) {
	var sb strings.Builder
	prev := -1
	var confSum float64
	var confN int
	for t := 0; t < steps; t++ {
		base := t * classes
		best := 0
		bestV := probs[base]
		for c := 1; c < classes; c++ {
			if v := probs[base+c]; v > bestV {
				bestV = v
				best = c
			}
		}
		if best != 0 && best != prev && best < len(e.chars) {
			sb.WriteString(e.chars[best])
			confSum += float64(bestV)
			confN++
		}
		prev = best
	}
	conf := 0.0
	if confN > 0 {
		conf = confSum / float64(confN)
	}
	return sb.String(), conf
}
