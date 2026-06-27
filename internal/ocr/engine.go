// Package ocr runs PP-OCR (PP-OCRv6) ONNX models natively in Go via
// onnxruntime_go, removing the Python perception sidecar.
//
// Detection uses a DBNet model (probability map -> binarize -> connected
// components -> boxes). Recognition uses a CRNN/CTC model with the character
// dictionary embedded in (and extracted from) the model metadata.
package ocr

import (
	"fmt"
	"image"
	"image/draw"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// Box is an axis-aligned detected text region in original-image coordinates.
type Box struct {
	X, Y, W, H int
	Score      float64
	Text       string
	Conf       float64
	Corners    [4]pt // rotated min-area rect (TL,TR,BR,BL) in image coords
}

// Engine holds the ONNX sessions and decoding assets.
type Engine struct {
	mu  sync.Mutex // serialize Run calls
	det *ort.DynamicAdvancedSession
	rec *ort.DynamicAdvancedSession

	recPath  string    // for hot-reload
	recMod   time.Time // last-seen rec model mtime
	provider string    // execution provider: cpu|directml|cuda
	deviceID int       // GPU device index

	chars []string // CTC index -> string; index 0 is blank

	// Detection tuning.
	DetThresh   float64
	BoxThresh   float64
	UnclipRatio float64
	MaxSide     int
	MinBox      int
}

var (
	initOnce sync.Once
	initErr  error
)

// InitRuntime initializes the global onnxruntime environment exactly once,
// loading the shared library from dllPath (if non-empty).
func InitRuntime(dllPath string) error {
	initOnce.Do(func() {
		if dllPath != "" {
			ort.SetSharedLibraryPath(dllPath)
		}
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

// NewEngine loads the detection + recognition models and the char dictionary.
// provider selects the execution backend: "cpu" (default), "directml" (any
// Windows GPU), or "cuda" (NVIDIA). deviceID picks the GPU index.
func NewEngine(detPath, recPath, keysPath, dllPath, provider string, deviceID int) (*Engine, error) {
	if err := InitRuntime(dllPath); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}
	opts, err := buildSessionOptions(provider, deviceID)
	if err != nil {
		return nil, err
	}
	if opts != nil {
		defer opts.Destroy()
	}
	det, err := ort.NewDynamicAdvancedSession(detPath, []string{"x"}, []string{"fetch_name_0"}, opts)
	if err != nil {
		return nil, fmt.Errorf("load det model: %w", err)
	}
	rec, err := ort.NewDynamicAdvancedSession(recPath, []string{"x"}, []string{"fetch_name_0"}, opts)
	if err != nil {
		det.Destroy()
		return nil, fmt.Errorf("load rec model: %w", err)
	}
	chars, err := loadChars(keysPath)
	if err != nil {
		det.Destroy()
		rec.Destroy()
		return nil, fmt.Errorf("load char dict: %w", err)
	}
	e := &Engine{
		det:         det,
		rec:         rec,
		recPath:     recPath,
		provider:    provider,
		deviceID:    deviceID,
		chars:       chars,
		DetThresh:   0.3,
		BoxThresh:   0.5,
		UnclipRatio: 1.6,
		MaxSide:     960,
		MinBox:      3,
	}
	if fi, err := os.Stat(recPath); err == nil {
		e.recMod = fi.ModTime()
	}
	return e, nil
}

// buildSessionOptions constructs ORT session options for the chosen execution
// provider. Returns (nil, nil) for CPU so the caller passes nil (ORT default).
func buildSessionOptions(provider string, deviceID int) (*ort.SessionOptions, error) {
	switch provider {
	case "", "cpu":
		return nil, nil
	case "directml", "dml":
		o, err := ort.NewSessionOptions()
		if err != nil {
			return nil, err
		}
		// DirectML requires sequential execution and disabled memory pattern.
		if err := o.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
			o.Destroy()
			return nil, err
		}
		if err := o.SetMemPattern(false); err != nil {
			o.Destroy()
			return nil, err
		}
		if err := o.AppendExecutionProviderDirectML(deviceID); err != nil {
			o.Destroy()
			return nil, fmt.Errorf("enable DirectML (need a DirectML onnxruntime.dll): %w", err)
		}
		return o, nil
	case "cuda":
		o, err := ort.NewSessionOptions()
		if err != nil {
			return nil, err
		}
		cu, err := ort.NewCUDAProviderOptions()
		if err != nil {
			o.Destroy()
			return nil, err
		}
		defer cu.Destroy()
		if err := cu.Update(map[string]string{"device_id": fmt.Sprintf("%d", deviceID)}); err != nil {
			o.Destroy()
			return nil, err
		}
		if err := o.AppendExecutionProviderCUDA(cu); err != nil {
			o.Destroy()
			return nil, fmt.Errorf("enable CUDA (need a CUDA onnxruntime build + CUDA/cuDNN): %w", err)
		}
		return o, nil
	default:
		return nil, fmt.Errorf("unknown execution provider %q (want cpu|directml|cuda)", provider)
	}
}

// ReloadRecIfChanged hot-swaps the recognition model if its file mtime advanced
// since it was last loaded. This is the runtime half of the data flywheel: a
// retrained model dropped in place is picked up without restarting the agent.
func (e *Engine) ReloadRecIfChanged() (bool, error) {
	fi, err := os.Stat(e.recPath)
	if err != nil {
		return false, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if !fi.ModTime().After(e.recMod) {
		return false, nil
	}
	sess, err := ort.NewDynamicAdvancedSession(e.recPath, []string{"x"}, []string{"fetch_name_0"}, nil)
	if err != nil {
		return false, fmt.Errorf("reload rec model: %w", err)
	}
	old := e.rec
	e.rec = sess
	e.recMod = fi.ModTime()
	old.Destroy()
	return true, nil
}

func (e *Engine) Close() error {
	if e.det != nil {
		e.det.Destroy()
	}
	if e.rec != nil {
		e.rec.Destroy()
	}
	return nil
}

// loadChars builds the CTC class list: [blank] + dict + [space].
func loadChars(keysPath string) ([]string, error) {
	raw, err := os.ReadFile(keysPath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
	// Drop a single trailing empty entry from a final newline.
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	chars := make([]string, 0, len(lines)+2)
	chars = append(chars, "") // index 0 = CTC blank
	chars = append(chars, lines...)
	chars = append(chars, " ") // trailing space class
	return chars, nil
}

// toRGBA returns an *image.RGBA view of img (copying if necessary).
func toRGBA(img image.Image) *image.RGBA {
	if r, ok := img.(*image.RGBA); ok {
		return r
	}
	b := img.Bounds()
	dst := image.NewRGBA(b)
	draw.Draw(dst, b, img, b.Min, draw.Src)
	return dst
}

// detResize computes the DBNet input size: each side a multiple of 32, longest
// side capped at MaxSide.
func (e *Engine) detResize(w, h int) (rw, rh int) {
	ratio := 1.0
	if m := max(w, h); m > e.MaxSide {
		ratio = float64(e.MaxSide) / float64(m)
	}
	rw = int(math.Round(float64(w)*ratio/32.0)) * 32
	rh = int(math.Round(float64(h)*ratio/32.0)) * 32
	if rw < 32 {
		rw = 32
	}
	if rh < 32 {
		rh = 32
	}
	return rw, rh
}

var (
	detMean = [3]float32{0.485, 0.456, 0.406}
	detStd  = [3]float32{0.229, 0.224, 0.225}
)

// Detect runs detection and returns axis-aligned boxes in original coordinates.
func (e *Engine) Detect(img image.Image) ([]Box, error) {
	rgba := toRGBA(img)
	b := rgba.Bounds()
	ow, oh := b.Dx(), b.Dy()
	if ow == 0 || oh == 0 {
		return nil, nil
	}
	rw, rh := e.detResize(ow, oh)

	data := make([]float32, 3*rh*rw)
	plane := rh * rw
	for y := 0; y < rh; y++ {
		sy := y * oh / rh
		for x := 0; x < rw; x++ {
			sx := x * ow / rw
			i := rgba.PixOffset(b.Min.X+sx, b.Min.Y+sy)
			r := float32(rgba.Pix[i]) / 255.0
			g := float32(rgba.Pix[i+1]) / 255.0
			bl := float32(rgba.Pix[i+2]) / 255.0
			o := y*rw + x
			data[o] = (r - detMean[0]) / detStd[0]
			data[plane+o] = (g - detMean[1]) / detStd[1]
			data[2*plane+o] = (bl - detMean[2]) / detStd[2]
		}
	}

	in, err := ort.NewTensor(ort.NewShape(1, 3, int64(rh), int64(rw)), data)
	if err != nil {
		return nil, fmt.Errorf("det input tensor: %w", err)
	}
	defer in.Destroy()

	outputs := []ort.Value{nil}
	e.mu.Lock()
	err = e.det.Run([]ort.Value{in}, outputs)
	e.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("det run: %w", err)
	}
	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected det output type")
	}
	defer out.Destroy()

	shape := out.GetShape() // [1,1,H,W]
	ph := int(shape[len(shape)-2])
	pw := int(shape[len(shape)-1])
	prob := out.GetData()

	boxes := e.boxesFromProbMap(prob, pw, ph)

	// Scale boxes (corners + bbox) from prob-map space back to original image.
	sx := float64(ow) / float64(pw)
	sy := float64(oh) / float64(ph)
	for i := range boxes {
		for j := range boxes[i].Corners {
			boxes[i].Corners[j].X *= sx
			boxes[i].Corners[j].Y *= sy
		}
		minBX, minBY, maxBX, maxBY := cornersBBox(boxes[i].Corners)
		boxes[i].X = int(math.Round(minBX))
		boxes[i].Y = int(math.Round(minBY))
		boxes[i].W = int(math.Round(maxBX - minBX))
		boxes[i].H = int(math.Round(maxBY - minBY))
		boxes[i] = clampBox(boxes[i], ow, oh)
	}
	return boxes, nil
}

// boxesFromProbMap binarizes the probability map and extracts boxes via
// 4-connected component labeling, scoring + unclip expansion.
func (e *Engine) boxesFromProbMap(prob []float32, w, h int) []Box {
	thr := float32(e.DetThresh)
	mask := make([]bool, w*h)
	for i, p := range prob {
		mask[i] = p > thr
	}
	visited := make([]bool, w*h)
	stack := make([]int, 0, 1024)
	var pts []pt
	var boxes []Box

	for start := 0; start < w*h; start++ {
		if !mask[start] || visited[start] {
			continue
		}
		minX, minY := w, h
		maxX, maxY := 0, 0
		var sum float64
		count := 0
		pts = pts[:0]

		stack = stack[:0]
		stack = append(stack, start)
		visited[start] = true
		for len(stack) > 0 {
			idx := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			cx := idx % w
			cy := idx / w
			if cx < minX {
				minX = cx
			}
			if cx > maxX {
				maxX = cx
			}
			if cy < minY {
				minY = cy
			}
			if cy > maxY {
				maxY = cy
			}
			sum += float64(prob[idx])
			count++

			left := cx > 0 && mask[idx-1]
			right := cx < w-1 && mask[idx+1]
			up := cy > 0 && mask[idx-w]
			down := cy < h-1 && mask[idx+w]
			// Keep only boundary pixels for the hull (much smaller set).
			if !left || !right || !up || !down {
				pts = append(pts, pt{float64(cx), float64(cy)})
			}
			if left && !visited[idx-1] {
				visited[idx-1] = true
				stack = append(stack, idx-1)
			}
			if right && !visited[idx+1] {
				visited[idx+1] = true
				stack = append(stack, idx+1)
			}
			if up && !visited[idx-w] {
				visited[idx-w] = true
				stack = append(stack, idx-w)
			}
			if down && !visited[idx+w] {
				visited[idx+w] = true
				stack = append(stack, idx+w)
			}
		}

		if (maxX-minX+1) < e.MinBox || (maxY-minY+1) < e.MinBox {
			continue
		}
		score := sum / float64(count)
		if score < e.BoxThresh {
			continue
		}

		// Rotated min-area rectangle from the component boundary, then unclip.
		corners := minAreaRect(convexHull(pts))
		rw := math.Hypot(corners[1].X-corners[0].X, corners[1].Y-corners[0].Y)
		rh := math.Hypot(corners[3].X-corners[0].X, corners[3].Y-corners[0].Y)
		if perim := 2.0 * (rw + rh); perim > 1e-6 {
			corners = unclipCorners(corners, rw*rh*e.UnclipRatio/perim)
		}

		minBX, minBY, maxBX, maxBY := cornersBBox(corners)
		bx := clampInt(int(math.Round(minBX)), 0, w-1)
		by := clampInt(int(math.Round(minBY)), 0, h-1)
		bxm := clampInt(int(math.Round(maxBX)), 0, w-1)
		bym := clampInt(int(math.Round(maxBY)), 0, h-1)

		boxes = append(boxes, Box{
			X:       bx,
			Y:       by,
			W:       bxm - bx + 1,
			H:       bym - by + 1,
			Score:   score,
			Corners: corners,
		})
	}
	return boxes
}

func clampBox(b Box, w, h int) Box {
	if b.X < 0 {
		b.X = 0
	}
	if b.Y < 0 {
		b.Y = 0
	}
	if b.X+b.W > w {
		b.W = w - b.X
	}
	if b.Y+b.H > h {
		b.H = h - b.Y
	}
	return b
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
