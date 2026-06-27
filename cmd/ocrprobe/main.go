// Command ocrprobe captures one screen frame and prints what the native ONNX
// OCR engine detects + recognizes. Diagnostic tool for the Stage 4 port.
//
// Run (cgo + gcc + onnxruntime.dll required):
//
//	go run ./cmd/ocrprobe
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"sort"
	"time"

	"visionagent/internal/capture"
	"visionagent/internal/ocr"
)

func main() {
	var (
		ortDLL   = flag.String("ort-dll", "assets/onnxruntime.dll", "onnxruntime shared library")
		detModel = flag.String("det-model", "assets/models/det.onnx", "detection ONNX model")
		recModel = flag.String("rec-model", "assets/models/rec.onnx", "recognition ONNX model")
		recKeys  = flag.String("rec-keys", "assets/ppocr_keys.txt", "recognition char dictionary")
		topN     = flag.Int("top", 25, "print at most this many boxes (by confidence)")
		ep       = flag.String("ep", "cpu", "execution provider: cpu | directml | cuda")
		epDevice = flag.Int("ep-device", 0, "GPU device index for directml/cuda")
	)
	flag.Parse()

	eng, err := ocr.NewEngine(*detModel, *recModel, *recKeys, *ortDLL, *ep, *epDevice)
	if err != nil {
		fmt.Fprintln(os.Stderr, "engine init:", err)
		os.Exit(1)
	}
	defer eng.Close()

	frame, err := capture.ScreenCapturer{Display: 0}.Capture(context.Background())
	if err != nil {
		fmt.Fprintln(os.Stderr, "capture:", err)
		os.Exit(1)
	}

	perc := ocr.NewNativePerceiver(eng, ocr.NativeOptions{Recognize: true})
	t0 := time.Now()
	res, err := perc.Perceive(context.Background(), frame)
	if err != nil {
		fmt.Fprintln(os.Stderr, "perceive:", err)
		os.Exit(1)
	}
	elapsed := time.Since(t0)

	boxes := res.Boxes
	sort.Slice(boxes, func(i, j int) bool { return boxes[i].Conf > boxes[j].Conf })

	fmt.Printf("detected %d boxes in %s\n", len(boxes), elapsed.Round(time.Millisecond))
	n := *topN
	if n > len(boxes) {
		n = len(boxes)
	}
	for i := 0; i < n; i++ {
		b := boxes[i]
		fmt.Printf("  [%3d,%3d %3dx%3d] conf=%.2f  %q\n", b.X, b.Y, b.W, b.H, b.Conf, b.Text)
	}
}
