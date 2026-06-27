package ocr

import (
	"bytes"
	"context"
	"image"
	_ "image/png" // register PNG decoder
	"log/slog"
	"strings"
	"time"

	"visionagent/internal/perception"
)

// NativeOptions configures the in-process ONNX perceiver.
type NativeOptions struct {
	Recognize bool       // run text recognition (false = detection boxes only)
	Watch     bool       // hot-reload the rec model when its file changes
	Collect   *Collector // if set, persist low-confidence crops for retraining
	Profile   bool       // log per-stage timings (decode/detect/recognize)
	Log       *slog.Logger
}

// NativePerceiver implements perception.Perceiver using the in-process ONNX
// engine (no Python sidecar).
type NativePerceiver struct {
	eng  *Engine
	opts NativeOptions
}

func NewNativePerceiver(eng *Engine, opts NativeOptions) *NativePerceiver {
	return &NativePerceiver{eng: eng, opts: opts}
}

func (n *NativePerceiver) Perceive(_ context.Context, framePNG []byte) (perception.Result, error) {
	if n.opts.Watch {
		if reloaded, err := n.eng.ReloadRecIfChanged(); err == nil && reloaded && n.opts.Log != nil {
			n.opts.Log.Info("rec model hot-reloaded", "path", n.eng.recPath)
		}
	}

	t0 := time.Now()
	img, _, err := image.Decode(bytes.NewReader(framePNG))
	if err != nil {
		return perception.Result{}, err
	}
	t1 := time.Now()
	boxes, err := n.eng.Detect(img)
	if err != nil {
		return perception.Result{}, err
	}
	t2 := time.Now()

	var texts []string
	var confs []float64
	if n.opts.Recognize {
		texts, confs = n.eng.RecognizeBatch(img, boxes)
	}
	t3 := time.Now()
	if n.opts.Profile && n.opts.Log != nil {
		n.opts.Log.Info("perceive profile",
			"decode_ms", t1.Sub(t0).Milliseconds(),
			"detect_ms", t2.Sub(t1).Milliseconds(),
			"recognize_ms", t3.Sub(t2).Milliseconds(),
			"boxes", len(boxes),
			"total_ms", t3.Sub(t0).Milliseconds())
	}

	out := make([]perception.Box, 0, len(boxes))
	var sb strings.Builder
	for i, b := range boxes {
		text := ""
		conf := b.Score
		if n.opts.Recognize {
			text = texts[i]
			conf = confs[i]
			if n.opts.Collect != nil {
				n.opts.Collect.Add(n.eng.CropImage(img, b), b, text, conf)
			}
		}
		out = append(out, perception.Box{
			X: b.X, Y: b.Y, W: b.W, H: b.H, Text: text, Conf: conf,
		})
		if text != "" {
			sb.WriteString(text)
			sb.WriteByte(' ')
		}
	}
	return perception.Result{Text: strings.TrimSpace(sb.String()), Boxes: out}, nil
}

func (n *NativePerceiver) Close() error {
	if n.opts.Collect != nil {
		_ = n.opts.Collect.Close()
	}
	return n.eng.Close()
}
