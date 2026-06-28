//go:build cgo

package main

import (
	"log/slog"

	"visionagent/internal/ocr"
	"visionagent/internal/perception"
)

// nativeConfig carries the flag values needed to construct the native ONNX
// perceiver.
type nativeConfig struct {
	detModel, recModel, recKeys, ortDLL, ep string
	epDevice                                int
	recognize, watch, profile, collect      bool
	collectDir                              string
	collectThresh                           float64
}

// newNativePerceiver builds the in-process onnxruntime perceiver. It is only
// compiled into CGO builds; see native_nocgo.go for the disabled stub.
func newNativePerceiver(log *slog.Logger, cfg nativeConfig) (perception.Perceiver, error) {
	eng, err := ocr.NewEngine(cfg.detModel, cfg.recModel, cfg.recKeys, cfg.ortDLL, cfg.ep, cfg.epDevice)
	if err != nil {
		return nil, err
	}
	log.Info("native ocr engine ready", "ep", cfg.ep, "device", cfg.epDevice, "dll", cfg.ortDLL)
	opts := ocr.NativeOptions{Recognize: cfg.recognize, Watch: cfg.watch, Profile: cfg.profile, Log: log}
	if cfg.collect {
		col, err := ocr.NewCollector(cfg.collectDir, cfg.collectThresh)
		if err != nil {
			return nil, err
		}
		opts.Collect = col
		log.Info("flywheel collection enabled", "dir", cfg.collectDir, "thresh", cfg.collectThresh)
	}
	log.Info("perception transport: native (onnxruntime)", "recognize", cfg.recognize, "watch", cfg.watch)
	return ocr.NewNativePerceiver(eng, opts), nil
}
