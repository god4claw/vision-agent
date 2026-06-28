//go:build !cgo

package main

import (
	"errors"
	"log/slog"

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

// newNativePerceiver is the non-CGO stub: the native ONNX OCR transport is not
// compiled into CGO-disabled release binaries (it needs onnxruntime_go plus the
// onnxruntime shared library and model assets). Use -transport http with the
// perception sidecar instead.
func newNativePerceiver(_ *slog.Logger, _ nativeConfig) (perception.Perceiver, error) {
	return nil, errors.New("native OCR transport is unavailable in this build (compiled without CGO); use -transport http with the perception sidecar")
}
