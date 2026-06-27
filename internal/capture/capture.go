package capture

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"

	"github.com/kbinani/screenshot"
)

// Capturer grabs a frame of the screen as PNG bytes.
type Capturer interface {
	Capture(ctx context.Context) ([]byte, error)
}

// ScreenCapturer captures a real display. On Windows it uses the win32 API
// via kbinani/screenshot (no CGo required).
//
// If W and H are > 0, only the region (X, Y, W, H) is captured (ROI), which
// shrinks the area the OCR engine has to process.
type ScreenCapturer struct {
	Display int
	X, Y    int
	W, H    int
}

// Origin returns the absolute screen coordinate of the captured frame's
// top-left pixel. OCR boxes are relative to the captured image, so the agent
// adds this offset to action coordinates to click the correct screen location.
// For an ROI it is (X, Y); for a full display it is that display's bounds origin.
func (c ScreenCapturer) Origin() image.Point {
	if c.W > 0 && c.H > 0 {
		return image.Pt(c.X, c.Y)
	}
	return screenshot.GetDisplayBounds(c.Display).Min
}

func (c ScreenCapturer) Capture(_ context.Context) ([]byte, error) {
	var (
		img *image.RGBA
		err error
	)
	if c.W > 0 && c.H > 0 {
		img, err = screenshot.Capture(c.X, c.Y, c.W, c.H)
	} else {
		img, err = screenshot.CaptureRect(screenshot.GetDisplayBounds(c.Display))
	}
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// FakeCapturer produces a synthetic frame for tests (no display needed).
type FakeCapturer struct {
	W, H int
}

func (f FakeCapturer) Capture(_ context.Context) ([]byte, error) {
	w, h := f.W, f.H
	if w == 0 {
		w = 16
	}
	if h == 0 {
		h = 16
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for x := 0; x < w; x++ {
		img.Set(x, 0, color.RGBA{R: uint8(x), A: 255})
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
