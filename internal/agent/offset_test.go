package agent

import (
	"context"
	"image"
	"testing"

	"visionagent/internal/capture"
	"visionagent/internal/embed"
	"visionagent/internal/executor"
	"visionagent/internal/memory"
	"visionagent/internal/perception"
)

// DoD: when capturing a region/display at a non-zero origin, the executed
// action's coordinates are shifted by that origin so the click lands at the
// correct absolute screen location (the bug behind the agent acting on the
// wrong monitor).
func TestCaptureOriginOffsetsActions(t *testing.T) {
	ctx := context.Background()
	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	rec := &capturingExecutor{}
	a := &Agent{
		Capturer: capture.FakeCapturer{},
		Perceiver: perception.FakePerceiver{Fixed: perception.Result{
			Text:  "x",
			Boxes: []perception.Box{{X: 0, Y: 0, W: 10, H: 10, Text: "a"}},
		}},
		Embedder:      embed.LocalEmbedder{},
		Store:         store,
		Executor:      rec,
		Reasoner:      fakeReasoner{act: executor.Action{Type: executor.ActionClick, X: 5, Y: 7}},
		CaptureOrigin: image.Pt(1920, 40),
	}
	if _, err := a.Run(ctx, 1); err != nil {
		t.Fatalf("run: %v", err)
	}
	// Relative (5,7) + origin (1920,40) = absolute (1925,47).
	if rec.last.X != 1925 || rec.last.Y != 47 {
		t.Fatalf("offset not applied: got (%d,%d), want (1925,47)", rec.last.X, rec.last.Y)
	}
}

// DoD: offsetAction shifts both endpoints of a drag and leaves key/none
// actions untouched.
func TestOffsetActionDragAndNonPositional(t *testing.T) {
	o := image.Pt(100, 200)
	drag := offsetAction(executor.Action{Type: executor.ActionDrag, X: 1, Y: 2, X2: 3, Y2: 4}, o)
	if drag.X != 101 || drag.Y != 202 || drag.X2 != 103 || drag.Y2 != 204 {
		t.Fatalf("drag offset wrong: %+v", drag)
	}
	key := offsetAction(executor.Action{Type: executor.ActionKey, Key: "Enter"}, o)
	if key.X != 0 || key.Y != 0 {
		t.Fatalf("key action should be unchanged: %+v", key)
	}
	// A zero origin is a no-op even for positional actions.
	same := offsetAction(executor.Action{Type: executor.ActionClick, X: 5, Y: 6}, image.Point{})
	if same.X != 5 || same.Y != 6 {
		t.Fatalf("zero origin should not shift: %+v", same)
	}
}
