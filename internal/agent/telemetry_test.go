package agent

import (
	"context"
	"math"
	"testing"

	"visionagent/internal/capture"
	"visionagent/internal/embed"
	"visionagent/internal/executor"
	"visionagent/internal/memory"
	"visionagent/internal/perception"
)

// DoD: Record buckets each scored action into progress/regress/stall and
// Snapshot derives the correct counts and rates.
func TestTelemetryRecordAndSnapshot(t *testing.T) {
	var tel Telemetry
	tel.Record(true, true)   // progress
	tel.Record(true, false)  // progress (judge said yes without a screen change)
	tel.Record(false, true)  // regress (changed, but not toward goal)
	tel.Record(false, false) // stall (no effect)

	s := tel.Snapshot()
	if s.Scored != 4 || s.Progress != 2 || s.Regress != 1 || s.Stall != 1 {
		t.Fatalf("counts wrong: %+v", s)
	}
	if s.Net != 1 {
		t.Fatalf("net = %d, want 1", s.Net)
	}
	if s.ProgressRate != 0.5 || s.RegressRate != 0.25 || s.StallRate != 0.25 {
		t.Fatalf("rates wrong: %+v", s)
	}
	// Efficiency = progress / (progress + regress) = 2/3, ignoring stalls.
	if math.Abs(s.Efficiency-2.0/3.0) > 1e-9 {
		t.Fatalf("efficiency = %v, want 2/3", s.Efficiency)
	}
}

// DoD: during a run, a scored action that the judge rejects while the screen
// changed is counted as a regress.
func TestTelemetryRecordsRegressInRun(t *testing.T) {
	ctx := context.Background()
	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	tel := &Telemetry{}
	a := &Agent{
		Capturer: capture.FakeCapturer{},
		Perceiver: &seqPerceiver{results: []perception.Result{
			{Text: "s0", Boxes: []perception.Box{{W: 10, H: 10, Text: "a"}}},
			{Text: "s1", Boxes: []perception.Box{{W: 10, H: 10, Text: "b"}}}, // screen changed
		}},
		Embedder:  embed.LocalEmbedder{},
		Store:     store,
		Executor:  &capturingExecutor{},
		Goal:      "g",
		Reasoner:  fakeReasoner{act: executor.Action{Type: executor.ActionClick, X: 1, Y: 1}},
		Scorer:    fakeScorer{result: false}, // judge: no progress
		Telemetry: tel,
	}
	if _, err := a.Run(ctx, 2); err != nil {
		t.Fatalf("run: %v", err)
	}
	// Tick 0's click is scored at tick 1, where s0 -> s1 changed but the judge
	// said no progress => one regress.
	s := tel.Snapshot()
	if s.Scored != 1 || s.Regress != 1 || s.Progress != 0 || s.Stall != 0 {
		t.Fatalf("expected exactly one regress: %+v", s)
	}
}
