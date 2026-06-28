package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"visionagent/internal/capture"
	"visionagent/internal/embed"
	"visionagent/internal/executor"
	"visionagent/internal/memory"
	"visionagent/internal/perception"
)

// TestAgentEndToEndOverHTTP exercises the full perceive -> memory -> decide ->
// act loop with REAL (non-network-external) components: a synthetic capturer, an
// httptest server standing in for the Python perception sidecar's /ocr endpoint
// driving the real HTTPPerceiver transport, the offline LocalEmbedder, a bounded
// in-memory episodic store, the dry-run executor and the built-in heuristic
// reasoner. It is deterministic and fast (no external dependencies, bounded
// iterations) and asserts the loop completes cleanly while accumulating memory
// and self-consistent telemetry.
func TestAgentEndToEndOverHTTP(t *testing.T) {
	var calls int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ocr" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			ImageB64 string `json:"image_b64"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.ImageB64 == "" {
			http.Error(w, "missing image_b64", http.StatusBadRequest)
			return
		}
		// Alternate the screen text each call so some actions register as
		// progress (the screen-change reward baseline credits an action when
		// the observed text changes), keeping the run non-trivial yet
		// deterministic.
		n := atomic.AddInt32(&calls, 1)
		text := "main menu start options quit"
		if n%2 == 0 {
			text = "battle screen attack defend run"
		}
		res := perception.Result{
			Text: text,
			Boxes: []perception.Box{
				{X: 10, Y: 20, W: 40, H: 12, Text: "START", Conf: 0.95},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(res)
	}))
	defer srv.Close()

	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	tele := &Telemetry{}
	a := &Agent{
		Capturer:  capture.FakeCapturer{},
		Perceiver: perception.NewHTTPPerceiver(srv.URL),
		Embedder:  embed.LocalEmbedder{},
		Store:     store,
		Executor:  executor.DryRunExecutor{},
		Telemetry: tele,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	const iters = 12
	st, err := a.Run(ctx, iters)
	if err != nil {
		t.Fatalf("run returned error: %v", err)
	}
	if st.Iterations != iters {
		t.Fatalf("want %d iterations, got %d", iters, st.Iterations)
	}
	if st.Errors != 0 {
		t.Fatalf("want 0 errors over real HTTP transport, got %d", st.Errors)
	}
	if st.Actions == 0 {
		t.Fatalf("want >0 executed actions, got 0")
	}
	if got := store.Len(); got == 0 {
		t.Fatalf("want >0 accumulated episodes, got %d", got)
	}

	// Telemetry must be internally consistent.
	s := tele.Snapshot()
	if s.Scored == 0 {
		t.Fatalf("expected at least one scored action")
	}
	if s.Scored != s.Progress+s.Regress+s.Stall {
		t.Errorf("scored (%d) != progress+regress+stall (%d+%d+%d)",
			s.Scored, s.Progress, s.Regress, s.Stall)
	}
	for name, rate := range map[string]float64{
		"progress_rate": s.ProgressRate,
		"regress_rate":  s.RegressRate,
		"stall_rate":    s.StallRate,
		"efficiency":    s.Efficiency,
	} {
		if rate < 0 || rate > 1 {
			t.Errorf("%s = %v, out of [0,1]", name, rate)
		}
	}
	if s.Net != s.Progress-s.Regress {
		t.Errorf("net (%d) != progress-regress (%d)", s.Net, s.Progress-s.Regress)
	}
}
