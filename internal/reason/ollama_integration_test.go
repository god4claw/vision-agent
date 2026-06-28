package reason

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"testing"
	"time"

	"visionagent/internal/executor"
	"visionagent/internal/perception"
)

// envOr returns the value of environment variable key, or def if it is unset
// or empty.
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// ollamaModelAvailable probes a live Ollama server at url and reports whether
// it is reachable and has model pulled. It returns (reachable, hasModel). A
// short timeout keeps CI fast when no server is listening.
func ollamaModelAvailable(ctx context.Context, url, model string) (bool, bool) {
	client := &http.Client{Timeout: 2 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url+"/api/tags", nil)
	if err != nil {
		return false, false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false, false
	}
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		// Server answered but the payload was unexpected; treat as reachable
		// but without a confirmable model so the test skips rather than fails.
		return true, false
	}
	for _, m := range tags.Models {
		if m.Name == model {
			return true, true
		}
	}
	return true, false
}

// allowedActionTypes is the set of action verbs the reasoner may legitimately
// return. The model only ever picks an element by index; the agent (ToAction)
// owns the resolved pixel coordinates.
var allowedActionTypes = map[executor.ActionType]bool{
	executor.ActionNone:  true,
	executor.ActionClick: true,
	executor.ActionMove:  true,
	executor.ActionDrag:  true,
	executor.ActionKey:   true,
}

// TestOllamaReasonerLive exercises the REAL OllamaReasoner against a locally
// running Ollama. It is a guarded integration test: if Ollama is unreachable
// or the chosen model is not pulled, the test SKIPS (it never fails on a
// machine without Ollama), keeping CI green. Configure with:
//
//	VA_OLLAMA_URL   (default http://127.0.0.1:11434)
//	VA_OLLAMA_MODEL (default llama3.2:3b)
//
// The assertions are deliberately structural to tolerate LLM variability: the
// decision must be a parseable Decision that resolves to a valid action whose
// chosen element index is in range. The model must NOT invent coordinates;
// ToAction resolves the center from the element index, so for pointer actions
// we verify the returned coordinates match one of the elements we presented.
func TestOllamaReasonerLive(t *testing.T) {
	url := envOr("VA_OLLAMA_URL", "http://127.0.0.1:11434")
	model := envOr("VA_OLLAMA_MODEL", "llama3.2:3b")

	probeCtx, cancelProbe := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancelProbe()
	reachable, hasModel := ollamaModelAvailable(probeCtx, url, model)
	if !reachable {
		t.Skipf("skipping real-Ollama test: %s unreachable (set VA_OLLAMA_URL to enable)", url)
	}
	if !hasModel {
		t.Skipf("skipping real-Ollama test: model %q not pulled at %s (set VA_OLLAMA_MODEL; `ollama pull %s`)", model, url, model)
	}

	// A tiny but realistic perception result: a few labeled text boxes laid out
	// on screen. Centers are deterministic so we can verify the resolved action
	// points at an element we actually presented.
	res := perception.Result{
		Boxes: []perception.Box{
			{X: 100, Y: 100, W: 120, H: 40, Text: "Start Game", Conf: 0.95},
			{X: 100, Y: 200, W: 120, H: 40, Text: "Settings", Conf: 0.93},
			{X: 100, Y: 300, W: 120, H: 40, Text: "Quit", Conf: 0.91},
		},
	}
	shown := selectElements(res.Boxes, defaultMaxElements)

	// Build the set of valid element centers so we can confirm the model picked
	// an index in range (the resolved coordinate must be one we presented).
	type pt struct{ x, y int }
	centers := make(map[pt]bool, len(shown))
	for _, b := range shown {
		centers[pt{b.X + b.W/2, b.Y + b.H/2}] = true
	}

	r := NewOllamaReasoner(url, model)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	act, why, err := r.Decide(ctx, "start a new game", res, nil)
	if err != nil {
		t.Fatalf("real Ollama Decide failed (model %q): %v", model, err)
	}
	t.Logf("model=%q decided action=%q x=%d y=%d reason=%q", model, act.Type, act.X, act.Y, why)

	if !allowedActionTypes[act.Type] {
		t.Fatalf("action type %q not in allowed set", act.Type)
	}

	switch act.Type {
	case executor.ActionClick, executor.ActionMove:
		if !centers[pt{act.X, act.Y}] {
			t.Fatalf("%s resolved to (%d,%d) which is not any presented element center: the model invented coordinates or picked an out-of-range index", act.Type, act.X, act.Y)
		}
	case executor.ActionDrag:
		if !centers[pt{act.X, act.Y}] || !centers[pt{act.X2, act.Y2}] {
			t.Fatalf("drag endpoints (%d,%d)->(%d,%d) must both be presented element centers", act.X, act.Y, act.X2, act.Y2)
		}
	case executor.ActionKey:
		if !executor.ValidKey(act.Key) {
			t.Fatalf("key action with invalid key %q", act.Key)
		}
	case executor.ActionNone:
		// Acceptable: the model may decline to act. Structural contract holds.
	}
}
