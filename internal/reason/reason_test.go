package reason

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"visionagent/internal/executor"
	"visionagent/internal/perception"
)

func sampleBoxes() []perception.Box {
	return []perception.Box{
		{X: 10, Y: 20, W: 40, H: 10, Text: "ATTACK", Conf: 0.9},
		{X: 100, Y: 200, W: 60, H: 20, Text: "Inventory", Conf: 0.8},
		{X: 0, Y: 0, W: 5, H: 5, Text: "", Conf: 0.5}, // empty text: must be filtered out
	}
}

func TestSelectElementsFiltersEmptyAndCaps(t *testing.T) {
	shown := selectElements(sampleBoxes(), 40)
	if len(shown) != 2 {
		t.Fatalf("want 2 non-empty elements, got %d", len(shown))
	}
	if shown[0].Text != "ATTACK" || shown[1].Text != "Inventory" {
		t.Fatalf("unexpected order/content: %+v", shown)
	}
	if capped := selectElements(sampleBoxes(), 1); len(capped) != 1 {
		t.Fatalf("cap not honored: got %d", len(capped))
	}
}

func TestToActionClickResolvesCoordinates(t *testing.T) {
	shown := selectElements(sampleBoxes(), 40)
	d := Decision{Action: "click", Target: 1}
	act, err := d.ToAction(shown)
	if err != nil {
		t.Fatalf("toAction: %v", err)
	}
	// Inventory center: x=100+60/2=130, y=200+20/2=210.
	if act.Type != executor.ActionClick || act.X != 130 || act.Y != 210 {
		t.Fatalf("wrong click action: %+v", act)
	}
}

func TestToActionMoveResolvesCoordinates(t *testing.T) {
	shown := selectElements(sampleBoxes(), 40)
	act, err := (Decision{Action: "move", Target: 1}).ToAction(shown)
	if err != nil {
		t.Fatalf("toAction move: %v", err)
	}
	// Inventory center: x=130, y=210 (same resolution as click, no click).
	if act.Type != executor.ActionMove || act.X != 130 || act.Y != 210 {
		t.Fatalf("wrong move action: %+v", act)
	}
	// "hover" is an alias for "move".
	if act, err := (Decision{Action: "hover", Target: 0}).ToAction(shown); err != nil || act.Type != executor.ActionMove {
		t.Fatalf("hover should map to move: %+v err=%v", act, err)
	}
	if _, err := (Decision{Action: "move", Target: 99}).ToAction(shown); err == nil {
		t.Fatal("expected out-of-range move error")
	}
}

func TestToActionDragResolvesBothEnds(t *testing.T) {
	shown := selectElements(sampleBoxes(), 40)
	act, err := (Decision{Action: "drag", Target: 0, To: 1}).ToAction(shown)
	if err != nil {
		t.Fatalf("toAction drag: %v", err)
	}
	// ATTACK center (30,25) -> Inventory center (130,210).
	if act.Type != executor.ActionDrag || act.X != 30 || act.Y != 25 || act.X2 != 130 || act.Y2 != 210 {
		t.Fatalf("wrong drag action: %+v", act)
	}
	if _, err := (Decision{Action: "drag", Target: 0, To: 99}).ToAction(shown); err == nil {
		t.Fatal("expected out-of-range drag destination error")
	}
	if _, err := (Decision{Action: "drag", Target: -1, To: 1}).ToAction(shown); err == nil {
		t.Fatal("expected out-of-range drag source error")
	}
}

func TestToActionKeyAndNone(t *testing.T) {
	if act, err := (Decision{Action: "key", Key: "2"}).ToAction(nil); err != nil || act.Type != executor.ActionKey || act.Key != "2" {
		t.Fatalf("key action wrong: %+v err=%v", act, err)
	}
	if act, err := (Decision{Action: "none"}).ToAction(nil); err != nil || act.Type != executor.ActionNone {
		t.Fatalf("none action wrong: %+v err=%v", act, err)
	}
	if act, err := (Decision{Action: ""}).ToAction(nil); err != nil || act.Type != executor.ActionNone {
		t.Fatalf("empty action should be none: %+v err=%v", act, err)
	}
}

func TestToActionErrors(t *testing.T) {
	shown := selectElements(sampleBoxes(), 40)
	if _, err := (Decision{Action: "click", Target: 99}).ToAction(shown); err == nil {
		t.Fatal("expected out-of-range click error")
	}
	if _, err := (Decision{Action: "click", Target: -1}).ToAction(shown); err == nil {
		t.Fatal("expected negative index click error")
	}
	if _, err := (Decision{Action: "key", Key: ""}).ToAction(shown); err == nil {
		t.Fatal("expected empty-key error")
	}
	if _, err := (Decision{Action: "key", Key: "Extra usage balance $0.00"}).ToAction(shown); err == nil {
		t.Fatal("expected garbage-key rejection")
	}
	if _, err := (Decision{Action: "frobnicate"}).ToAction(shown); err == nil {
		t.Fatal("expected unknown-action error")
	}
}

func TestParseDecisionMalformed(t *testing.T) {
	if _, err := ParseDecision("not json"); err == nil {
		t.Fatal("expected parse error on non-json")
	}
}

// End-to-end: a mocked Ollama returns a click decision; Decide must POST to
// /api/generate and resolve the chosen element to real coordinates.
func TestDecideEndToEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		inner, _ := json.Marshal(Decision{Action: "click", Target: 0, Reason: "press attack"})
		_ = json.NewEncoder(w).Encode(map[string]string{"response": string(inner)})
	}))
	defer srv.Close()

	r := NewOllamaReasoner(srv.URL, "test-model")
	act, why, err := r.Decide(context.Background(), "kill the boss", perception.Result{Boxes: sampleBoxes()}, nil)
	if err != nil {
		t.Fatalf("decide: %v", err)
	}
	if act.Type != executor.ActionClick || act.X != 30 || act.Y != 25 {
		t.Fatalf("wrong action from reasoner: %+v", act)
	}
	if why != "press attack" {
		t.Fatalf("reason not propagated: %q", why)
	}
}

func TestDecideServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	r := NewOllamaReasoner(srv.URL, "test-model")
	if _, _, err := r.Decide(context.Background(), "g", perception.Result{Boxes: sampleBoxes()}, nil); err == nil {
		t.Fatal("expected error on 500 from ollama")
	}
}

func scorerServer(t *testing.T, progress bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		inner, _ := json.Marshal(ScoreDecision{Progress: progress, Reason: "judged"})
		_ = json.NewEncoder(w).Encode(map[string]string{"response": string(inner)})
	}))
}

func TestScorerProgress(t *testing.T) {
	for _, want := range []bool{true, false} {
		srv := scorerServer(t, want)
		s := NewOllamaScorer(srv.URL, "test-model")
		got, err := s.Score(context.Background(), "open file menu", "before text", "after text")
		srv.Close()
		if err != nil {
			t.Fatalf("score: %v", err)
		}
		if got != want {
			t.Fatalf("progress: want %v got %v", want, got)
		}
	}
}

func TestScorerServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	s := NewOllamaScorer(srv.URL, "test-model")
	if _, err := s.Score(context.Background(), "g", "a", "b"); err == nil {
		t.Fatal("expected error on 500 from ollama")
	}
}

func TestPlannerPlan(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		inner, _ := json.Marshal(PlanDecision{Steps: []string{"click File", "click Open", "   "}})
		_ = json.NewEncoder(w).Encode(map[string]string{"response": string(inner)})
	}))
	defer srv.Close()
	p := NewOllamaPlanner(srv.URL, "test-model")
	steps, err := p.Plan(context.Background(), "open a file")
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	// blank step filtered out.
	if len(steps) != 2 || steps[0] != "click File" || steps[1] != "click Open" {
		t.Fatalf("unexpected steps: %v", steps)
	}
}

func TestPlannerStepComplete(t *testing.T) {
	for _, want := range []bool{true, false} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			inner, _ := json.Marshal(struct {
				Complete bool `json:"complete"`
			}{want})
			_ = json.NewEncoder(w).Encode(map[string]string{"response": string(inner)})
		}))
		p := NewOllamaPlanner(srv.URL, "test-model")
		got, err := p.StepComplete(context.Background(), "click File", "some screen text")
		srv.Close()
		if err != nil {
			t.Fatalf("step-complete: %v", err)
		}
		if got != want {
			t.Fatalf("complete: want %v got %v", want, got)
		}
	}
}
