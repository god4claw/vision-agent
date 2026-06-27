package agent

import (
	"context"
	"strings"
	"testing"

	"visionagent/internal/capture"
	"visionagent/internal/embed"
	"visionagent/internal/executor"
	"visionagent/internal/memory"
	"visionagent/internal/perception"
	"visionagent/internal/reason"
)

// fakePlanner is a deterministic planner for tests.
type fakePlanner struct {
	steps []string
	done  func(step, screen string) bool
}

func (p fakePlanner) Plan(context.Context, string) ([]string, error) { return p.steps, nil }
func (p fakePlanner) StepComplete(_ context.Context, step, screen string) (bool, error) {
	return p.done(step, screen), nil
}

var _ Planner = fakePlanner{}

// recordReasoner records the goal handed to it each tick.
type recordReasoner struct {
	goals []string
	act   executor.Action
}

func (r *recordReasoner) Decide(_ context.Context, goal string, _ perception.Result, _ []memory.Episode) (executor.Action, string, error) {
	r.goals = append(r.goals, goal)
	return r.act, "rec", nil
}

var _ reason.Reasoner = (*recordReasoner)(nil)

// fakeReasoner is a deterministic stand-in for the LLM policy.
type fakeReasoner struct {
	act executor.Action
	err error
}

func (f fakeReasoner) Decide(context.Context, string, perception.Result, []memory.Episode) (executor.Action, string, error) {
	return f.act, "because test", f.err
}

// capturingExecutor records the last action it was asked to perform.
type capturingExecutor struct{ last executor.Action }

func (c *capturingExecutor) Do(_ context.Context, a executor.Action) error {
	c.last = a
	return nil
}

// seqPerceiver returns a fixed sequence of results, one per call, holding the
// last once exhausted. Lets tests drive screen changes across ticks.
type seqPerceiver struct {
	results []perception.Result
	i       int
}

func (s *seqPerceiver) Perceive(context.Context, []byte) (perception.Result, error) {
	r := s.results[s.i]
	if s.i < len(s.results)-1 {
		s.i++
	}
	return r, nil
}

// fakeScorer is a deterministic goal-progress judge for tests.
type fakeScorer struct {
	result bool
	err    error
}

func (f fakeScorer) Score(context.Context, string, string, string) (bool, error) {
	return f.result, f.err
}

var _ RewardScorer = fakeScorer{}

func newTestAgent(t *testing.T, maxEpisodes int) *Agent {
	t.Helper()
	store, err := memory.NewStore(embed.LocalEmbedder{}, maxEpisodes)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	return &Agent{
		Capturer: capture.FakeCapturer{},
		Perceiver: perception.FakePerceiver{Fixed: perception.Result{
			Text: "ATTACK target hp 80",
			Boxes: []perception.Box{
				{X: 10, Y: 20, W: 40, H: 10, Text: "ATTACK", Conf: 0.9},
			},
		}},
		Embedder: embed.LocalEmbedder{},
		Store:    store,
		Executor: executor.DryRunExecutor{},
	}
}

// DoD: cycle runs >=100 iterations without crash, with 0 errors, and memory
// stays bounded.
func Test100IterationsNoCrashBounded(t *testing.T) {
	a := newTestAgent(t, 50)
	st, err := a.Run(context.Background(), 100)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if st.Iterations != 100 {
		t.Fatalf("want 100 iterations, got %d", st.Iterations)
	}
	if st.Errors != 0 {
		t.Fatalf("want 0 errors, got %d", st.Errors)
	}
	if got := a.Store.Len(); got > 50 {
		t.Fatalf("memory not bounded: len=%d > 50", got)
	}
}

// DoD: watchdog — a failing component does not crash the loop; errors are
// counted and iterations continue to completion.
func TestWatchdogContinuesOnComponentFailure(t *testing.T) {
	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	a := &Agent{
		Capturer:  capture.FakeCapturer{},
		Perceiver: perception.FakePerceiver{Err: context.DeadlineExceeded},
		Embedder:  embed.LocalEmbedder{},
		Store:     store,
		Executor:  executor.DryRunExecutor{},
	}
	st, err := a.Run(context.Background(), 20)
	if err != nil {
		t.Fatalf("run should not return error on component failure: %v", err)
	}
	if st.Iterations != 20 {
		t.Fatalf("want 20 iterations despite failures, got %d", st.Iterations)
	}
	if st.Errors != 20 {
		t.Fatalf("want 20 counted errors, got %d", st.Errors)
	}
}

// DoD: episode is written and retrieval returns the nearest matching action.
func TestRecallAt1(t *testing.T) {
	ctx := context.Background()
	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	want := memory.Episode{
		ID:        "ep-boss",
		Stage:     "boss",
		StateText: "BOSS phase two enrage soon",
		Action:    executor.Action{Type: executor.ActionKey, Key: "2"},
		Success:   true,
	}
	if err := store.Add(ctx, want); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := store.Add(ctx, memory.Episode{
		ID:        "ep-loot",
		Stage:     "loot",
		StateText: "loot window gold items vendor",
		Action:    executor.Action{Type: executor.ActionClick, X: 5, Y: 5},
		Success:   true,
	}); err != nil {
		t.Fatalf("add: %v", err)
	}

	got, err := store.Nearest(ctx, "BOSS phase two enrage soon", 1)
	if err != nil {
		t.Fatalf("nearest: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("recall@1: want 1 result, got %d", len(got))
	}
	if got[0].Action.Key != "2" {
		t.Fatalf("recall@1 returned wrong action: %+v", got[0].Action)
	}
}

// DoD: when a Reasoner is configured, its chosen action is the one executed.
func TestReasonerDecisionIsExecuted(t *testing.T) {
	a := newTestAgent(t, 0)
	exec := &capturingExecutor{}
	a.Executor = exec
	a.Goal = "press the attack key"
	a.Reasoner = fakeReasoner{act: executor.Action{Type: executor.ActionKey, Key: "5"}}

	if _, err := a.Run(context.Background(), 1); err != nil {
		t.Fatalf("run: %v", err)
	}
	if exec.last.Type != executor.ActionKey || exec.last.Key != "5" {
		t.Fatalf("reasoner action not executed: %+v", exec.last)
	}
}

// DoD: a failing reasoner never stalls the loop; the agent degrades to the
// heuristic (click the first detected box) and counts no error.
func TestReasonerErrorFallsBackToHeuristic(t *testing.T) {
	a := newTestAgent(t, 0)
	exec := &capturingExecutor{}
	a.Executor = exec
	a.Reasoner = fakeReasoner{err: context.DeadlineExceeded}

	st, err := a.Run(context.Background(), 1)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if st.Errors != 0 {
		t.Fatalf("reasoner failure should not count as step error, got %d", st.Errors)
	}
	// FakePerceiver yields one ATTACK box at (10,20,40x10) -> center (30,25).
	if exec.last.Type != executor.ActionClick || exec.last.X != 30 || exec.last.Y != 25 {
		t.Fatalf("fallback heuristic action wrong: %+v", exec.last)
	}
}

var _ reason.Reasoner = fakeReasoner{}

// DoD: the reward signal credits an action only when the screen actually
// changed; a no-op never counts.
func TestScoreSuccess(t *testing.T) {
	none := executor.Action{Type: executor.ActionNone}
	click := executor.Action{Type: executor.ActionClick}
	if scoreSuccess(none, "a", "b") {
		t.Fatal("no-op must never be a success")
	}
	if !scoreSuccess(click, "a", "b") {
		t.Fatal("changed screen should score success")
	}
	if scoreSuccess(click, "a", "a") {
		t.Fatal("unchanged screen should not score success")
	}
}

// DoD: deferred reward persists the PREVIOUS action's episode with its real
// success, measured by whether the next observation differs.
func TestDeferredRewardPersistsRealSuccess(t *testing.T) {
	ctx := context.Background()
	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	a := &Agent{
		Capturer: capture.FakeCapturer{},
		Perceiver: &seqPerceiver{results: []perception.Result{
			{Text: "screen one alpha", Boxes: []perception.Box{{W: 10, H: 10, Text: "alpha"}}},
			{Text: "screen two beta", Boxes: []perception.Box{{W: 10, H: 10, Text: "beta"}}},
			{Text: "screen two beta", Boxes: []perception.Box{{W: 10, H: 10, Text: "beta"}}},
		}},
		Embedder: embed.LocalEmbedder{},
		Store:    store,
		Executor: &capturingExecutor{},
	}
	if _, err := a.Run(ctx, 3); err != nil {
		t.Fatalf("run: %v", err)
	}

	// Action taken on "screen one alpha" was followed by a different screen =>
	// success true.
	got, err := store.Nearest(ctx, "screen one alpha", 1)
	if err != nil || len(got) == 0 {
		t.Fatalf("nearest alpha: %v (n=%d)", err, len(got))
	}
	if !got[0].Success {
		t.Fatalf("changed screen should be success=true: %+v", got[0])
	}

	// Action taken on "screen two beta" was followed by the same screen =>
	// success false.
	got2, err := store.Nearest(ctx, "screen two beta", 1)
	if err != nil || len(got2) == 0 {
		t.Fatalf("nearest beta: %v (n=%d)", err, len(got2))
	}
	if got2[0].Success {
		t.Fatalf("unchanged screen should be success=false: %+v", got2[0])
	}
}

// DoD: with a goal + scorer, reward follows the judge, not the screen-change
// baseline. Here the screen does NOT change (baseline => false) but the scorer
// reports progress => the episode must be success=true.
func TestGoalAwareRewardUsesScorer(t *testing.T) {
	ctx := context.Background()
	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	a := &Agent{
		Capturer: capture.FakeCapturer{},
		Perceiver: &seqPerceiver{results: []perception.Result{
			{Text: "same screen", Boxes: []perception.Box{{W: 10, H: 10, Text: "x"}}},
			{Text: "same screen", Boxes: []perception.Box{{W: 10, H: 10, Text: "x"}}},
		}},
		Embedder: embed.LocalEmbedder{},
		Store:    store,
		Executor: &capturingExecutor{},
		Goal:     "do the thing",
		Scorer:   fakeScorer{result: true},
	}
	if _, err := a.Run(ctx, 2); err != nil {
		t.Fatalf("run: %v", err)
	}
	got, err := store.Nearest(ctx, "same screen", 1)
	if err != nil || len(got) == 0 {
		t.Fatalf("nearest: %v (n=%d)", err, len(got))
	}
	if !got[0].Success {
		t.Fatalf("scorer said progress but episode not success: %+v", got[0])
	}
}

// DoD: a failing scorer degrades to the screen-change baseline without error.
func TestGoalScorerErrorFallsBackToBaseline(t *testing.T) {
	ctx := context.Background()
	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	a := &Agent{
		Capturer: capture.FakeCapturer{},
		Perceiver: &seqPerceiver{results: []perception.Result{
			{Text: "alpha state", Boxes: []perception.Box{{W: 10, H: 10, Text: "alpha"}}},
			{Text: "beta state", Boxes: []perception.Box{{W: 10, H: 10, Text: "beta"}}},
			{Text: "beta state", Boxes: []perception.Box{{W: 10, H: 10, Text: "beta"}}},
		}},
		Embedder: embed.LocalEmbedder{},
		Store:    store,
		Executor: &capturingExecutor{},
		Goal:     "do the thing",
		Scorer:   fakeScorer{err: context.DeadlineExceeded},
	}
	st, err := a.Run(ctx, 3)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if st.Errors != 0 {
		t.Fatalf("scorer error should not count as step error, got %d", st.Errors)
	}
	// alpha -> beta is a real screen change, so the baseline credits success.
	got, err := store.Nearest(ctx, "alpha state", 1)
	if err != nil || len(got) == 0 {
		t.Fatalf("nearest: %v (n=%d)", err, len(got))
	}
	if !got[0].Success {
		t.Fatalf("baseline should credit changed screen: %+v", got[0])
	}
}

// DoD: the Plan cursor walks the sub-steps and falls back to the goal when done.
func TestPlanCursor(t *testing.T) {
	p := &Plan{Goal: "G", Steps: []string{"a", "b"}}
	if p.Current() != "a" || p.Done() {
		t.Fatalf("start: current=%q done=%v", p.Current(), p.Done())
	}
	p.Advance()
	if p.Current() != "b" {
		t.Fatalf("after one advance: current=%q", p.Current())
	}
	p.Advance()
	if !p.Done() || p.Current() != "G" {
		t.Fatalf("exhausted: done=%v current=%q", p.Done(), p.Current())
	}
}

// DoD: with a planner, the reasoner is handed the current sub-step as its goal,
// and the plan advances as the screen satisfies each step.
func TestMultiStepPlanAdvances(t *testing.T) {
	ctx := context.Background()
	store, err := memory.NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("store init: %v", err)
	}
	rr := &recordReasoner{act: executor.Action{Type: executor.ActionKey, Key: "a"}}
	a := &Agent{
		Capturer: capture.FakeCapturer{},
		Perceiver: &seqPerceiver{results: []perception.Result{
			{Text: "x", Boxes: []perception.Box{{W: 10, H: 10, Text: "x"}}},
			{Text: "alpha-done", Boxes: []perception.Box{{W: 10, H: 10, Text: "y"}}},
			{Text: "beta", Boxes: []perception.Box{{W: 10, H: 10, Text: "z"}}},
		}},
		Embedder: embed.LocalEmbedder{},
		Store:    store,
		Executor: &capturingExecutor{},
		Goal:     "do alpha then beta",
		Reasoner: rr,
		Planner: fakePlanner{
			steps: []string{"alpha", "beta"},
			done:  func(step, screen string) bool { return strings.Contains(screen, step+"-done") },
		},
	}
	if _, err := a.Run(ctx, 3); err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(rr.goals) < 2 {
		t.Fatalf("reasoner saw too few goals: %v", rr.goals)
	}
	if rr.goals[0] != "alpha" {
		t.Fatalf("first sub-goal = %q, want alpha", rr.goals[0])
	}
	if rr.goals[1] != "beta" {
		t.Fatalf("second sub-goal = %q, want beta (plan should advance)", rr.goals[1])
	}
}
