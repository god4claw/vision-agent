package agent

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"image"
	"log/slog"
	"strings"
	"time"

	"visionagent/internal/capture"
	"visionagent/internal/embed"
	"visionagent/internal/executor"
	"visionagent/internal/memory"
	"visionagent/internal/perception"
	"visionagent/internal/reason"
)

// Stats summarizes a run segment.
type Stats struct {
	Iterations    int
	Errors        int
	Actions       int
	PerceiveTotal time.Duration
}

// AvgPerceive returns the mean perceive latency across iterations.
func (s Stats) AvgPerceive() time.Duration {
	if s.Iterations == 0 {
		return 0
	}
	return s.PerceiveTotal / time.Duration(s.Iterations)
}

// Agent wires the perception -> memory -> decision -> action loop.
type Agent struct {
	Capturer  capture.Capturer
	Perceiver perception.Perceiver
	Embedder  embed.Embedder
	Store     *memory.Store
	Executor  executor.Executor
	Log       *slog.Logger

	// Reasoner, if set, is the LLM policy that decides each action. When it is
	// nil or fails, the agent falls back to the built-in heuristic decide().
	Reasoner reason.Reasoner
	Goal     string // objective handed to the reasoner

	// ReplayThreshold is the minimum memory similarity (0..1) required before
	// the heuristic replays a past successful action; <=0 uses a default.
	ReplayThreshold float64

	// Scorer, if set and a Goal is configured, judges reward by whether an
	// action moved the screen closer to the goal. When nil/erroring, the agent
	// falls back to the screen-change baseline (scoreSuccess).
	Scorer RewardScorer

	// Planner, if set with a Goal, decomposes the goal into ordered sub-steps;
	// the agent then pursues one sub-step at a time, handing the current step to
	// the reasoner and advancing when the planner judges it complete.
	Planner Planner
	plan    *Plan
	planned bool

	// Telemetry, if set, accumulates progress/regress/stall counts per scored
	// action so the operator can track how the agent is advancing.
	Telemetry *Telemetry

	// CaptureOrigin is the absolute screen coordinate of the captured frame's
	// top-left pixel. OCR boxes (and thus action coordinates) are relative to
	// the captured image, so this offset is added back before real input so a
	// click lands at the correct screen location. Zero for a primary full-screen
	// capture; set it to the ROI/display origin to target a region or a second
	// (or virtual) display.
	CaptureOrigin image.Point

	TickDelay time.Duration
}

// RewardScorer judges whether an action made progress toward the goal, given
// the goal and the observed screen state before and after the action.
type RewardScorer interface {
	Score(ctx context.Context, goal, before, after string) (bool, error)
}

// Planner decomposes a goal into ordered sub-steps and judges sub-step
// completion from the current screen.
type Planner interface {
	Plan(ctx context.Context, goal string) ([]string, error)
	StepComplete(ctx context.Context, step, screen string) (bool, error)
}

// Plan is an ordered list of sub-steps with a cursor at the current step.
type Plan struct {
	Goal  string
	Steps []string
	idx   int
}

// Current returns the active sub-step, or the overall goal once the plan is
// exhausted.
func (p *Plan) Current() string {
	if p.Done() {
		return p.Goal
	}
	return p.Steps[p.idx]
}

func (p *Plan) Advance()             { p.idx++ }
func (p *Plan) Done() bool           { return p.idx >= len(p.Steps) }
func (p *Plan) Progress() (int, int) { return p.idx, len(p.Steps) }

// rewardState carries one pending action across ticks so its on-screen effect
// can be scored on the following observation (deferred reward).
type rewardState struct {
	prevGoal   string
	prevState  string
	prevAction executor.Action
	have       bool
}

// Run executes the loop for maxIters iterations (or until ctx is cancelled).
// Each step is wrapped with recover so a single failure cannot crash the loop
// (step-level watchdog); errors are counted and the loop continues.
func (a *Agent) Run(ctx context.Context, maxIters int) (Stats, error) {
	var st Stats
	var rw rewardState
	for i := 0; i < maxIters; i++ {
		select {
		case <-ctx.Done():
			return st, ctx.Err()
		default:
		}
		if err := a.step(ctx, i, &st, &rw); err != nil {
			st.Errors++
			if a.Log != nil {
				a.Log.Warn("step failed, continuing", "iter", i, "err", err)
			}
		}
		st.Iterations++
		if a.Telemetry != nil && a.Log != nil && st.Iterations%50 == 0 {
			s := a.Telemetry.Snapshot()
			a.Log.Info("telemetry", "scored", s.Scored, "progress", s.Progress,
				"regress", s.Regress, "stall", s.Stall, "net", s.Net)
		}
		if a.TickDelay > 0 {
			time.Sleep(a.TickDelay)
		}
	}
	return st, nil
}

func (a *Agent) step(ctx context.Context, i int, st *Stats, rw *rewardState) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic in step: %v", r)
		}
	}()

	frame, err := a.Capturer.Capture(ctx)
	if err != nil {
		return fmt.Errorf("capture: %w", err)
	}
	t0 := time.Now()
	res, err := a.Perceiver.Perceive(ctx, frame)
	st.PerceiveTotal += time.Since(t0)
	if err != nil {
		return fmt.Errorf("perceive: %w", err)
	}

	state := buildState(res)

	// Deferred reward: the previous action's effect is only observable now, by
	// comparing the screen before it (rw.prevState) with the screen after it
	// (state). Resolve and persist that episode with its real success signal,
	// scored against the sub-step that was active when the action was taken.
	if rw.have {
		success := a.scoreReward(ctx, rw.prevGoal, rw.prevState, state, rw.prevAction)
		ep := memory.Episode{
			ID:        episodeID(rw.prevState, i),
			Stage:     "default",
			StateText: rw.prevState,
			Action:    rw.prevAction,
			Variables: map[string]string{},
			Success:   success,
		}
		if err := a.Store.Add(ctx, ep); err != nil {
			return fmt.Errorf("store: %w", err)
		}
		if a.Telemetry != nil && rw.prevAction.Type != executor.ActionNone {
			a.Telemetry.Record(success, rw.prevState != state)
		}
	}

	// Multi-step planning: build the plan once, then advance past any sub-steps
	// the current screen already satisfies. The reasoner pursues the current
	// sub-step as its goal.
	a.ensurePlan(ctx)
	a.advancePlan(ctx, state)
	goalNow := a.effectiveGoal()

	near, err := a.Store.Nearest(ctx, state, 1)
	if err != nil {
		return fmt.Errorf("retrieve: %w", err)
	}

	action := a.choose(ctx, goalNow, res, near)
	if err := a.Executor.Do(ctx, offsetAction(action, a.CaptureOrigin)); err != nil {
		return fmt.Errorf("execute: %w", err)
	}
	if action.Type != executor.ActionNone {
		st.Actions++
	}

	// Defer this action's episode to the next tick, when its effect on the
	// screen can be measured and scored.
	rw.prevGoal = goalNow
	rw.prevState = state
	rw.prevAction = action
	rw.have = true
	return nil
}

// ensurePlan lazily decomposes the goal into sub-steps the first time it is
// needed. A planning failure leaves plan nil, so the agent pursues the whole
// goal directly.
func (a *Agent) ensurePlan(ctx context.Context) {
	if a.Planner == nil || a.planned || strings.TrimSpace(a.Goal) == "" {
		return
	}
	a.planned = true
	steps, err := a.Planner.Plan(ctx, a.Goal)
	if err != nil || len(steps) == 0 {
		if a.Log != nil {
			a.Log.Warn("planning failed; pursuing goal directly", "err", err)
		}
		return
	}
	a.plan = &Plan{Goal: a.Goal, Steps: steps}
	if a.Log != nil {
		a.Log.Info("plan built", "steps", len(steps))
	}
}

// advancePlan moves the cursor past every leading sub-step the current screen
// already satisfies. A check failure stops advancement for this tick.
func (a *Agent) advancePlan(ctx context.Context, state string) {
	if a.Planner == nil || a.plan == nil {
		return
	}
	for !a.plan.Done() {
		done, err := a.Planner.StepComplete(ctx, a.plan.Current(), state)
		if err != nil {
			if a.Log != nil {
				a.Log.Warn("step-complete check failed", "err", err)
			}
			return
		}
		if !done {
			return
		}
		if a.Log != nil {
			cur, total := a.plan.Progress()
			a.Log.Info("plan step complete", "step", a.plan.Current(), "index", cur, "of", total)
		}
		a.plan.Advance()
	}
}

// effectiveGoal is the sub-step the agent is currently pursuing, falling back
// to the overall goal when there is no active plan.
func (a *Agent) effectiveGoal() string {
	if a.plan != nil && !a.plan.Done() {
		return a.plan.Current()
	}
	return a.Goal
}

// choose picks the next action. If a Reasoner (LLM policy) is configured it is
// consulted first; any reasoner error degrades gracefully to the heuristic so a
// flaky/unavailable model never stalls the loop.
func (a *Agent) choose(ctx context.Context, goal string, res perception.Result, near []memory.Episode) executor.Action {
	if a.Reasoner == nil {
		return decide(res, near, a.replayThreshold())
	}
	action, why, err := a.Reasoner.Decide(ctx, goal, res, near)
	if err != nil {
		if a.Log != nil {
			a.Log.Warn("reasoner failed, falling back to heuristic", "err", err)
		}
		return decide(res, near, a.replayThreshold())
	}
	if a.Log != nil {
		a.Log.Info("reasoner decision", "action", string(action.Type), "key", action.Key, "reason", why)
	}
	return action
}

// decide is the heuristic policy: replay the nearest past action only when it
// succeeded AND the current state is similar enough (>= minSim); otherwise
// explore by proposing a click on the first detected element.
func decide(res perception.Result, near []memory.Episode, minSim float64) executor.Action {
	if len(near) > 0 && near[0].Success && near[0].Action.Type != executor.ActionNone &&
		float64(near[0].Similarity) >= minSim {
		return near[0].Action
	}
	if len(res.Boxes) > 0 {
		b := res.Boxes[0]
		return executor.Action{Type: executor.ActionClick, X: b.X + b.W/2, Y: b.Y + b.H/2}
	}
	return executor.Action{Type: executor.ActionNone}
}

// replayThreshold returns the configured replay similarity gate, defaulting to
// 0.7 when unset so an unrelated state never triggers a stale replay.
func (a *Agent) replayThreshold() float64 {
	if a.ReplayThreshold > 0 {
		return a.ReplayThreshold
	}
	return 0.7
}

// scoreReward picks the reward signal: a goal-aware judge when a Scorer and
// Goal are configured (success = the action moved closer to the goal), else
// the screen-change baseline. A scorer error degrades to the baseline so a
// flaky judge never stalls learning.
func (a *Agent) scoreReward(ctx context.Context, goal, before, after string, action executor.Action) bool {
	if a.Scorer != nil && strings.TrimSpace(goal) != "" && action.Type != executor.ActionNone {
		ok, err := a.Scorer.Score(ctx, goal, before, after)
		if err == nil {
			return ok
		}
		if a.Log != nil {
			a.Log.Warn("goal scorer failed, using screen-change baseline", "err", err)
		}
	}
	return scoreSuccess(action, before, after)
}

// scoreSuccess is the baseline reward signal: a real action "succeeded" if the
// observed screen changed after it (it did something). A no-op is never a
// useful success. This needs no goal oracle and works on any screen.
func scoreSuccess(action executor.Action, before, after string) bool {
	if action.Type == executor.ActionNone {
		return false
	}
	return before != after
}

// offsetAction shifts a positional action's coordinates by the capture origin
// so screen-relative box coordinates become absolute screen coordinates. Key
// and no-op actions are returned unchanged.
func offsetAction(a executor.Action, origin image.Point) executor.Action {
	if origin.X == 0 && origin.Y == 0 {
		return a
	}
	switch a.Type {
	case executor.ActionClick, executor.ActionMove:
		a.X += origin.X
		a.Y += origin.Y
	case executor.ActionDrag:
		a.X += origin.X
		a.Y += origin.Y
		a.X2 += origin.X
		a.Y2 += origin.Y
	}
	return a
}

func buildState(res perception.Result) string {
	if s := strings.TrimSpace(res.Text); s != "" {
		return s
	}
	var sb strings.Builder
	for _, b := range res.Boxes {
		sb.WriteString(b.Text)
		sb.WriteByte(' ')
	}
	if s := strings.TrimSpace(sb.String()); s != "" {
		return s
	}
	return "empty-screen"
}

func episodeID(state string, i int) string {
	h := sha1.Sum([]byte(fmt.Sprintf("%s#%d", state, i)))
	return hex.EncodeToString(h[:])
}
