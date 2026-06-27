package agent

import (
	"context"
	"errors"
	"testing"
)

// countingPlanner records how many times the underlying StepComplete ran.
type countingPlanner struct {
	stepCalls int
	planCalls int
	result    bool
	err       error
}

func (c *countingPlanner) Plan(context.Context, string) ([]string, error) {
	c.planCalls++
	return []string{"s1", "s2"}, nil
}

func (c *countingPlanner) StepComplete(context.Context, string, string) (bool, error) {
	c.stepCalls++
	return c.result, c.err
}

// DoD: an identical (step, screen) check hits the cache and does not re-invoke
// the underlying planner; a new check does. Plan always delegates.
func TestCachingPlannerMemoizes(t *testing.T) {
	ctx := context.Background()
	inner := &countingPlanner{result: true}
	c := NewCachingPlanner(inner, 0)

	if _, err := c.Plan(ctx, "goal"); err != nil {
		t.Fatalf("plan: %v", err)
	}
	for i := 0; i < 3; i++ {
		got, err := c.StepComplete(ctx, "step", "screen")
		if err != nil || !got {
			t.Fatalf("check #%d: got=%v err=%v", i, got, err)
		}
	}
	if inner.stepCalls != 1 {
		t.Fatalf("StepComplete called %d times, want 1 (cached)", inner.stepCalls)
	}
	if _, err := c.StepComplete(ctx, "step", "other-screen"); err != nil {
		t.Fatalf("new check: %v", err)
	}
	if inner.stepCalls != 2 {
		t.Fatalf("new (step,screen) should invoke planner: calls=%d", inner.stepCalls)
	}

	hits, misses := c.Stats()
	if hits != 2 || misses != 2 {
		t.Fatalf("stats: hits=%d misses=%d, want 2/2", hits, misses)
	}
}

// DoD: planner errors are never cached.
func TestCachingPlannerDoesNotCacheErrors(t *testing.T) {
	ctx := context.Background()
	inner := &countingPlanner{err: errors.New("boom")}
	c := NewCachingPlanner(inner, 0)

	if _, err := c.StepComplete(ctx, "s", "scr"); err == nil {
		t.Fatal("expected error from inner")
	}
	inner.err = nil
	inner.result = true
	got, err := c.StepComplete(ctx, "s", "scr")
	if err != nil || !got {
		t.Fatalf("second call: got=%v err=%v", got, err)
	}
	if inner.stepCalls != 2 {
		t.Fatalf("errored check must not be cached: calls=%d, want 2", inner.stepCalls)
	}
}
