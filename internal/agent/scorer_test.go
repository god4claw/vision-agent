package agent

import (
	"context"
	"errors"
	"testing"
)

// countingScorer records how many times the underlying judge was invoked.
type countingScorer struct {
	calls  int
	result bool
	err    error
}

func (c *countingScorer) Score(context.Context, string, string, string) (bool, error) {
	c.calls++
	return c.result, c.err
}

// DoD: an identical (goal, before, after) transition hits the cache and does
// not re-invoke the underlying judge; a new transition does.
func TestCachingScorerMemoizes(t *testing.T) {
	ctx := context.Background()
	inner := &countingScorer{result: true}
	c := NewCachingScorer(inner, 0)

	for i := 0; i < 3; i++ {
		got, err := c.Score(ctx, "goal", "before", "after")
		if err != nil || !got {
			t.Fatalf("score #%d: got=%v err=%v", i, got, err)
		}
	}
	if inner.calls != 1 {
		t.Fatalf("inner judge called %d times, want 1 (cached)", inner.calls)
	}

	if _, err := c.Score(ctx, "goal", "before", "different-after"); err != nil {
		t.Fatalf("score new transition: %v", err)
	}
	if inner.calls != 2 {
		t.Fatalf("new transition should invoke judge: calls=%d", inner.calls)
	}

	hits, misses := c.Stats()
	if hits != 2 || misses != 2 {
		t.Fatalf("stats: hits=%d misses=%d, want 2/2", hits, misses)
	}
}

// DoD: judge errors are never cached, so a later success is still consulted.
func TestCachingScorerDoesNotCacheErrors(t *testing.T) {
	ctx := context.Background()
	inner := &countingScorer{err: errors.New("boom")}
	c := NewCachingScorer(inner, 0)

	if _, err := c.Score(ctx, "g", "b", "a"); err == nil {
		t.Fatal("expected error from inner")
	}
	inner.err = nil
	inner.result = true
	got, err := c.Score(ctx, "g", "b", "a")
	if err != nil || !got {
		t.Fatalf("second call: got=%v err=%v", got, err)
	}
	if inner.calls != 2 {
		t.Fatalf("errored call must not be cached: calls=%d, want 2", inner.calls)
	}
}

// DoD: the cache is bounded; once evicted, a transition is judged again.
func TestCachingScorerBounded(t *testing.T) {
	ctx := context.Background()
	inner := &countingScorer{result: true}
	c := NewCachingScorer(inner, 2)

	_, _ = c.Score(ctx, "g", "s", "a") // key A
	_, _ = c.Score(ctx, "g", "s", "b") // key B
	_, _ = c.Score(ctx, "g", "s", "c") // key C -> evicts A
	if inner.calls != 3 {
		t.Fatalf("setup calls=%d, want 3", inner.calls)
	}
	// A was evicted -> judged again.
	_, _ = c.Score(ctx, "g", "s", "a")
	if inner.calls != 4 {
		t.Fatalf("evicted key should re-invoke judge: calls=%d, want 4", inner.calls)
	}
	// C is still cached -> no new call.
	_, _ = c.Score(ctx, "g", "s", "c")
	if inner.calls != 4 {
		t.Fatalf("cached key should not re-invoke judge: calls=%d, want 4", inner.calls)
	}
}
