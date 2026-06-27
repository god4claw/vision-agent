package agent

import (
	"context"
	"sync"
)

// CachingPlanner memoizes a Planner's StepComplete verdicts so an identical
// (step, screen) check is judged by the underlying planner (e.g. an LLM) at
// most once. Plan is delegated unchanged (it runs once per goal). Errors are
// never cached. The cache is bounded with FIFO eviction.
type CachingPlanner struct {
	inner Planner
	max   int

	mu     sync.Mutex
	cache  map[string]bool
	order  []string
	hits   int
	misses int
}

// NewCachingPlanner wraps inner with a bounded StepComplete cache. max <= 0
// means unbounded.
func NewCachingPlanner(inner Planner, max int) *CachingPlanner {
	return &CachingPlanner{inner: inner, max: max, cache: make(map[string]bool)}
}

func (c *CachingPlanner) Plan(ctx context.Context, goal string) ([]string, error) {
	return c.inner.Plan(ctx, goal)
}

func (c *CachingPlanner) StepComplete(ctx context.Context, step, screen string) (bool, error) {
	key := scoreKey(step, screen, "")

	c.mu.Lock()
	if v, ok := c.cache[key]; ok {
		c.hits++
		c.mu.Unlock()
		return v, nil
	}
	c.misses++
	c.mu.Unlock()

	v, err := c.inner.StepComplete(ctx, step, screen)
	if err != nil {
		return false, err // do not cache failures
	}

	c.mu.Lock()
	c.put(key, v)
	c.mu.Unlock()
	return v, nil
}

// Stats returns cache hit and miss counts for observability.
func (c *CachingPlanner) Stats() (hits, misses int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses
}

// put inserts key=v and evicts the oldest entries while over capacity. Caller
// must hold c.mu.
func (c *CachingPlanner) put(key string, v bool) {
	if _, ok := c.cache[key]; ok {
		return
	}
	c.cache[key] = v
	c.order = append(c.order, key)
	for c.max > 0 && len(c.order) > c.max {
		oldest := c.order[0]
		c.order = c.order[1:]
		delete(c.cache, oldest)
	}
}
