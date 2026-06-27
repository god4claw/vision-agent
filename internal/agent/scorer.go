package agent

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"io"
	"sync"
)

// CachingScorer memoizes a RewardScorer's verdicts so an identical
// (goal, before, after) transition is judged by the underlying scorer (e.g. an
// LLM) at most once. Static or repeating screens then cost no LLM calls.
// Errors are never cached. The cache is bounded with FIFO eviction.
type CachingScorer struct {
	inner RewardScorer
	max   int

	mu     sync.Mutex
	cache  map[string]bool
	order  []string
	hits   int
	misses int
}

// NewCachingScorer wraps inner with a bounded verdict cache. max <= 0 means
// unbounded.
func NewCachingScorer(inner RewardScorer, max int) *CachingScorer {
	return &CachingScorer{inner: inner, max: max, cache: make(map[string]bool)}
}

func (c *CachingScorer) Score(ctx context.Context, goal, before, after string) (bool, error) {
	key := scoreKey(goal, before, after)

	c.mu.Lock()
	if v, ok := c.cache[key]; ok {
		c.hits++
		c.mu.Unlock()
		return v, nil
	}
	c.misses++
	c.mu.Unlock()

	v, err := c.inner.Score(ctx, goal, before, after)
	if err != nil {
		return false, err // do not cache failures
	}

	c.mu.Lock()
	c.put(key, v)
	c.mu.Unlock()
	return v, nil
}

// Stats returns cache hit and miss counts for observability.
func (c *CachingScorer) Stats() (hits, misses int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses
}

// put inserts key=v and evicts the oldest entries while over capacity. Caller
// must hold c.mu.
func (c *CachingScorer) put(key string, v bool) {
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

// scoreKey hashes the transition into a compact, collision-resistant key so the
// map does not retain full (possibly large) screen texts.
func scoreKey(goal, before, after string) string {
	h := sha1.New()
	io.WriteString(h, goal)
	h.Write([]byte{0})
	io.WriteString(h, before)
	h.Write([]byte{0})
	io.WriteString(h, after)
	return hex.EncodeToString(h.Sum(nil))
}
