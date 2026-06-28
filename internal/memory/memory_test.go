package memory

import (
	"context"
	"testing"

	"visionagent/internal/embed"
	"visionagent/internal/executor"
)

func TestStoreAddNearest(t *testing.T) {
	ctx := context.Background()
	s, err := NewStore(embed.LocalEmbedder{}, 0)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	episodes := []Episode{
		{ID: "a", Stage: "s1", StateText: "open the file menu", Action: executor.Action{Type: executor.ActionClick}, Success: true},
		{ID: "b", Stage: "s1", StateText: "save the document now", Action: executor.Action{Type: executor.ActionKey, Key: "Ctrl+S"}, Success: true},
	}
	for _, ep := range episodes {
		if err := s.Add(ctx, ep); err != nil {
			t.Fatalf("Add %s: %v", ep.ID, err)
		}
	}
	if got := s.Len(); got != 2 {
		t.Fatalf("Len = %d, want 2", got)
	}

	near, err := s.Nearest(ctx, "open the file menu", 1)
	if err != nil {
		t.Fatalf("Nearest: %v", err)
	}
	if len(near) != 1 {
		t.Fatalf("Nearest returned %d episodes, want 1", len(near))
	}
	if near[0].ID != "a" {
		t.Fatalf("Nearest matched %q, want %q", near[0].ID, "a")
	}
}

func TestStoreBoundedEviction(t *testing.T) {
	ctx := context.Background()
	const max = 3
	s, err := NewStore(embed.LocalEmbedder{}, max)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	ids := []string{"e0", "e1", "e2", "e3", "e4"}
	for i, id := range ids {
		ep := Episode{ID: id, Stage: "s", StateText: "state text " + id, Action: executor.Action{Type: executor.ActionClick, X: i}}
		if err := s.Add(ctx, ep); err != nil {
			t.Fatalf("Add %s: %v", id, err)
		}
	}

	if got := s.Len(); got != max {
		t.Fatalf("Len = %d, want bounded to %d", got, max)
	}

	// Oldest entries (e0, e1) must have been evicted; e4 (newest) must remain.
	near, err := s.Nearest(ctx, "state text e4", max)
	if err != nil {
		t.Fatalf("Nearest: %v", err)
	}
	got := map[string]bool{}
	for _, ep := range near {
		got[ep.ID] = true
	}
	if got["e0"] || got["e1"] {
		t.Fatalf("evicted episodes still present: %v", got)
	}
	if !got["e4"] {
		t.Fatalf("newest episode e4 missing after eviction: %v", got)
	}
}

func TestPersistentStoreRoundTrip(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	// First store: persist an episode, then drop the handle.
	s1, err := NewPersistentStore(embed.LocalEmbedder{}, 0, dir)
	if err != nil {
		t.Fatalf("NewPersistentStore (1): %v", err)
	}
	want := Episode{
		ID:        "persisted",
		Stage:     "s1",
		StateText: "open the settings panel",
		Action:    executor.Action{Type: executor.ActionClick, X: 10, Y: 20},
		Variables: map[string]string{"k": "v"},
		Success:   true,
	}
	if err := s1.Add(ctx, want); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Second store on the SAME dir simulates a process restart.
	s2, err := NewPersistentStore(embed.LocalEmbedder{}, 0, dir)
	if err != nil {
		t.Fatalf("NewPersistentStore (2): %v", err)
	}
	if got := s2.Len(); got != 1 {
		t.Fatalf("reloaded Len = %d, want 1", got)
	}
	near, err := s2.Nearest(ctx, "open the settings panel", 1)
	if err != nil {
		t.Fatalf("Nearest: %v", err)
	}
	if len(near) != 1 {
		t.Fatalf("Nearest returned %d episodes, want 1", len(near))
	}
	if near[0].ID != want.ID {
		t.Fatalf("persisted episode ID = %q, want %q", near[0].ID, want.ID)
	}
	if near[0].StateText != want.StateText {
		t.Fatalf("persisted StateText = %q, want %q", near[0].StateText, want.StateText)
	}
	if near[0].Action.Type != want.Action.Type || near[0].Action.X != want.Action.X {
		t.Fatalf("persisted Action = %+v, want %+v", near[0].Action, want.Action)
	}
	if !near[0].Success {
		t.Fatalf("persisted Success = false, want true")
	}
}
