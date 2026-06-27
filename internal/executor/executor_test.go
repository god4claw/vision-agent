package executor

import (
	"context"
	"image"
	"testing"
)

// recordExec records every action forwarded to it.
type recordExec struct{ calls []Action }

func (r *recordExec) Do(_ context.Context, a Action) error {
	r.calls = append(r.calls, a)
	return nil
}

// DoD: the SafeExecutor forwards only actions that pass every guardrail and
// suppresses (does not forward, does not error) the rest.
func TestSafeExecutorGuardrails(t *testing.T) {
	ctx := context.Background()
	rec := &recordExec{}
	s := NewSafeExecutor(rec, Guardrails{
		Bounds: image.Rect(0, 0, 100, 100),
		NoGo:   []image.Rectangle{image.Rect(40, 40, 60, 60)},
	})

	must := func(a Action) {
		if err := s.Do(ctx, a); err != nil {
			t.Fatalf("Do(%+v): %v", a, err)
		}
	}
	must(Action{Type: ActionClick, X: 10, Y: 10})     // valid -> forwarded
	must(Action{Type: ActionClick, X: 200, Y: 10})    // out of bounds -> blocked
	must(Action{Type: ActionClick, X: 50, Y: 50})     // no-go zone -> blocked
	must(Action{Type: ActionKey, Key: "hello world"}) // invalid key -> blocked
	must(Action{Type: ActionKey, Key: "Ctrl+L"})      // valid -> forwarded
	must(Action{Type: ActionNone})                    // ignored

	if len(rec.calls) != 2 {
		t.Fatalf("forwarded %d actions, want 2: %+v", len(rec.calls), rec.calls)
	}
	done, blocked := s.Stats()
	if done != 2 || blocked != 3 {
		t.Fatalf("stats done=%d blocked=%d, want 2/3", done, blocked)
	}
}

// DoD: positional guardrails (bounds, no-go) apply to ActionMove just like
// clicks, so the cursor can never be parked out of bounds or in a forbidden
// zone.
func TestSafeExecutorMoveGuardrails(t *testing.T) {
	ctx := context.Background()
	rec := &recordExec{}
	s := NewSafeExecutor(rec, Guardrails{
		Bounds: image.Rect(0, 0, 100, 100),
		NoGo:   []image.Rectangle{image.Rect(40, 40, 60, 60)},
	})
	_ = s.Do(ctx, Action{Type: ActionMove, X: 10, Y: 10})  // valid -> forwarded
	_ = s.Do(ctx, Action{Type: ActionMove, X: 200, Y: 10}) // out of bounds -> blocked
	_ = s.Do(ctx, Action{Type: ActionMove, X: 50, Y: 50})  // no-go -> blocked
	if len(rec.calls) != 1 {
		t.Fatalf("forwarded %d moves, want 1: %+v", len(rec.calls), rec.calls)
	}
	done, blocked := s.Stats()
	if done != 1 || blocked != 2 {
		t.Fatalf("stats done=%d blocked=%d, want 1/2", done, blocked)
	}
}

// DoD: a drag is forwarded only when BOTH its press and release points clear
// the positional guardrails.
func TestSafeExecutorDragGuardrails(t *testing.T) {
	ctx := context.Background()
	rec := &recordExec{}
	s := NewSafeExecutor(rec, Guardrails{
		Bounds: image.Rect(0, 0, 100, 100),
		NoGo:   []image.Rectangle{image.Rect(40, 40, 60, 60)},
	})
	_ = s.Do(ctx, Action{Type: ActionDrag, X: 10, Y: 10, X2: 80, Y2: 80})  // both clear -> forwarded
	_ = s.Do(ctx, Action{Type: ActionDrag, X: 10, Y: 10, X2: 50, Y2: 50})  // dest no-go -> blocked
	_ = s.Do(ctx, Action{Type: ActionDrag, X: 200, Y: 10, X2: 80, Y2: 80}) // src oob -> blocked
	if len(rec.calls) != 1 {
		t.Fatalf("forwarded %d drags, want 1: %+v", len(rec.calls), rec.calls)
	}
	done, blocked := s.Stats()
	if done != 1 || blocked != 2 {
		t.Fatalf("stats done=%d blocked=%d, want 1/2", done, blocked)
	}
}

// DoD: with an action-type allowlist (e.g. move-only), disallowed types are
// suppressed even when otherwise valid, so a live run can be restricted to safe
// actions.
func TestSafeExecutorAllowlist(t *testing.T) {
	ctx := context.Background()
	rec := &recordExec{}
	s := NewSafeExecutor(rec, Guardrails{Allow: map[ActionType]bool{ActionMove: true}})

	_ = s.Do(ctx, Action{Type: ActionMove, X: 10, Y: 10})  // allowed -> forwarded
	_ = s.Do(ctx, Action{Type: ActionClick, X: 10, Y: 10}) // not allowed -> blocked
	_ = s.Do(ctx, Action{Type: ActionKey, Key: "a"})       // not allowed -> blocked
	if len(rec.calls) != 1 || rec.calls[0].Type != ActionMove {
		t.Fatalf("allowlist failed, forwarded: %+v", rec.calls)
	}
	done, blocked := s.Stats()
	if done != 1 || blocked != 2 {
		t.Fatalf("stats done=%d blocked=%d, want 1/2", done, blocked)
	}
}

// DoD: while the kill-switch is engaged no action is forwarded; once released
// actions pass again.
func TestSafeExecutorKillSwitch(t *testing.T) {
	ctx := context.Background()
	rec := &recordExec{}
	abort := true
	s := NewSafeExecutor(rec, Guardrails{Abort: func() bool { return abort }})

	_ = s.Do(ctx, Action{Type: ActionClick, X: 1, Y: 1})
	if len(rec.calls) != 0 {
		t.Fatal("kill-switch should suppress all actions")
	}
	abort = false
	_ = s.Do(ctx, Action{Type: ActionClick, X: 1, Y: 1})
	if len(rec.calls) != 1 {
		t.Fatal("action should pass once kill-switch is released")
	}
}

func TestValidKey(t *testing.T) {
	valid := []string{
		"a", "Z", "5", "/", ".",
		"Enter", "enter", "Tab", "Esc", "Space", "Backspace",
		"Up", "Down", "Left", "Right", "Home", "End", "PageUp",
		"F1", "F5", "F12", "f24",
		"Ctrl+L", "ctrl+l", "Alt+F4", "Ctrl+Shift+P", "Win+R",
	}
	for _, s := range valid {
		if !ValidKey(s) {
			t.Errorf("ValidKey(%q) = false, want true", s)
		}
	}

	invalid := []string{
		"",
		"   ",
		"Extra usage balance $0.00", // LLM free text
		"hello",                     // multi-char word
		"Ctrl",                      // modifier only, no terminal key
		"Ctrl+Shift",                // modifiers only
		"Ctrl+",                     // empty terminal
		"+a",                        // empty leading part
		"foo+bar",                   // unknown tokens
		"F0", "F25",                 // out of range function keys
		"open the File menu",
	}
	for _, s := range invalid {
		if ValidKey(s) {
			t.Errorf("ValidKey(%q) = true, want false", s)
		}
	}
}
