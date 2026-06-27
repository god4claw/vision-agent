package executor

import (
	"context"
	"image"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ActionType string

const (
	ActionNone  ActionType = "none"
	ActionClick ActionType = "click"
	ActionKey   ActionType = "key"
	ActionMove  ActionType = "move" // move the cursor to (X,Y) without clicking
	ActionDrag  ActionType = "drag" // press at (X,Y), drag to (X2,Y2), release
)

// Action is what the agent decides to do this tick. For a drag, (X,Y) is the
// press point and (X2,Y2) is the release point.
type Action struct {
	Type ActionType `json:"type"`
	X    int        `json:"x"`
	Y    int        `json:"y"`
	X2   int        `json:"x2"`
	Y2   int        `json:"y2"`
	Key  string     `json:"key"`
}

// Executor performs (or simulates) an action.
type Executor interface {
	Do(ctx context.Context, a Action) error
}

// DryRunExecutor logs the intended action but performs nothing real.
type DryRunExecutor struct {
	Log *slog.Logger
}

func (d DryRunExecutor) Do(_ context.Context, a Action) error {
	if d.Log != nil && a.Type != ActionNone {
		d.Log.Info("DRY-RUN", "type", string(a.Type), "x", a.X, "y", a.Y, "key", a.Key)
	}
	return nil
}

// RealExecutor performs real OS input (mouse/keyboard) via platform syscalls
// (see input_windows.go). It is only constructed when the operator passes
// -live, and is normally wrapped by a SafeExecutor that enforces guardrails.
// On non-Windows builds sendInput returns an error.
type RealExecutor struct{}

func (RealExecutor) Do(_ context.Context, a Action) error {
	return sendInput(a)
}

// Guardrails constrain a wrapped executor before any real input is performed.
type Guardrails struct {
	Bounds      image.Rectangle     // permitted click region; zero value => no bounds check
	NoGo        []image.Rectangle   // forbidden click regions (e.g. taskbar, close buttons)
	Allow       map[ActionType]bool // if non-empty, only these action types are forwarded
	MinInterval time.Duration       // minimum delay between forwarded actions
	Abort       func() bool         // kill-switch; while it returns true, actions are suppressed
	Log         *slog.Logger
}

// SafeExecutor wraps an executor and suppresses any action that violates the
// guardrails: kill-switch held, click out of bounds or inside a no-go zone, or
// an invalid key. Suppressed actions are counted and logged, never forwarded,
// and are not treated as errors. It also rate-limits forwarded actions.
type SafeExecutor struct {
	inner Executor
	g     Guardrails

	mu      sync.Mutex
	last    time.Time
	done    int
	blocked int
}

func NewSafeExecutor(inner Executor, g Guardrails) *SafeExecutor {
	return &SafeExecutor{inner: inner, g: g}
}

func (s *SafeExecutor) Do(ctx context.Context, a Action) error {
	if a.Type == ActionNone {
		return nil
	}
	if s.g.Abort != nil && s.g.Abort() {
		return s.suppress("kill-switch active", a)
	}
	if len(s.g.Allow) > 0 && !s.g.Allow[a.Type] {
		return s.suppress("action type not allowed", a)
	}
	switch a.Type {
	case ActionClick, ActionMove:
		if reason := s.checkPoint(a.X, a.Y); reason != "" {
			return s.suppress(reason, a)
		}
	case ActionDrag:
		// Both the press and release points must clear the guardrails.
		if reason := s.checkPoint(a.X, a.Y); reason != "" {
			return s.suppress(reason, a)
		}
		if reason := s.checkPoint(a.X2, a.Y2); reason != "" {
			return s.suppress(reason, a)
		}
	case ActionKey:
		if !ValidKey(a.Key) {
			return s.suppress("invalid key", a)
		}
	}

	if s.g.MinInterval > 0 {
		s.mu.Lock()
		wait := s.g.MinInterval - time.Since(s.last)
		s.mu.Unlock()
		if wait > 0 {
			time.Sleep(wait)
		}
	}

	err := s.inner.Do(ctx, a)
	s.mu.Lock()
	s.last = time.Now()
	s.done++
	s.mu.Unlock()
	return err
}

// checkPoint returns a suppression reason if (x,y) violates the positional
// guardrails (bounds, no-go zones), or "" if it is allowed.
func (s *SafeExecutor) checkPoint(x, y int) string {
	p := image.Pt(x, y)
	if !s.g.Bounds.Empty() && !p.In(s.g.Bounds) {
		return "position out of bounds"
	}
	for _, z := range s.g.NoGo {
		if p.In(z) {
			return "position in no-go zone"
		}
	}
	return ""
}

func (s *SafeExecutor) suppress(reason string, a Action) error {
	s.mu.Lock()
	s.blocked++
	s.mu.Unlock()
	if s.g.Log != nil {
		s.g.Log.Warn("action suppressed by guardrail", "reason", reason,
			"type", string(a.Type), "x", a.X, "y", a.Y, "key", a.Key)
	}
	return nil
}

// Stats returns counts of forwarded and suppressed actions.
func (s *SafeExecutor) Stats() (done, blocked int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.done, s.blocked
}

// keyModifiers are the allowed modifier tokens in a key combination.
var keyModifiers = map[string]bool{
	"ctrl": true, "control": true, "alt": true, "option": true,
	"shift": true, "win": true, "super": true, "cmd": true, "meta": true,
}

// keyNames are the allowed named (non-printable) keys.
var keyNames = map[string]bool{
	"enter": true, "return": true, "tab": true, "esc": true, "escape": true,
	"space": true, "backspace": true, "delete": true, "del": true,
	"insert": true, "ins": true, "up": true, "down": true, "left": true,
	"right": true, "home": true, "end": true, "pageup": true, "pagedown": true,
	"pgup": true, "pgdn": true, "capslock": true,
}

// ValidKey reports whether s is a real, pressable key or key combination such
// as "a", "5", "Enter", "Ctrl+L", or "Ctrl+Shift+P" -- and NOT free text like a
// sentence accidentally emitted by an LLM. A combination is modifiers joined by
// '+' with exactly one terminal key (single printable char, named key, or
// F1..F24). This is the whitelist gate applied before any real input.
func ValidKey(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	hasMain := false
	for _, part := range strings.Split(s, "+") {
		p := strings.ToLower(strings.TrimSpace(part))
		switch {
		case p == "":
			return false
		case keyModifiers[p]:
			continue
		case isKeyToken(p):
			hasMain = true
		default:
			return false
		}
	}
	return hasMain
}

func isKeyToken(p string) bool {
	if keyNames[p] {
		return true
	}
	if len([]rune(p)) == 1 {
		return true
	}
	return isFunctionKey(p)
}

func isFunctionKey(p string) bool {
	if len(p) < 2 || p[0] != 'f' {
		return false
	}
	n, err := strconv.Atoi(p[1:])
	return err == nil && n >= 1 && n <= 24
}
