// Package reason turns a perceived screen state plus a goal into the single
// next action, using a local LLM (Ollama) as the policy. It is deliberately
// transport-light: the model only ever picks an element by index and an action
// verb, so the agent (not the model) owns the real pixel coordinates.
package reason

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"visionagent/internal/executor"
	"visionagent/internal/memory"
	"visionagent/internal/perception"
)

// Reasoner chooses the next action for the given goal and perceived state.
// It returns the action, a short human-readable rationale, and an error.
type Reasoner interface {
	Decide(ctx context.Context, goal string, res perception.Result, recent []memory.Episode) (executor.Action, string, error)
}

// Decision is the strict JSON contract the model must emit.
type Decision struct {
	Action string `json:"action"` // "click" | "move" | "drag" | "key" | "none"
	Target int    `json:"target"` // index into the presented element list (-1 if N/A)
	To     int    `json:"to"`     // drag destination index into the element list (-1 if N/A)
	Key    string `json:"key"`    // keystrokes when Action == "key"
	Reason string `json:"reason"` // short rationale for logging
}

const defaultMaxElements = 40

// OllamaReasoner calls a local Ollama chat/generate model to decide actions.
type OllamaReasoner struct {
	URL         string
	Model       string
	MaxElements int // cap on UI elements shown to the model (0 => default)
	client      *http.Client
}

// NewOllamaReasoner builds a reasoner against a local Ollama server.
// A generous timeout is used because local generation can be slow.
func NewOllamaReasoner(url, model string) *OllamaReasoner {
	return &OllamaReasoner{
		URL:         url,
		Model:       model,
		MaxElements: defaultMaxElements,
		client:      &http.Client{Timeout: 60 * time.Second},
	}
}

func (o *OllamaReasoner) maxElements() int {
	if o.MaxElements > 0 {
		return o.MaxElements
	}
	return defaultMaxElements
}

func (o *OllamaReasoner) Decide(ctx context.Context, goal string, res perception.Result, recent []memory.Episode) (executor.Action, string, error) {
	shown := selectElements(res.Boxes, o.maxElements())
	prompt := buildPrompt(goal, shown, recent)

	raw, err := ollamaGenerateJSON(ctx, o.client, o.URL, o.Model, prompt)
	if err != nil {
		return executor.Action{}, "", err
	}
	d, err := ParseDecision(raw)
	if err != nil {
		return executor.Action{}, "", err
	}
	act, err := d.ToAction(shown)
	if err != nil {
		return executor.Action{}, d.Reason, err
	}
	return act, d.Reason, nil
}

// ollamaGenerateJSON posts a prompt to Ollama's /api/generate with JSON mode
// forced and returns the model's raw response string (itself JSON). Shared by
// the action reasoner and the goal-progress scorer.
func ollamaGenerateJSON(ctx context.Context, client *http.Client, url, model, prompt string) (string, error) {
	reqBody, err := json.Marshal(map[string]any{
		"model":   model,
		"prompt":  prompt,
		"stream":  false,
		"format":  "json",
		"options": map[string]any{"temperature": 0.1},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/api/generate", bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama generate: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama generate status %d (model %q pulled?)", resp.StatusCode, model)
	}
	var out struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode ollama envelope: %w", err)
	}
	return out.Response, nil
}

// ParseDecision unmarshals the model's JSON string into a Decision.
func ParseDecision(raw string) (Decision, error) {
	var d Decision
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &d); err != nil {
		return Decision{}, fmt.Errorf("parse decision json: %w (raw=%q)", err, raw)
	}
	return d, nil
}

// ToAction maps a model Decision onto a concrete executor.Action, resolving the
// target index against the element list that was shown to the model. The model
// never invents coordinates; it only references an element by index.
func (d Decision) ToAction(shown []perception.Box) (executor.Action, error) {
	switch strings.ToLower(strings.TrimSpace(d.Action)) {
	case "", "none", "wait", "noop":
		return executor.Action{Type: executor.ActionNone}, nil
	case "click":
		if d.Target < 0 || d.Target >= len(shown) {
			return executor.Action{}, fmt.Errorf("click target index %d out of range [0,%d)", d.Target, len(shown))
		}
		b := shown[d.Target]
		return executor.Action{Type: executor.ActionClick, X: b.X + b.W/2, Y: b.Y + b.H/2}, nil
	case "move", "hover":
		if d.Target < 0 || d.Target >= len(shown) {
			return executor.Action{}, fmt.Errorf("move target index %d out of range [0,%d)", d.Target, len(shown))
		}
		b := shown[d.Target]
		return executor.Action{Type: executor.ActionMove, X: b.X + b.W/2, Y: b.Y + b.H/2}, nil
	case "drag":
		if d.Target < 0 || d.Target >= len(shown) {
			return executor.Action{}, fmt.Errorf("drag source index %d out of range [0,%d)", d.Target, len(shown))
		}
		if d.To < 0 || d.To >= len(shown) {
			return executor.Action{}, fmt.Errorf("drag destination index %d out of range [0,%d)", d.To, len(shown))
		}
		src, dst := shown[d.Target], shown[d.To]
		return executor.Action{
			Type: executor.ActionDrag,
			X:    src.X + src.W/2, Y: src.Y + src.H/2,
			X2: dst.X + dst.W/2, Y2: dst.Y + dst.H/2,
		}, nil
	case "key", "type":
		if !executor.ValidKey(d.Key) {
			return executor.Action{}, fmt.Errorf("invalid key %q (not a pressable key/combo)", d.Key)
		}
		return executor.Action{Type: executor.ActionKey, Key: d.Key}, nil
	default:
		return executor.Action{}, fmt.Errorf("unknown action %q", d.Action)
	}
}

// selectElements filters out empty-text boxes (not useful to a text-driven
// policy) and caps the list so the prompt stays small on busy screens. The
// returned slice is exactly what indices in the prompt refer to.
func selectElements(boxes []perception.Box, max int) []perception.Box {
	shown := make([]perception.Box, 0, len(boxes))
	for _, b := range boxes {
		if strings.TrimSpace(b.Text) == "" {
			continue
		}
		shown = append(shown, b)
		if max > 0 && len(shown) >= max {
			break
		}
	}
	return shown
}

// buildPrompt renders the goal, the visible elements (indexed, with click
// coordinates), and recent successful actions into a compact instruction that
// asks for a single strict-JSON decision.
func buildPrompt(goal string, shown []perception.Box, recent []memory.Episode) string {
	var sb strings.Builder
	sb.WriteString("You control a computer by looking at the screen and choosing ONE next action.\n")
	if g := strings.TrimSpace(goal); g != "" {
		sb.WriteString("Goal: ")
		sb.WriteString(g)
		sb.WriteByte('\n')
	}
	sb.WriteString("\nVisible UI elements (index: \"text\" @ x,y):\n")
	if len(shown) == 0 {
		sb.WriteString("(no text elements detected)\n")
	}
	for i, b := range shown {
		sb.WriteString(fmt.Sprintf("%d: %q @ %d,%d\n", i, b.Text, b.X+b.W/2, b.Y+b.H/2))
	}
	if len(recent) > 0 {
		sb.WriteString("\nRecent actions that worked before (for reference):\n")
		for _, e := range recent {
			if !e.Success || e.Action.Type == executor.ActionNone {
				continue
			}
			st := strings.TrimSpace(e.StateText)
			if len(st) > 60 {
				st = st[:60]
			}
			sb.WriteString(fmt.Sprintf("- on %q did %s %s\n", st, e.Action.Type, e.Action.Key))
		}
	}
	sb.WriteString("\nReply with ONLY a JSON object (no markdown, no prose) with these fields:\n")
	sb.WriteString("  \"action\": exactly one word, one of: click, move, drag, key, none\n")
	sb.WriteString("  \"target\": the index of the element to click/move to, or the drag source; use -1 otherwise\n")
	sb.WriteString("  \"to\": the drag destination element index when action is \"drag\"; otherwise -1\n")
	sb.WriteString("  \"key\": the keys to press when action is \"key\"; otherwise \"\"\n")
	sb.WriteString("  \"reason\": a short phrase, max 12 words\n")
	sb.WriteString("\nExample reply: {\"action\":\"click\",\"target\":0,\"key\":\"\",\"reason\":\"open the file menu\"}\n")
	sb.WriteString("\nIMPORTANT: \"click\" opens/activates/selects an element; \"move\" only positions the cursor and does NOT open or activate anything. To open, press, select, or click something, you MUST use \"click\".\n")
	sb.WriteString("\nNow choose the single best action toward the goal. If nothing helps, use action \"none\".")
	return sb.String()
}

// ScoreDecision is the strict JSON contract for the goal-progress judge.
type ScoreDecision struct {
	Progress bool   `json:"progress"`
	Reason   string `json:"reason"`
}

// OllamaScorer judges whether an action moved the screen closer to the goal,
// using a local Ollama model. Its method set satisfies the agent's reward
// scorer contract (structural interface), enabling goal-aware reward.
type OllamaScorer struct {
	URL    string
	Model  string
	client *http.Client
}

func NewOllamaScorer(url, model string) *OllamaScorer {
	return &OllamaScorer{URL: url, Model: model, client: &http.Client{Timeout: 60 * time.Second}}
}

// Score returns true when the AFTER screen is closer to achieving the goal than
// the BEFORE screen, as judged by the model.
func (o *OllamaScorer) Score(ctx context.Context, goal, before, after string) (bool, error) {
	raw, err := ollamaGenerateJSON(ctx, o.client, o.URL, o.Model, buildScorePrompt(goal, before, after))
	if err != nil {
		return false, err
	}
	var d ScoreDecision
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &d); err != nil {
		return false, fmt.Errorf("parse score json: %w (raw=%q)", err, raw)
	}
	return d.Progress, nil
}

func buildScorePrompt(goal, before, after string) string {
	var sb strings.Builder
	sb.WriteString("You judge whether a UI action made progress toward a goal.\n")
	sb.WriteString("Goal: ")
	sb.WriteString(strings.TrimSpace(goal))
	sb.WriteString("\n\nScreen text BEFORE the action:\n")
	sb.WriteString(truncate(before, 800))
	sb.WriteString("\n\nScreen text AFTER the action:\n")
	sb.WriteString(truncate(after, 800))
	sb.WriteString("\n\nReply with ONLY a JSON object: {\"progress\": true or false, \"reason\": \"<max 12 words>\"}\n")
	sb.WriteString("Set \"progress\" to true ONLY if the AFTER screen is closer to achieving the goal than BEFORE.")
	return sb.String()
}

func truncate(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) > max {
		return s[:max]
	}
	return s
}
