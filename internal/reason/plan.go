package reason

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const defaultMaxSteps = 8

// PlanDecision is the strict JSON contract for goal decomposition.
type PlanDecision struct {
	Steps []string `json:"steps"`
}

// OllamaPlanner decomposes a goal into ordered sub-steps and judges when a
// sub-step is complete, using a local Ollama model. Its method set satisfies
// the agent's Planner contract (structural interface).
type OllamaPlanner struct {
	URL      string
	Model    string
	MaxSteps int
	client   *http.Client
}

func NewOllamaPlanner(url, model string) *OllamaPlanner {
	return &OllamaPlanner{
		URL:      url,
		Model:    model,
		MaxSteps: defaultMaxSteps,
		client:   &http.Client{Timeout: 60 * time.Second},
	}
}

func (o *OllamaPlanner) maxSteps() int {
	if o.MaxSteps > 0 {
		return o.MaxSteps
	}
	return defaultMaxSteps
}

// Plan breaks goal into a short ordered list of concrete UI sub-steps.
func (o *OllamaPlanner) Plan(ctx context.Context, goal string) ([]string, error) {
	raw, err := ollamaGenerateJSON(ctx, o.client, o.URL, o.Model, buildPlanPrompt(goal, o.maxSteps()))
	if err != nil {
		return nil, err
	}
	var d PlanDecision
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &d); err != nil {
		return nil, fmt.Errorf("parse plan json: %w (raw=%q)", err, raw)
	}
	steps := make([]string, 0, len(d.Steps))
	for _, s := range d.Steps {
		if s = strings.TrimSpace(s); s != "" {
			steps = append(steps, s)
		}
		if len(steps) >= o.maxSteps() {
			break
		}
	}
	return steps, nil
}

// StepComplete reports whether the given sub-step is already satisfied by the
// current screen text.
func (o *OllamaPlanner) StepComplete(ctx context.Context, step, screen string) (bool, error) {
	raw, err := ollamaGenerateJSON(ctx, o.client, o.URL, o.Model, buildStepCompletePrompt(step, screen))
	if err != nil {
		return false, err
	}
	var d struct {
		Complete bool `json:"complete"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &d); err != nil {
		return false, fmt.Errorf("parse step-complete json: %w (raw=%q)", err, raw)
	}
	return d.Complete, nil
}

func buildPlanPrompt(goal string, maxSteps int) string {
	var sb strings.Builder
	sb.WriteString("You break a high-level computer task into a short ordered list of concrete UI steps.\n")
	sb.WriteString("Goal: ")
	sb.WriteString(strings.TrimSpace(goal))
	sb.WriteString("\n\nReply with ONLY a JSON object: {\"steps\": [\"step 1\", \"step 2\", ...]}\n")
	sb.WriteString(fmt.Sprintf("Use at most %d steps. Each step is one concrete action phrased imperatively, e.g. \"click the File menu\".", maxSteps))
	return sb.String()
}

func buildStepCompletePrompt(step, screen string) string {
	var sb strings.Builder
	sb.WriteString("You judge whether a UI step is already done, based on the current screen text.\n")
	sb.WriteString("Step: ")
	sb.WriteString(strings.TrimSpace(step))
	sb.WriteString("\n\nCurrent screen text:\n")
	sb.WriteString(truncate(screen, 800))
	sb.WriteString("\n\nReply with ONLY a JSON object: {\"complete\": true or false}\n")
	sb.WriteString("Set \"complete\" to true only if the screen shows the step has already been accomplished.")
	return sb.String()
}
