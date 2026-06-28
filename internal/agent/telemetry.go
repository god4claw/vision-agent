package agent

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
)

// Telemetry accumulates progress/regress/stall counts over a run so the
// operator can track how effectively the agent is advancing toward its goal.
// Each scored (non-no-op) action falls into exactly one bucket:
//
//   - progress: the reward judged the action moved closer to the goal.
//   - regress:  no progress, but the screen changed (the agent did something
//     that did not help, i.e. moved the wrong way).
//   - stall:    no progress and the screen did not change (the action had no
//     visible effect).
//
// The derived rates and Net (= progress - regress) summarise the trend.
type Telemetry struct {
	mu       sync.Mutex
	scored   int
	progress int
	regress  int
	stall    int
}

// Record classifies one scored action. progress is the reward verdict (closer
// to the goal); screenChanged indicates the screen text differed afterward.
func (t *Telemetry) Record(progress, screenChanged bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.scored++
	switch {
	case progress:
		t.progress++
	case screenChanged:
		t.regress++
	default:
		t.stall++
	}
}

// Snapshot is an immutable view of the accumulated telemetry with derived rates.
type Snapshot struct {
	Scored       int     `json:"scored"`
	Progress     int     `json:"progress"`
	Regress      int     `json:"regress"`
	Stall        int     `json:"stall"`
	ProgressRate float64 `json:"progress_rate"` // Progress / Scored
	RegressRate  float64 `json:"regress_rate"`  // Regress / Scored
	StallRate    float64 `json:"stall_rate"`    // Stall / Scored
	Net          int     `json:"net"`           // Progress - Regress (net forward motion)

	// Efficiency is the share of *effective* (screen-changing) actions that
	// were progress: Progress / (Progress + Regress). It ignores stalls, so it
	// measures decision quality once the agent actually did something. High is
	// good (1.0 = every effective action helped); 0 when no effective actions.
	Efficiency float64 `json:"efficiency"`
}

// snapshotColumns is the canonical column order shared by the CSV header and
// row so they always line up.
var snapshotColumns = []string{
	"scored", "progress", "regress", "stall",
	"progress_rate", "regress_rate", "stall_rate", "efficiency", "net",
}

// JSON renders the snapshot as indented JSON (with a trailing newline).
func (s Snapshot) JSON() []byte {
	b, _ := json.MarshalIndent(s, "", "  ")
	return append(b, '\n')
}

// CSV renders the snapshot as a two-line CSV document: a header row followed by
// a single data row, in snapshotColumns order. Rates use 4 decimal places.
func (s Snapshot) CSV() []byte {
	rate := func(f float64) string { return strconv.FormatFloat(f, 'f', 4, 64) }
	row := []string{
		strconv.Itoa(s.Scored),
		strconv.Itoa(s.Progress),
		strconv.Itoa(s.Regress),
		strconv.Itoa(s.Stall),
		rate(s.ProgressRate),
		rate(s.RegressRate),
		rate(s.StallRate),
		rate(s.Efficiency),
		strconv.Itoa(s.Net),
	}
	var b strings.Builder
	b.WriteString(strings.Join(snapshotColumns, ","))
	b.WriteByte('\n')
	b.WriteString(strings.Join(row, ","))
	b.WriteByte('\n')
	return []byte(b.String())
}

// Marshal renders the snapshot in the format implied by the file extension of
// path (".csv" -> CSV, ".json" -> JSON). Unknown extensions error.
func (s Snapshot) Marshal(path string) ([]byte, error) {
	switch {
	case strings.HasSuffix(strings.ToLower(path), ".csv"):
		return s.CSV(), nil
	case strings.HasSuffix(strings.ToLower(path), ".json"):
		return s.JSON(), nil
	default:
		return nil, fmt.Errorf("telemetry: unsupported export extension for %q (want .csv or .json)", path)
	}
}

// Snapshot returns the current counts and rates.
func (t *Telemetry) Snapshot() Snapshot {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := Snapshot{
		Scored:   t.scored,
		Progress: t.progress,
		Regress:  t.regress,
		Stall:    t.stall,
		Net:      t.progress - t.regress,
	}
	if t.scored > 0 {
		inv := 1.0 / float64(t.scored)
		s.ProgressRate = float64(t.progress) * inv
		s.RegressRate = float64(t.regress) * inv
		s.StallRate = float64(t.stall) * inv
	}
	if effective := t.progress + t.regress; effective > 0 {
		s.Efficiency = float64(t.progress) / float64(effective)
	}
	return s
}
