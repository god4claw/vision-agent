package agent

import "sync"

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
	Scored       int
	Progress     int
	Regress      int
	Stall        int
	ProgressRate float64 // Progress / Scored
	RegressRate  float64 // Regress / Scored
	StallRate    float64 // Stall / Scored
	Net          int     // Progress - Regress (net forward motion)

	// Efficiency is the share of *effective* (screen-changing) actions that
	// were progress: Progress / (Progress + Regress). It ignores stalls, so it
	// measures decision quality once the agent actually did something. High is
	// good (1.0 = every effective action helped); 0 when no effective actions.
	Efficiency float64
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
