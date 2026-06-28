package agent

import (
	"strconv"
	"strings"
	"testing"
)

// TestSnapshotPrometheus asserts the exposition format is well-formed: every
// metric has its # HELP and # TYPE lines, the declared type matches the metric
// suffix, the sample lines parse, and the values equal a known Snapshot.
func TestSnapshotPrometheus(t *testing.T) {
	s := buildSnapshot() // 4 scored: 2 progress, 1 regress, 1 stall.
	out := s.Prometheus()

	if !strings.HasSuffix(out, "\n") {
		t.Errorf("exposition output should end with a newline")
	}

	// Parse the document into help text, declared type and sample value per
	// metric, validating the structural rules as we go.
	type metric struct {
		help, typ, sample string
		hasSample         bool
	}
	metrics := map[string]*metric{}
	for _, line := range strings.Split(strings.TrimRight(out, "\n"), "\n") {
		switch {
		case strings.HasPrefix(line, "# HELP "):
			f := strings.SplitN(strings.TrimPrefix(line, "# HELP "), " ", 2)
			if len(f) != 2 || f[1] == "" {
				t.Fatalf("malformed HELP line: %q", line)
			}
			m := metrics[f[0]]
			if m == nil {
				m = &metric{}
				metrics[f[0]] = m
			}
			m.help = f[1]
		case strings.HasPrefix(line, "# TYPE "):
			f := strings.SplitN(strings.TrimPrefix(line, "# TYPE "), " ", 2)
			if len(f) != 2 {
				t.Fatalf("malformed TYPE line: %q", line)
			}
			m := metrics[f[0]]
			if m == nil {
				m = &metric{}
				metrics[f[0]] = m
			}
			m.typ = f[1]
		case strings.HasPrefix(line, "#"):
			t.Fatalf("unexpected comment line: %q", line)
		default:
			f := strings.SplitN(line, " ", 2)
			if len(f) != 2 {
				t.Fatalf("malformed sample line: %q", line)
			}
			if _, err := strconv.ParseFloat(f[1], 64); err != nil {
				t.Fatalf("sample value not parseable: %q (%v)", line, err)
			}
			m := metrics[f[0]]
			if m == nil {
				t.Fatalf("sample without preceding HELP/TYPE: %q", line)
			}
			m.sample = f[1]
			m.hasSample = true
		}
	}

	wantType := map[string]string{
		"visionagent_scored_total":   "counter",
		"visionagent_progress_total": "counter",
		"visionagent_regress_total":  "counter",
		"visionagent_stall_total":    "counter",
		"visionagent_progress_rate":  "gauge",
		"visionagent_regress_rate":   "gauge",
		"visionagent_stall_rate":     "gauge",
		"visionagent_efficiency":     "gauge",
		"visionagent_net":            "gauge",
	}
	for name, typ := range wantType {
		m := metrics[name]
		if m == nil {
			t.Errorf("missing metric %q", name)
			continue
		}
		if m.help == "" {
			t.Errorf("metric %q missing HELP", name)
		}
		if m.typ != typ {
			t.Errorf("metric %q TYPE = %q, want %q", name, m.typ, typ)
		}
		if !m.hasSample {
			t.Errorf("metric %q missing sample line", name)
		}
	}

	wantValue := map[string]string{
		"visionagent_scored_total":   "4",
		"visionagent_progress_total": "2",
		"visionagent_regress_total":  "1",
		"visionagent_stall_total":    "1",
		"visionagent_progress_rate":  "0.5",
		"visionagent_regress_rate":   "0.25",
		"visionagent_stall_rate":     "0.25",
		"visionagent_net":            "1",
	}
	for name, want := range wantValue {
		if got := metrics[name].sample; got != want {
			t.Errorf("metric %q value = %q, want %q", name, got, want)
		}
	}
	// Efficiency = 2/(2+1) = 0.666... ; assert it round-trips numerically.
	eff, err := strconv.ParseFloat(metrics["visionagent_efficiency"].sample, 64)
	if err != nil {
		t.Fatalf("efficiency parse: %v", err)
	}
	if eff < 0.6666 || eff > 0.6667 {
		t.Errorf("efficiency = %v, want ~0.6667", eff)
	}
}
