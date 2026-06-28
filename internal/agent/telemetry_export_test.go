package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// buildSnapshot drives a Telemetry through a known sequence and returns the
// resulting Snapshot so the serialization tests have a non-trivial input.
func buildSnapshot() Snapshot {
	var t Telemetry
	// 4 scored: 2 progress, 1 regress, 1 stall.
	t.Record(true, true)   // progress
	t.Record(true, false)  // progress
	t.Record(false, true)  // regress
	t.Record(false, false) // stall
	return t.Snapshot()
}

func TestSnapshotCSV(t *testing.T) {
	s := buildSnapshot()
	out := string(s.CSV())
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("CSV want 2 lines (header+row), got %d: %q", len(lines), out)
	}
	header := strings.Split(lines[0], ",")
	row := strings.Split(lines[1], ",")
	if len(header) != len(row) {
		t.Fatalf("CSV header/row column mismatch: %d vs %d", len(header), len(row))
	}
	if !strings.HasSuffix(out, "\n") {
		t.Errorf("CSV should end with newline")
	}
	want := map[string]string{
		"scored":        "4",
		"progress":      "2",
		"regress":       "1",
		"stall":         "1",
		"progress_rate": "0.5000",
		"regress_rate":  "0.2500",
		"stall_rate":    "0.2500",
		"efficiency":    "0.6667",
		"net":           "1",
	}
	got := map[string]string{}
	for i, col := range header {
		got[col] = row[i]
	}
	for col, exp := range want {
		if got[col] != exp {
			t.Errorf("CSV column %q = %q, want %q", col, got[col], exp)
		}
	}
}

func TestSnapshotJSON(t *testing.T) {
	s := buildSnapshot()
	out := s.JSON()
	if !strings.HasSuffix(string(out), "\n") {
		t.Errorf("JSON should end with newline")
	}
	var back Snapshot
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatalf("JSON unmarshal: %v", err)
	}
	if back != s {
		t.Errorf("JSON round-trip mismatch:\n got  %+v\n want %+v", back, s)
	}
}

func TestSnapshotMarshalByExtension(t *testing.T) {
	s := buildSnapshot()
	tests := []struct {
		path    string
		wantErr bool
		marker  string // substring expected in output
	}{
		{"out.csv", false, "scored,progress"},
		{"OUT.CSV", false, "scored,progress"},
		{"out.json", false, "\"scored\": 4"},
		{"data/run.json", false, "\"scored\": 4"},
		{"out.txt", true, ""},
		{"out", true, ""},
	}
	for _, tc := range tests {
		got, err := s.Marshal(tc.path)
		if tc.wantErr {
			if err == nil {
				t.Errorf("Marshal(%q) want error, got nil", tc.path)
			}
			continue
		}
		if err != nil {
			t.Errorf("Marshal(%q) unexpected error: %v", tc.path, err)
			continue
		}
		if !strings.Contains(string(got), tc.marker) {
			t.Errorf("Marshal(%q) output missing %q: %s", tc.path, tc.marker, got)
		}
	}
}
