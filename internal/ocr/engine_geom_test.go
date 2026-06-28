package ocr

import (
	"math"
	"testing"
)

const eps = 1e-9

func almostEqual(a, b float64) bool { return math.Abs(a-b) <= 1e-6 }

// hullContains reports whether the hull contains a point (within eps).
func hullContains(hull []pt, p pt) bool {
	for _, h := range hull {
		if math.Abs(h.X-p.X) <= eps && math.Abs(h.Y-p.Y) <= eps {
			return true
		}
	}
	return false
}

func TestConvexHull(t *testing.T) {
	tests := []struct {
		name   string
		in     []pt
		want   []pt // corners that MUST be on the hull
		absent []pt // points that MUST NOT be on the hull (interior)
		size   int  // expected hull vertex count (0 = skip exact check)
	}{
		{
			name: "fewer than 3 returned as-is",
			in:   []pt{{0, 0}, {1, 1}},
			want: []pt{{0, 0}, {1, 1}},
			size: 2,
		},
		{
			name:   "unit square with interior point",
			in:     []pt{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0.5, 0.5}},
			want:   []pt{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			absent: []pt{{0.5, 0.5}},
			size:   4,
		},
		{
			name:   "collinear points collapse to extremes",
			in:     []pt{{0, 0}, {1, 0}, {2, 0}, {3, 0}},
			want:   []pt{{0, 0}, {3, 0}},
			absent: []pt{{1, 0}, {2, 0}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// copy input since convexHull sorts in place
			in := append([]pt(nil), tc.in...)
			hull := convexHull(in)
			if tc.size > 0 && len(hull) != tc.size {
				t.Errorf("hull size = %d, want %d (%v)", len(hull), tc.size, hull)
			}
			for _, w := range tc.want {
				if !hullContains(hull, w) {
					t.Errorf("hull missing required corner %v: %v", w, hull)
				}
			}
			for _, a := range tc.absent {
				if hullContains(hull, a) {
					t.Errorf("hull should not contain interior/collinear point %v: %v", a, hull)
				}
			}
		})
	}
}

func rectArea(c [4]pt) float64 {
	w := math.Hypot(c[1].X-c[0].X, c[1].Y-c[0].Y)
	h := math.Hypot(c[3].X-c[0].X, c[3].Y-c[0].Y)
	return w * h
}

func TestMinAreaRect(t *testing.T) {
	tests := []struct {
		name     string
		hull     []pt
		wantArea float64
	}{
		{
			name:     "unit square",
			hull:     []pt{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			wantArea: 1,
		},
		{
			name:     "axis-aligned 2x1 rectangle",
			hull:     []pt{{0, 0}, {2, 0}, {2, 1}, {0, 1}},
			wantArea: 2,
		},
		{
			name:     "diamond (45-degree square, side sqrt2)",
			hull:     []pt{{1, 0}, {2, 1}, {1, 2}, {0, 1}},
			wantArea: 2,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := minAreaRect(tc.hull)
			if got := rectArea(c); !almostEqual(got, tc.wantArea) {
				t.Errorf("area = %v, want %v (corners %v)", got, tc.wantArea, c)
			}
			// First edge (TL->TR) must be the longer-or-equal side.
			w := math.Hypot(c[1].X-c[0].X, c[1].Y-c[0].Y)
			h := math.Hypot(c[3].X-c[0].X, c[3].Y-c[0].Y)
			if w+1e-9 < h {
				t.Errorf("TL->TR (%.3f) shorter than TL->BL (%.3f); not text-major", w, h)
			}
		})
	}
}

func TestBBoxCorners(t *testing.T) {
	got := bboxCorners([]pt{{3, 5}, {-1, 2}, {4, -2}, {0, 0}})
	want := [4]pt{{-1, -2}, {4, -2}, {4, 5}, {-1, 5}}
	if got != want {
		t.Errorf("bboxCorners = %v, want %v", got, want)
	}
}

func TestCornersBBox(t *testing.T) {
	minX, minY, maxX, maxY := cornersBBox([4]pt{{1, 2}, {5, 1}, {6, 8}, {0, 7}})
	if !almostEqual(minX, 0) || !almostEqual(minY, 1) || !almostEqual(maxX, 6) || !almostEqual(maxY, 8) {
		t.Errorf("cornersBBox = (%v,%v,%v,%v), want (0,1,6,8)", minX, minY, maxX, maxY)
	}
}

func TestOrderCorners(t *testing.T) {
	// A wide rectangle given in scrambled order should come back as
	// [TL, TR, BR, BL] with the long edge first.
	in := [4]pt{{4, 1}, {0, 1}, {0, 0}, {4, 0}}
	got := orderCorners(in)
	want := [4]pt{{0, 0}, {4, 0}, {4, 1}, {0, 1}}
	if got != want {
		t.Errorf("orderCorners = %v, want %v", got, want)
	}
}

func TestUnclipCorners(t *testing.T) {
	in := [4]pt{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	d := 0.5
	out := unclipCorners(in, d)

	// Center is preserved.
	cx := (out[0].X + out[1].X + out[2].X + out[3].X) / 4
	cy := (out[0].Y + out[1].Y + out[2].Y + out[3].Y) / 4
	if !almostEqual(cx, 0.5) || !almostEqual(cy, 0.5) {
		t.Errorf("center moved to (%v,%v), want (0.5,0.5)", cx, cy)
	}

	// Each corner moved outward by exactly d from its original position.
	for i := range in {
		moved := math.Hypot(out[i].X-in[i].X, out[i].Y-in[i].Y)
		if !almostEqual(moved, d) {
			t.Errorf("corner %d moved %v, want %v", i, moved, d)
		}
	}

	// The unclipped rect is strictly larger than the original.
	if rectArea(out) <= rectArea(in) {
		t.Errorf("unclip did not enlarge: in=%v out=%v", rectArea(in), rectArea(out))
	}
}

// TestUnclipDegenerate ensures a zero-size box (all points coincident) is
// returned unchanged rather than producing NaNs.
func TestUnclipDegenerate(t *testing.T) {
	in := [4]pt{{2, 2}, {2, 2}, {2, 2}, {2, 2}}
	out := unclipCorners(in, 1.0)
	if out != in {
		t.Errorf("degenerate unclip = %v, want unchanged %v", out, in)
	}
}
