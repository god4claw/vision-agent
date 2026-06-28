package embed

import (
	"context"
	"math"
	"testing"
)

func TestLocalEmbedderDeterminism(t *testing.T) {
	ctx := context.Background()
	var e LocalEmbedder
	text := "open the file menu and click new"

	a, err := e.Embed(ctx, text)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	b, err := e.Embed(ctx, text)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(a) != len(b) {
		t.Fatalf("dim mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at index %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func BenchmarkLocalEmbedderEmbed(b *testing.B) {
	ctx := context.Background()
	var e LocalEmbedder
	const text = "open the file menu and click new then save the document as report"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Embed(ctx, text); err != nil {
			b.Fatalf("Embed: %v", err)
		}
	}
}

func TestLocalEmbedderFixedDimAndNorm(t *testing.T) {
	ctx := context.Background()
	var e LocalEmbedder

	for _, text := range []string{"hello world", "a completely different sentence", ""} {
		v, err := e.Embed(ctx, text)
		if err != nil {
			t.Fatalf("Embed(%q): %v", text, err)
		}
		if len(v) != localDim {
			t.Fatalf("Embed(%q) dim = %d, want %d", text, len(v), localDim)
		}
		var sumSq float64
		for _, x := range v {
			sumSq += float64(x) * float64(x)
		}
		norm := math.Sqrt(sumSq)
		if math.Abs(norm-1.0) > 1e-5 {
			t.Fatalf("Embed(%q) L2 norm = %v, want ~1.0", text, norm)
		}
	}
}
