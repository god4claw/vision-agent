package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"net/http"
	"strings"
	"time"
	"unicode"
)

// Embedder turns text into a vector for similarity search.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

const localDim = 256

// LocalEmbedder is a deterministic, offline, dependency-free embedder.
// It hashes tokens into a fixed-dimension bag-of-words vector and L2-normalizes.
// Good enough for recall@1 on similar state text; replace with a real model later.
type LocalEmbedder struct{}

func (LocalEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	vec := make([]float32, localDim)
	tokens := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	for _, tok := range tokens {
		h := fnv.New32a()
		_, _ = h.Write([]byte(tok))
		vec[h.Sum32()%localDim]++
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		vec[0] = 1
		return vec, nil
	}
	for i := range vec {
		vec[i] = float32(float64(vec[i]) / norm)
	}
	return vec, nil
}

// OllamaEmbedder uses a local Ollama model for embeddings.
// Requires a pulled embedding model, e.g. `ollama pull nomic-embed-text`.
type OllamaEmbedder struct {
	URL    string
	Model  string
	client *http.Client
}

func NewOllamaEmbedder(url, model string) *OllamaEmbedder {
	return &OllamaEmbedder{URL: url, Model: model, client: &http.Client{Timeout: 10 * time.Second}}
}

func (o *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(map[string]string{"model": o.Model, "prompt": text})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.URL+"/api/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama embeddings status %d", resp.StatusCode)
	}
	var out struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding from ollama (model %q pulled?)", o.Model)
	}
	vec := make([]float32, len(out.Embedding))
	for i, v := range out.Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}
