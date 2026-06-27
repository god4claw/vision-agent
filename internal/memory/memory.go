package memory

import (
	"context"
	"encoding/json"
	"strconv"
	"sync"

	chromem "github.com/philippgille/chromem-go"

	"visionagent/internal/embed"
	"visionagent/internal/executor"
)

// Episode is one remembered "what / how / at which stage / under which variables".
type Episode struct {
	ID        string
	Stage     string
	StateText string
	Action    executor.Action
	Variables map[string]string
	Success   bool

	// Similarity is the cosine similarity to the query, populated only on
	// retrieval (Nearest); it is not persisted.
	Similarity float32
}

// Store is the episodic memory backed by chromem-go (vector similarity search).
type Store struct {
	mu          sync.Mutex
	coll        *chromem.Collection
	maxEpisodes int
	order       []string // insertion order, for bounded eviction
}

// NewStore creates an in-memory vector store using the given embedder.
// maxEpisodes <= 0 means unbounded.
func NewStore(emb embed.Embedder, maxEpisodes int) (*Store, error) {
	db := chromem.NewDB()
	ef := chromem.EmbeddingFunc(func(ctx context.Context, text string) ([]float32, error) {
		return emb.Embed(ctx, text)
	})
	coll, err := db.CreateCollection("episodes", nil, ef)
	if err != nil {
		return nil, err
	}
	return &Store{coll: coll, maxEpisodes: maxEpisodes}, nil
}

func (s *Store) Add(ctx context.Context, ep Episode) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	actJSON, _ := json.Marshal(ep.Action)
	varsJSON, _ := json.Marshal(ep.Variables)
	doc := chromem.Document{
		ID:      ep.ID,
		Content: ep.StateText,
		Metadata: map[string]string{
			"stage":   ep.Stage,
			"action":  string(actJSON),
			"vars":    string(varsJSON),
			"success": strconv.FormatBool(ep.Success),
		},
	}
	if err := s.coll.AddDocument(ctx, doc); err != nil {
		return err
	}
	s.order = append(s.order, ep.ID)

	// Bounded memory: evict oldest so an infinite loop cannot grow forever.
	for s.maxEpisodes > 0 && s.coll.Count() > s.maxEpisodes && len(s.order) > 0 {
		oldest := s.order[0]
		s.order = s.order[1:]
		if err := s.coll.Delete(ctx, nil, nil, oldest); err != nil {
			return err
		}
	}
	return nil
}

// Nearest returns up to k closest past episodes to the given state text.
func (s *Store) Nearest(ctx context.Context, stateText string, k int) ([]Episode, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := s.coll.Count()
	if n == 0 {
		return nil, nil
	}
	if k > n {
		k = n
	}
	results, err := s.coll.Query(ctx, stateText, k, nil, nil)
	if err != nil {
		return nil, err
	}
	eps := make([]Episode, 0, len(results))
	for _, r := range results {
		eps = append(eps, resultToEpisode(r))
	}
	return eps, nil
}

func (s *Store) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.coll.Count()
}

func resultToEpisode(r chromem.Result) Episode {
	var act executor.Action
	_ = json.Unmarshal([]byte(r.Metadata["action"]), &act)
	var vars map[string]string
	_ = json.Unmarshal([]byte(r.Metadata["vars"]), &vars)
	success, _ := strconv.ParseBool(r.Metadata["success"])
	return Episode{
		ID:         r.ID,
		Stage:      r.Metadata["stage"],
		StateText:  r.Content,
		Action:     act,
		Variables:  vars,
		Success:    success,
		Similarity: r.Similarity,
	}
}
