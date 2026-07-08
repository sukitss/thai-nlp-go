package vector

import (
	"sync"
	"testing"
)

// seqIDs returns [0,1,...,n-1] as the caller ids used throughout these tests.
func seqIDs(n int) []uint32 {
	ids := make([]uint32, n)
	for i := range ids {
		ids[i] = uint32(i)
	}
	return ids
}

func buildHNSW(vecs [][]float32, ids []uint32, dim int, opts ...Option) *HNSW {
	h := NewHNSW(dim, opts...)
	for i, v := range vecs {
		h.Add(ids[i], v)
	}
	return h
}

// recallAt measures mean recall@k of an HNSW against the exact top-k.
func recallAt(h *HNSW, vecs [][]float32, ids []uint32, queries [][]float32, k int) float64 {
	var hit, total int
	for _, q := range queries {
		exact := idSet(exactTopK(vecs, ids, q, k))
		got := h.TopK(q, k)
		for _, g := range got {
			if exact[g.ID] {
				hit++
			}
		}
		total += len(exact)
	}
	return float64(hit) / float64(total)
}

// TestHNSWRecall is the accuracy gate: HNSW top-k must overlap the Flat-exact
// top-k above a recall threshold on clustered embeddings.
func TestHNSWRecall(t *testing.T) {
	const (
		n   = 3000
		dim = 64
		k   = 10
	)
	vecs := randVecs(n, dim, 12, 42)
	ids := seqIDs(n)
	queries := randVecs(200, dim, 12, 7)

	h := buildHNSW(vecs, ids, dim, WithEfSearch(100))
	recall := recallAt(h, vecs, ids, queries, k)
	t.Logf("HNSW recall@%d = %.4f (n=%d dim=%d ef=100)", k, recall, n, dim)
	if recall < 0.95 {
		t.Errorf("recall@%d = %.4f, want >= 0.95", k, recall)
	}
}

// TestHNSWEfSearchImprovesRecall checks recall is monotone-ish in efSearch.
func TestHNSWEfSearchImprovesRecall(t *testing.T) {
	const (
		n   = 2000
		dim = 48
		k   = 10
	)
	vecs := randVecs(n, dim, 8, 11)
	ids := seqIDs(n)
	queries := randVecs(100, dim, 8, 99)
	h := buildHNSW(vecs, ids, dim)

	h.SetEfSearch(10)
	low := recallAt(h, vecs, ids, queries, k)
	h.SetEfSearch(200)
	high := recallAt(h, vecs, ids, queries, k)
	t.Logf("recall ef=10 %.4f, ef=200 %.4f", low, high)
	if high < low {
		t.Errorf("recall dropped with larger efSearch: ef=10 %.4f > ef=200 %.4f", low, high)
	}
	if high < 0.95 {
		t.Errorf("recall@10 ef=200 = %.4f, want >= 0.95", high)
	}
}

// TestHNSWDeterministic verifies same seed + insertion order + query yields
// byte-identical results across two independent builds.
func TestHNSWDeterministic(t *testing.T) {
	vecs := randVecs(500, 32, 6, 3)
	ids := seqIDs(500)
	queries := randVecs(20, 32, 6, 4)

	a := buildHNSW(vecs, ids, 32, WithSeed(123))
	b := buildHNSW(vecs, ids, 32, WithSeed(123))

	for _, q := range queries {
		ha := a.TopK(q, 10)
		hb := b.TopK(q, 10)
		if len(ha) != len(hb) {
			t.Fatalf("length mismatch: %d vs %d", len(ha), len(hb))
		}
		for i := range ha {
			if ha[i] != hb[i] {
				t.Fatalf("determinism broken at %d: %+v vs %+v", i, ha[i], hb[i])
			}
		}
	}
}

// TestHNSWResultOrdering checks TopK results are strictly ordered by betterHit
// (score desc, id asc) with no duplicate ids.
func TestHNSWResultOrdering(t *testing.T) {
	vecs := randVecs(1000, 32, 6, 5)
	ids := seqIDs(1000)
	h := buildHNSW(vecs, ids, 32)
	q := randVecs(1, 32, 6, 6)[0]
	hits := h.TopK(q, 20)
	seen := map[uint32]bool{}
	for i, hit := range hits {
		if seen[hit.ID] {
			t.Fatalf("duplicate id %d in results", hit.ID)
		}
		seen[hit.ID] = true
		if i > 0 && !betterHit(hits[i-1], hits[i]) {
			t.Fatalf("not sorted at %d: %+v then %+v", i, hits[i-1], hits[i])
		}
	}
}

// TestHNSWConcurrentRead runs TopK from many goroutines and checks results match
// the serial answer (guards -race and read-only concurrency safety).
func TestHNSWConcurrentRead(t *testing.T) {
	vecs := randVecs(1500, 48, 8, 8)
	ids := seqIDs(1500)
	queries := randVecs(50, 48, 8, 9)
	h := buildHNSW(vecs, ids, 48)

	want := make([][]Hit, len(queries))
	for i, q := range queries {
		want[i] = h.TopK(q, 10)
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i, q := range queries {
				got := h.TopK(q, 10)
				if len(got) != len(want[i]) {
					t.Errorf("q%d len %d != %d", i, len(got), len(want[i]))
					return
				}
				for j := range got {
					if got[j] != want[i][j] {
						t.Errorf("q%d pos%d %+v != %+v", i, j, got[j], want[i][j])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}

func TestHNSWEdgeCases(t *testing.T) {
	dim := 16

	// Empty corpus.
	empty := NewHNSW(dim)
	if got := empty.TopK(randVecs(1, dim, 1, 1)[0], 5); got != nil {
		t.Errorf("empty corpus: got %v, want nil", got)
	}

	vecs := randVecs(50, dim, 4, 2)
	ids := seqIDs(50)
	h := buildHNSW(vecs, ids, dim)
	q := randVecs(1, dim, 4, 3)[0]

	// k <= 0.
	if got := h.TopK(q, 0); got != nil {
		t.Errorf("k=0: got %v, want nil", got)
	}
	if got := h.TopK(q, -1); got != nil {
		t.Errorf("k=-1: got %v, want nil", got)
	}

	// k > n returns all n.
	if got := h.TopK(q, 1000); len(got) != 50 {
		t.Errorf("k>n: got %d hits, want 50", len(got))
	}

	// Single node.
	one := NewHNSW(dim)
	one.Add(99, vecs[0])
	got := one.TopK(q, 5)
	if len(got) != 1 || got[0].ID != 99 {
		t.Errorf("single node: got %+v", got)
	}

	// Zero vector is a legitimate member (scores ~0, never panics).
	z := NewHNSW(dim)
	z.Add(1, make([]float32, dim))
	z.Add(2, vecs[0])
	if got := z.TopK(vecs[0], 2); len(got) != 2 {
		t.Errorf("zero-vector corpus: got %d hits, want 2", len(got))
	}
}

// TestHNSWAllSameVector: a degenerate corpus of identical vectors must still
// return k distinct ids without panicking (all sims tie).
func TestHNSWAllSameVector(t *testing.T) {
	dim := 24
	base := randVecs(1, dim, 1, 1)[0]
	h := NewHNSW(dim)
	for i := 0; i < 100; i++ {
		h.Add(uint32(i), base)
	}
	hits := h.TopK(base, 10)
	if len(hits) != 10 {
		t.Fatalf("got %d hits, want 10", len(hits))
	}
	seen := map[uint32]bool{}
	for _, hit := range hits {
		if seen[hit.ID] {
			t.Fatalf("duplicate id %d", hit.ID)
		}
		seen[hit.ID] = true
	}
}

func TestHNSWDimMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on dim mismatch")
		}
	}()
	h := NewHNSW(8)
	h.Add(1, make([]float32, 4))
}
