package vector

import (
	"sort"
	"sync"
)

// Flat is a brute-force [Matcher]: TopK scans every stored code. With the
// [Float32] quantizer the result is the exact top-k; with a lossy quantizer it
// is exact over the codes (the approximation is entirely in the quantizer).
// Codes are held in one contiguous byte slice for cache-friendly, low-alloc
// scanning.
//
// Ranking uses a strict total order — higher score first, then lower id — so
// results are deterministic and [Flat.TopKParallel] is byte-identical to
// [Flat.TopK].
type Flat struct {
	q       Quantizer
	dim     int
	codeLen int
	ids     []uint32
	codes   []byte

	idxOnce sync.Once
	idx     map[uint32]int // id → storage position, for Rescore (built lazily)
}

// NewFlat returns an empty Flat matcher backed by quantizer q.
func NewFlat(q Quantizer) *Flat {
	return &Flat{q: q, dim: q.Dim(), codeLen: q.CodeLen()}
}

// Add encodes vec and stores it under id. Not safe for concurrent use.
func (f *Flat) Add(id uint32, vec []float32) {
	if len(vec) != f.dim {
		panic("vector: Flat.Add dim mismatch")
	}
	f.ids = append(f.ids, id)
	f.codes = append(f.codes, f.q.Encode(vec)...)
}

// Len reports how many vectors have been added.
func (f *Flat) Len() int { return len(f.ids) }

// code returns the i-th stored code (no copy).
func (f *Flat) code(i int) []byte {
	return f.codes[i*f.codeLen : (i+1)*f.codeLen]
}

// TopK returns the k vectors most similar to query, best first. If k exceeds
// the corpus size all vectors are returned. Returns nil for k <= 0 or an empty
// corpus.
func (f *Flat) TopK(query []float32, k int) []Hit {
	n := len(f.ids)
	if k <= 0 || n == 0 {
		return nil
	}
	k = min(k, n)
	sc := f.q.Query(query)
	tk := newTopK(k)
	for i := 0; i < n; i++ {
		tk.push(Hit{ID: f.ids[i], Score: sc.Score(f.code(i))})
	}
	return tk.result()
}

// Rescore re-ranks a candidate set (the ids from a coarse stage) with THIS
// matcher's quantizer and returns the top-k, best first. This is the second
// stage of the standard two-stage retrieval pattern: a cheap, lossy matcher
// (e.g. Binary) returns top-N candidates, then an exact matcher (Float32 over
// the same ids) Rescores them — recovering the tail the coarse stage lost while
// keeping the coarse stage's speed/memory. Ids not present are skipped.
//
//	coarse := NewFlat(NewBinary(dim)); exact := NewFlat(NewFloat32(dim))
//	// Add every (id, vec) to both.
//	cand := coarse.TopKParallel(query, 200, workers)   // fast, lossy
//	ids  := make([]uint32, len(cand))
//	for i, h := range cand { ids[i] = h.ID }
//	final := exact.Rescore(query, ids, 10)             // exact over 200 → top-10
//
// Not safe for concurrent use with Add (it builds an id index on first call).
func (f *Flat) Rescore(query []float32, ids []uint32, k int) []Hit {
	if k <= 0 || len(ids) == 0 {
		return nil
	}
	f.idxOnce.Do(func() {
		f.idx = make(map[uint32]int, len(f.ids))
		for i, id := range f.ids {
			f.idx[id] = i
		}
	})
	sc := f.q.Query(query)
	tk := newTopK(min(k, len(ids)))
	for _, id := range ids {
		if i, ok := f.idx[id]; ok {
			tk.push(Hit{ID: id, Score: sc.Score(f.code(i))})
		}
	}
	return tk.result()
}

// TopKParallel is TopK sharded across the given number of goroutines: each
// scans a disjoint range and the partial top-k lists are merged. The result is
// identical to TopK. workers <= 1 (or a tiny corpus) runs serially.
func (f *Flat) TopKParallel(query []float32, k, workers int) []Hit {
	n := len(f.ids)
	if k <= 0 || n == 0 {
		return nil
	}
	k = min(k, n)
	if workers <= 1 || n < workers*2 {
		return f.TopK(query, k)
	}
	sc := f.q.Query(query) // read-only, shared across workers

	parts := make([][]Hit, workers)
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= n {
			break
		}
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			tk := newTopK(k)
			for i := lo; i < hi; i++ {
				tk.push(Hit{ID: f.ids[i], Score: sc.Score(f.code(i))})
			}
			parts[w] = tk.hits
		}(w, lo, hi)
	}
	wg.Wait()

	// Merge partial top-k lists into a final top-k.
	tk := newTopK(k)
	for _, p := range parts {
		for _, h := range p {
			tk.push(h)
		}
	}
	return tk.result()
}

// betterHit is the strict total order used for ranking: higher score wins;
// ties break to the lower id. Distinct ids make it a strict total order, so the
// top-k set is unique.
func betterHit(a, b Hit) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	return a.ID < b.ID
}

// topK is a bounded max-selection heap keeping the k best hits. It is a
// min-heap under betterHit, so hits[0] is the worst kept hit and is evicted
// first. Allocation-free after construction.
type topK struct {
	k    int
	hits []Hit
}

func newTopK(k int) *topK { return &topK{k: k, hits: make([]Hit, 0, k)} }

func (t *topK) push(h Hit) {
	if len(t.hits) < t.k {
		t.hits = append(t.hits, h)
		t.up(len(t.hits) - 1)
		return
	}
	if betterHit(h, t.hits[0]) { // better than the worst kept
		t.hits[0] = h
		t.down(0)
	}
}

func (t *topK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !betterHit(t.hits[p], t.hits[i]) { // parent not better => heap ok
			break
		}
		t.hits[p], t.hits[i] = t.hits[i], t.hits[p]
		i = p
	}
}

func (t *topK) down(i int) {
	n := len(t.hits)
	for {
		l, r := 2*i+1, 2*i+2
		worst := i
		if l < n && betterHit(t.hits[worst], t.hits[l]) {
			worst = l
		}
		if r < n && betterHit(t.hits[worst], t.hits[r]) {
			worst = r
		}
		if worst == i {
			break
		}
		t.hits[i], t.hits[worst] = t.hits[worst], t.hits[i]
		i = worst
	}
}

func (t *topK) result() []Hit {
	out := make([]Hit, len(t.hits))
	copy(out, t.hits)
	sort.Slice(out, func(i, j int) bool { return betterHit(out[i], out[j]) })
	return out
}
