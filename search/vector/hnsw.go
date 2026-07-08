package vector

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

// HNSW is a Hierarchical Navigable Small World graph [Matcher]: an approximate
// nearest-neighbour index whose TopK is sub-linear in the corpus size, unlike
// [Flat] which scans every vector. It is the mechanism a Qdrant collection or a
// Lucene HNSW field wraps, reduced to a pure-Go primitive.
//
// # Accuracy
//
// Unlike Flat+Float32, HNSW is APPROXIMATE: it explores a navigable subgraph
// rather than the whole corpus, so its top-k can miss a true neighbour. Its
// accuracy is reported as recall@k against the Flat+Float32 exact result and is
// tuned by efSearch (larger = more accurate, slower). With sensible parameters
// (M=16, efConstruction=200, efSearch≥k·4) recall@10 is typically ≥0.95 on real
// embeddings. Measure it for your corpus — do not assume.
//
// # Distance
//
// This first implementation stores vectors as full float32 (L2-normalized) and
// scores with exact cosine similarity, so the graph is built and searched on
// exact distances; only the graph traversal is approximate. It does not use a
// [Quantizer]: HNSW needs a symmetric vector-to-vector distance to link nodes at
// build time, which the query-bound [Scorer] cannot express. Quantized HNSW
// (binary/PQ codes in the graph) is a planned extension.
//
// # Determinism
//
// Level assignment is the only randomness; it is driven by a seeded RNG, so the
// same vectors added in the same order with the same seed produce an identical
// graph and identical query results. This makes recall reproducible in tests.
//
// # Concurrency
//
// Add mutates the graph and is single-goroutine (like [Flat.Add]). Once building
// is done TopK is read-only and safe for concurrent callers: it allocates no
// shared state beyond a pooled scratch buffer that each call checks out
// exclusively.
type HNSW struct {
	dim   int
	nodes []hnswNode

	// Representation. Exactly one is used: with no quantizer (sq == nil, the
	// default) vectors are stored as normalized float32 in vecs and scored by
	// exact cosine; with a quantizer they are stored as compact codes and scored
	// by the quantizer's symmetric code-to-code similarity.
	sq    SymQuantizer
	vecs  [][]float32 // node index → normalized vector   (sq == nil)
	codes [][]byte    // node index → quantized code        (sq != nil)

	entry    int32 // internal index of the top-layer entry point
	maxLevel int

	m              int     // target neighbours per node on layers >0
	mmax           int     // neighbour cap on layers >0
	mmax0          int     // neighbour cap on layer 0
	efConstruction int     // candidate-list size while inserting
	efSearch       int     // candidate-list size while querying (>= k)
	mL             float64 // level-generation normalization = 1/ln(M)

	rng *rand.Rand
}

// hnswNode is one corpus vector's id and its per-layer adjacency. neighbors[l]
// holds the internal indices of the node's neighbours on layer l; the slice has
// len(node.level)+1 layers. The vector itself lives in HNSW.vecs or HNSW.codes
// at the same index.
type hnswNode struct {
	id        uint32    // caller-supplied id
	neighbors [][]int32 // neighbors[level] = internal indices
}

// queryRep is a query prepared for one representation: a normalized float32
// vector (exact path) or a quantized code (quantized path). Built once per query
// and passed down through searchLayer.
type queryRep struct {
	vec  []float32
	code []byte
}

// Option configures a [HNSW] at construction.
type Option func(*HNSW)

// WithM sets the target number of neighbours per node on the upper layers (and,
// doubled, the cap on layer 0). Larger M builds a denser graph: better recall
// and more memory. Default 16.
func WithM(m int) Option { return func(h *HNSW) { h.m = m } }

// WithEfConstruction sets the candidate-list size used while inserting. Larger
// values build a higher-quality graph at higher build cost. Default 200.
func WithEfConstruction(ef int) Option { return func(h *HNSW) { h.efConstruction = ef } }

// WithEfSearch sets the default candidate-list size used while querying. TopK
// raises it to k if k is larger. Larger values trade latency for recall.
// Default 50.
func WithEfSearch(ef int) Option { return func(h *HNSW) { h.efSearch = ef } }

// WithSeed sets the RNG seed for level assignment, making the graph
// deterministic. Default 1.
func WithSeed(seed int64) Option { return func(h *HNSW) { h.rng = rand.New(rand.NewSource(seed)) } }

// WithQuantizer stores the graph over compact codes produced by sq instead of
// full float32 vectors, trading recall for 4–32× less memory and cheaper
// distances (e.g. Hamming popcount for [NewBinarySym]). The graph is built AND
// searched entirely in the code space using sq's symmetric code-to-code
// similarity, so both build and query see the quantization. Passing nil (the
// default) keeps the exact float32 representation. sq.Dim() must equal the
// HNSW's dim.
func WithQuantizer(sq SymQuantizer) Option { return func(h *HNSW) { h.sq = sq } }

// NewHNSW returns an empty HNSW index for dim-dimensional vectors.
func NewHNSW(dim int, opts ...Option) *HNSW {
	h := &HNSW{
		dim:            dim,
		m:              16,
		efConstruction: 200,
		efSearch:       50,
		rng:            rand.New(rand.NewSource(1)),
	}
	for _, opt := range opts {
		opt(h)
	}
	if h.m < 1 {
		h.m = 1
	}
	if h.sq != nil && h.sq.Dim() != dim {
		panic("vector: HNSW quantizer dim mismatch")
	}
	h.mmax = h.m
	h.mmax0 = 2 * h.m
	h.mL = 1 / math.Log(float64(h.m)+1e-9)
	return h
}

// Len reports how many vectors have been added.
func (h *HNSW) Len() int { return len(h.nodes) }

// SetEfSearch changes the query-time candidate-list size. Call it before
// concurrent TopK readers start; it is not safe to call concurrently with TopK.
func (h *HNSW) SetEfSearch(ef int) { h.efSearch = ef }

// Add inserts vec under id. Not safe for concurrent use (with Add or TopK).
//
// As with [Flat.Add], a zero vector (empty or all-out-of-vocabulary text) is a
// legitimate member: it scores ~0 and never wins, but adding it preserves the
// id↔corpus mapping. It is simply an isolated graph node.
func (h *HNSW) Add(id uint32, vec []float32) {
	if len(vec) != h.dim {
		panic("vector: HNSW.Add dim mismatch")
	}
	nv := normalized(vec)
	newIdx := int32(len(h.nodes))
	level := h.randomLevel()
	h.nodes = append(h.nodes, hnswNode{
		id:        id,
		neighbors: make([][]int32, level+1),
	})
	var qr queryRep
	if h.sq != nil {
		code := h.sq.Encode(nv)
		h.codes = append(h.codes, code)
		qr = queryRep{code: code}
	} else {
		h.vecs = append(h.vecs, nv)
		qr = queryRep{vec: nv}
	}

	if newIdx == 0 { // first node becomes the entry point
		h.entry = 0
		h.maxLevel = level
		return
	}

	ep := []int32{h.entry}
	// Greedy descent through the layers above this node's top level: one step
	// each, moving the entry point closer to the query.
	for l := h.maxLevel; l > level; l-- {
		w := h.searchLayer(qr, ep, 1, l)
		ep = ep[:0]
		ep = append(ep, w[0].node)
	}
	// From this node's top level down to 0: ef-search, pick neighbours, link.
	for l := min(h.maxLevel, level); l >= 0; l-- {
		w := h.searchLayer(qr, ep, h.efConstruction, l)
		mmax := h.mmax
		if l == 0 {
			mmax = h.mmax0
		}
		selected := h.selectNeighbors(w, h.m)
		h.nodes[newIdx].neighbors[l] = selected
		for _, nb := range selected {
			h.nodes[nb].neighbors[l] = append(h.nodes[nb].neighbors[l], newIdx)
			if len(h.nodes[nb].neighbors[l]) > mmax {
				h.nodes[nb].neighbors[l] = h.pruneNeighbors(nb, l, mmax)
			}
		}
		ep = candNodes(w, ep[:0])
	}

	if level > h.maxLevel {
		h.maxLevel = level
		h.entry = newIdx
	}
}

// TopK returns the k vectors most similar to query, best first (approximate).
// Returns nil for k <= 0 or an empty corpus. Safe for concurrent readers.
func (h *HNSW) TopK(query []float32, k int) []Hit {
	if k <= 0 || len(h.nodes) == 0 {
		return nil
	}
	qr := h.prepareQuery(query)
	ep := []int32{h.entry}
	for l := h.maxLevel; l >= 1; l-- {
		w := h.searchLayer(qr, ep, 1, l)
		ep = ep[:0]
		ep = append(ep, w[0].node)
	}
	ef := h.efSearch
	if ef < k {
		ef = k
	}
	w := h.searchLayer(qr, ep, ef, 0)
	hits := make([]Hit, len(w))
	for i, c := range w {
		hits[i] = Hit{ID: c.id, Score: c.sim}
	}
	sort.Slice(hits, func(i, j int) bool { return betterHit(hits[i], hits[j]) })
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits
}

// randomLevel draws a node's top level from a geometric-like distribution:
// level = floor(-ln(U) * mL), so higher levels are exponentially rarer.
func (h *HNSW) randomLevel() int {
	r := h.rng.Float64()
	if r <= 0 {
		r = math.SmallestNonzeroFloat64
	}
	return int(-math.Log(r) * h.mL)
}

// prepareQuery builds the query representation matching this index: a normalized
// float32 vector (exact path) or a quantized code (quantized path).
func (h *HNSW) prepareQuery(query []float32) queryRep {
	nv := normalized(query)
	if h.sq != nil {
		return queryRep{code: h.sq.Encode(nv)}
	}
	return queryRep{vec: nv}
}

// simQ is the similarity between a prepared query and a stored node: exact cosine
// for the float32 path, the quantizer's symmetric code similarity otherwise.
func (h *HNSW) simQ(q queryRep, node int32) float32 {
	if h.sq != nil {
		return h.sq.SimCodes(q.code, h.codes[node])
	}
	return dot(q.vec, h.vecs[node])
}

// simNodes is the similarity between two stored nodes, in whichever
// representation the index uses.
func (h *HNSW) simNodes(a, b int32) float32 {
	if h.sq != nil {
		return h.sq.SimCodes(h.codes[a], h.codes[b])
	}
	return dot(h.vecs[a], h.vecs[b])
}

// searchLayer runs the HNSW greedy best-first search on one layer: starting from
// entry, it expands the ef most promising nodes and returns the ef nearest found
// (as an unordered slice). q is the prepared query for the index's representation.
func (h *HNSW) searchLayer(q queryRep, entry []int32, ef, level int) []cand {
	vs := visitedPool.Get().(*visitedSet)
	vs.reset(len(h.nodes))
	defer visitedPool.Put(vs)

	candidates := candHeap{lt: candBetter}                              // top = best (max sim)
	w := candHeap{lt: func(x, y cand) bool { return candBetter(y, x) }} // top = worst (min sim)

	for _, e := range entry {
		c := cand{node: e, id: h.nodes[e].id, sim: h.simQ(q, e)}
		vs.add(e)
		candidates.push(c)
		w.push(c)
	}

	for candidates.len() > 0 {
		c := candidates.pop()
		if w.len() >= ef && c.sim < w.top().sim {
			break // best remaining candidate is worse than the worst kept
		}
		for _, nb := range h.nodes[c.node].neighbors[level] {
			if vs.seen(nb) {
				continue
			}
			vs.add(nb)
			nc := cand{node: nb, id: h.nodes[nb].id, sim: h.simQ(q, nb)}
			if w.len() < ef || nc.sim > w.top().sim {
				candidates.push(nc)
				w.push(nc)
				if w.len() > ef {
					w.pop() // drop the worst
				}
			}
		}
	}
	return append([]cand(nil), w.a...)
}

// selectNeighbors picks up to m diverse neighbours from candidates (each scored
// against the base the search was run for) using the HNSW heuristic: keep a
// candidate only if it is closer to the base than to any already-kept neighbour,
// which spreads links across directions instead of clustering them. If the
// heuristic keeps fewer than m, the closest discarded candidates fill the rest
// (keepPrunedConnections) to avoid under-connecting the graph.
func (h *HNSW) selectNeighbors(candidates []cand, m int) []int32 {
	sort.Slice(candidates, func(i, j int) bool { return candBetter(candidates[i], candidates[j]) })
	result := make([]int32, 0, m)
	for _, e := range candidates {
		if len(result) >= m {
			break
		}
		keep := true
		for _, r := range result {
			if h.simNodes(e.node, r) > e.sim { // r is closer to e than the base is
				keep = false
				break
			}
		}
		if keep {
			result = append(result, e.node)
		}
	}
	if len(result) < m { // top up with the closest discarded candidates
		for _, e := range candidates {
			if len(result) >= m {
				break
			}
			if containsInt32(result, e.node) {
				continue
			}
			result = append(result, e.node)
		}
	}
	return result
}

// pruneNeighbors re-selects node's layer-l neighbours down to mmax using the
// same diversity heuristic, keeping the most useful links after a new reverse
// edge pushed the count over the cap.
func (h *HNSW) pruneNeighbors(node int32, level, mmax int) []int32 {
	cur := h.nodes[node].neighbors[level]
	cs := make([]cand, len(cur))
	for i, x := range cur {
		cs[i] = cand{node: x, id: h.nodes[x].id, sim: h.simNodes(node, x)}
	}
	return h.selectNeighbors(cs, mmax)
}

// cand is a node reached during search: its internal index, external id, and
// similarity to the search's query/base (larger = closer).
type cand struct {
	node int32
	id   uint32
	sim  float32
}

// candBetter is the strict total order matching [betterHit]: higher sim wins,
// ties break to the lower id.
func candBetter(a, b cand) bool {
	if a.sim != b.sim {
		return a.sim > b.sim
	}
	return a.id < b.id
}

// candNodes appends the internal indices of cs to dst and returns it.
func candNodes(cs []cand, dst []int32) []int32 {
	for _, c := range cs {
		dst = append(dst, c.node)
	}
	return dst
}

func containsInt32(s []int32, x int32) bool {
	for _, v := range s {
		if v == x {
			return true
		}
	}
	return false
}

// candHeap is a binary heap of cand ordered by lt: lt(x,y) reports whether x
// belongs nearer the top than y. Used as a max-heap (best on top) for the
// candidate frontier and a min-heap (worst on top) for the result set.
type candHeap struct {
	a  []cand
	lt func(x, y cand) bool
}

func (h *candHeap) len() int  { return len(h.a) }
func (h *candHeap) top() cand { return h.a[0] }

func (h *candHeap) push(c cand) {
	h.a = append(h.a, c)
	i := len(h.a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if !h.lt(h.a[i], h.a[p]) {
			break
		}
		h.a[i], h.a[p] = h.a[p], h.a[i]
		i = p
	}
}

func (h *candHeap) pop() cand {
	n := len(h.a) - 1
	h.a[0], h.a[n] = h.a[n], h.a[0]
	c := h.a[n]
	h.a = h.a[:n]
	i, sz := 0, len(h.a)
	for {
		l, r := 2*i+1, 2*i+2
		top := i
		if l < sz && h.lt(h.a[l], h.a[top]) {
			top = l
		}
		if r < sz && h.lt(h.a[r], h.a[top]) {
			top = r
		}
		if top == i {
			break
		}
		h.a[i], h.a[top] = h.a[top], h.a[i]
		i = top
	}
	return c
}

// visitedSet is an O(1)-reset seen-set using a monotonic version stamp: a node
// is marked visited for the current epoch when mark[node]==cur. Pooled and
// reused across searches so a query does not allocate an O(n) buffer.
type visitedSet struct {
	mark []uint32
	cur  uint32
}

func (v *visitedSet) reset(n int) {
	if cap(v.mark) < n {
		v.mark = make([]uint32, n)
		v.cur = 0
	}
	v.mark = v.mark[:n]
	v.cur++
	if v.cur == 0 { // wrapped: clear and restart
		for i := range v.mark {
			v.mark[i] = 0
		}
		v.cur = 1
	}
}

func (v *visitedSet) seen(x int32) bool { return v.mark[x] == v.cur }
func (v *visitedSet) add(x int32)       { v.mark[x] = v.cur }

var visitedPool = sync.Pool{New: func() any { return &visitedSet{} }}
