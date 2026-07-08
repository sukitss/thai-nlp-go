package invidx

import (
	"math"
	"sort"

	"github.com/sukitss/thai-nlp-go/search/weight"
)

// sentinel is the past-the-end document id for an exhausted cursor. Internal
// document indices are 0..n-1 < math.MaxUint32, so sentinel sorts after every
// real document.
const sentinel = math.MaxUint32

// Prepare computes and caches sc's per-term upper bounds — the maximum
// contribution each term can make to any document — by scanning every postings
// list once. This is the one-time, per-scorer cost WAND amortizes over queries
// (the analogue of Lucene's index-time max impacts); [Index.Search] calls it
// lazily on first use, but calling it up front keeps the query path off the
// build mutex. Cached by scorer VALUE, so pass comparable value scorers
// (weight.BM25{...} etc.); the cache is shared safely across goroutines.
//
// Bounds are clamped at 0: a term whose contributions are all negative gets an
// upper bound of 0, which still over-estimates (0 >= any negative), keeping WAND
// correct — see the package "scorer contract".
func (idx *Index) Prepare(sc weight.CorpusScorer) []float64 {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if ub, ok := idx.ubCache[sc]; ok {
		return ub
	}
	ub := make([]float64, idx.numTerms)
	for t := 0; t < idx.numTerms; t++ {
		p := idx.post[t]
		var mx float64
		for i, d := range p.docs {
			v := sc.ScoreStats(idx.stats(uint32(t), float64(p.tfs[i]), idx.docLen[d]))
			if v > mx {
				mx = v
			}
		}
		ub[t] = mx
	}
	idx.ubCache[sc] = ub
	return ub
}

// cursor is one query term's walk through its postings list. weight is the
// term's query multiplicity (a term repeated in the query is weighted that many
// times, matching weight.ScoreDoc); ub is weight × the term's precomputed
// maximum contribution.
//
// blkMax/blkSize are set only on the block-max query path ([Index.SearchBlockMax]):
// blkMax[i] is the maximum RAW (unweighted) contribution over postings block i,
// so weight × blkMax[pos/blkSize] is a tighter, position-local upper bound than
// the term-global ub. They are nil/0 on the WAND and brute paths.
type cursor struct {
	docs   []uint32
	tfs    []uint32
	pos    int
	term   uint32
	weight float64
	ub     float64

	blkMax  []float64
	blkSize int
}

// doc returns the current document, or sentinel when the cursor is exhausted.
func (c *cursor) doc() uint32 {
	if c.pos < len(c.docs) {
		return c.docs[c.pos]
	}
	return sentinel
}

// skipTo advances the cursor to the first document >= target (a no-op if it is
// already there), via binary search over the remaining postings — the galloping
// skip that makes pruning cheap on long lists.
func (c *cursor) skipTo(target uint32) {
	if c.pos < len(c.docs) && c.docs[c.pos] < target {
		rest := c.docs[c.pos:]
		c.pos += sort.Search(len(rest), func(i int) bool { return rest[i] >= target })
	}
}

// buildCursors maps a query (term ids, with repeats) to one cursor per DISTINCT
// term that the index actually contains. ub is the per-term upper-bound table
// from Prepare, or nil for the brute path which needs no bounds. The returned
// cursors are ordered by term id so both query paths sum a document's per-term
// contributions in the same canonical order — making their scores bit-identical,
// not merely close.
func (idx *Index) buildCursors(query []uint32, ub []float64) []*cursor {
	weights := make(map[uint32]float64, len(query))
	for _, t := range query {
		weights[t]++
	}
	terms := make([]uint32, 0, len(weights))
	for t := range weights {
		terms = append(terms, t)
	}
	sort.Slice(terms, func(i, j int) bool { return terms[i] < terms[j] })

	// One backing array for the cursors (no per-cursor heap allocation) plus a
	// slice of pointers into it for the caller to reorder.
	backing := make([]cursor, 0, len(terms))
	for _, t := range terms {
		if int(t) >= idx.numTerms || len(idx.post[t].docs) == 0 {
			continue // term not in the index — no document matches it
		}
		w := weights[t]
		c := cursor{docs: idx.post[t].docs, tfs: idx.post[t].tfs, term: t, weight: w}
		if ub != nil {
			c.ub = w * ub[t]
		}
		backing = append(backing, c)
	}
	cursors := make([]*cursor, len(backing))
	for i := range backing {
		cursors[i] = &backing[i]
	}
	return cursors
}

// sortByDoc orders cursors ascending by current document with an insertion sort.
// Query term counts are tiny (a handful) and the slice is already nearly sorted
// between WAND rounds (one cursor moved), so this is O(m) in practice and, unlike
// sort.Slice, allocates nothing — the difference between WAND pruning paying off
// and drowning in per-round closure allocations.
func sortByDoc(cursors []*cursor) {
	for i := 1; i < len(cursors); i++ {
		c := cursors[i]
		d := c.doc()
		j := i - 1
		for j >= 0 && cursors[j].doc() > d {
			cursors[j+1] = cursors[j]
			j--
		}
		cursors[j+1] = c
	}
}

// scoreDoc sums the contributions of every cursor currently positioned on doc
// and advances those cursors past it. cursors is in term-id order, so the sum is
// evaluated in a fixed, canonical order shared by both query paths. docLen is
// the document's length, looked up once by the caller.
func (idx *Index) scoreDoc(cursors []*cursor, doc uint32, docLen float64, sc weight.CorpusScorer, tr *trace) float64 {
	var s float64
	for _, c := range cursors {
		if c.pos < len(c.docs) && c.docs[c.pos] == doc {
			s += c.weight * sc.ScoreStats(idx.stats(c.term, float64(c.tfs[c.pos]), docLen))
			c.pos++
			if tr != nil {
				tr.scoreCalls++
			}
		}
	}
	if tr != nil {
		tr.evaluated++
	}
	return s
}

// trace records how much work a query did, for the fair-comparison harness. nil
// on the public query paths.
type trace struct {
	evaluated  int // documents fully scored
	scoreCalls int // per-(term,document) ScoreStats calls
	union      int // documents containing >= 1 query term (the brute candidate set)
}

// SearchBrute is the document-at-a-time baseline and correctness oracle: it
// merges the query terms' postings and fully scores EVERY document containing
// any query term, keeping the top k. Result is ordered best-first by the strict
// (score desc, id asc) order. Returns nil for k <= 0, an empty query, or a query
// whose terms are all absent from the index. If fewer than k documents match,
// all of them are returned.
func (idx *Index) SearchBrute(query []uint32, sc weight.CorpusScorer, k int) []Hit {
	hits, _ := idx.searchBrute(query, sc, k, nil)
	return hits
}

func (idx *Index) searchBrute(query []uint32, sc weight.CorpusScorer, k int, tr *trace) ([]Hit, *trace) {
	if k <= 0 || len(query) == 0 {
		return nil, tr
	}
	cursors := idx.buildCursors(query, nil)
	if len(cursors) == 0 {
		return nil, tr
	}
	tk := newTopK(k)
	for {
		// Smallest current document across all cursors.
		min := uint32(sentinel)
		for _, c := range cursors {
			if d := c.doc(); d < min {
				min = d
			}
		}
		if min == sentinel {
			break
		}
		score := idx.scoreDoc(cursors, min, idx.docLen[min], sc, tr)
		tk.push(Hit{ID: idx.docID[min], Score: score})
	}
	if tr != nil {
		tr.union = tr.evaluated
	}
	return tk.result(), tr
}

// Search is WAND: it returns the same top-k as [Index.SearchBrute] — identical
// ids, scores and order — but skips documents whose score upper bound cannot
// beat the current k-th best, so on a large corpus it scores a fraction of the
// candidates. The scorer must satisfy the package "scorer contract" (additive,
// non-negative, present-only); for others use SearchBrute. Same nil/edge-case
// behaviour as SearchBrute.
func (idx *Index) Search(query []uint32, sc weight.CorpusScorer, k int) []Hit {
	hits, _ := idx.search(query, sc, k, nil)
	return hits
}

func (idx *Index) search(query []uint32, sc weight.CorpusScorer, k int, tr *trace) ([]Hit, *trace) {
	if k <= 0 || len(query) == 0 {
		return nil, tr
	}
	ub := idx.Prepare(sc)
	cursors := idx.buildCursors(query, ub)
	if len(cursors) == 0 {
		return nil, tr
	}
	if tr != nil {
		seen := map[uint32]struct{}{}
		for _, c := range cursors {
			for _, d := range c.docs {
				seen[d] = struct{}{}
			}
		}
		tr.union = len(seen)
	}

	// ord is reordered by current document each round for the pivot search; the
	// cursors it points at are the same objects scoreDoc reads in term-id order.
	ord := make([]*cursor, len(cursors))
	copy(ord, cursors)

	tk := newTopK(k)
	for {
		sortByDoc(ord)
		if ord[0].doc() == sentinel {
			break // every cursor exhausted
		}

		// threshold: a document must be able to reach this score to matter. The
		// heap's worst kept score once it is full; -inf while it is filling, so
		// nothing is pruned until there are k candidates.
		//
		// A tiny slack is subtracted so a document whose TRUE score ties the
		// threshold is always evaluated. The upper bound (cum) is summed in
		// document order while a document's real score is summed in term-id
		// order, so floating-point rounding can leave cum a ULP below a genuine
		// tie; without the slack such a tie could be pruned and the id tiebreak
		// decided wrongly. Widening the evaluation band only ever scores a few
		// EXTRA documents — the heap still keeps the exact top-k by real score —
		// so the result stays identical to the brute scan regardless of slack.
		theta := math.Inf(-1)
		if tk.full() {
			w := tk.worst()
			theta = w - 1e-12*(math.Abs(w)+1)
		}

		// Pivot: the first cursor at which the cumulative upper bound reaches
		// theta. Documents before it cannot reach theta (their max is the prefix
		// before the pivot, which is < theta) and are safe to skip. >= keeps a
		// document whose bound exactly equals theta — it could still tie and win
		// on the id tiebreak.
		var cum float64
		pivot := -1
		for i, c := range ord {
			if c.doc() == sentinel {
				break
			}
			cum += c.ub
			if cum >= theta {
				pivot = i
				break
			}
		}
		if pivot < 0 {
			break // no remaining document can reach the threshold
		}
		pivotDoc := ord[pivot].doc()

		if ord[0].doc() == pivotDoc {
			// All cursors before the pivot sit on pivotDoc; scoreDoc scores every
			// cursor at it (in canonical order) and advances them.
			score := idx.scoreDoc(cursors, pivotDoc, idx.docLen[pivotDoc], sc, tr)
			tk.push(Hit{ID: idx.docID[pivotDoc], Score: score})
		} else {
			// Some cursor before the pivot lags behind pivotDoc; skip it forward.
			// ord[0] is the laggard with the smallest document.
			ord[0].skipTo(pivotDoc)
		}
	}
	return tk.result(), tr
}
