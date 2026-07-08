package invidx

import (
	"math"
	"sort"

	"github.com/sukitss/thai-nlp-go/search/weight"
)

// DefaultBlockSize is the postings block granularity [Index.SearchBlockMax] uses
// for its per-block max impacts. 128 matches Lucene's impacts block size: small
// enough that a block's max bound is tight, large enough that the bound table
// stays cheap (one float64 per 128 postings).
const DefaultBlockSize = 128

// bmKey caches block-max impact tables by (scorer value, block size): the same
// scorer at a different granularity is a different table.
type bmKey struct {
	sc weight.CorpusScorer
	bs int
}

// blockData holds, per term, the maximum RAW (unweighted) contribution within
// each fixed-size block of that term's postings. max[term][i] bounds the score
// any document in block i of the term can receive from it, which is tighter than
// the term-global bound Prepare computes and is what lets Block-Max WAND skip
// whole blocks. A term's block i covers postings [i*blockSize : (i+1)*blockSize).
type blockData struct {
	blockSize int
	max       [][]float64 // term id → per-block max raw contribution
}

// PrepareBlockMax computes and caches sc's per-block max impact table at the
// given block size, scanning every postings list once — the block-granular
// analogue of [Index.Prepare]. [Index.SearchBlockMax] calls it lazily; call it up
// front to keep the query path off the build mutex. Cached by (scorer, blockSize).
// A blockSize <= 0 is treated as [DefaultBlockSize].
func (idx *Index) PrepareBlockMax(sc weight.CorpusScorer, blockSize int) *blockData {
	if blockSize <= 0 {
		blockSize = DefaultBlockSize
	}
	key := bmKey{sc, blockSize}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if bd, ok := idx.bmCache[key]; ok {
		return bd
	}
	bd := &blockData{blockSize: blockSize, max: make([][]float64, idx.numTerms)}
	for t := 0; t < idx.numTerms; t++ {
		p := idx.post[t]
		if len(p.docs) == 0 {
			continue
		}
		nb := (len(p.docs) + blockSize - 1) / blockSize
		mx := make([]float64, nb)
		for i, d := range p.docs {
			v := sc.ScoreStats(idx.stats(uint32(t), float64(p.tfs[i]), idx.docLen[d]))
			b := i / blockSize
			if v > mx[b] {
				mx[b] = v
			}
		}
		bd.max[t] = mx
	}
	idx.bmCache[key] = bd
	return bd
}

// buildCursorsBM is [Index.buildCursors] with the per-cursor block-max table
// attached, for the block-max query path.
func (idx *Index) buildCursorsBM(query []uint32, ub []float64, bd *blockData) []*cursor {
	cursors := idx.buildCursors(query, ub)
	for _, c := range cursors {
		c.blkMax = bd.max[c.term]
		c.blkSize = bd.blockSize
	}
	return cursors
}

// blockAt returns the index of the postings block that would contain target —
// the block of the first posting at or after the cursor's position with document
// >= target — and whether such a posting exists. The block-local bound must be
// read at the block covering the pivot document, NOT the cursor's current block:
// a cursor lagging behind the pivot sits in an earlier block whose max may be
// lower, and using it would under-estimate the pivot's score and wrongly prune it.
func (c *cursor) blockAt(target uint32) (int, bool) {
	rest := c.docs[c.pos:]
	off := sort.Search(len(rest), func(i int) bool { return rest[i] >= target })
	p := c.pos + off
	if p >= len(c.docs) {
		return 0, false
	}
	return p / c.blkSize, true
}

// blockMaxUBAt is the weighted upper bound on this cursor's contribution to
// document target, read from the block that would contain target. 0 if the
// cursor has no posting at or after target (it cannot contribute).
func (c *cursor) blockMaxUBAt(target uint32) float64 {
	b, ok := c.blockAt(target)
	if !ok {
		return 0
	}
	return c.weight * c.blkMax[b]
}

// blockLastDocAt is the largest document id in the block that would contain
// target — the last document blockMaxUBAt(target) covers, hence the first
// document past which a new (possibly higher) block bound applies. sentinel if
// the cursor has no posting at or after target.
func (c *cursor) blockLastDocAt(target uint32) uint32 {
	b, ok := c.blockAt(target)
	if !ok {
		return sentinel
	}
	end := (b + 1) * c.blkSize
	if end > len(c.docs) {
		end = len(c.docs)
	}
	return c.docs[end-1]
}

// SearchBlockMax is Block-Max WAND (Ding & Suel 2011): like [Index.Search] it
// returns the IDENTICAL top-k as [Index.SearchBrute] — same ids, scores and
// order — but it prunes with per-block max impacts as well as the term-global
// bounds, so it can skip whole postings blocks WAND would have to walk. The win
// grows with k (where global bounds are loose) and on long postings lists. Same
// scorer contract and nil/edge-case behaviour as [Index.Search]. Uses
// [DefaultBlockSize]; see [Index.SearchBlockMaxSized] to tune the granularity.
func (idx *Index) SearchBlockMax(query []uint32, sc weight.CorpusScorer, k int) []Hit {
	return idx.searchBlockMax(query, sc, k, DefaultBlockSize, nil)
}

// SearchBlockMaxSized is [Index.SearchBlockMax] with an explicit block size (the
// research/tuning knob). Smaller blocks give tighter bounds and more skipping at
// the cost of a larger impact table and more per-round bookkeeping.
func (idx *Index) SearchBlockMaxSized(query []uint32, sc weight.CorpusScorer, k, blockSize int) []Hit {
	return idx.searchBlockMax(query, sc, k, blockSize, nil)
}

func (idx *Index) searchBlockMax(query []uint32, sc weight.CorpusScorer, k, blockSize int, tr *trace) []Hit {
	if k <= 0 || len(query) == 0 {
		return nil
	}
	ub := idx.Prepare(sc) // term-global bounds drive pivot selection
	bd := idx.PrepareBlockMax(sc, blockSize)
	cursors := idx.buildCursorsBM(query, ub, bd)
	if len(cursors) == 0 {
		return nil
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

	ord := make([]*cursor, len(cursors))
	copy(ord, cursors)

	tk := newTopK(k)
	for {
		sortByDoc(ord)
		if ord[0].doc() == sentinel {
			break
		}

		// threshold with the same tie-preserving slack as WAND: a document whose
		// TRUE score ties the k-th best must still be evaluated (see search.go).
		theta := math.Inf(-1)
		if tk.full() {
			w := tk.worst()
			theta = w - 1e-12*(math.Abs(w)+1)
		}

		// Stage 1 — WAND pivot on the term-GLOBAL bounds: the first cursor at
		// which the cumulative global upper bound reaches theta. Documents before
		// it cannot reach theta and are skipped for free.
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

		// The cursors that could contain pivotDoc are exactly those with current
		// document <= pivotDoc (postings are ascending, so a cursor holding
		// pivotDoc at a later position already has doc() <= pivotDoc). WAND's pivot
		// can fall on the first of several cursors tied on pivotDoc, so extend the
		// prefix over the ties: every tied cursor genuinely contributes to
		// pivotDoc and must be in the bound, or it would under-estimate.
		le := pivot
		for le+1 < len(ord) && ord[le+1].doc() <= pivotDoc {
			le++
		}

		// Stage 2 — Block-Max refinement: sum each prefix cursor's bound read at
		// the block that would contain pivotDoc. This is a true upper bound on
		// pivotDoc's score, so if it cannot reach theta, pivotDoc — and every
		// document up to the earliest of those blocks' ends — is safe to skip.
		var bcum float64
		for i := 0; i <= le; i++ {
			bcum += ord[i].blockMaxUBAt(pivotDoc)
		}
		if bcum < theta {
			// Jump just past the earliest block end among the prefix, but never
			// past the next cursor beyond the prefix (a new term could add score
			// there) and always make progress past pivotDoc.
			minEnd := uint32(sentinel)
			pick := 0
			for i := 0; i <= le; i++ {
				if e := ord[i].blockLastDocAt(pivotDoc); e < minEnd {
					minEnd = e
					pick = i
				}
			}
			next := minEnd + 1
			if next <= pivotDoc {
				next = pivotDoc + 1
			}
			if le+1 < len(ord) {
				if nd := ord[le+1].doc(); nd < next { // nd > pivotDoc by prefix def
					next = nd
				}
			}
			ord[pick].skipTo(next)
			continue
		}

		// The blocks can reach theta: fall back to WAND's evaluate/advance step.
		if ord[0].doc() == pivotDoc {
			score := idx.scoreDoc(cursors, pivotDoc, idx.docLen[pivotDoc], sc, tr)
			tk.push(Hit{ID: idx.docID[pivotDoc], Score: score})
		} else {
			ord[0].skipTo(pivotDoc)
		}
	}
	return tk.result()
}
