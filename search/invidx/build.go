package invidx

import (
	"sync"

	"github.com/sukitss/thai-nlp-go/search/weight"
)

// postings is one term's inverted list: the internal indices of the documents
// containing the term (ascending) and the parallel term frequencies. Internal
// indices are assigned in Add order, so appending as documents arrive keeps docs
// sorted with no explicit sort — the invariant document-at-a-time scanning and
// WAND both rely on.
type postings struct {
	docs []uint32 // internal doc index, ascending
	tfs  []uint32 // term frequency, parallel to docs
}

// Index is an immutable inverted index. It is built by a [Builder] and never
// mutated afterwards (apart from the internally-synchronized upper-bound cache),
// so it is safe for concurrent SearchBrute/Search once building is done.
//
// The index holds the corpus statistics itself rather than borrowing a
// [weight.Collection]: Collection is keyed by string terms, whereas an inverted
// index is keyed by [vocab] term ids, and every statistic a scheme needs is
// already implied by the postings — document frequency is a postings list's
// length, collection frequency the sum of its term frequencies, document length
// the sum of a document's term frequencies. Recomputing them term-side would
// duplicate the index; instead [Index.stats] assembles the same [weight.Stats] a
// Collection would, from the index's own arrays.
type Index struct {
	post      []postings // term id → postings (dense; index by term id)
	docID     []uint32   // internal index → caller-supplied document id
	docLen    []float64  // internal index → document length in tokens
	df        []int      // term id → document frequency (len of its postings)
	cf        []int      // term id → collection frequency (sum of its tfs)
	n         int        // number of documents
	avgDocLen float64    // mean document length
	collLen   float64    // total tokens across the corpus
	numTerms  int        // len(post): highest term id + 1 seen at build

	mu      sync.Mutex
	ubCache map[weight.CorpusScorer][]float64 // scorer → per-term max contribution
	bmCache map[bmKey]*blockData              // (scorer, blockSize) → per-block max impacts
}

// N reports the number of documents in the index.
func (idx *Index) N() int { return idx.n }

// NumTerms reports the number of distinct term ids the index has postings for
// (one past the highest term id seen at build time).
func (idx *Index) NumTerms() int { return idx.numTerms }

// AvgDocLen reports the mean document length in tokens.
func (idx *Index) AvgDocLen() float64 { return idx.avgDocLen }

// DF returns term's document frequency (0 for a term with no postings or out of
// range).
func (idx *Index) DF(term uint32) int {
	if int(term) >= idx.numTerms {
		return 0
	}
	return idx.df[term]
}

// CF returns term's collection frequency: its total occurrences across the
// corpus (0 for a term with no postings or out of range).
func (idx *Index) CF(term uint32) int {
	if int(term) >= idx.numTerms {
		return 0
	}
	return idx.cf[term]
}

// stats assembles the [weight.Stats] for one term in one document, from the
// index's own arrays — the same value a [weight.Collection] would hand a scheme.
func (idx *Index) stats(term uint32, tf, docLen float64) weight.Stats {
	return weight.Stats{
		TF:        tf,
		DF:        idx.df[term],
		N:         idx.n,
		DocLen:    docLen,
		AvgDocLen: idx.avgDocLen,
		CF:        float64(idx.cf[term]),
		CollLen:   idx.collLen,
	}
}

// Builder accumulates documents and produces an immutable [Index]. Add
// documents in whatever id order you like — internal indices are assigned in
// Add order and postings stay sorted automatically. NOT safe for concurrent
// use; build from one goroutine, then Build and hand out the Index.
type Builder struct {
	post     []postings
	docID    []uint32
	docLen   []float64
	numTerms int
}

// NewBuilder returns an empty Builder.
func NewBuilder() *Builder { return &Builder{} }

// Add records one document: docID is the caller's id (returned in [Hit]s), terms
// are its DISTINCT term ids and tfs the parallel term frequencies (tfs[i] is how
// many times terms[i] occurs). The document length is the sum of tfs. terms and
// tfs must have equal length; a term id repeated within one call double-counts,
// so pass distinct ids (use [Count] to derive them from a token stream).
//
// Panics if len(terms) != len(tfs).
func (b *Builder) Add(docID uint32, terms []uint32, tfs []uint32) {
	if len(terms) != len(tfs) {
		panic("invidx: Builder.Add terms/tfs length mismatch")
	}
	d := uint32(len(b.docID)) // internal index for this document
	var dl float64
	for i, t := range terms {
		if int(t) >= b.numTerms {
			b.numTerms = int(t) + 1
		}
		if int(t) >= len(b.post) {
			grown := make([]postings, int(t)+1)
			copy(grown, b.post)
			b.post = grown
		}
		p := &b.post[t]
		p.docs = append(p.docs, d)
		p.tfs = append(p.tfs, tfs[i])
		dl += float64(tfs[i])
	}
	b.docID = append(b.docID, docID)
	b.docLen = append(b.docLen, dl)
}

// Build finalizes the accumulated documents into an immutable [Index],
// computing the corpus aggregates. The Builder must not be used afterwards.
func (b *Builder) Build() *Index {
	idx := &Index{
		post:     b.post,
		docID:    b.docID,
		docLen:   b.docLen,
		n:        len(b.docID),
		numTerms: b.numTerms,
		df:       make([]int, b.numTerms),
		cf:       make([]int, b.numTerms),
		ubCache:  map[weight.CorpusScorer][]float64{},
		bmCache:  map[bmKey]*blockData{},
	}
	for t := 0; t < b.numTerms; t++ {
		p := b.post[t]
		idx.df[t] = len(p.docs)
		var c int
		for _, tf := range p.tfs {
			c += int(tf)
		}
		idx.cf[t] = c
	}
	var total float64
	for _, dl := range b.docLen {
		total += dl
	}
	idx.collLen = total
	if idx.n > 0 {
		idx.avgDocLen = total / float64(idx.n)
	}
	return idx
}

// Count turns a stream of term ids (a tokenized, vocab-mapped document, with
// repeats) into the distinct-terms / term-frequencies pair [Builder.Add] takes.
// Term order in the result is unspecified but the two slices are parallel.
func Count(tokenIDs []uint32) (terms []uint32, tfs []uint32) {
	seen := make(map[uint32]int, len(tokenIDs))
	for _, id := range tokenIDs {
		if pos, ok := seen[id]; ok {
			tfs[pos]++
			continue
		}
		seen[id] = len(terms)
		terms = append(terms, id)
		tfs = append(tfs, 1)
	}
	return terms, tfs
}
