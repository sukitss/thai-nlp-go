// Package vocab maps terms to dense sequential ids (0, 1, 2, ...) and tracks
// document frequency, the vocabulary-dictionary mechanism used by classic
// lexical search engines (Lucene/Elasticsearch term dictionaries, Qdrant
// miniCOIL) for sparse-vector and BM25-style retrieval.
//
// Why not just hash terms to dimensions? Hashing (e.g. FNV-32) maps an
// unbounded term space onto a fixed integer range, so distinct terms can and
// do collide — a query for one term then matches documents containing an
// unrelated term. Sequential assignment is collision-free by construction:
// every distinct term gets its own dimension, and Len() tells you exactly how
// many dimensions exist. The per-term document frequency lets the consumer
// compute IDF weights (see IDF) without a second pass over the corpus.
//
// The package is a stateless library mechanism: Builder accumulates terms in
// memory, Save/Load provide a versioned binary snapshot, and Vocab.Extend
// resumes building from a snapshot with all existing ids unchanged. Where the
// snapshot lives and how it is synchronized across processes is policy that
// belongs to the caller.
//
// Terms are arbitrary byte strings: Thai, emoji, and even the empty string ""
// are legal, distinct terms — the package does not police tokenization. Ids
// are uint32, so a vocabulary holds at most 1<<32 terms.
package vocab

import (
	"io"
	"math"
)

// table is the shared term→id / id→(term, df) storage behind Builder and
// Vocab; both embed it and promote its read methods.
type table struct {
	ids   map[string]uint32
	terms []string // id → term, in assignment order
	df    []int    // id → document frequency
	docs  int
}

// ID returns the id assigned to term, and whether the term is present.
func (t *table) ID(term string) (uint32, bool) {
	id, ok := t.ids[term]
	return id, ok
}

// DF returns term's document frequency: the number of documents passed to
// AddDoc that contained it. Unknown terms (and terms only ever registered via
// GetOrAssign) have DF 0.
func (t *table) DF(term string) int {
	if id, ok := t.ids[term]; ok {
		return t.df[id]
	}
	return 0
}

// DocCount reports how many documents have been added with AddDoc.
func (t *table) DocCount() int { return t.docs }

// Len reports the number of distinct terms. Ids are always exactly
// 0..Len()-1.
func (t *table) Len() int { return len(t.terms) }

// IDF returns the BM25 inverse document frequency of term:
//
//	ln(1 + (N - df + 0.5) / (df + 0.5))
//
// where N is DocCount() and df is DF(term). An unseen term uses df = 0 and so
// gets the maximum weight for the current corpus; with no documents at all
// every term scores ln(2). The +0.5 smoothing keeps the value positive and
// finite for every df in 0..N.
func (t *table) IDF(term string) float64 {
	df := float64(t.DF(term))
	return math.Log(1 + (float64(t.docs)-df+0.5)/(df+0.5))
}

// Terms calls fn for every term in id order (id 0 first). Iteration stops
// early if fn returns false.
func (t *table) Terms(fn func(term string, id uint32, df int) bool) {
	for id, term := range t.terms {
		if !fn(term, uint32(id), t.df[id]) {
			return
		}
	}
}

// Builder accumulates a vocabulary: it assigns sequential ids to new terms and
// counts document frequency. It is NOT safe for concurrent use — build from a
// single goroutine, then Save (or hand out the loaded Vocab) for readers.
type Builder struct {
	table
	seen []int // id → stamp of the last doc that counted it (docs are 1-based)
}

// NewBuilder returns an empty Builder.
func NewBuilder() *Builder {
	return &Builder{table: table{ids: map[string]uint32{}}}
}

// GetOrAssign returns term's id, assigning the next sequential id (starting
// at 0, in first-seen order) if the term is new. It never touches document
// frequency — use it for query-side lookups or pre-registering terms; only
// AddDoc counts DF. The same insert order always yields the same ids.
func (b *Builder) GetOrAssign(term string) uint32 {
	if id, ok := b.ids[term]; ok {
		return id
	}
	id := uint32(len(b.terms))
	b.ids[term] = id
	b.terms = append(b.terms, term)
	b.df = append(b.df, 0)
	b.seen = append(b.seen, 0)
	return id
}

// AddDoc records one document's terms: new terms get ids (as GetOrAssign) and
// every distinct term has its document frequency incremented once, no matter
// how many times it repeats within the document.
func (b *Builder) AddDoc(terms []string) {
	b.docs++
	for _, t := range terms {
		id := b.GetOrAssign(t)
		if b.seen[id] != b.docs {
			b.seen[id] = b.docs
			b.df[id]++
		}
	}
}

// Save writes the builder's state to w in the versioned binary format read by
// Load. Output is deterministic: the same builder state always produces
// byte-identical output.
func (b *Builder) Save(w io.Writer) error { return b.table.save(w) }

// Vocab is an immutable snapshot of a vocabulary, as returned by Load. It has
// the same read methods as Builder (ID, DF, DocCount, Len, IDF, Terms) and is
// safe for concurrent readers.
type Vocab struct {
	table
}

// Extend returns a new Builder seeded with a copy of the vocabulary: every
// existing term keeps its id and document frequency, and new terms are
// appended after Len(). The Vocab itself is never modified. Copying is O(Len).
func (v *Vocab) Extend() *Builder {
	b := &Builder{
		table: table{
			ids:   make(map[string]uint32, len(v.ids)),
			terms: append(make([]string, 0, len(v.terms)), v.terms...),
			df:    append(make([]int, 0, len(v.df)), v.df...),
			docs:  v.docs,
		},
		seen: make([]int, len(v.terms)),
	}
	for term, id := range v.ids {
		b.ids[term] = id
	}
	return b
}
