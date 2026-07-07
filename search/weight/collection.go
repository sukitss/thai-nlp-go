package weight

// Collection holds the corpus-level statistics the schemes read: per-term
// document frequency and collection frequency plus the document-length
// aggregates (mean length, total tokens). It is the piece [vocab.Vocab] does not
// provide — vocab tracks document frequency and DocCount but not collection
// frequency or lengths — so a Collection completes the [Stats] every scheme,
// including the collection-frequency schemes (DFR/PL2, QLDirichlet), needs.
//
// Build it from the tokenized corpus with [NewCollection], using the SAME tokens
// you index (multi/tokenize output). It is read-only after construction and safe
// for concurrent readers.
type Collection struct {
	n         int
	avgDocLen float64
	collLen   float64
	df        map[string]int
	cf        map[string]int
}

// NewCollection computes the corpus statistics from docs, each document already
// tokenized. Document frequency counts a term once per document (however many
// times it repeats there); collection frequency counts every occurrence.
func NewCollection(docs [][]string) *Collection {
	c := &Collection{
		n:  len(docs),
		df: make(map[string]int),
		cf: make(map[string]int),
	}
	seen := make(map[string]int, 64) // term → 1-based index of the last doc that counted its df
	var total float64
	for i, doc := range docs {
		total += float64(len(doc))
		di := i + 1
		for _, t := range doc {
			c.cf[t]++
			if seen[t] != di {
				seen[t] = di
				c.df[t]++
			}
		}
	}
	c.collLen = total
	if c.n > 0 {
		c.avgDocLen = total / float64(c.n)
	}
	return c
}

// N reports the number of documents.
func (c *Collection) N() int { return c.n }

// AvgDocLen reports the mean document length in tokens (0 for an empty corpus).
func (c *Collection) AvgDocLen() float64 { return c.avgDocLen }

// CollLen reports the total number of tokens across the corpus.
func (c *Collection) CollLen() float64 { return c.collLen }

// DF returns term's document frequency (0 for an unseen term).
func (c *Collection) DF(term string) int { return c.df[term] }

// CF returns term's collection frequency: its total occurrences across the
// corpus (0 for an unseen term).
func (c *Collection) CF(term string) int { return c.cf[term] }

// Stats assembles the [Stats] for one query term in one document, given the
// term's frequency tf in that document and the document's length docLen. It
// fills df, cf, N, mean length and corpus token total from the collection, so a
// caller rarely constructs a Stats by hand.
func (c *Collection) Stats(term string, tf, docLen float64) Stats {
	return Stats{
		TF:        tf,
		DF:        c.df[term],
		N:         c.n,
		DocLen:    docLen,
		AvgDocLen: c.avgDocLen,
		CF:        float64(c.cf[term]),
		CollLen:   c.collLen,
	}
}
