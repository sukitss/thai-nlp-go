// Package weight provides pluggable term-weighting / ranking schemes over a
// term–document collection, so lexical retrieval is not locked to a single
// BM25. Each scheme scores one query term's contribution to a document from the
// standard information-retrieval statistics; summing the contributions over the
// query terms gives the document's score. It is the ranking mechanism a
// Lucene/Elasticsearch Similarity or a Terrier weighting model wraps, reduced to
// a pure-Go primitive that any caller can embed directly.
//
// # The interface(s)
//
// Most schemes fit the minimal per-term signature:
//
//	type Scorer interface {
//	    Score(tf, df, n int, docLen, avgDocLen float64) float64
//	}
//
// where tf is the term frequency in the document, df its document frequency, n
// the number of documents, and docLen/avgDocLen the document and mean document
// lengths in tokens. BM25, BM25Plus, BM25L and TFIDF implement it.
//
// Two schemes need MORE than df: a divergence-from-randomness model (DFR/PL2)
// and a query-likelihood language model (QLDirichlet) both read the term's
// COLLECTION frequency — its total number of occurrences across the whole
// corpus, not just the number of documents it appears in — and QLDirichlet also
// reads the corpus token total. Document frequency cannot supply either. Rather
// than bolt those onto the five-argument signature (where they would be dead
// weight for BM25), the package carries every statistic a scheme might read in a
// single [Stats] value and defines the superset interface
//
//	type CorpusScorer interface {
//	    ScoreStats(s Stats) float64
//	}
//
// EVERY scheme implements CorpusScorer; the df-only schemes additionally
// implement Scorer. [ScoreDoc] and the whole-document helpers take a
// CorpusScorer, so all six schemes are used through one uniform path. This is
// the interface deviation the design notes anticipated: PL2 and QLDirichlet are
// CorpusScorer-only because they genuinely need collection frequency.
//
// # Building on vocab
//
// [vocab.Vocab] supplies term→id, document frequency and DocCount but tracks
// only df — not collection frequency or document lengths. [Collection] adds
// exactly those: build it from the tokenized corpus (the same tokens you index
// with multi/tokenize) and it yields df, cf, N, mean length and corpus token
// total, assembling the [Stats] each scheme reads.
//
// # Concurrency
//
// A Scorer/CorpusScorer is immutable after construction and safe for concurrent
// Score calls. A [Collection] is read-only once built and safe for concurrent
// readers; building it (NewCollection) is single-goroutine.
package weight

import "math"

// Stats bundles every corpus statistic a scheme in this package might read, so
// schemes that need only document frequency (the BM25 family, TFIDF) and those
// that need collection frequency (DFR/PL2, QLDirichlet) share one call shape.
// The df-only schemes ignore CF and CollLen.
type Stats struct {
	TF        float64 // term frequency in this document
	DF        int     // document frequency: documents containing the term
	N         int     // number of documents in the corpus
	DocLen    float64 // this document's length in tokens
	AvgDocLen float64 // mean document length over the corpus
	CF        float64 // collection frequency: total occurrences of the term in the whole corpus (DFR/QL only)
	CollLen   float64 // total tokens in the corpus, i.e. sum of all document lengths (QL only)
}

// Scorer scores one query term's contribution to a document from the minimal
// document-frequency statistics. Implemented by [BM25], [BM25Plus], [BM25L] and
// [TFIDF]. Schemes that need collection frequency ([DFR], [QLDirichlet]) do not
// implement Scorer — use [CorpusScorer].
type Scorer interface {
	// Score returns the term's contribution. tf is its frequency in the
	// document, df its document frequency, n the number of documents, and
	// docLen/avgDocLen the document and mean document lengths in tokens. A term
	// absent from the document (tf == 0) contributes 0.
	Score(tf, df, n int, docLen, avgDocLen float64) float64
}

// CorpusScorer scores one query term's contribution from the full [Stats]. It is
// the superset interface every scheme implements; the whole-document helpers
// ([ScoreDoc]) take it so BM25 and the collection-frequency schemes (DFR/PL2,
// QLDirichlet) are driven identically.
type CorpusScorer interface {
	ScoreStats(s Stats) float64
}

// bm25IDF is the Robertson/vocab BM25 inverse document frequency,
//
//	ln(1 + (N - df + 0.5) / (df + 0.5)),
//
// identical to vocab.IDF so a Collection built from a vocab and one built here
// weight terms the same. It is non-negative for every df in 0..N and is shared
// by the whole BM25 family, so those schemes differ only in TF normalization.
func bm25IDF(df, n int) float64 {
	d := float64(df)
	return math.Log(1 + (float64(n)-d+0.5)/(d+0.5))
}

// lengthNorm is the BM25 document-length factor 1 - b + b*docLen/avgDocLen. A
// zero or unknown avgDocLen degrades to 1 (no length normalization) rather than
// dividing by zero.
func lengthNorm(b, docLen, avgDocLen float64) float64 {
	if avgDocLen <= 0 {
		return 1 - b + b
	}
	return 1 - b + b*docLen/avgDocLen
}

// ScoreDoc sums a scheme's per-term contributions over the query terms, scoring
// one document. query is the list of query terms (duplicates count, as a
// repeated query term is weighted repeatedly); docTF maps a term to its
// frequency in this document (absent ⇒ 0); docLen is this document's length in
// tokens; and c supplies the corpus statistics (df, cf, N, mean length, token
// total).
//
// Every query term is scored, including terms absent from the document: the
// BM25 family and TFIDF contribute 0 for an absent term, but a query-likelihood
// model ([QLDirichlet]) contributes its smoothed background probability, which
// is the whole point of smoothing — so the helper must not skip tf == 0 terms.
func ScoreDoc(sc CorpusScorer, query []string, docTF map[string]int, docLen float64, c *Collection) float64 {
	var total float64
	for _, term := range query {
		total += sc.ScoreStats(c.Stats(term, float64(docTF[term]), docLen))
	}
	return total
}

// TermFreq counts how many times each token occurs, the per-document term
// frequency source ScoreDoc reads. len(tokens) is the document length to pass
// alongside it.
func TermFreq(tokens []string) map[string]int {
	tf := make(map[string]int, len(tokens))
	for _, t := range tokens {
		tf[t]++
	}
	return tf
}
