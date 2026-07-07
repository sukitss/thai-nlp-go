package weight

import "math"

// TFIDF is the classic tf·idf scheme, the reference every other scheme is read
// against. It follows the SMART lnc.ltc lineage: a logarithmically dampened term
// frequency times an inverse document frequency. A query term contributes
//
//	(1 + ln(tf)) · ln(N/df)
//
// for tf > 0, and 0 for an absent term. The 1+ln(tf) damping is the "l"
// (logarithmic) term-frequency of lnc.ltc; ln(N/df) is the textbook idf ("t").
//
// Deviation from full lnc.ltc: the canonical scheme divides each document (and
// the query) vector by its Euclidean norm — cosine normalization ("c"). That is
// a document-level operation over the whole term vector, which a per-term
// [Scorer] cannot see, so TFIDF here omits it. The effect is that longer
// documents accumulate larger raw scores; if you want the cosine-normalized
// ranking, divide [ScoreDoc]'s result by the document's term-weight norm. The
// logarithm base does not affect the ranking. This scheme is included as a
// familiar reference point, not because it is expected to win.
type TFIDF struct{}

// NewTFIDF returns the classic tf·idf reference scorer.
func NewTFIDF() TFIDF { return TFIDF{} }

// Score implements [Scorer]. With df == 0 (an unseen term) or df >= N the idf
// factor ln(N/df) is clamped to 0, so an out-of-corpus term contributes nothing
// rather than a negative or infinite weight.
func (TFIDF) Score(tf, df, n int, docLen, avgDocLen float64) float64 {
	if tf <= 0 || df <= 0 || n <= 0 || df >= n {
		return 0
	}
	return (1 + math.Log(float64(tf))) * math.Log(float64(n)/float64(df))
}

// ScoreStats implements [CorpusScorer].
func (t TFIDF) ScoreStats(s Stats) float64 {
	return t.Score(int(s.TF), s.DF, s.N, s.DocLen, s.AvgDocLen)
}
