package weight

// BM25 is the Robertson–Zaragoza Okapi BM25 scheme, the lexical-retrieval
// baseline. A query term contributes
//
//	IDF · tf·(K1+1) / (tf + K1·(1 - B + B·docLen/avgDocLen))
//
// where IDF = ln(1 + (N-df+0.5)/(df+0.5)) is the vocab BM25 idf. K1 controls how
// fast the term-frequency contribution saturates and B how strongly long
// documents are penalized. An absent term (tf == 0) contributes 0.
//
// Defaults K1 = 1.2, B = 0.75 (the classic Okapi settings). Construct with
// [NewBM25] for the defaults or set the fields directly.
type BM25 struct {
	K1 float64
	B  float64
}

// NewBM25 returns BM25 with the standard K1 = 1.2, B = 0.75.
func NewBM25() BM25 { return BM25{K1: 1.2, B: 0.75} }

// Score implements [Scorer].
func (m BM25) Score(tf, df, n int, docLen, avgDocLen float64) float64 {
	if tf <= 0 {
		return 0
	}
	f := float64(tf)
	k := m.K1 * lengthNorm(m.B, docLen, avgDocLen)
	return bm25IDF(df, n) * (f * (m.K1 + 1)) / (f + k)
}

// ScoreStats implements [CorpusScorer].
func (m BM25) ScoreStats(s Stats) float64 {
	return m.Score(int(s.TF), s.DF, s.N, s.DocLen, s.AvgDocLen)
}

// BM25Plus is the Lv–Zhai BM25+ variant (2011). It adds a constant lower bound
// Delta to the (already length-normalized) term-frequency component, so a term
// present in a very long document keeps a floor of score instead of being driven
// toward zero — BM25's residual over-penalization of long documents. A query
// term contributes
//
//	IDF · ( tf·(K1+1) / (tf + K1·(1 - B + B·docLen/avgDocLen)) + Delta )
//
// for tf > 0, and 0 for an absent term (the Delta floor applies only to terms
// the document actually contains — it is a lower bound on a match, not a reward
// for a miss). IDF and the length normalization are BM25's.
//
// Defaults K1 = 1.2, B = 0.75, Delta = 1.0 (the paper's setting). The IDF here
// is the vocab BM25 idf, so the family is directly comparable; the original
// paper uses IDF = ln((N+1)/df), which changes the scale but not the ranking of
// documents scored under one scheme.
type BM25Plus struct {
	K1    float64
	B     float64
	Delta float64
}

// NewBM25Plus returns BM25+ with K1 = 1.2, B = 0.75, Delta = 1.0.
func NewBM25Plus() BM25Plus { return BM25Plus{K1: 1.2, B: 0.75, Delta: 1.0} }

// Score implements [Scorer].
func (m BM25Plus) Score(tf, df, n int, docLen, avgDocLen float64) float64 {
	if tf <= 0 {
		return 0
	}
	f := float64(tf)
	k := m.K1 * lengthNorm(m.B, docLen, avgDocLen)
	return bm25IDF(df, n) * (f*(m.K1+1)/(f+k) + m.Delta)
}

// ScoreStats implements [CorpusScorer].
func (m BM25Plus) ScoreStats(s Stats) float64 {
	return m.Score(int(s.TF), s.DF, s.N, s.DocLen, s.AvgDocLen)
}

// BM25L is the Lv–Zhai BM25L variant (2011). Instead of adding a floor after
// normalization (BM25+), it reshapes the length normalization itself: the raw
// term frequency is first divided by the length factor,
//
//	ctf = tf / (1 - B + B·docLen/avgDocLen),
//
// then shifted by Delta before the BM25 saturation, so that long documents are
// no longer systematically ranked below short ones for the same normalized term
// frequency. A query term contributes
//
//	IDF · (K1+1)·(ctf + Delta) / (K1 + ctf + Delta)
//
// for tf > 0, and 0 for an absent term.
//
// Defaults K1 = 1.2, B = 0.75, Delta = 0.5 (the paper's recommended shift). IDF
// is the vocab BM25 idf (see [BM25Plus] on the IDF choice).
type BM25L struct {
	K1    float64
	B     float64
	Delta float64
}

// NewBM25L returns BM25L with K1 = 1.2, B = 0.75, Delta = 0.5.
func NewBM25L() BM25L { return BM25L{K1: 1.2, B: 0.75, Delta: 0.5} }

// Score implements [Scorer].
func (m BM25L) Score(tf, df, n int, docLen, avgDocLen float64) float64 {
	if tf <= 0 {
		return 0
	}
	ctf := float64(tf) / lengthNorm(m.B, docLen, avgDocLen)
	c := ctf + m.Delta
	return bm25IDF(df, n) * (m.K1 + 1) * c / (m.K1 + c)
}

// ScoreStats implements [CorpusScorer].
func (m BM25L) ScoreStats(s Stats) float64 {
	return m.Score(int(s.TF), s.DF, s.N, s.DocLen, s.AvgDocLen)
}
