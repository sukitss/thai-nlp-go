package weight

import "math"

// log2e is log2(e) = 1/ln(2), the factor that converts the natural-log terms of
// the Stirling approximation in PL2 into bits.
const log2e = 1.4426950408889634

// DFR is a divergence-from-randomness weighting model, implemented as PL2 —
// Poisson estimation (P) with Laplace after-effect normalization (L) and term-
// frequency normalization 2. DFR scores a term by how much its occurrence in a
// document DIVERGES from what a random (Poisson) distribution over the whole
// collection would predict: the more surprising the count, the higher the score.
//
// PL2 needs the term's COLLECTION frequency — its total number of occurrences
// across the corpus — because the Poisson mean is
//
//	lambda = cf / N
//
// (mean occurrences per document). Document frequency cannot supply that, so DFR
// implements [CorpusScorer] only, not [Scorer]: it reads Stats.CF. This is the
// same interface deviation as [QLDirichlet].
//
// The contribution of a query term, with the normalized term frequency
//
//	tfn = tf · log2(1 + C·avgDocLen/docLen)                       (normalization 2)
//
// is the Laplace-normalized Poisson information
//
//	(1/(tfn+1)) · ( tfn·log2(tfn/lambda)
//	              + (lambda + 1/(12·tfn) - tfn)·log2(e)
//	              + 0.5·log2(2π·tfn) )
//
// (the middle and last terms are Stirling's approximation of the Poisson
// surprisal). An absent term (tf == 0, hence tfn == 0) contributes 0.
//
// C is the normalization-2 length parameter; default 1.0. It is the analogue of
// BM25's B and is tuned per collection in practice (Terrier reports values from
// ~1 to ~7 depending on corpus); expose it via the field.
type DFR struct {
	C float64
}

// NewDFR returns a PL2 divergence-from-randomness scorer with C = 1.0.
func NewDFR() DFR { return DFR{C: 1.0} }

// ScoreStats implements [CorpusScorer]. It reads Stats.CF (collection frequency)
// and Stats.N to form the Poisson mean; Stats.DF is unused.
func (m DFR) ScoreStats(s Stats) float64 {
	if s.TF <= 0 || s.CF <= 0 || s.N <= 0 {
		return 0
	}
	dl := s.DocLen
	if dl <= 0 {
		dl = s.AvgDocLen
	}
	tfn := s.TF * math.Log2(1+m.C*s.AvgDocLen/dl)
	if tfn <= 0 {
		return 0
	}
	lambda := s.CF / float64(s.N)
	surprise := tfn*math.Log2(tfn/lambda) +
		(lambda+1.0/(12.0*tfn)-tfn)*log2e +
		0.5*math.Log2(2*math.Pi*tfn)
	return surprise / (tfn + 1)
}
