package weight

import "math"

// QLDirichlet is the query-likelihood language model with Dirichlet smoothing
// (Zhai–Lafferty 2001) — a different paradigm from the tf·idf family. Instead of
// weighting a term by rarity, it models each document as a language model and
// ranks documents by the log-probability that they GENERATED the query:
//
//	log p(t|d) = log( (tf + Mu·p(t|C)) / (docLen + Mu) )
//
// where p(t|C) = cf/collLen is the term's probability in the whole collection
// (its collection frequency over the total token count). Mu is the Dirichlet
// prior — effectively the number of pseudo-tokens of the background model mixed
// into every document; larger Mu smooths more (helps short documents), smaller
// Mu trusts the document counts. Default Mu = 2000, the standard setting.
//
// Because the smoothing gives every term positive probability, a query term
// ABSENT from a document (tf == 0) still contributes log(Mu·p(t|C)/(docLen+Mu)),
// a negative number that penalizes the document for the miss — this background
// term is the point of smoothing, so [ScoreDoc] scores absent query terms too.
// Document scores are sums of log-probabilities and are therefore negative;
// larger (closer to zero) means more likely, i.e. better, so they order
// documents the same direction as the other schemes.
//
// QLDirichlet needs the term's collection frequency AND the corpus token total,
// neither of which document frequency provides, so it implements [CorpusScorer]
// only (not [Scorer]): it reads Stats.CF and Stats.CollLen. This is the primary
// interface deviation the design notes flagged.
type QLDirichlet struct {
	Mu float64
}

// NewQLDirichlet returns a Dirichlet-smoothed query-likelihood scorer with the
// standard Mu = 2000.
func NewQLDirichlet() QLDirichlet { return QLDirichlet{Mu: 2000} }

// ScoreStats implements [CorpusScorer]. It reads Stats.CF and Stats.CollLen for
// the background model and Stats.DocLen for the document model; Stats.DF and
// Stats.N are unused. A term with no collection probability (CF == 0, e.g. a
// query term never seen in the corpus) contributes 0 rather than -inf.
func (m QLDirichlet) ScoreStats(s Stats) float64 {
	if s.CollLen <= 0 || s.CF <= 0 {
		return 0
	}
	pC := s.CF / s.CollLen
	return math.Log((s.TF + m.Mu*pC) / (s.DocLen + m.Mu))
}
