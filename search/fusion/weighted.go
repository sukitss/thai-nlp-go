package fusion

import "math"

// Normalizer maps one list's raw scores onto a comparable range so lists on
// different scales can be added. It takes the scores in list order and returns a
// parallel slice of normalized values (same length, same order). The input is
// never mutated. [MinMax] and [ZScore] are the built-ins; a nil Normalizer in
// [WeightedSum] means "no normalization" (add the raw scores as-is).
type Normalizer func(scores []float64) []float64

// MinMax scales scores to [0, 1] as (x - min) / (max - min): the best hit in the
// list maps to 1, the worst to 0. When every score is equal (max == min, e.g. a
// single-hit list) the range is zero and every score maps to 1 — the list has no
// internal ordering to preserve, so each member gets full presence credit rather
// than collapsing to 0. This is the workhorse normalizer for weighted fusion.
func MinMax(scores []float64) []float64 {
	out := make([]float64, len(scores))
	if len(scores) == 0 {
		return out
	}
	min, max := scores[0], scores[0]
	for _, s := range scores[1:] {
		if s < min {
			min = s
		}
		if s > max {
			max = s
		}
	}
	rng := max - min
	if rng == 0 {
		for i := range out {
			out[i] = 1
		}
		return out
	}
	for i, s := range scores {
		out[i] = (s - min) / rng
	}
	return out
}

// ZScore standardizes scores to zero mean and unit variance ((x - mean) / stddev,
// population standard deviation). Unlike [MinMax] it keeps the shape of the
// distribution (outliers stay far out), which suits lists whose scores are
// roughly normal. A zero-variance list (all scores equal) maps every score to 0.
func ZScore(scores []float64) []float64 {
	out := make([]float64, len(scores))
	n := len(scores)
	if n == 0 {
		return out
	}
	var sum float64
	for _, s := range scores {
		sum += s
	}
	mean := sum / float64(n)
	var varsum float64
	for _, s := range scores {
		d := s - mean
		varsum += d * d
	}
	std := math.Sqrt(varsum / float64(n))
	if std == 0 {
		return out // all equal → all zero
	}
	for i, s := range scores {
		out[i] = (s - mean) / std
	}
	return out
}

// WeightedSum fuses the lists by normalizing each list's scores with norm, then
// adding the normalized scores with a per-list weight: a document's fused score
// is Σ weights[i] * norm(list[i])[rank of the doc in list i], over the lists that
// contain it. Pass [MinMax] (or [ZScore]) for norm; a nil norm adds the raw
// scores unchanged — only meaningful when the lists are already on the same
// scale.
//
// weights supplies one weight per list; a nil weights slice weights every list
// equally (1.0). Panics if weights is non-nil and its length differs from the
// number of lists. The result is ordered by fused score descending, ties broken
// by lower id. Empty or all-empty input returns nil.
func WeightedSum(lists [][]Hit, weights []float64, norm Normalizer) []Hit {
	if weights != nil && len(weights) != len(lists) {
		panic("fusion: WeightedSum weights length must match number of lists")
	}
	scores := make(map[uint32]float64)
	for i, list := range lists {
		if len(list) == 0 {
			continue
		}
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		var norms []float64
		if norm != nil {
			raw := make([]float64, len(list))
			for j, h := range list {
				raw[j] = h.Score
			}
			norms = norm(raw)
		}
		for j, h := range list {
			s := h.Score
			if norms != nil {
				s = norms[j]
			}
			scores[h.ID] += w * s
		}
	}
	return rank(scores)
}
