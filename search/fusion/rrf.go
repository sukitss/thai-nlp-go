package fusion

// DefaultRRFK is the standard reciprocal-rank-fusion constant (Cormack et al.
// 2009). It damps the influence of the very top ranks so no single list can
// dominate on rank 1 alone; 60 is the value from the original paper and the
// common default in search stacks.
const DefaultRRFK = 60

// RRF fuses the lists by Reciprocal Rank Fusion: a document's fused score is the
// sum over the lists it appears in of 1/(k + rank), where rank is its 1-based
// position in that list (the best hit is rank 1). k <= 0 uses [DefaultRRFK]
// (60); a larger k flattens the rank curve (later ranks matter relatively more),
// a smaller k sharpens the advantage of the top ranks.
//
// RRF is RANK-based, so it needs no score normalization and is unaffected by the
// lists being on wildly different scales (BM25 vs cosine) — the property that
// makes it the robust default for sparse+dense fusion. Each list must be ordered
// best-first (every search/* primitive returns that order); RRF reads position
// as rank. The result is ordered by fused score descending, ties broken by lower
// id. Empty or all-empty input returns nil.
func RRF(lists [][]Hit, k int) []Hit {
	if k <= 0 {
		k = DefaultRRFK
	}
	kf := float64(k)
	scores := make(map[uint32]float64)
	for _, list := range lists {
		for pos, h := range list {
			scores[h.ID] += 1 / (kf + float64(pos+1)) // rank = pos+1 (1-based)
		}
	}
	return rank(scores)
}
