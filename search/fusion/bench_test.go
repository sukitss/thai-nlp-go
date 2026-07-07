package fusion

import (
	"math/rand"
	"testing"
)

// makeList builds a best-first list of n hits with descending scores over an id
// space of span, so two lists partially overlap.
func makeList(rng *rand.Rand, n, span int) []Hit {
	out := make([]Hit, n)
	score := float64(n)
	for i := range out {
		out[i] = Hit{ID: uint32(rng.Intn(span)), Score: score}
		score-- // already descending → best-first
	}
	return out
}

func benchLists(seed int64, n, span int) [][]Hit {
	rng := rand.New(rand.NewSource(seed))
	return [][]Hit{makeList(rng, n, span), makeList(rng, n, span)}
}

func BenchmarkRRF(b *testing.B) {
	lists := benchLists(1, 100, 300)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RRF(lists, 0)
	}
}

func BenchmarkWeightedSumMinMax(b *testing.B) {
	lists := benchLists(2, 100, 300)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = WeightedSum(lists, []float64{0.6, 0.4}, MinMax)
	}
}
