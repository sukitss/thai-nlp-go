package invidx

import (
	"math/rand"
	"testing"

	"github.com/sukitss/thai-nlp-go/search/weight"
)

// benchIndex builds a Zipf-ish random corpus: term ids drawn from a skewed
// distribution so a few terms have long postings (the case pruning is built
// for), the rest short.
func benchIndex(n, v, avgLen int) (*Index, *rand.Rand) {
	rng := rand.New(rand.NewSource(42))
	b := NewBuilder()
	for i := 0; i < n; i++ {
		nt := 1 + rng.Intn(2*avgLen)
		seen := map[uint32]int{}
		var terms, tfs []uint32
		for len(terms) < nt {
			// Skew toward small ids: square a uniform to bias low.
			f := rng.Float64()
			t := uint32(float64(v) * f * f)
			if t >= uint32(v) {
				t = uint32(v - 1)
			}
			if _, ok := seen[t]; ok {
				continue
			}
			seen[t] = len(terms)
			terms = append(terms, t)
			tfs = append(tfs, uint32(1+rng.Intn(4)))
		}
		b.Add(uint32(i), terms, tfs)
	}
	return b.Build(), rng
}

func benchQuery(rng *rand.Rand, qlen, v int) []uint32 {
	q := make([]uint32, qlen)
	for i := range q {
		f := rng.Float64()
		q[i] = uint32(float64(v) * f * f)
		if q[i] >= uint32(v) {
			q[i] = uint32(v - 1)
		}
	}
	return q
}

var hitSink []Hit

// benchSearch times Search vs SearchBrute across k and query length on a corpus
// large enough for pruning to bite. Queries are drawn up front and cycled so
// every iteration does real work on a fresh query.
func benchSearch(b *testing.B, wand bool, n, qlen, k int) {
	const v = 5000
	idx, rng := benchIndex(n, v, 12)
	sc := weight.NewBM25()
	idx.Prepare(sc) // exclude the one-time upper-bound precompute from the timing
	queries := make([][]uint32, 256)
	for i := range queries {
		queries[i] = benchQuery(rng, qlen, v)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i&255]
		if wand {
			hitSink = idx.Search(q, sc, k)
		} else {
			hitSink = idx.SearchBrute(q, sc, k)
		}
	}
}

func BenchmarkWAND_100k_q3_k10(b *testing.B)   { benchSearch(b, true, 100000, 3, 10) }
func BenchmarkBrute_100k_q3_k10(b *testing.B)  { benchSearch(b, false, 100000, 3, 10) }
func BenchmarkWAND_100k_q3_k100(b *testing.B)  { benchSearch(b, true, 100000, 3, 100) }
func BenchmarkBrute_100k_q3_k100(b *testing.B) { benchSearch(b, false, 100000, 3, 100) }
func BenchmarkWAND_100k_q6_k10(b *testing.B)   { benchSearch(b, true, 100000, 6, 10) }
func BenchmarkBrute_100k_q6_k10(b *testing.B)  { benchSearch(b, false, 100000, 6, 10) }
