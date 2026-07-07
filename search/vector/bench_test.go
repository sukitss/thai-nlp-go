package vector

import (
	"fmt"
	"testing"
)

// benchDim / benchN are a realistic-ish in-memory corpus: BGE-m3-scale
// dimensionality at a mid-size corpus, small enough to keep `go test` fast.
const (
	benchDim = 1024
	benchN   = 20000
)

func benchCorpus() ([][]float32, []float32) {
	vecs := randVecs(benchN, benchDim, 32, 100)
	query := randVecs(1, benchDim, 32, 101)[0]
	return vecs, query
}

func benchQuantizers(tb testing.TB, vecs [][]float32) []Quantizer {
	pq, err := TrainPQ(vecs[:min(4000, len(vecs))], 64, 8, 1)
	if err != nil {
		tb.Fatal(err)
	}
	return []Quantizer{NewFloat32(benchDim), NewBinary(benchDim), NewScalar8(benchDim), pq}
}

// BenchmarkEncode: per-vector encode throughput for each quantizer.
func BenchmarkEncode(b *testing.B) {
	vecs, _ := benchCorpus()
	for _, q := range benchQuantizers(b, vecs) {
		b.Run(q.Name(), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = q.Encode(vecs[i%len(vecs)])
			}
		})
	}
}

// BenchmarkTopK: single-goroutine top-10 latency per quantizer over benchN
// vectors.
func BenchmarkTopK(b *testing.B) {
	vecs, query := benchCorpus()
	for _, q := range benchQuantizers(b, vecs) {
		f := NewFlat(q)
		for i, v := range vecs {
			f.Add(uint32(i), v)
		}
		b.Run(q.Name(), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = f.TopK(query, 10)
			}
		})
	}
}

// BenchmarkTopKParallel: top-10 latency with several worker counts.
func BenchmarkTopKParallel(b *testing.B) {
	vecs, query := benchCorpus()
	for _, q := range benchQuantizers(b, vecs) {
		f := NewFlat(q)
		for i, v := range vecs {
			f.Add(uint32(i), v)
		}
		for _, w := range []int{2, 4, 8} {
			b.Run(fmt.Sprintf("%s/w%d", q.Name(), w), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = f.TopKParallel(query, 10, w)
				}
			})
		}
	}
}
