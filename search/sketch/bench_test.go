package sketch

import (
	"fmt"
	"testing"
)

// benchShingles is a realistic ~250-shingle document set reused across the
// MinHash benchmarks so they measure hashing+folding, not setup.
func benchShingles() []string {
	s := make([]string, 250)
	for i := range s {
		s[i] = fmt.Sprintf("shingle-token-%d", i)
	}
	return s
}

var (
	benchSigSink []uint64
	benchU64Sink uint64
	benchF64Sink float64
	benchCandSnk []uint32
)

func BenchmarkMinHashSignature128(b *testing.B) {
	mh := NewMinHash(128, 1)
	sh := benchShingles()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSigSink = mh.SignatureStrings(sh)
	}
}

func BenchmarkMinHashSignature256(b *testing.B) {
	mh := NewMinHash(256, 1)
	sh := benchShingles()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSigSink = mh.SignatureStrings(sh)
	}
}

func BenchmarkEstJaccard128(b *testing.B) {
	mh := NewMinHash(128, 1)
	sa := mh.SignatureStrings(benchShingles())
	sb := mh.SignatureStrings(benchShingles()[:200])
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchF64Sink = EstJaccard(sa, sb)
	}
}

func BenchmarkLSHQuery(b *testing.B) {
	mh := NewMinHash(128, 1)
	l := NewLSH(32, 4, 1)
	for i := 0; i < 10000; i++ {
		l.Add(uint32(i), mh.SignatureStrings([]string{fmt.Sprintf("doc%d", i), "shared", "tail"}))
	}
	q := mh.SignatureStrings([]string{"doc42", "shared", "tail"})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCandSnk = l.Query(q)
	}
}

func BenchmarkSimHashTokens(b *testing.B) {
	sh := NewSimHash(1)
	toks := benchShingles() // 250 distinct "tokens"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchU64Sink = sh.HashTokens(toks)
	}
}

func BenchmarkSimHashHamming(b *testing.B) {
	x, y := uint64(0xdeadbeefcafef00d), uint64(0x0123456789abcdef)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchU64Sink += uint64(HammingDistance(x, y))
	}
}

func BenchmarkCountMinAdd(b *testing.B) {
	cm := NewCountMinParams(0.001, 0.001, 1)
	key := []byte("a-representative-term")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cm.Add(key, 1)
	}
}

func BenchmarkCountMinEstimate(b *testing.B) {
	cm := NewCountMinParams(0.001, 0.001, 1)
	cm.AddString("a-representative-term", 100)
	key := []byte("a-representative-term")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchU64Sink += cm.Estimate(key)
	}
}
