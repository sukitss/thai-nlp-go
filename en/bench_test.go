package en

import "testing"

func BenchmarkCut(b *testing.B) {
	const s = "The quick brown fox jumps over the lazy dog, don't you know?"
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(Cut(s))
	}
	_ = n
}
