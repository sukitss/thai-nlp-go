package translit

import "testing"

func benchWords(b *testing.B) []string { return readLines(b, "testdata/soundex.txt") }

func BenchmarkUdom83(b *testing.B) {
	words := benchWords(b)
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, w := range words {
			sink += len(Udom83(w))
		}
	}
	_ = sink
}

func BenchmarkLK82(b *testing.B) {
	words := benchWords(b)
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, w := range words {
			sink += len(LK82(w))
		}
	}
	_ = sink
}

func BenchmarkMetasound(b *testing.B) {
	words := benchWords(b)
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, w := range words {
			sink += len(Key(w))
		}
	}
	_ = sink
}
