package crf

import "testing"

func BenchmarkSplit(b *testing.B) {
	lines := readLines(b, "testdata/crfcut.txt")
	Split(lines[0]) // warm the model load out of the timed loop
	var nbytes int64
	for _, ln := range lines {
		nbytes += int64(len(ln)) + 1
	}
	b.SetBytes(nbytes)
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, ln := range lines {
			sink += len(Split(ln))
		}
	}
	_ = sink
}
