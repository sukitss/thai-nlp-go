package normalize

import "testing"

// BenchmarkNormalize runs over the mixed edge+corpus test set (real-world mix of
// already-normal and needs-normalizing lines).
func BenchmarkNormalize(b *testing.B) {
	lines := readLines(b, "testdata/norm.txt")
	var nbytes int64
	for _, ln := range lines {
		nbytes += int64(len(ln)) + 1
	}
	b.SetBytes(nbytes)
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, ln := range lines {
			sink += len(Normalize(ln))
		}
	}
	_ = sink
}

// BenchmarkNormalizeFastPath measures the fast path (mark-free text that no rule
// can change — returned untouched, zero allocation).
func BenchmarkNormalizeFastPath(b *testing.B) {
	const s = "the quick brown fox jumps over the lazy dog"
	if Normalize(s) != s {
		b.Fatalf("expected fast-path no-op, got change")
	}
	b.SetBytes(int64(len(s)))
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		sink += len(Normalize(s))
	}
	_ = sink
}
