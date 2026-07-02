package tokenize

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

func benchInput(b *testing.B) ([]string, int64) {
	lines := readLines(b, "testdata/stress.txt")
	var n int64
	for _, ln := range lines {
		n += int64(len(ln)) + 1
	}
	return lines, n
}

// BenchmarkSerial — single goroutine, tokens as []string.
func BenchmarkSerial(b *testing.B) {
	lines, nbytes := benchInput(b)
	seg := defaultSeg(b)
	b.SetBytes(nbytes)
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, ln := range lines {
			sink += len(seg.SegmentNoWS(ln))
		}
	}
	_ = sink
}

// BenchmarkSerialBytes — single goroutine, []byte output with a reused buffer
// (the zero-steady-state-allocation index path).
func BenchmarkSerialBytes(b *testing.B) {
	lines, nbytes := benchInput(b)
	seg := defaultSeg(b)
	buf := make([]byte, 0, 4096)
	b.SetBytes(nbytes)
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, ln := range lines {
			buf = seg.AppendBytes(buf[:0], ln, ' ')
			sink += len(buf)
		}
	}
	_ = sink
}

// BenchmarkParallel — one Segmenter view per goroutine over a shared dict.
func BenchmarkParallel(b *testing.B) {
	lines, nbytes := benchInput(b)
	base := defaultSeg(b)
	workers := runtime.GOMAXPROCS(0)
	b.SetBytes(nbytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var idx int64 = -1
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				seg := New(base.d) // shares read-only dict, own scratch
				for {
					k := atomic.AddInt64(&idx, 1)
					if int(k) >= len(lines) {
						return
					}
					seg.SegmentNoWS(lines[k])
				}
			}()
		}
		wg.Wait()
	}
}

// BenchmarkSession — overlay dictionary overhead on the hot path.
func BenchmarkSession(b *testing.B) {
	lines, nbytes := benchInput(b)
	seg := defaultSeg(b).Session([]string{"อาริน", "เวธกา", "กขคง"})
	b.SetBytes(nbytes)
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, ln := range lines {
			sink += len(seg.SegmentNoWS(ln))
		}
	}
	_ = sink
}
