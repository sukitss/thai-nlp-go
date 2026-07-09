package charseg_test

import (
	"bufio"
	"os"
	"testing"

	"github.com/sukitss/thai-nlp-go/sentence"
	"github.com/sukitss/thai-nlp-go/sentence/charseg"
	"github.com/sukitss/thai-nlp-go/sentence/crf"
)

func loadLines(tb testing.TB) []string {
	f, err := os.Open("testdata/bench.txt")
	if err != nil {
		tb.Skip("no bench corpus:", err)
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		if t := sc.Text(); len(t) > 0 {
			lines = append(lines, t)
		}
	}
	return lines
}

func corpusBytes(lines []string) int64 {
	var n int64
	for _, l := range lines {
		n += int64(len(l)) + 1
	}
	return n
}

// engines under test — all satisfy func(string) []string.
func engines() []struct {
	name string
	fn   func(string) []string
} {
	crf.Split("warm")       // warm default model + tokenizer
	crf.Dialogue().Split("warm")
	charseg.Split("warm")
	return []struct {
		name string
		fn   func(string) []string
	}{
		{"whitespace", sentence.Split},
		{"heuristic", sentence.SplitHeuristic},
		{"crf.Default", crf.Split},
		{"crf.Dialogue", func(s string) []string { return crf.Dialogue().Split(s) }},
		{"charseg", charseg.Split},
	}
}

// BenchmarkEngines: single-goroutine latency/throughput per engine.
func BenchmarkEngines(b *testing.B) {
	lines := loadLines(b)
	nb := corpusBytes(lines)
	for _, e := range engines() {
		b.Run(e.name, func(b *testing.B) {
			b.SetBytes(nb)
			b.ReportAllocs()
			b.ResetTimer()
			var sink int
			for i := 0; i < b.N; i++ {
				for _, ln := range lines {
					sink += len(e.fn(ln))
				}
			}
			_ = sink
		})
	}
}

// BenchmarkEnginesParallel: throughput with all GOMAXPROCS goroutines (run with
// -cpu=4 to model a 4-core box). Each iteration processes the whole corpus.
func BenchmarkEnginesParallel(b *testing.B) {
	lines := loadLines(b)
	nb := corpusBytes(lines)
	for _, e := range engines() {
		b.Run(e.name, func(b *testing.B) {
			b.SetBytes(nb)
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				var sink int
				for pb.Next() {
					for _, ln := range lines {
						sink += len(e.fn(ln))
					}
				}
				_ = sink
			})
		})
	}
}
