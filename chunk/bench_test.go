package chunk

import (
	"os"
	"strings"
	"testing"
)

const benchTargetBytes = 1 << 20

// BenchmarkSplit chunks ~1 MB of real Thai corpus text (one paragraph per
// line, novel-style) with the default rune Measure and built-in splitter.
func BenchmarkSplit(b *testing.B) {
	data, err := os.ReadFile("../sentence/crf/testdata/crfcut.txt")
	if err != nil {
		b.Fatal(err)
	}
	doc := strings.Repeat(string(data), benchTargetBytes/len(data)+1)
	o := Options{MaxUnits: 512, OverlapUnits: 64}
	b.SetBytes(int64(len(doc)))
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		cs, err := Split(doc, o)
		if err != nil {
			b.Fatal(err)
		}
		sink += len(cs)
	}
	_ = sink
}
