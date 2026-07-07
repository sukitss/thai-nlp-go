package weight

import "testing"

// benchStats is a representative single-term input reused across the per-term
// benchmarks so they measure only the arithmetic, not setup.
var benchStats = Stats{TF: 3, DF: 40, N: 10000, DocLen: 120, AvgDocLen: 95, CF: 260, CollLen: 950000}

var benchSink float64

func benchScorer(b *testing.B, sc CorpusScorer) {
	b.ReportAllocs()
	var s float64
	for i := 0; i < b.N; i++ {
		s += sc.ScoreStats(benchStats)
	}
	benchSink = s
}

func BenchmarkBM25(b *testing.B)        { benchScorer(b, NewBM25()) }
func BenchmarkBM25Plus(b *testing.B)    { benchScorer(b, NewBM25Plus()) }
func BenchmarkBM25L(b *testing.B)       { benchScorer(b, NewBM25L()) }
func BenchmarkDFR(b *testing.B)         { benchScorer(b, NewDFR()) }
func BenchmarkQLDirichlet(b *testing.B) { benchScorer(b, NewQLDirichlet()) }
func BenchmarkTFIDF(b *testing.B)       { benchScorer(b, NewTFIDF()) }

// BenchmarkScoreDoc measures the whole-document helper on a realistic 5-term
// query against a small in-memory collection: the map lookups and per-term sum
// that dominate a real ranking loop's inner cost.
func BenchmarkScoreDoc(b *testing.B) {
	docs := [][]string{
		{"a", "b", "c", "d", "a", "e", "f", "b"},
		{"a", "c", "g", "h", "b"},
		{"x", "y", "z", "a", "c"},
		{"b", "c", "c", "d", "e", "f", "g"},
	}
	c := NewCollection(docs)
	tf := TermFreq(docs[0])
	dl := float64(len(docs[0]))
	query := []string{"a", "b", "c", "z", "q"} // z rare, q unseen
	sc := NewBM25()
	b.ReportAllocs()
	b.ResetTimer()
	var s float64
	for i := 0; i < b.N; i++ {
		s += ScoreDoc(sc, query, tf, dl, c)
	}
	benchSink = s
}
