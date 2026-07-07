package weight

import (
	"math"
	"testing"
)

const eps = 1e-9

func close(a, b float64) bool { return math.Abs(a-b) <= eps }

// Golden tests pin each scheme to a value worked out by hand on one shared,
// fully specified example, so a change to a formula is caught as a numeric
// regression rather than a subtle ranking drift.
//
// Shared example (one query term, one document):
//
//	tf = 3   df = 2   N = 10   docLen = 100   avgDocLen = 80
//	cf = 5   collLen = 800   (= avgDocLen · N, a self-consistent corpus)
//
// Derived once and reused below:
//
//	IDF   = ln(1 + (N-df+0.5)/(df+0.5)) = ln(1 + 8.5/2.5) = ln(4.4) = 1.4816045409…
//	B'    = 1 - b + b·docLen/avg = 0.25 + 0.75·1.25 = 1.1875          (b = 0.75)
//	K     = k1·B' = 1.2·1.1875 = 1.425                                 (k1 = 1.2)

func TestGoldenBM25(t *testing.T) {
	// BM25 = IDF · tf·(k1+1)/(tf + K)
	//      = 1.4816045409 · 6.6/4.425
	//      = 1.4816045409 · 1.4915254237 = 2.209850840700525
	got := NewBM25().Score(3, 2, 10, 100, 80)
	if want := 2.209850840700525; !close(got, want) {
		t.Fatalf("BM25 = %.15f, want %.15f", got, want)
	}
}

func TestGoldenBM25Plus(t *testing.T) {
	// BM25+ = IDF · ( tf·(k1+1)/(tf + K) + Delta )   Delta = 1.0
	//       = 1.4816045409 · (1.4915254237 + 1.0) = 3.691455381624741
	got := NewBM25Plus().Score(3, 2, 10, 100, 80)
	if want := 3.691455381624741; !close(got, want) {
		t.Fatalf("BM25+ = %.15f, want %.15f", got, want)
	}
}

func TestGoldenBM25L(t *testing.T) {
	// ctf   = tf/B' = 3/1.1875 = 2.5263157895
	// c     = ctf + Delta = 2.5263157895 + 0.5 = 3.0263157895   Delta = 0.5
	// BM25L = IDF · (k1+1)·c/(k1 + c)
	//       = 1.4816045409 · 2.2·3.0263157895/4.2263157895 = 2.334034550771025
	got := NewBM25L().Score(3, 2, 10, 100, 80)
	if want := 2.334034550771025; !close(got, want) {
		t.Fatalf("BM25L = %.15f, want %.15f", got, want)
	}
}

func TestGoldenDFR(t *testing.T) {
	// PL2, C = 1.0, cf = 5:
	//   tfn    = tf·log2(1 + C·avg/docLen) = 3·log2(1.8) = 2.5439907197
	//   lambda = cf/N = 5/10 = 0.5
	//   score  = (1/(tfn+1))·( tfn·log2(tfn/lambda)
	//                        + (lambda + 1/(12·tfn) - tfn)·log2(e)
	//                        + 0.5·log2(2π·tfn) )
	//          = 1.430218645271038
	got := NewDFR().ScoreStats(Stats{TF: 3, DF: 2, N: 10, DocLen: 100, AvgDocLen: 80, CF: 5, CollLen: 800})
	if want := 1.430218645271038; !close(got, want) {
		t.Fatalf("DFR/PL2 = %.15f, want %.15f", got, want)
	}
}

func TestGoldenQLDirichlet(t *testing.T) {
	// p(t|C) = cf/collLen = 5/800 = 0.00625,  Mu = 2000
	// present: log( (tf + Mu·p) / (docLen + Mu) ) = log(15.5/2100)  = -4.908852599786313
	// absent : log( (0  + Mu·p) / (docLen + Mu) ) = log(12.5/2100)  = -5.123963979403259
	s := Stats{TF: 3, DF: 2, N: 10, DocLen: 100, AvgDocLen: 80, CF: 5, CollLen: 800}
	got := NewQLDirichlet().ScoreStats(s)
	if want := -4.908852599786313; !close(got, want) {
		t.Fatalf("QL present = %.15f, want %.15f", got, want)
	}
	s.TF = 0
	got = NewQLDirichlet().ScoreStats(s)
	if want := -5.123963979403259; !close(got, want) {
		t.Fatalf("QL absent = %.15f, want %.15f", got, want)
	}
}

func TestGoldenTFIDF(t *testing.T) {
	// TFIDF = (1 + ln(tf))·ln(N/df) = (1 + ln 3)·ln 5 = 3.377586180882552
	got := NewTFIDF().Score(3, 2, 10, 100, 80)
	if want := 3.377586180882552; !close(got, want) {
		t.Fatalf("TFIDF = %.15f, want %.15f", got, want)
	}
}

// TestScoreStatsMatchesScore checks the CorpusScorer path delegates to the
// five-argument Score identically for the df-only schemes.
func TestScoreStatsMatchesScore(t *testing.T) {
	s := Stats{TF: 4, DF: 3, N: 20, DocLen: 55, AvgDocLen: 40}
	cases := []struct {
		name string
		sc   interface {
			Scorer
			CorpusScorer
		}
	}{
		{"bm25", NewBM25()},
		{"bm25+", NewBM25Plus()},
		{"bm25l", NewBM25L()},
		{"tfidf", NewTFIDF()},
	}
	for _, c := range cases {
		a := c.sc.Score(int(s.TF), s.DF, s.N, s.DocLen, s.AvgDocLen)
		b := c.sc.ScoreStats(s)
		if !close(a, b) {
			t.Errorf("%s: Score=%v ScoreStats=%v", c.name, a, b)
		}
	}
}

// TestAbsentTermContribution documents the tf==0 behaviour that ScoreDoc relies
// on: the tf·idf family contributes exactly 0 for a term the document lacks,
// while QLDirichlet contributes a negative background (the smoothing term).
func TestAbsentTermContribution(t *testing.T) {
	dfOnly := []Scorer{NewBM25(), NewBM25Plus(), NewBM25L(), NewTFIDF()}
	for _, sc := range dfOnly {
		if got := sc.Score(0, 2, 10, 100, 80); got != 0 {
			t.Errorf("%T absent term = %v, want 0", sc, got)
		}
	}
	if got := NewDFR().ScoreStats(Stats{TF: 0, CF: 5, N: 10, DocLen: 100, AvgDocLen: 80}); got != 0 {
		t.Errorf("DFR absent term = %v, want 0", got)
	}
	if got := NewQLDirichlet().ScoreStats(Stats{TF: 0, CF: 5, CollLen: 800, DocLen: 100}); got >= 0 {
		t.Errorf("QL absent term = %v, want negative background", got)
	}
}

// TestEdgeCases exercises degenerate inputs: empty corpus, unseen terms, zero
// lengths. No scheme should NaN/Inf or panic.
func TestEdgeCases(t *testing.T) {
	all := []CorpusScorer{NewBM25(), NewBM25Plus(), NewBM25L(), NewDFR(), NewQLDirichlet(), NewTFIDF()}
	stats := []Stats{
		{}, // all zero
		{TF: 1, DF: 0, N: 0, DocLen: 0, AvgDocLen: 0},                          // unseen term, empty corpus
		{TF: 2, DF: 5, N: 5, DocLen: 10, AvgDocLen: 10, CF: 8, CollLen: 50},    // df == N (idf 0)
		{TF: 1, DF: 1, N: 100, DocLen: 0, AvgDocLen: 20, CF: 1, CollLen: 2000}, // zero docLen
	}
	for _, sc := range all {
		for i, s := range stats {
			v := sc.ScoreStats(s)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("%T case %d: non-finite %v for %+v", sc, i, v, s)
			}
		}
	}
}

// TestScoreDoc checks the whole-document helper sums per-term contributions and
// scores absent query terms (so QL sees its background) via a small collection.
func TestScoreDoc(t *testing.T) {
	docs := [][]string{
		{"ก", "ข", "ข", "ค"},
		{"ก", "ค", "ง"},
		{"ข", "จ"},
	}
	c := NewCollection(docs)
	if c.N() != 3 {
		t.Fatalf("N = %d, want 3", c.N())
	}
	if c.DF("ก") != 2 || c.CF("ข") != 3 {
		t.Fatalf("DF/CF wrong: DF(ก)=%d CF(ข)=%d", c.DF("ก"), c.CF("ข"))
	}
	if want := 9.0 / 3.0; c.AvgDocLen() != want {
		t.Fatalf("AvgDocLen = %v, want %v", c.AvgDocLen(), want)
	}

	doc := docs[0]
	tf := TermFreq(doc)
	query := []string{"ข", "จ"} // จ is absent from doc0

	// BM25: only present terms contribute; equals summing Score over them.
	bm := NewBM25()
	want := bm.ScoreStats(c.Stats("ข", 2, 4)) + bm.ScoreStats(c.Stats("จ", 0, 4))
	if got := ScoreDoc(bm, query, tf, float64(len(doc)), c); !close(got, want) {
		t.Fatalf("ScoreDoc BM25 = %v, want %v", got, want)
	}

	// QL: the absent term "จ" must lower the score (background penalty), so the
	// full-query score is strictly less than scoring only the present term.
	ql := NewQLDirichlet()
	full := ScoreDoc(ql, query, tf, float64(len(doc)), c)
	presentOnly := ScoreDoc(ql, []string{"ข"}, tf, float64(len(doc)), c)
	if !(full < presentOnly) {
		t.Fatalf("QL absent term did not penalize: full=%v presentOnly=%v", full, presentOnly)
	}
}
