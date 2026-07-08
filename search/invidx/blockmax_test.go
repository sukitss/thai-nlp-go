package invidx

import (
	"math/rand"
	"testing"

	"github.com/sukitss/thai-nlp-go/search/weight"
)

// TestBlockMaxMatchesBrute is THE correctness gate for Block-Max WAND: over many
// random corpora, queries, scorers, k values AND block sizes, SearchBlockMax must
// return exactly the same top-k as the brute oracle — identical ids AND
// bit-identical scores, in the same order.
func TestBlockMaxMatchesBrute(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	ks := []int{1, 5, 10, 50, 100, 1000}
	blockSizes := []int{1, 4, 16, 128}
	for iter := 0; iter < 300; iter++ {
		n := 1 + rng.Intn(400)
		v := 2 + rng.Intn(60)
		idx, _, _ := randIndex(rng, n, v)
		for _, s := range allScorers() {
			for q := 0; q < 4; q++ {
				qlen := 1 + rng.Intn(8)
				query := randQuery(rng, qlen, v)
				for _, k := range ks {
					brute := idx.SearchBrute(query, s.sc, k)
					for _, bs := range blockSizes {
						bmw := idx.SearchBlockMaxSized(query, s.sc, k, bs)
						if !sameHits(brute, bmw) {
							t.Fatalf("BMW != Brute: scorer=%s n=%d v=%d k=%d bs=%d query=%v\nbrute=%v\nbmw  =%v",
								s.name, n, v, k, bs, query, brute, bmw)
						}
					}
				}
			}
		}
	}
}

// TestBlockMaxMatchesBruteDFR: BMW stays exact on DFR (PL2) just as WAND does —
// the block bounds clamp at 0 like the global bounds, staying valid over-estimates.
func TestBlockMaxMatchesBruteDFR(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	sc := weight.NewDFR()
	for iter := 0; iter < 150; iter++ {
		n := 1 + rng.Intn(400)
		v := 2 + rng.Intn(50)
		idx, _, _ := randIndex(rng, n, v)
		for q := 0; q < 4; q++ {
			query := randQuery(rng, 1+rng.Intn(6), v)
			for _, k := range []int{1, 10, 100} {
				brute := idx.SearchBrute(query, sc, k)
				bmw := idx.SearchBlockMaxSized(query, sc, k, 8)
				if !sameHits(brute, bmw) {
					t.Fatalf("BMW(DFR) != Brute: n=%d v=%d k=%d query=%v\nbrute=%v\nbmw=%v",
						n, v, k, query, brute, bmw)
				}
			}
		}
	}
}

// TestBlockMaxMatchesWAND checks BMW and WAND agree (both equal brute), covering
// the default block size and the public default method.
func TestBlockMaxMatchesWAND(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for iter := 0; iter < 100; iter++ {
		idx, _, _ := randIndex(rng, 1+rng.Intn(500), 2+rng.Intn(80))
		for _, s := range allScorers() {
			query := randQuery(rng, 1+rng.Intn(6), 40)
			for _, k := range []int{1, 10, 100} {
				wand := idx.Search(query, s.sc, k)
				bmw := idx.SearchBlockMax(query, s.sc, k) // default block size
				if !sameHits(wand, bmw) {
					t.Fatalf("BMW != WAND: scorer=%s k=%d query=%v\nwand=%v\nbmw=%v", s.name, k, query, wand, bmw)
				}
			}
		}
	}
}

// TestBlockMaxEdgeCases covers empty query, k<=0, all-absent terms, single doc,
// duplicate query terms, and k>N.
func TestBlockMaxEdgeCases(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	idx, _, _ := randIndex(rng, 200, 30)
	sc := weight.NewBM25()

	if got := idx.SearchBlockMax(nil, sc, 10); got != nil {
		t.Errorf("empty query: got %v, want nil", got)
	}
	if got := idx.SearchBlockMax([]uint32{1, 2}, sc, 0); got != nil {
		t.Errorf("k=0: got %v, want nil", got)
	}
	if got := idx.SearchBlockMax([]uint32{9999, 8888}, sc, 10); got != nil {
		t.Errorf("all-absent terms: got %v, want nil", got)
	}
	// duplicate query terms == brute (weighting by multiplicity).
	dupq := []uint32{3, 3, 5}
	if !sameHits(idx.SearchBrute(dupq, sc, 10), idx.SearchBlockMax(dupq, sc, 10)) {
		t.Error("duplicate query terms: BMW != Brute")
	}
	// k > N returns all matching docs, same as brute.
	if !sameHits(idx.SearchBrute([]uint32{1}, sc, 100000), idx.SearchBlockMax([]uint32{1}, sc, 100000)) {
		t.Error("k>N: BMW != Brute")
	}

	// single-document index.
	b := NewBuilder()
	b.Add(42, []uint32{0, 1}, []uint32{2, 1})
	one := b.Build()
	if !sameHits(one.SearchBrute([]uint32{0, 1}, sc, 5), one.SearchBlockMax([]uint32{0, 1}, sc, 5)) {
		t.Error("single doc: BMW != Brute")
	}
}

// TestBlockMaxPrunesAtLargeK measures that on a corpus with long postings BMW
// evaluates no more documents than WAND and strictly fewer than brute — the
// block bounds are meant to help most where WAND's global bounds are loose.
func TestBlockMaxPrunesAtLargeK(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	// Dense corpus: few terms, many docs → long postings lists where blocks matter.
	b := NewBuilder()
	const n, v = 20000, 12
	for i := 0; i < n; i++ {
		nt := 1 + rng.Intn(v)
		seen := map[uint32]bool{}
		var terms, tfs []uint32
		for len(terms) < nt {
			tm := uint32(rng.Intn(v))
			if seen[tm] {
				continue
			}
			seen[tm] = true
			terms = append(terms, tm)
			tfs = append(tfs, uint32(1+rng.Intn(6)))
		}
		b.Add(uint32(i), terms, tfs)
	}
	idx := b.Build()
	sc := weight.NewBM25()
	query := []uint32{0, 1, 2, 3}

	for _, k := range []int{10, 100} {
		var bt, wt, bmt trace
		idx.searchBrute(query, sc, k, &bt)
		idx.search(query, sc, k, &wt)
		idx.searchBlockMax(query, sc, k, DefaultBlockSize, &bmt)
		t.Logf("k=%d: brute evaluated %d, WAND %d, BMW %d (BMW skipped %.1f%% vs brute)",
			k, bt.evaluated, wt.evaluated, bmt.evaluated,
			100*(1-float64(bmt.evaluated)/float64(bt.evaluated)))
		if bmt.evaluated > wt.evaluated {
			t.Errorf("k=%d: BMW evaluated %d > WAND %d (block bounds should not do worse)", k, bmt.evaluated, wt.evaluated)
		}
		if bmt.evaluated >= bt.evaluated {
			t.Errorf("k=%d: BMW evaluated %d did not prune vs brute %d", k, bmt.evaluated, bt.evaluated)
		}
	}
}

// TestBlockMaxConcurrent runs SearchBlockMax from many goroutines (shared,
// read-only index; the impact table is built once under the cache mutex).
func TestBlockMaxConcurrent(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	idx, _, _ := randIndex(rng, 3000, 40)
	sc := weight.NewBM25()
	idx.PrepareBlockMax(sc, DefaultBlockSize) // warm the cache
	queries := make([][]uint32, 40)
	for i := range queries {
		queries[i] = randQuery(rng, 3, 40)
	}
	want := make([][]Hit, len(queries))
	for i, q := range queries {
		want[i] = idx.SearchBlockMax(q, sc, 10)
	}
	done := make(chan bool)
	for g := 0; g < 8; g++ {
		go func() {
			for i, q := range queries {
				if !sameHits(idx.SearchBlockMax(q, sc, 10), want[i]) {
					t.Errorf("concurrent BMW mismatch on query %d", i)
				}
			}
			done <- true
		}()
	}
	for g := 0; g < 8; g++ {
		<-done
	}
}
