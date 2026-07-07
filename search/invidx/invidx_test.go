package invidx

import (
	"math"
	"math/rand"
	"sort"
	"strconv"
	"sync"
	"testing"

	"github.com/sukitss/thai-nlp-go/search/weight"
)

// allScorers is the WAND-compatible set (additive, non-negative, present-only)
// exercised by the correctness gate. QLDirichlet is deliberately excluded — it
// violates the contract and is a SearchBrute-only scheme.
func allScorers() []struct {
	name string
	sc   weight.CorpusScorer
} {
	return []struct {
		name string
		sc   weight.CorpusScorer
	}{
		{"BM25", weight.NewBM25()},
		{"BM25Plus", weight.NewBM25Plus()},
		{"BM25L", weight.NewBM25L()},
		{"TFIDF", weight.NewTFIDF()},
	}
}

// randIndex builds a random corpus: n documents over a vocabulary of v terms,
// each document a random subset of terms with random small term frequencies.
// Document ids are shuffled so internal order != id order (exercising the id
// tiebreak). Returns the index and the docs as term-id lists for oracle use.
func randIndex(rng *rand.Rand, n, v int) (*Index, [][]uint32, []uint32) {
	b := NewBuilder()
	docTerms := make([][]uint32, n)
	ids := make([]uint32, n)
	perm := rng.Perm(n)
	for i := 0; i < n; i++ {
		id := uint32(perm[i]) * 7 // sparse, non-sequential ids
		ids[i] = id
		nt := 1 + rng.Intn(v) // 1..v distinct terms
		seen := map[uint32]int{}
		var terms, tfs []uint32
		var flat []uint32
		for len(terms) < nt {
			t := uint32(rng.Intn(v))
			if _, ok := seen[t]; ok {
				continue
			}
			tf := uint32(1 + rng.Intn(4))
			seen[t] = len(terms)
			terms = append(terms, t)
			tfs = append(tfs, tf)
			for j := uint32(0); j < tf; j++ {
				flat = append(flat, t)
			}
		}
		b.Add(id, terms, tfs)
		docTerms[i] = flat
	}
	return b.Build(), docTerms, ids
}

// randQuery returns qlen random term ids in 0..v-1 (with possible repeats, and
// possibly some out-of-vocabulary ids to exercise absent terms).
func randQuery(rng *rand.Rand, qlen, v int) []uint32 {
	q := make([]uint32, qlen)
	for i := range q {
		if rng.Float64() < 0.1 {
			q[i] = uint32(v + rng.Intn(v)) // out of vocabulary
		} else {
			q[i] = uint32(rng.Intn(v))
		}
	}
	return q
}

// TestWANDMatchesBrute is THE correctness gate: over many random corpora,
// queries, scorers and k, WAND must return exactly the same top-k as the brute
// oracle — identical ids AND bit-identical scores, in the same order.
func TestWANDMatchesBrute(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	ks := []int{1, 5, 10, 50, 100, 1000}
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
					wand := idx.Search(query, s.sc, k)
					if !sameHits(brute, wand) {
						t.Fatalf("WAND != Brute: scorer=%s n=%d v=%d k=%d query=%v\nbrute=%v\nwand =%v",
							s.name, n, v, k, query, brute, wand)
					}
				}
			}
		}
	}
}

// TestWANDMatchesBruteDFR substantiates the package doc's claim that WAND stays
// correct on DFR (PL2): its per-term contribution is non-negative in the regime
// that matters but can dip slightly negative, and the index's upper bounds clamp
// at 0 — still a valid over-estimate, so pruning never drops a true top-k
// document. If this ever fails, the doc's DFR guidance must change.
func TestWANDMatchesBruteDFR(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	sc := weight.NewDFR()
	for iter := 0; iter < 150; iter++ {
		n := 1 + rng.Intn(400)
		v := 2 + rng.Intn(50)
		idx, _, _ := randIndex(rng, n, v)
		for q := 0; q < 4; q++ {
			query := randQuery(rng, 1+rng.Intn(6), v)
			for _, k := range []int{1, 10, 100} {
				if !sameHits(idx.SearchBrute(query, sc, k), idx.Search(query, sc, k)) {
					t.Fatalf("WAND != Brute (DFR): n=%d v=%d k=%d query=%v", n, v, k, query)
				}
			}
		}
	}
}

// sameHits requires identical length, ids, and bit-identical scores in order.
func sameHits(a, b []Hit) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Score != b[i].Score {
			return false
		}
	}
	return true
}

// TestBruteMatchesNaive validates the brute oracle ITSELF against a fully
// independent implementation built on weight.Collection (string-keyed), so a
// shared bug between brute and WAND cannot pass unnoticed. Ids must match
// exactly; scores match within a float tolerance (the naive path sums in query
// order, brute in canonical term-id order).
func TestBruteMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for iter := 0; iter < 120; iter++ {
		n := 1 + rng.Intn(200)
		v := 2 + rng.Intn(40)
		idx, docTerms, ids := randIndex(rng, n, v)
		// Independent oracle over the same corpus, terms as strings.
		strDocs := make([][]string, n)
		for i, dt := range docTerms {
			s := make([]string, len(dt))
			for j, t := range dt {
				s[j] = strconv.FormatUint(uint64(t), 10)
			}
			strDocs[i] = s
		}
		coll := weight.NewCollection(strDocs)
		for _, s := range allScorers() {
			for q := 0; q < 4; q++ {
				query := randQuery(rng, 1+rng.Intn(6), v)
				qstr := make([]string, len(query))
				for i, t := range query {
					qstr[i] = strconv.FormatUint(uint64(t), 10)
				}
				k := 1 + rng.Intn(30)
				got := idx.SearchBrute(query, s.sc, k)
				want := naiveTopK(strDocs, ids, coll, s.sc, qstr, query, k)
				if msg := compareApprox(got, want); msg != "" {
					t.Fatalf("scorer=%s k=%d: %s\ngot =%v\nwant=%v", s.name, k, msg, got, want)
				}
			}
		}
	}
}

// compareApprox checks brute against the independent oracle. The two sum a
// document's per-term contributions in different orders (canonical term-id vs
// query order), so scores can differ by rounding and, at a tie boundary, the id
// tiebreak can select a different member of a tie group. It therefore requires:
// equal length; each rank's scores within a float tolerance (a real ranking or
// scoring bug shows up here as a gap beyond rounding); and any id disagreement
// confined to documents whose score sits within the tolerance of the k-th
// boundary score (a genuine tie, ambiguous by construction). Returns "" on
// agreement or a description of the first violation.
func compareApprox(got, want []Hit) string {
	const eps = 1e-9
	if len(got) != len(want) {
		return "length mismatch"
	}
	if len(got) == 0 {
		return ""
	}
	for i := range got {
		if math.Abs(got[i].Score-want[i].Score) > eps {
			return "score/order mismatch at rank " + strconv.Itoa(i)
		}
	}
	boundary := got[len(got)-1].Score
	gs := map[uint32]float64{}
	ws := map[uint32]float64{}
	for _, h := range got {
		gs[h.ID] = h.Score
	}
	for _, h := range want {
		ws[h.ID] = h.Score
	}
	for id, sc := range gs {
		if _, ok := ws[id]; !ok && math.Abs(sc-boundary) > eps {
			return "extra id " + strconv.FormatUint(uint64(id), 10) + " not near the tie boundary"
		}
	}
	for id, sc := range ws {
		if _, ok := gs[id]; !ok && math.Abs(sc-boundary) > eps {
			return "missing id " + strconv.FormatUint(uint64(id), 10) + " not near the tie boundary"
		}
	}
	return ""
}

// naiveTopK is the independent oracle: score every document that shares a term
// with the query using weight.ScoreDoc, then rank by the strict order.
func naiveTopK(strDocs [][]string, ids []uint32, coll *weight.Collection, sc weight.CorpusScorer, qstr []string, query []uint32, k int) []Hit {
	qset := map[string]bool{}
	for _, t := range qstr {
		qset[t] = true
	}
	var hits []Hit
	for i, doc := range strDocs {
		matches := false
		for _, t := range doc {
			if qset[t] {
				matches = true
				break
			}
		}
		if !matches {
			continue
		}
		score := weight.ScoreDoc(sc, qstr, weight.TermFreq(doc), float64(len(doc)), coll)
		hits = append(hits, Hit{ID: ids[i], Score: score})
	}
	sort.Slice(hits, func(a, b int) bool { return betterHit(hits[a], hits[b]) })
	if k < len(hits) {
		hits = hits[:k]
	}
	return hits
}

// TestDeterminism checks both query paths are reproducible across repeated calls
// (independent of the upper-bound cache being warm or cold).
func TestDeterminism(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	idx, _, _ := randIndex(rng, 500, 50)
	sc := weight.NewBM25()
	query := []uint32{1, 5, 5, 12, 30}
	first := idx.Search(query, sc, 20)
	for i := 0; i < 5; i++ {
		if !sameHits(first, idx.Search(query, sc, 20)) {
			t.Fatal("Search not deterministic across calls")
		}
		if !sameHits(first, idx.SearchBrute(query, sc, 20)) {
			t.Fatal("SearchBrute disagrees with Search")
		}
	}
}

// TestEdgeCases exercises degenerate inputs on both paths.
func TestEdgeCases(t *testing.T) {
	sc := weight.NewBM25()

	// Empty index.
	empty := NewBuilder().Build()
	if got := empty.Search([]uint32{1}, sc, 10); got != nil {
		t.Errorf("empty index Search = %v, want nil", got)
	}

	b := NewBuilder()
	b.Add(100, []uint32{0, 1, 2}, []uint32{1, 2, 1})
	b.Add(200, []uint32{1, 3}, []uint32{1, 1})
	b.Add(300, []uint32{2, 3, 4}, []uint32{2, 1, 3})
	idx := b.Build()

	cases := []struct {
		name  string
		query []uint32
		k     int
	}{
		{"k>N", []uint32{1, 2, 3}, 100},
		{"k=0", []uint32{1}, 0},
		{"k<0", []uint32{1}, -3},
		{"empty query", nil, 10},
		{"term not in index", []uint32{99}, 10},
		{"mixed present/absent", []uint32{99, 1, 99}, 10},
		{"single term", []uint32{3}, 2},
		{"duplicate query terms", []uint32{1, 1, 1}, 10},
	}
	for _, c := range cases {
		brute := idx.SearchBrute(c.query, sc, c.k)
		wand := idx.Search(c.query, sc, c.k)
		if !sameHits(brute, wand) {
			t.Errorf("%s: WAND != Brute\nbrute=%v\nwand=%v", c.name, brute, wand)
		}
	}

	// k > number of matching docs returns exactly the matching docs.
	got := idx.Search([]uint32{4}, sc, 10) // only doc 300 has term 4
	if len(got) != 1 || got[0].ID != 300 {
		t.Errorf("single-match query = %v, want [{300 ...}]", got)
	}
}

// TestSingleDoc covers the smallest non-empty index.
func TestSingleDoc(t *testing.T) {
	b := NewBuilder()
	b.Add(42, []uint32{0, 1}, []uint32{3, 1})
	idx := b.Build()
	sc := weight.NewBM25()
	for _, k := range []int{1, 5} {
		w := idx.Search([]uint32{0, 1}, sc, k)
		br := idx.SearchBrute([]uint32{0, 1}, sc, k)
		if !sameHits(w, br) || len(w) != 1 || w[0].ID != 42 {
			t.Errorf("single doc k=%d: wand=%v brute=%v", k, w, br)
		}
	}
}

// TestAllSameScore forces a corpus where every matching document scores
// identically, so the entire ranking is decided by the id tiebreak — the
// hardest case for WAND's threshold to get right.
func TestAllSameScore(t *testing.T) {
	// Every document is identical (same single term, same tf), so every doc's
	// score for a query on that term is the same; top-k must be the k smallest
	// ids.
	b := NewBuilder()
	const n = 200
	ids := make([]uint32, n)
	for i := 0; i < n; i++ {
		id := uint32((i * 13) % 1000) // distinct-ish, unsorted
		ids[i] = id
		b.Add(id, []uint32{0}, []uint32{1})
	}
	idx := b.Build()
	sc := weight.NewBM25()
	for _, k := range []int{1, 10, 50, n} {
		w := idx.Search([]uint32{0}, sc, k)
		br := idx.SearchBrute([]uint32{0}, sc, k)
		if !sameHits(w, br) {
			t.Fatalf("all-same-score k=%d: WAND != Brute", k)
		}
		// Verify it really is the k smallest ids.
		sortedIDs := append([]uint32(nil), ids...)
		sort.Slice(sortedIDs, func(a, b int) bool { return sortedIDs[a] < sortedIDs[b] })
		want := sortedIDs[:min(k, n)]
		for i, h := range w {
			if h.ID != want[i] {
				t.Fatalf("all-same-score k=%d rank %d: id %d want %d", k, i, h.ID, want[i])
			}
		}
	}
}

// TestPruningHappens confirms WAND actually skips work on a corpus large enough
// for it to matter: it must evaluate strictly fewer documents than the brute
// union for a selective (multi-term) query, while returning the same result.
func TestPruningHappens(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	idx, _, _ := randIndex(rng, 20000, 400)
	sc := weight.NewBM25()
	idx.Prepare(sc)
	query := randQuery(rng, 4, 400)

	_, bt := idx.searchBrute(query, sc, 10, &trace{})
	_, wt := idx.search(query, sc, 10, &trace{})
	if wt.evaluated >= bt.evaluated {
		t.Fatalf("WAND did not prune: wand evaluated %d, brute evaluated %d", wt.evaluated, bt.evaluated)
	}
	t.Logf("brute evaluated %d docs, WAND evaluated %d (%.1f%% skipped)",
		bt.evaluated, wt.evaluated, 100*(1-float64(wt.evaluated)/float64(bt.evaluated)))
}

// TestConcurrentSearch runs many goroutines through Search on one shared index
// (with -race) to prove the read path and lazy upper-bound cache are safe.
func TestConcurrentSearch(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	idx, _, _ := randIndex(rng, 2000, 100)
	scorers := allScorers()
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			r := rand.New(rand.NewSource(seed))
			for i := 0; i < 50; i++ {
				s := scorers[r.Intn(len(scorers))]
				q := randQuery(r, 1+r.Intn(5), 100)
				w := idx.Search(q, s.sc, 10)
				br := idx.SearchBrute(q, s.sc, 10)
				if !sameHits(w, br) {
					t.Errorf("concurrent mismatch scorer=%s", s.name)
					return
				}
			}
		}(int64(g))
	}
	wg.Wait()
}

// TestStats checks the corpus aggregates the index derives from its own
// postings: df, cf, N, mean length, and the Count helper.
func TestStats(t *testing.T) {
	b := NewBuilder()
	b.Add(1, []uint32{0, 1, 2}, []uint32{1, 2, 1}) // len 4
	b.Add(2, []uint32{0, 2}, []uint32{3, 1})       // len 4
	b.Add(3, []uint32{1}, []uint32{2})             // len 2
	idx := b.Build()

	if idx.N() != 3 {
		t.Errorf("N = %d, want 3", idx.N())
	}
	if idx.DF(0) != 2 || idx.DF(1) != 2 || idx.DF(2) != 2 {
		t.Errorf("DF wrong: %d %d %d", idx.DF(0), idx.DF(1), idx.DF(2))
	}
	if idx.CF(0) != 4 || idx.CF(1) != 4 || idx.CF(2) != 2 {
		t.Errorf("CF wrong: %d %d %d", idx.CF(0), idx.CF(1), idx.CF(2))
	}
	if want := 10.0 / 3.0; math.Abs(idx.AvgDocLen()-want) > 1e-12 {
		t.Errorf("AvgDocLen = %v, want %v", idx.AvgDocLen(), want)
	}
	if idx.DF(999) != 0 || idx.CF(999) != 0 {
		t.Errorf("out-of-range term should have zero df/cf")
	}

	terms, tfs := Count([]uint32{5, 3, 5, 5, 3, 8})
	m := map[uint32]uint32{}
	for i, tm := range terms {
		m[tm] = tfs[i]
	}
	if m[5] != 3 || m[3] != 2 || m[8] != 1 || len(terms) != 3 {
		t.Errorf("Count wrong: %v / %v", terms, tfs)
	}
}
