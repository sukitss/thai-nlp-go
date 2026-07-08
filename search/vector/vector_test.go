package vector

import (
	"math"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// randVecs returns n deterministic dim-dim vectors drawn from a small number of
// Gaussian clusters, so nearest-neighbour structure is meaningful.
func randVecs(n, dim, clusters int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for j := range centers[c] {
			centers[c][j] = float32(rng.NormFloat64())
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(clusters)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64())*0.35
		}
		out[i] = v
	}
	return out
}

// exactTopK is a naive O(n*d) cosine reference: the ground truth. It scores with
// the package dot — the same kernel float32Scorer.Score uses via dotF32Bytes — so
// Flat+Float32 reproduces it bit-for-bit regardless of the SIMD path (both are
// dot(normalized(query), normalized(v))). dot's own correctness against a strict
// sequential sum is covered separately by TestDotFloat32MatchesGeneric.
func exactTopK(vecs [][]float32, ids []uint32, query []float32, k int) []Hit {
	nq := normalized(query)
	hits := make([]Hit, len(vecs))
	for i, v := range vecs {
		hits[i] = Hit{ID: ids[i], Score: dot(nq, normalized(v))}
	}
	sort.Slice(hits, func(i, j int) bool { return betterHit(hits[i], hits[j]) })
	if k > len(hits) {
		k = len(hits)
	}
	return hits[:k]
}

func idSet(hits []Hit) map[uint32]bool {
	m := make(map[uint32]bool, len(hits))
	for _, h := range hits {
		m[h.ID] = true
	}
	return m
}

// TestFloat32FlatExact: Float32+Flat must reproduce the naive reference top-k
// exactly (ids and order), for several queries and k values.
func TestFloat32FlatExact(t *testing.T) {
	const dim = 48
	vecs := randVecs(500, dim, 6, 1)
	ids := make([]uint32, len(vecs))
	f := NewFlat(NewFloat32(dim))
	for i, v := range vecs {
		ids[i] = uint32(i * 7) // non-sequential ids
		f.Add(ids[i], v)
	}
	queries := randVecs(20, dim, 6, 2)
	for _, k := range []int{1, 10, 50} {
		for qi, q := range queries {
			got := f.TopK(q, k)
			want := exactTopK(vecs, ids, q, k)
			if len(got) != len(want) {
				t.Fatalf("q%d k=%d: len %d, want %d", qi, k, len(got), len(want))
			}
			for i := range want {
				if got[i].ID != want[i].ID {
					t.Fatalf("q%d k=%d pos %d: id %d, want %d", qi, k, i, got[i].ID, want[i].ID)
				}
				if math.Abs(float64(got[i].Score-want[i].Score)) > 1e-5 {
					t.Fatalf("q%d k=%d pos %d: score %v, want %v", qi, k, i, got[i].Score, want[i].Score)
				}
			}
		}
	}
}

// TestParallelMatchesSerial: TopKParallel is byte-identical to TopK for every
// quantizer, at various worker counts.
func TestParallelMatchesSerial(t *testing.T) {
	const dim = 64
	vecs := randVecs(2000, dim, 8, 3)
	train := vecs
	pq, err := TrainPQ(train, 8, 8, 42)
	if err != nil {
		t.Fatal(err)
	}
	quants := []Quantizer{NewFloat32(dim), NewBinary(dim), NewScalar8(dim), pq}
	query := randVecs(1, dim, 8, 4)[0]
	for _, q := range quants {
		f := NewFlat(q)
		for i, v := range vecs {
			f.Add(uint32(i), v)
		}
		serial := f.TopK(query, 20)
		for _, w := range []int{2, 4, 7, 16} {
			par := f.TopKParallel(query, 20, w)
			if !reflect.DeepEqual(serial, par) {
				t.Fatalf("%s: parallel(w=%d) != serial\n serial=%v\n par=%v", q.Name(), w, serial, par)
			}
		}
	}
}

// TestRoundTripSanity: encoding a vector then querying with the same vector
// should rank it at (or very near) the top for every quantizer.
func TestRoundTripSanity(t *testing.T) {
	const dim = 96
	vecs := randVecs(1000, dim, 10, 5)
	pq, err := TrainPQ(vecs, 12, 8, 7)
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []Quantizer{NewFloat32(dim), NewBinary(dim), NewScalar8(dim), pq} {
		f := NewFlat(q)
		for i, v := range vecs {
			f.Add(uint32(i), v)
		}
		// Query with a stored vector; its own id should be in the top-3.
		for _, target := range []int{0, 250, 999} {
			hits := f.TopK(vecs[target], 3)
			if !idSet(hits)[uint32(target)] {
				t.Errorf("%s: self-query of vec %d not in top-3: %v", q.Name(), target, hits)
			}
		}
	}
}

// TestCodeLen: Encode always produces CodeLen bytes.
func TestCodeLen(t *testing.T) {
	const dim = 50
	vecs := randVecs(300, dim, 4, 11)
	pq, err := TrainPQ(vecs, 10, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []Quantizer{NewFloat32(dim), NewBinary(dim), NewScalar8(dim), pq} {
		code := q.Encode(vecs[0])
		if len(code) != q.CodeLen() {
			t.Errorf("%s: Encode len %d, CodeLen %d", q.Name(), len(code), q.CodeLen())
		}
	}
}

// TestPQDeterministic: same seed and data ⇒ identical codebook and codes.
func TestPQDeterministic(t *testing.T) {
	const dim = 32
	vecs := randVecs(800, dim, 5, 9)
	a, err := TrainPQ(vecs, 8, 8, 123)
	if err != nil {
		t.Fatal(err)
	}
	b, err := TrainPQ(vecs, 8, 8, 123)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range vecs[:50] {
		if !reflect.DeepEqual(a.Encode(v), b.Encode(v)) {
			t.Fatal("PQ not deterministic for fixed seed")
		}
	}
	// A different seed should (almost surely) differ somewhere.
	c, err := TrainPQ(vecs, 8, 8, 999)
	if err != nil {
		t.Fatal(err)
	}
	diff := false
	for _, v := range vecs[:200] {
		if !reflect.DeepEqual(a.Encode(v), c.Encode(v)) {
			diff = true
			break
		}
	}
	if !diff {
		t.Error("different seeds produced identical encodings (suspicious)")
	}
}

// TestBinaryHamming: exact Hamming against a hand computation.
func TestBinaryHamming(t *testing.T) {
	q := NewBinary(4)
	a := q.Encode([]float32{1, -1, 1, -1}) // bits 1010 -> byte 0x05 (low bit first)
	b := q.Encode([]float32{1, 1, -1, -1}) // bits 1100 -> byte 0x03
	if h := q.Hamming(a, a); h != 0 {
		t.Errorf("Hamming(a,a) = %d, want 0", h)
	}
	if h := q.Hamming(a, b); h != 2 {
		t.Errorf("Hamming(a,b) = %d, want 2", h)
	}
}

// TestEdgeCases: k>n, k<=0, empty corpus, zero query.
func TestEdgeCases(t *testing.T) {
	const dim = 16
	f := NewFlat(NewFloat32(dim))
	// empty corpus
	if got := f.TopK(make([]float32, dim), 5); got != nil {
		t.Errorf("empty corpus: got %v, want nil", got)
	}
	vecs := randVecs(3, dim, 2, 13)
	for i, v := range vecs {
		f.Add(uint32(i), v)
	}
	// k > n returns all n
	if got := f.TopK(vecs[0], 100); len(got) != 3 {
		t.Errorf("k>n: len %d, want 3", len(got))
	}
	// k <= 0
	if got := f.TopK(vecs[0], 0); got != nil {
		t.Errorf("k=0: got %v, want nil", got)
	}
	// zero query: no panic, returns k hits with score 0
	zero := make([]float32, dim)
	got := f.TopK(zero, 2)
	if len(got) != 2 {
		t.Fatalf("zero query: len %d, want 2", len(got))
	}
	for _, h := range got {
		if h.Score != 0 {
			t.Errorf("zero query: score %v, want 0", h.Score)
		}
	}
}

// TestDimMismatchPanics: Add/Query with a wrong-length vector panics clearly.
func TestDimMismatchPanics(t *testing.T) {
	f := NewFlat(NewFloat32(8))
	assertPanics(t, "Add", func() { f.Add(0, make([]float32, 4)) })
	assertPanics(t, "Encode", func() { NewBinary(8).Encode(make([]float32, 4)) })
	assertPanics(t, "Query", func() { NewScalar8(8).Query(make([]float32, 4)) })
}

func assertPanics(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

// TestPQTrainErrors: invalid configs are rejected, not panicked.
func TestPQTrainErrors(t *testing.T) {
	vecs := randVecs(300, 30, 3, 1)
	cases := []struct {
		name     string
		m, nbits int
		vecs     [][]float32
	}{
		{"dim not divisible by m", 7, 8, vecs}, // 30 % 7 != 0
		{"nbits too large", 10, 9, vecs},
		{"nbits too small", 10, 0, vecs},
		{"too few vectors", 10, 8, vecs[:100]}, // need >= 256
		{"empty", 10, 8, nil},
	}
	for _, c := range cases {
		if _, err := TrainPQ(c.vecs, c.m, c.nbits, 1); err == nil {
			t.Errorf("%s: expected error, got nil", c.name)
		}
	}
}
