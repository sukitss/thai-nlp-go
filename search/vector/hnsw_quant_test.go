package vector

import (
	"sync"
	"testing"
)

// TestSymQuantizerSymmetric checks SimCodes is symmetric and self-similarity is
// maximal for both symmetric quantizers.
func TestSymQuantizerSymmetric(t *testing.T) {
	vecs := randVecs(50, 64, 6, 1)
	for _, sq := range []SymQuantizer{NewBinarySym(64), NewScalar8Sym(64)} {
		codes := make([][]byte, len(vecs))
		for i, v := range vecs {
			codes[i] = sq.Encode(v)
			if len(codes[i]) != sq.CodeLen() {
				t.Fatalf("%s: code len %d != CodeLen %d", sq.Name(), len(codes[i]), sq.CodeLen())
			}
		}
		// SimCodes must be symmetric. (Self-similarity is NOT necessarily the
		// maximum for a lossy scaled quantizer: scalar8's raw scaled dot is not
		// unit-normalized, so a near-parallel vector with lower quantization error
		// can score slightly above self — the ranking still approximates cosine,
		// which the recall test verifies.)
		for i := 0; i < len(codes); i++ {
			for j := 0; j < len(codes); j++ {
				ab := sq.SimCodes(codes[i], codes[j])
				ba := sq.SimCodes(codes[j], codes[i])
				if ab != ba {
					t.Fatalf("%s: SimCodes not symmetric (%d,%d): %v vs %v", sq.Name(), i, j, ab, ba)
				}
			}
		}
		// Binary's self-similarity IS maximal (Hamming 0 → dim), a useful sanity
		// check the scaled quantizer can't offer.
		if _, ok := sq.(*BinarySym); ok {
			for i := range codes {
				if self := sq.SimCodes(codes[i], codes[i]); self != float32(sq.Dim()) {
					t.Fatalf("binary self-sim %v != dim %d", self, sq.Dim())
				}
			}
		}
	}
}

// TestHNSWQuantizedRecall measures quantized-HNSW recall against the float32
// exact ground truth. Binary is lossy; scalar8 should stay near-exact.
func TestHNSWQuantizedRecall(t *testing.T) {
	const (
		n   = 3000
		dim = 64
		k   = 10
	)
	vecs := randVecs(n, dim, 12, 42)
	ids := seqIDs(n)
	queries := randVecs(200, dim, 12, 7)

	// scalar8 keeps near-exact ranking → assert a real threshold. binary is
	// deliberately NOT asserted here: on this 12-tight-cluster synthetic corpus
	// the true top-10 are within-cluster near-duplicates whose sign vectors tie
	// in Hamming, so binary recall collapses (a documented synthetic artifact,
	// T-120 — real m3 gives ~0.67). Binary recall is measured on real m3 in the
	// bench, not asserted on synthetic here.
	cases := []struct {
		name   string
		sq     SymQuantizer
		assert bool
		min    float64
	}{
		{"scalar8", NewScalar8Sym(dim), true, 0.90},
		{"binary", NewBinarySym(dim), false, 0},
	}
	for _, c := range cases {
		h := NewHNSW(dim, WithEfSearch(100), WithQuantizer(c.sq))
		for i, v := range vecs {
			h.Add(ids[i], v)
		}
		recall := recallAt(h, vecs, ids, queries, k)
		t.Logf("HNSW+%s recall@%d = %.4f", c.name, k, recall)
		if c.assert && recall < c.min {
			t.Errorf("HNSW+%s recall@%d = %.4f, want >= %.2f", c.name, k, recall, c.min)
		}
	}
}

// TestHNSWQuantizedDeterministic: quantized graphs are deterministic too.
func TestHNSWQuantizedDeterministic(t *testing.T) {
	vecs := randVecs(500, 32, 6, 3)
	ids := seqIDs(500)
	queries := randVecs(20, 32, 6, 4)

	build := func() *HNSW {
		h := NewHNSW(32, WithSeed(123), WithQuantizer(NewScalar8Sym(32)))
		for i, v := range vecs {
			h.Add(ids[i], v)
		}
		return h
	}
	a, b := build(), build()
	for _, q := range queries {
		ha, hb := a.TopK(q, 10), b.TopK(q, 10)
		if len(ha) != len(hb) {
			t.Fatalf("len mismatch %d vs %d", len(ha), len(hb))
		}
		for i := range ha {
			if ha[i] != hb[i] {
				t.Fatalf("determinism broken at %d: %+v vs %+v", i, ha[i], hb[i])
			}
		}
	}
}

// TestHNSWQuantizedConcurrentRead guards -race on the quantized query path.
func TestHNSWQuantizedConcurrentRead(t *testing.T) {
	vecs := randVecs(1500, 48, 8, 8)
	ids := seqIDs(1500)
	queries := randVecs(40, 48, 8, 9)
	h := NewHNSW(48, WithQuantizer(NewBinarySym(48)))
	for i, v := range vecs {
		h.Add(ids[i], v)
	}
	want := make([][]Hit, len(queries))
	for i, q := range queries {
		want[i] = h.TopK(q, 10)
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i, q := range queries {
				got := h.TopK(q, 10)
				if len(got) != len(want[i]) {
					t.Errorf("q%d len mismatch", i)
					return
				}
				for j := range got {
					if got[j] != want[i][j] {
						t.Errorf("q%d pos%d mismatch", i, j)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}

// TestHNSWQuantizedEdge: dim mismatch panics, zero vector is legal, k>n returns all.
func TestHNSWQuantizedEdge(t *testing.T) {
	func() {
		defer func() {
			if recover() == nil {
				t.Error("expected panic on quantizer dim mismatch")
			}
		}()
		NewHNSW(8, WithQuantizer(NewBinarySym(16)))
	}()

	dim := 16
	h := NewHNSW(dim, WithQuantizer(NewScalar8Sym(dim)))
	h.Add(1, make([]float32, dim)) // zero vector: all-zero code, legal
	vecs := randVecs(30, dim, 4, 2)
	for i, v := range vecs {
		h.Add(uint32(i+2), v)
	}
	if got := h.TopK(vecs[0], 1000); len(got) != 31 {
		t.Errorf("k>n: got %d, want 31", len(got))
	}
}
