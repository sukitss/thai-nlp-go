package vector

import (
	"reflect"
	"testing"
)

// Rescore over a candidate set must equal an exact scan restricted to the same
// candidates — i.e. the second stage of two-stage retrieval is exact over what
// the coarse stage hands it.
func TestRescoreEqualsExactOverCandidates(t *testing.T) {
	const dim = 64
	vecs := randVecs(500, dim, 6, 21)
	coarse := NewFlat(NewBinary(dim))
	exact := NewFlat(NewFloat32(dim))
	for i, v := range vecs {
		coarse.Add(uint32(i), v)
		exact.Add(uint32(i), v)
	}
	q := randVecs(1, dim, 6, 22)[0]

	cand := coarse.TopK(q, 200) // fast, lossy candidate set
	ids := make([]uint32, len(cand))
	candVecs := make([][]float32, len(cand)) // parallel to ids for the reference
	for i, h := range cand {
		ids[i] = h.ID
		candVecs[i] = vecs[h.ID]
	}

	got := exact.Rescore(q, ids, 10)
	want := exactTopK(candVecs, ids, q, 10) // exact scan restricted to the candidates
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Rescore != exact over candidates\n got=%v\nwant=%v", got, want)
	}
}

func TestRescoreEdgeCases(t *testing.T) {
	const dim = 32
	vecs := randVecs(50, dim, 3, 31)
	f := NewFlat(NewFloat32(dim))
	for i, v := range vecs {
		f.Add(uint32(i), v)
	}
	q := randVecs(1, dim, 3, 32)[0]
	ids := []uint32{0, 1, 2}

	if r := f.Rescore(q, nil, 5); r != nil {
		t.Errorf("Rescore(nil ids) = %v, want nil", r)
	}
	if r := f.Rescore(q, ids, 0); r != nil {
		t.Errorf("Rescore(k=0) = %v, want nil", r)
	}
	if r := f.Rescore(q, []uint32{99999}, 5); len(r) != 0 {
		t.Errorf("Rescore(unknown id) = %v, want empty (skipped)", r)
	}
	// k larger than candidate count returns all candidates, ranked.
	if r := f.Rescore(q, ids, 100); len(r) != len(ids) {
		t.Errorf("Rescore(k>len) returned %d, want %d", len(r), len(ids))
	}
}
