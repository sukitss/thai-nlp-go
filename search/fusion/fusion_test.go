package fusion

import (
	"math"
	"testing"
)

const eps = 1e-12

func approx(a, b float64) bool { return math.Abs(a-b) <= eps }

// ids extracts the id order of a result, for order assertions.
func ids(hits []Hit) []uint32 {
	out := make([]uint32, len(hits))
	for i, h := range hits {
		out[i] = h.ID
	}
	return out
}

func eqIDs(a []uint32, b ...uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestRRFGolden pins RRF to hand-computed reciprocal-rank values.
//
//	list A (best-first): [1, 2, 3]
//	list B (best-first): [2, 3, 4]
//	k = 60, rank is 1-based.
//	id1 = 1/61                      = 0.016393442622950820
//	id2 = 1/62 + 1/61               = 0.032522474881015340
//	id3 = 1/63 + 1/62               = 0.032002048131080388
//	id4 = 1/63                      = 0.015873015873015872
//	order by score desc, id asc: [2, 3, 1, 4]
func TestRRFGolden(t *testing.T) {
	a := []Hit{{1, 9}, {2, 8}, {3, 7}} // scores are irrelevant to RRF (rank-based)
	b := []Hit{{2, 0.5}, {3, 0.4}, {4, 0.3}}
	got := RRF([][]Hit{a, b}, 0) // 0 → default k = 60

	want := map[uint32]float64{
		1: 1.0 / 61,
		2: 1.0/62 + 1.0/61,
		3: 1.0/63 + 1.0/62,
		4: 1.0 / 63,
	}
	if !eqIDs(ids(got), 2, 3, 1, 4) {
		t.Fatalf("RRF order = %v, want [2 3 1 4]", ids(got))
	}
	for _, h := range got {
		if !approx(h.Score, want[h.ID]) {
			t.Errorf("RRF id %d score = %.18f, want %.18f", h.ID, h.Score, want[h.ID])
		}
	}
	if DefaultRRFK != 60 {
		t.Errorf("DefaultRRFK = %d, want 60", DefaultRRFK)
	}
}

// TestRRFCustomK checks a non-default k and that a doc in one list only still
// scores.
func TestRRFCustomK(t *testing.T) {
	a := []Hit{{10, 1}, {20, 1}}
	got := RRF([][]Hit{a}, 1) // k=1: ranks 1,2 → 1/2, 1/3
	if len(got) != 2 || got[0].ID != 10 || got[1].ID != 20 {
		t.Fatalf("order = %v", ids(got))
	}
	if !approx(got[0].Score, 1.0/2) || !approx(got[1].Score, 1.0/3) {
		t.Errorf("scores = %.6f, %.6f want 0.5, 0.3333", got[0].Score, got[1].Score)
	}
}

// TestWeightedSumMinMaxGolden pins WeightedSum+MinMax to hand-computed values.
//
//	list A: [(1,10), (2,6), (3,2)]  min=2 max=10 range=8 → 1:1.0  2:0.5  3:0.0
//	list B: [(2,0.9),(3,0.6),(4,0.3)] min=.3 max=.9 range=.6 → 2:1.0 3:0.5 4:0.0
//	weights = [1, 1]:
//	  id1=1.0  id2=0.5+1.0=1.5  id3=0.0+0.5=0.5  id4=0.0
//	order: [2, 1, 3, 4]
func TestWeightedSumMinMaxGolden(t *testing.T) {
	a := []Hit{{1, 10}, {2, 6}, {3, 2}}
	b := []Hit{{2, 0.9}, {3, 0.6}, {4, 0.3}}
	got := WeightedSum([][]Hit{a, b}, []float64{1, 1}, MinMax)

	want := map[uint32]float64{1: 1.0, 2: 1.5, 3: 0.5, 4: 0.0}
	if !eqIDs(ids(got), 2, 1, 3, 4) {
		t.Fatalf("order = %v, want [2 1 3 4]", ids(got))
	}
	for _, h := range got {
		if !approx(h.Score, want[h.ID]) {
			t.Errorf("id %d score = %.6f, want %.6f", h.ID, h.Score, want[h.ID])
		}
	}
}

// TestWeightedSumWeightsAndTiebreak: weights [2,1] make id1 and id2 tie at 2.0,
// and the id-ascending tiebreak must order id1 before id2.
//
//	id1 = 2*1.0            = 2.0
//	id2 = 2*0.5 + 1*1.0    = 2.0
//	id3 = 2*0.0 + 1*0.5    = 0.5
//	id4 = 1*0.0            = 0.0
func TestWeightedSumWeightsAndTiebreak(t *testing.T) {
	a := []Hit{{1, 10}, {2, 6}, {3, 2}}
	b := []Hit{{2, 0.9}, {3, 0.6}, {4, 0.3}}
	got := WeightedSum([][]Hit{a, b}, []float64{2, 1}, MinMax)
	if !eqIDs(ids(got), 1, 2, 3, 4) {
		t.Fatalf("order = %v, want [1 2 3 4] (id-asc tiebreak on the 2.0 tie)", ids(got))
	}
	if !approx(got[0].Score, 2.0) || !approx(got[1].Score, 2.0) {
		t.Errorf("top two scores = %.6f, %.6f want 2.0, 2.0", got[0].Score, got[1].Score)
	}
}

// TestWeightedSumNilWeights: nil weights ⇒ equal weight 1.0 (same as explicit
// [1,1]).
func TestWeightedSumNilWeights(t *testing.T) {
	a := []Hit{{1, 10}, {2, 6}, {3, 2}}
	b := []Hit{{2, 0.9}, {3, 0.6}, {4, 0.3}}
	nilW := WeightedSum([][]Hit{a, b}, nil, MinMax)
	oneW := WeightedSum([][]Hit{a, b}, []float64{1, 1}, MinMax)
	if !eqIDs(ids(nilW), ids(oneW)...) {
		t.Fatalf("nil weights %v != [1,1] %v", ids(nilW), ids(oneW))
	}
}

// TestMinMaxConstantList: a constant list (max==min) maps every score to 1.0, so
// a single-hit list still contributes full presence credit.
func TestMinMaxConstantList(t *testing.T) {
	got := MinMax([]float64{5, 5, 5})
	for i, v := range got {
		if v != 1 {
			t.Errorf("MinMax constant[%d] = %v, want 1", i, v)
		}
	}
	if len(MinMax(nil)) != 0 {
		t.Error("MinMax(nil) should be empty")
	}
	single := WeightedSum([][]Hit{{{7, 3.3}}}, nil, MinMax)
	if len(single) != 1 || single[0].ID != 7 || !approx(single[0].Score, 1.0) {
		t.Errorf("single-hit MinMax = %+v, want id7 score 1.0", single)
	}
}

// TestZScore standardizes to mean 0 / unit variance and preserves order.
func TestZScore(t *testing.T) {
	got := ZScore([]float64{1, 2, 3})
	// mean 2, pop-variance 2/3, std = sqrt(2/3).
	std := math.Sqrt(2.0 / 3.0)
	want := []float64{-1 / std, 0, 1 / std}
	for i := range got {
		if !approx(got[i], want[i]) {
			t.Errorf("ZScore[%d] = %.6f, want %.6f", i, got[i], want[i])
		}
	}
	for _, v := range ZScore([]float64{4, 4, 4}) { // zero variance → all 0
		if v != 0 {
			t.Errorf("ZScore constant = %v, want 0", v)
		}
	}
}

// TestNilNorm: nil normalizer adds raw scores unchanged.
func TestNilNorm(t *testing.T) {
	a := []Hit{{1, 3}, {2, 1}}
	b := []Hit{{2, 5}, {1, 0.5}}
	got := WeightedSum([][]Hit{a, b}, nil, nil)
	// id1 = 3 + 0.5 = 3.5 ; id2 = 1 + 5 = 6.0
	want := map[uint32]float64{1: 3.5, 2: 6.0}
	if !eqIDs(ids(got), 2, 1) {
		t.Fatalf("order = %v, want [2 1]", ids(got))
	}
	for _, h := range got {
		if !approx(h.Score, want[h.ID]) {
			t.Errorf("id %d = %v want %v", h.ID, h.Score, want[h.ID])
		}
	}
}

// TestAdapt projects a foreign hit type and preserves order.
func TestAdapt(t *testing.T) {
	type ext struct {
		DocID uint32
		Sim   float32
	}
	src := []ext{{5, 0.9}, {6, 0.1}}
	got := Adapt(src, func(e ext) Hit { return Hit{ID: e.DocID, Score: float64(e.Sim)} })
	if len(got) != 2 || got[0].ID != 5 || got[1].ID != 6 {
		t.Fatalf("Adapt order/ids wrong: %+v", got)
	}
	if !approx(got[0].Score, float64(float32(0.9))) {
		t.Errorf("Adapt score = %v", got[0].Score)
	}
	if Adapt([]ext(nil), func(e ext) Hit { return Hit{} }) != nil {
		t.Error("Adapt(nil) should be nil")
	}
}

// TestEmptyInputs: empty and all-empty inputs return nil, no panic.
func TestEmptyInputs(t *testing.T) {
	if RRF(nil, 0) != nil {
		t.Error("RRF(nil) should be nil")
	}
	if RRF([][]Hit{{}, {}}, 0) != nil {
		t.Error("RRF of empty lists should be nil")
	}
	if WeightedSum(nil, nil, MinMax) != nil {
		t.Error("WeightedSum(nil) should be nil")
	}
	if WeightedSum([][]Hit{{}, {}}, nil, MinMax) != nil {
		t.Error("WeightedSum of empty lists should be nil")
	}
}

// TestWeightedSumPanicsOnLenMismatch.
func TestWeightedSumPanicsOnLenMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on weights/lists length mismatch")
		}
	}()
	WeightedSum([][]Hit{{{1, 1}}}, []float64{1, 2}, MinMax)
}
