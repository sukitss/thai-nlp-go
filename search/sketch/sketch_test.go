package sketch

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// --- hashing / modular arithmetic -------------------------------------------

// TestModArithVsBigInt proves mulModP and addModP against big.Int over random
// operands below 2^61, the correctness foundation the whole MinHash rests on.
func TestModArithVsBigInt(t *testing.T) {
	p := new(big.Int).SetUint64(mPrime)
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 200000; i++ {
		a := rng.Uint64() % mPrime
		x := rng.Uint64() % mPrime
		b := rng.Uint64() % mPrime
		got := addModP(mulModP(a, x), b)
		want := new(big.Int).Mul(new(big.Int).SetUint64(a), new(big.Int).SetUint64(x))
		want.Add(want, new(big.Int).SetUint64(b))
		want.Mod(want, p)
		if got != want.Uint64() {
			t.Fatalf("a=%d x=%d b=%d: got %d want %d", a, x, b, got, want.Uint64())
		}
	}
}

func TestHashDeterministicAndSeeded(t *testing.T) {
	if hashString("ปากหวาน", 7) != hashString("ปากหวาน", 7) {
		t.Fatal("hashString not deterministic")
	}
	if hashString("ปากหวาน", 7) == hashString("ปากหวาน", 8) {
		t.Fatal("hashString ignores seed")
	}
	if hashBytes([]byte("abc"), 3) != hashString("abc", 3) {
		t.Fatal("hashBytes and hashString disagree")
	}
}

// --- shingles ---------------------------------------------------------------

func TestWordShingles(t *testing.T) {
	toks := []string{"a", "b", "c", "d"}
	got := WordShingles(toks, 2)
	want := []string{"a\x00b", "b\x00c", "c\x00d"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got %q want %q", got, want)
	}
	// dedup: repeated window collapses
	if s := WordShingles([]string{"x", "x", "x"}, 1); len(s) != 1 {
		t.Fatalf("dedup failed: %q", s)
	}
	// fewer tokens than k -> single shingle of all
	if s := WordShingles([]string{"a", "b"}, 5); len(s) != 1 || s[0] != "a\x00b" {
		t.Fatalf("short input: %q", s)
	}
	if s := WordShingles(nil, 2); s != nil {
		t.Fatalf("empty: %q", s)
	}
}

func TestCharNGrams(t *testing.T) {
	got := CharNGrams("กขคง", 2) // 4 runes -> 3 bigrams
	want := []string{"กข", "ขค", "คง"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got %q want %q", got, want)
	}
	if s := CharNGrams("ก", 3); len(s) != 1 || s[0] != "ก" {
		t.Fatalf("short: %q", s)
	}
	if s := CharNGrams("", 2); s != nil {
		t.Fatalf("empty: %q", s)
	}
}

// --- MinHash ----------------------------------------------------------------

// setStrings returns the integers [lo,hi) as string elements.
func setStrings(lo, hi int) []string {
	s := make([]string, 0, hi-lo)
	for i := lo; i < hi; i++ {
		s = append(s, fmt.Sprintf("e%d", i))
	}
	return s
}

func trueJaccard(loA, hiA, loB, hiB int) float64 {
	interLo, interHi := max(loA, loB), min(hiA, hiB)
	inter := 0
	if interHi > interLo {
		inter = interHi - interLo
	}
	union := (hiA - loA) + (hiB - loB) - inter
	return float64(inter) / float64(union)
}

// TestMinHashConverges checks EstJaccard lands within the expected
// 1/sqrt(numHashes) error band on fixed sets of known Jaccard.
func TestMinHashConverges(t *testing.T) {
	const nh = 256
	band := 3.0 / math.Sqrt(nh) // generous 3-sigma-ish envelope
	mh := NewMinHash(nh, 42)
	cases := []struct{ loA, hiA, loB, hiB int }{
		{0, 1000, 0, 1000},    // identical -> 1.0
		{0, 1000, 500, 1500},  // J = 1/3
		{0, 1000, 750, 1750},  // J = 250/1750 ~ 0.1429
		{0, 1000, 900, 1900},  // J = 100/1900 ~ 0.0526
		{0, 1000, 2000, 3000}, // disjoint -> 0.0
	}
	for _, c := range cases {
		sa := mh.SignatureStrings(setStrings(c.loA, c.hiA))
		sb := mh.SignatureStrings(setStrings(c.loB, c.hiB))
		est := EstJaccard(sa, sb)
		want := trueJaccard(c.loA, c.hiA, c.loB, c.hiB)
		if math.Abs(est-want) > band {
			t.Errorf("sets %v: est=%.4f want=%.4f (band=%.4f)", c, est, want, band)
		}
	}
}

func TestMinHashExactEdges(t *testing.T) {
	mh := NewMinHash(128, 1)
	a := mh.SignatureStrings([]string{"x", "y", "z"})
	// identical set -> exactly 1.0
	if got := EstJaccard(a, mh.SignatureStrings([]string{"z", "y", "x"})); got != 1.0 {
		t.Fatalf("identical set EstJaccard=%v want 1.0", got)
	}
	// two empty sets -> 1.0 (both all-sentinel)
	e1, e2 := mh.SignatureStrings(nil), mh.SignatureStrings(nil)
	if got := EstJaccard(e1, e2); got != 1.0 {
		t.Fatalf("empty vs empty EstJaccard=%v want 1.0", got)
	}
	// empty vs non-empty -> 0.0
	if got := EstJaccard(e1, a); got != 0.0 {
		t.Fatalf("empty vs non-empty EstJaccard=%v want 0.0", got)
	}
	// duplicate shingles do not change the signature
	dup := mh.SignatureStrings([]string{"x", "x", "y", "y", "z"})
	if EstJaccard(a, dup) != 1.0 {
		t.Fatal("duplicate shingles changed signature")
	}
}

func TestMinHashDeterministic(t *testing.T) {
	a := NewMinHash(64, 99).SignatureStrings([]string{"a", "b", "c"})
	b := NewMinHash(64, 99).SignatureStrings([]string{"a", "b", "c"})
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("same seed produced different signatures")
		}
	}
	c := NewMinHash(64, 100).SignatureStrings([]string{"a", "b", "c"})
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
		}
	}
	if same {
		t.Fatal("different seed produced identical signatures")
	}
}

func TestMinHashByteAndStringAgree(t *testing.T) {
	mh := NewMinHash(64, 5)
	s := mh.SignatureStrings([]string{"alpha", "beta"})
	b := mh.Signature([][]byte{[]byte("alpha"), []byte("beta")})
	for i := range s {
		if s[i] != b[i] {
			t.Fatal("Signature and SignatureStrings disagree")
		}
	}
}

// --- LSH --------------------------------------------------------------------

// TestLSHRecoversNearDups checks that near-duplicate documents (high Jaccard)
// are returned as candidates while unrelated ones mostly are not.
func TestLSHRecoversNearDups(t *testing.T) {
	const nh = 128
	mh := NewMinHash(nh, 7)
	l := NewLSH(32, 4, 7) // 32*4 = 128, threshold ~ (1/32)^(1/4) ~ 0.42
	if l.SigLen() != nh {
		t.Fatalf("SigLen=%d", l.SigLen())
	}
	// Build 100 base docs, each a distinct 200-element set.
	base := make([][]uint64, 100)
	for i := 0; i < 100; i++ {
		base[i] = mh.SignatureStrings(setStrings(i*1000, i*1000+200))
		l.Add(uint32(i), base[i])
	}
	// A near-dup of doc 0: 90% overlap (J ~ 0.82) must be a candidate.
	near := mh.SignatureStrings(setStrings(0, 180)) // shares 180 of union ~220
	cands := l.Query(near)
	found := false
	for _, id := range cands {
		if id == 0 {
			found = true
		}
	}
	if !found {
		t.Fatalf("near-dup of doc0 not among %d candidates", len(cands))
	}
	// A totally unrelated doc should rarely collide with all 100.
	far := mh.SignatureStrings(setStrings(9_000_000, 9_000_200))
	if c := l.Query(far); len(c) > 5 {
		t.Fatalf("unrelated doc had %d candidates (expected few)", len(c))
	}
	// sorted, unique
	prev := int64(-1)
	for _, id := range cands {
		if int64(id) <= prev {
			t.Fatal("candidates not strictly sorted/unique")
		}
		prev = int64(id)
	}
}

func TestLSHThreshold(t *testing.T) {
	// 20 bands x 5 rows -> threshold ~ (1/20)^(1/5)
	l := NewLSH(20, 5, 1)
	want := math.Pow(1.0/20, 1.0/5)
	if math.Abs(l.Threshold()-want) > 1e-9 {
		t.Fatalf("Threshold=%v want %v", l.Threshold(), want)
	}
}

// --- SimHash ----------------------------------------------------------------

// TestSimHashCorrelatesWithOverlap checks Hamming distance grows as documents
// share fewer features: identical=0, small edit small, disjoint ~ large.
func TestSimHashCorrelatesWithOverlap(t *testing.T) {
	sh := NewSimHash(3)
	base := make([]string, 200)
	for i := range base {
		base[i] = fmt.Sprintf("w%d", i)
	}
	h0 := sh.HashTokens(base)
	if HammingDistance(h0, sh.HashTokens(base)) != 0 {
		t.Fatal("identical docs must have Hamming 0")
	}
	// replace a growing fraction of tokens with fresh ones.
	dist := func(frac float64) int {
		doc := make([]string, len(base))
		copy(doc, base)
		n := int(frac * float64(len(base)))
		for i := 0; i < n; i++ {
			doc[i] = fmt.Sprintf("x%d", i)
		}
		return HammingDistance(h0, sh.HashTokens(doc))
	}
	d10, d50, d100 := dist(0.10), dist(0.50), dist(1.0)
	if !(d10 < d50 && d50 < d100) {
		t.Fatalf("Hamming not monotone in edit fraction: 10%%=%d 50%%=%d 100%%=%d", d10, d50, d100)
	}
	if d10 > 12 {
		t.Errorf("10%% edit Hamming=%d unexpectedly large", d10)
	}
}

func TestSimHashDeterministicAndSeeded(t *testing.T) {
	toks := []string{"a", "b", "c", "a"}
	if NewSimHash(1).HashTokens(toks) != NewSimHash(1).HashTokens(toks) {
		t.Fatal("SimHash not deterministic")
	}
	if NewSimHash(1).HashTokens(toks) == NewSimHash(2).HashTokens(toks) {
		t.Fatal("SimHash ignores seed")
	}
	if NewSimHash(1).HashTokens(nil) != 0 {
		t.Fatal("empty tokens must hash to 0")
	}
}

func TestSimHashWeightedFeatures(t *testing.T) {
	sh := NewSimHash(0)
	// HashTokens must equal Hash over frequency-weighted features.
	toks := []string{"a", "b", "a", "c", "a"}
	feats := []Feature{
		{Hash: hashString("a", 0), Weight: 3},
		{Hash: hashString("b", 0), Weight: 1},
		{Hash: hashString("c", 0), Weight: 1},
	}
	if sh.HashTokens(toks) != sh.Hash(feats) {
		t.Fatal("HashTokens != weighted Hash")
	}
}

// --- CountMin ---------------------------------------------------------------

// TestCountMinNeverUnderAndBounded verifies the two count-min guarantees on a
// fixed stream: Estimate >= true count for every key, and at least (1-delta) of
// keys are within epsilon*total.
func TestCountMinNeverUnderAndBounded(t *testing.T) {
	const eps, delta = 0.01, 0.01
	cm := NewCountMinParams(eps, delta, 12345)
	rng := rand.New(rand.NewSource(7))
	exact := make(map[string]uint64)
	// Skewed stream: many distinct keys, Zipf-ish counts -> forces collisions.
	for i := 0; i < 20000; i++ {
		key := fmt.Sprintf("term-%d", rng.Intn(5000))
		n := uint64(rng.Intn(5) + 1)
		cm.AddString(key, n)
		exact[key] += n
	}
	total := cm.Total()
	bound := uint64(eps * float64(total))
	within, over := 0, 0
	for k, want := range exact {
		got := cm.EstimateString(k)
		if got < want {
			t.Fatalf("UNDER-estimate for %q: got %d < true %d", k, got, want)
		}
		if got-want <= bound {
			within++
		}
		over++
	}
	frac := float64(within) / float64(over)
	if frac < 1-delta {
		t.Fatalf("only %.4f of keys within eps*total (want >= %.4f)", frac, 1-delta)
	}
	// A never-added key still never under-estimates (true count 0).
	if cm.EstimateString("never-seen-key") < 0 {
		t.Fatal("impossible")
	}
}

func TestCountMinDeterministicAndParams(t *testing.T) {
	a := NewCountMin(100, 4, 9)
	b := NewCountMin(100, 4, 9)
	for _, k := range []string{"x", "y", "z"} {
		a.AddString(k, 3)
		b.AddString(k, 3)
	}
	if a.EstimateString("x") != b.EstimateString("x") {
		t.Fatal("same seed/params diverged")
	}
	cm := NewCountMinParams(0.001, 0.001, 1)
	if cm.Width() != int(math.Ceil(math.E/0.001)) {
		t.Fatalf("width=%d", cm.Width())
	}
	if cm.Depth() != int(math.Ceil(math.Log(1/0.001))) {
		t.Fatalf("depth=%d", cm.Depth())
	}
	if cm.SizeBytes() != cm.Width()*cm.Depth()*8 {
		t.Fatal("SizeBytes wrong")
	}
}

func TestCountMinExactWhenNoCollisions(t *testing.T) {
	// Wide sketch + few keys -> minimum row is exact.
	cm := NewCountMin(100000, 5, 1)
	cm.AddString("a", 10)
	cm.AddString("b", 7)
	cm.AddString("a", 5)
	if got := cm.EstimateString("a"); got != 15 {
		t.Fatalf("a=%d want 15", got)
	}
	if got := cm.EstimateString("b"); got != 7 {
		t.Fatalf("b=%d want 7", got)
	}
	if got := cm.EstimateString("c"); got != 0 {
		t.Fatalf("unseen c=%d want 0", got)
	}
}
