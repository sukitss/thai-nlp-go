package sketch

import (
	"math"
	"math/bits"
	"sort"
)

// mPrime is the Mersenne prime 2^61 - 1, the modulus of the universal hash
// family used for MinHash permutations. It is large enough that collisions among
// distinct 61-bit shingle hashes are negligible, and small enough that a product
// of two operands below it fits the fast reduction in mulModP.
const mPrime uint64 = (1 << 61) - 1

// MinHash computes k-permutation MinHash signatures. Each of the numHashes
// permutations is a universal hash h_i(x) = (a_i*x + b_i) mod (2^61-1) drawn
// deterministically from the seed; a signature stores, per permutation, the
// minimum h_i over a document's shingles. The fraction of signature positions
// two documents agree on is an unbiased estimator of the Jaccard similarity of
// their shingle sets, with standard error ~1/sqrt(numHashes).
//
// This is Broder's classic construction (min-wise independent permutations),
// not the "k hashes from two" double-hashing approximation, so the variance
// matches the theoretical 1/sqrt(numHashes) bound.
type MinHash struct {
	k    int
	seed uint64
	a    []uint64 // multipliers, in [1, mPrime)
	b    []uint64 // addends, in [0, mPrime)
}

// NewMinHash returns a MinHash with numHashes permutations derived from seed.
// numHashes must be >= 1; larger values reduce the Jaccard estimation error
// (~1/sqrt(numHashes)) at proportional time and signature-size cost. Typical
// values are 64-256.
func NewMinHash(numHashes int, seed uint64) *MinHash {
	if numHashes < 1 {
		panic("sketch: NewMinHash numHashes must be >= 1")
	}
	m := &MinHash{k: numHashes, seed: seed, a: make([]uint64, numHashes), b: make([]uint64, numHashes)}
	rng := splitmix64{state: seed}
	for i := 0; i < numHashes; i++ {
		// a in [1, mPrime): non-zero so the map is a bijection mod the prime.
		m.a[i] = rng.next()%(mPrime-1) + 1
		m.b[i] = rng.next() % mPrime
	}
	return m
}

// NumHashes reports the signature length.
func (m *MinHash) NumHashes() int { return m.k }

// mulModP returns a*x mod (2^61-1). a and x must both be < 2^61. It uses a
// 128-bit product and the fast Mersenne reduction 2^61 == 1 (mod 2^61-1).
func mulModP(a, x uint64) uint64 {
	hi, lo := bits.Mul64(a, x)
	// value = hi*2^64 + lo; since 2^61 == 1 (mod p), fold the high bits down.
	v := (lo & mPrime) + ((lo >> 61) | (hi << 3))
	v = (v & mPrime) + (v >> 61)
	if v >= mPrime {
		v -= mPrime
	}
	return v
}

// addModP returns a+b mod (2^61-1), with a, b < 2^61.
func addModP(a, b uint64) uint64 {
	v := a + b
	v = (v & mPrime) + (v >> 61)
	if v >= mPrime {
		v -= mPrime
	}
	return v
}

// newSig returns a signature slice initialised to the max sentinel.
func (m *MinHash) newSig() []uint64 {
	sig := make([]uint64, m.k)
	for i := range sig {
		sig[i] = mPrime // any real hash is < mPrime, so this is a proper +inf
	}
	return sig
}

// fold updates every signature position with one shingle's base hash x.
func (m *MinHash) fold(sig []uint64, x uint64) {
	for i := 0; i < m.k; i++ {
		hv := addModP(mulModP(m.a[i], x), m.b[i])
		if hv < sig[i] {
			sig[i] = hv
		}
	}
}

// Signature returns the MinHash signature of a set of shingles (raw byte
// slices). Duplicate shingles do not affect the result (min is idempotent), so
// the caller need not deduplicate. An empty set yields the all-sentinel
// signature, which reports Jaccard 1.0 against another empty set and 0.0 against
// any non-empty one.
func (m *MinHash) Signature(shingles [][]byte) []uint64 {
	sig := m.newSig()
	for _, sh := range shingles {
		m.fold(sig, hashBytes(sh, m.seed)%mPrime)
	}
	return sig
}

// SignatureStrings is Signature for string shingles (e.g. from [WordShingles] or
// [CharNGrams]); it hashes without allocating.
func (m *MinHash) SignatureStrings(shingles []string) []uint64 {
	sig := m.newSig()
	for _, sh := range shingles {
		m.fold(sig, hashString(sh, m.seed)%mPrime)
	}
	return sig
}

// EstJaccard estimates the Jaccard similarity of the two sets whose signatures
// are a and b: the fraction of positions at which they agree. The signatures
// must have the same length (same MinHash). The estimate is unbiased with
// standard error ~1/sqrt(len).
func EstJaccard(a, b []uint64) float64 {
	if len(a) != len(b) {
		panic("sketch: EstJaccard signature length mismatch")
	}
	if len(a) == 0 {
		return 0
	}
	eq := 0
	for i := range a {
		if a[i] == b[i] {
			eq++
		}
	}
	return float64(eq) / float64(len(a))
}

// LSH is a locality-sensitive-hash index over MinHash signatures for sub-linear
// near-duplicate candidate generation. It splits each length bands*rows
// signature into bands consecutive bands of rows values; two signatures land in
// the same bucket of a band when those rows are identical. Add stores an id in
// every band bucket its signature falls in; Query returns the ids that share at
// least one band bucket with the query — the candidate set to verify with
// [EstJaccard] or an exact check.
//
// The number of bands and rows tunes the S-curve: a pair with true Jaccard s is
// a candidate with probability 1 - (1 - s^rows)^bands, which rises sharply near
// the threshold ~ (1/bands)^(1/rows) ([LSH.Threshold]). More bands / fewer rows
// lowers the threshold (higher recall, more candidates).
type LSH struct {
	bands, rows int
	seed        uint64
	buckets     []map[uint64][]uint32
}

// NewLSH returns an LSH index with the given band/row layout; every signature
// added or queried must have length bands*rows. bands and rows must be >= 1.
func NewLSH(bands, rows int, seed uint64) *LSH {
	if bands < 1 || rows < 1 {
		panic("sketch: NewLSH bands and rows must be >= 1")
	}
	l := &LSH{bands: bands, rows: rows, seed: seed, buckets: make([]map[uint64][]uint32, bands)}
	for i := range l.buckets {
		l.buckets[i] = make(map[uint64][]uint32)
	}
	return l
}

// SigLen is the required signature length, bands*rows.
func (l *LSH) SigLen() int { return l.bands * l.rows }

// Threshold is the approximate Jaccard similarity at which a pair becomes
// likely to be a candidate, (1/bands)^(1/rows) — the midpoint of the S-curve.
func (l *LSH) Threshold() float64 {
	return math.Pow(1/float64(l.bands), 1/float64(l.rows))
}

// bandKey hashes band bi of sig into a bucket key.
func (l *LSH) bandKey(bi int, sig []uint64) uint64 {
	start := bi * l.rows
	return hashWords(sig[start:start+l.rows], l.seed+uint64(bi))
}

// Add indexes id under sig. sig must have length SigLen(). Adding the same id
// twice duplicates it in the candidate lists; callers that re-add should track
// membership themselves.
func (l *LSH) Add(id uint32, sig []uint64) {
	if len(sig) != l.SigLen() {
		panic("sketch: LSH.Add signature length mismatch")
	}
	for bi := 0; bi < l.bands; bi++ {
		k := l.bandKey(bi, sig)
		l.buckets[bi][k] = append(l.buckets[bi][k], id)
	}
}

// Query returns the sorted, de-duplicated ids that share at least one band
// bucket with sig (including id itself if it was added). sig must have length
// SigLen(). The result is the candidate set; verify true similarity with
// [EstJaccard] on the signatures or an exact comparison.
func (l *LSH) Query(sig []uint64) []uint32 {
	if len(sig) != l.SigLen() {
		panic("sketch: LSH.Query signature length mismatch")
	}
	set := make(map[uint32]struct{})
	for bi := 0; bi < l.bands; bi++ {
		k := l.bandKey(bi, sig)
		for _, id := range l.buckets[bi][k] {
			set[id] = struct{}{}
		}
	}
	out := make([]uint32, 0, len(set))
	for id := range set {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
