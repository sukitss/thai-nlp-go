// Package sketch provides small, probabilistic data-structure primitives for
// near-duplicate detection and approximate frequency counting at scale, entirely
// in pure Go with no dependencies. They are the mechanism behind "did someone
// copy this chapter?" and "roughly how often does this term occur?" across
// corpora too large to compare exhaustively or index with a full vocabulary.
//
// Three sketches are provided, each trading a bounded amount of accuracy for a
// large, fixed reduction in memory or time:
//
//	MinHash  — a fixed-length signature of a set (a document's shingles) whose
//	           Hamming-style agreement estimates the Jaccard similarity of the
//	           original sets. With optional LSH banding it generates near-duplicate
//	           candidate pairs in sub-linear time instead of comparing all pairs.
//	SimHash  — a single 64-bit fingerprint of a weighted feature bag; two documents
//	           are near-duplicates when their fingerprints differ in few bits
//	           (Hamming distance via hardware popcount).
//	CountMin — a fixed-size count-min sketch: it counts how often keys occur in a
//	           stream using a small table instead of a per-key map, never
//	           under-estimates, and stays within a tunable error bound with high
//	           probability.
//
// # Determinism
//
// Every sketch is seeded. Two sketches built with the same constructor arguments
// (including seed) produce byte-identical results for the same input, so a
// signature computed on one machine matches one computed on another. Comparisons
// (EstJaccard, Hamming, LSH bucketing) are only meaningful between sketches that
// share the same parameters and seed.
//
// # Shingles
//
// [WordShingles] and [CharNGrams] turn a document into the SET of overlapping
// pieces that MinHash and SimHash operate on. They are convenience helpers: the
// sketches accept raw [][]byte / []string / []uint64 too, so a caller with its
// own tokenizer or feature extractor is not forced through them.
//
// # Concurrency
//
// Building a sketch mutates it and is single-goroutine (MinHash/SimHash have no
// build step; CountMin.Add and LSH.Add do). Once built, the read-only methods
// (Signature, Hash, Estimate, EstJaccard, LSH.Query) are safe for concurrent use.
package sketch

// hashing --------------------------------------------------------------------
//
// A seeded 64-bit hash with good avalanche: FNV-1a mixing followed by the
// splitmix64 finalizer so single-bit input changes spread across the output.
// Separate []byte and string variants avoid a []byte(string) allocation on the
// hot paths.

const (
	fnvOffset64 = 1469598103934665603
	fnvPrime64  = 1099511628211
)

// mix is the splitmix64 finalizer; it turns FNV-1a's weak avalanche into a
// well-distributed 64-bit value.
func mix(h uint64) uint64 {
	h ^= h >> 30
	h *= 0xbf58476d1ce4e5b9
	h ^= h >> 27
	h *= 0x94d049bb133111eb
	h ^= h >> 31
	return h
}

// hashBytes hashes b with the given seed.
func hashBytes(b []byte, seed uint64) uint64 {
	h := fnvOffset64 ^ seed
	for _, c := range b {
		h ^= uint64(c)
		h *= fnvPrime64
	}
	return mix(h)
}

// hashString hashes s with the given seed without allocating.
func hashString(s string, seed uint64) uint64 {
	h := fnvOffset64 ^ seed
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= fnvPrime64
	}
	return mix(h)
}

// hashWords hashes a slice of uint64 words with the given seed, byte by byte in
// little-endian order, without allocating. Used to bucket LSH bands.
func hashWords(vals []uint64, seed uint64) uint64 {
	h := fnvOffset64 ^ seed
	for _, v := range vals {
		for j := 0; j < 8; j++ {
			h ^= (v >> (8 * j)) & 0xff
			h *= fnvPrime64
		}
	}
	return mix(h)
}

// splitmix64 is a tiny deterministic PRNG used to derive per-permutation
// parameters from a single user seed.
type splitmix64 struct{ state uint64 }

func (s *splitmix64) next() uint64 {
	s.state += 0x9e3779b97f4a7c15
	return mix(s.state)
}

// shingles -------------------------------------------------------------------

// shingleSep separates joined tokens in a word shingle. NUL never appears inside
// a token, so distinct token sequences map to distinct shingle strings.
const shingleSep = "\x00"

// WordShingles returns the SET of overlapping k-token windows of tokens, joined
// by a separator. It is the standard document representation for MinHash /
// SimHash near-duplicate detection over word sequences. Duplicate windows are
// collapsed (Jaccard and MinHash are defined over sets). If there are fewer than
// k tokens, the whole token sequence is returned as a single shingle; an empty
// input returns nil. k must be >= 1.
func WordShingles(tokens []string, k int) []string {
	if k < 1 {
		panic("sketch: WordShingles k must be >= 1")
	}
	if len(tokens) == 0 {
		return nil
	}
	if len(tokens) < k {
		return []string{joinTokens(tokens)}
	}
	seen := make(map[string]struct{}, len(tokens))
	out := make([]string, 0, len(tokens)-k+1)
	for i := 0; i+k <= len(tokens); i++ {
		sh := joinTokens(tokens[i : i+k])
		if _, ok := seen[sh]; ok {
			continue
		}
		seen[sh] = struct{}{}
		out = append(out, sh)
	}
	return out
}

func joinTokens(tokens []string) string {
	if len(tokens) == 1 {
		return tokens[0]
	}
	n := 0
	for _, t := range tokens {
		n += len(t)
	}
	n += len(shingleSep) * (len(tokens) - 1)
	b := make([]byte, 0, n)
	for i, t := range tokens {
		if i > 0 {
			b = append(b, shingleSep...)
		}
		b = append(b, t...)
	}
	return string(b)
}

// CharNGrams returns the SET of overlapping n-character (rune) windows of s.
// It is the standard document representation for character-level near-duplicate
// detection, robust to tokenization differences and small edits — well suited to
// Thai text, which is not space-segmented. Duplicate windows are collapsed. If s
// has fewer than n runes, the whole string is returned as a single shingle (nil
// for empty s). n must be >= 1.
func CharNGrams(s string, n int) []string {
	if n < 1 {
		panic("sketch: CharNGrams n must be >= 1")
	}
	if s == "" {
		return nil
	}
	runes := []rune(s)
	if len(runes) < n {
		return []string{s}
	}
	seen := make(map[string]struct{}, len(runes))
	out := make([]string, 0, len(runes)-n+1)
	for i := 0; i+n <= len(runes); i++ {
		sh := string(runes[i : i+n])
		if _, ok := seen[sh]; ok {
			continue
		}
		seen[sh] = struct{}{}
		out = append(out, sh)
	}
	return out
}
