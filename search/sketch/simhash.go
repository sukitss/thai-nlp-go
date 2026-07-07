package sketch

import "math/bits"

// SimHash computes Charikar's 64-bit SimHash fingerprint of a weighted feature
// bag. Each feature contributes its weight, added or subtracted per output bit
// according to the bits of the feature's hash; the fingerprint bit is 1 where
// the signed sum is positive. Documents that share most of their weighted
// features produce fingerprints that differ in few bits, so a small Hamming
// distance ([HammingDistance], hardware popcount) means near-duplicate.
//
// Unlike MinHash, a SimHash fingerprint is a single uint64: cheap to store
// (8 bytes/doc), compare, and index by Hamming distance. It captures cosine-like
// similarity of the feature bag rather than Jaccard of a set.
type SimHash struct {
	seed uint64
}

// NewSimHash returns a SimHash whose token hashing is seeded by seed. Only
// fingerprints built with the same seed are comparable.
func NewSimHash(seed uint64) *SimHash { return &SimHash{seed: seed} }

// Feature is one weighted feature: a 64-bit hash of the feature and its weight
// (e.g. a term and its frequency or tf-idf). Weights should be non-negative;
// a zero weight contributes nothing.
type Feature struct {
	Hash   uint64
	Weight float64
}

// Hash returns the 64-bit SimHash of pre-hashed weighted features. The seed is
// not used here (the caller supplies the feature hashes); use [SimHash.HashTokens]
// to hash and weight tokens with this SimHash's seed.
func (s *SimHash) Hash(features []Feature) uint64 {
	var acc [64]float64
	for _, f := range features {
		h := f.Hash
		w := f.Weight
		for j := 0; j < 64; j++ {
			if h&(1<<uint(j)) != 0 {
				acc[j] += w
			} else {
				acc[j] -= w
			}
		}
	}
	var out uint64
	for j := 0; j < 64; j++ {
		if acc[j] > 0 {
			out |= 1 << uint(j)
		}
	}
	return out
}

// HashTokens returns the SimHash of a token slice, weighting each distinct token
// by how often it occurs (term frequency). Tokens are hashed with this SimHash's
// seed. This is the common document fingerprint; for custom weights (tf-idf,
// shingle weights) build []Feature and call [SimHash.Hash].
func (s *SimHash) HashTokens(tokens []string) uint64 {
	if len(tokens) == 0 {
		return 0
	}
	freq := make(map[string]int, len(tokens))
	for _, t := range tokens {
		freq[t]++
	}
	feats := make([]Feature, 0, len(freq))
	for t, c := range freq {
		feats = append(feats, Feature{Hash: hashString(t, s.seed), Weight: float64(c)})
	}
	return s.Hash(feats)
}

// HammingDistance returns the number of differing bits between two SimHash
// fingerprints via hardware popcount. Near-duplicate is HammingDistance(a, b) <=
// threshold; a threshold of 3-4 bits out of 64 is a common near-duplicate cut.
func HammingDistance(a, b uint64) int {
	return bits.OnesCount64(a ^ b)
}
