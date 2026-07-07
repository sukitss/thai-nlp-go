// Package vector provides in-memory dense-vector nearest-neighbour matching:
// given a corpus of embedding vectors it finds the top-k most similar to a
// query, entirely on the CPU with no vector database, embedding server, or
// GPU. It is the mechanism a Lucene/Elasticsearch dense-vector field or a
// Qdrant collection wraps, reduced to a pure-Go primitive that any caller can
// embed directly.
//
// # Quantization
//
// Storing every vector as raw float32 is exact but expensive: a 1024-dim
// corpus costs 4 KiB per vector. A [Quantizer] trades a little accuracy for a
// lot of memory by compressing each vector to a compact code and scoring a
// query against those codes:
//
//	Float32 — no compression, exact cosine. The ground-truth baseline every
//	          other quantizer's recall is measured against.
//	Binary  — one sign bit per dimension; distance is Hamming via hardware
//	          popcount. ~32x smaller.
//	Scalar8 — one int8 per dimension with a per-vector scale. ~4x smaller,
//	          more accurate than Binary.
//	PQ      — product quantization: split the vector into m subvectors, learn
//	          a k-means codebook per subspace, store one centroid id per
//	          subspace. Tunable memory/accuracy; scored with asymmetric
//	          distance (the query stays float, only the corpus is quantized).
//
// # Scoring convention
//
// Every quantizer scores a stored code against a query so that a LARGER score
// means MORE similar (closer). This lets [Flat] keep a single k-element
// max-selection heap that works identically for cosine similarity, negated
// Hamming distance, and PQ table lookups. For the vector families here the
// score approximates cosine similarity, so it is directly comparable across
// quantizers.
//
// This is the reason the interface exposes a query-bound [Scorer] rather than
// the symmetric Distance(a, b []byte) sketched in the design notes: PQ's
// asymmetric distance keeps the query in full precision and precomputes a
// per-query lookup table, which a symmetric code-to-code function cannot
// express. [Binary] additionally exposes an exact [Binary.Hamming] for callers
// that want the raw symmetric primitive.
//
// # Matching
//
// [Flat] is a brute-force matcher: it scans every stored code. With the
// Float32 quantizer its top-k is exact by construction. [Flat.TopKParallel]
// shards the scan across goroutines and returns byte-identical results to the
// serial [Flat.TopK]. Graph indexes (HNSW) are a planned extension; the
// [Matcher] interface is the seam they will plug into.
//
// # Concurrency
//
// Building a matcher (Add) is single-goroutine. Once built, TopK and
// TopKParallel are read-only and safe for concurrent callers.
package vector

import "math"

// Hit is one matched vector: its caller-supplied id and its similarity score
// (larger = more similar).
type Hit struct {
	ID    uint32
	Score float32
}

// Scorer scores stored codes against a single fixed query vector. It is
// returned by [Quantizer.Query] and encapsulates whatever per-query state a
// quantizer needs (a normalized query, a bitset, or a PQ lookup table). A
// Scorer is read-only and may be shared across goroutines scanning disjoint
// code ranges.
type Scorer interface {
	// Score reports the similarity of code to the query the Scorer was built
	// for. Larger = more similar. code must be exactly [Quantizer.CodeLen]
	// bytes, as produced by [Quantizer.Encode].
	Score(code []byte) float32
}

// Quantizer compresses float32 vectors into fixed-length byte codes and scores
// a query against them. All vectors it handles share one dimensionality
// ([Quantizer.Dim]); Encode always emits [Quantizer.CodeLen] bytes.
type Quantizer interface {
	// Encode compresses vec (length Dim) into a code of length CodeLen.
	Encode(vec []float32) []byte
	// Dim is the vector dimensionality the quantizer was built for.
	Dim() int
	// CodeLen is the byte length of every code Encode produces.
	CodeLen() int
	// Query returns a Scorer bound to query (length Dim).
	Query(query []float32) Scorer
	// Name identifies the quantizer for reporting, e.g. "float32" or "pq".
	Name() string
}

// Matcher holds a corpus of encoded vectors and returns the k nearest to a
// query. Add is single-goroutine; TopK is safe for concurrent readers once
// building is done.
type Matcher interface {
	Add(id uint32, vec []float32)
	TopK(query []float32, k int) []Hit
}

// l2norm returns the Euclidean length of v as a float64 for precision.
func l2norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

// normalized returns a unit-length copy of v. A zero vector is returned
// unchanged (all zeros), which scores 0 against every query.
func normalized(v []float32) []float32 {
	out := make([]float32, len(v))
	n := l2norm(v)
	if n == 0 {
		return out
	}
	inv := float32(1 / n)
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// dot returns the dot product of two equal-length float32 vectors, accumulated
// in float32 (the hot path — callers that need precision normalize first).
func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
