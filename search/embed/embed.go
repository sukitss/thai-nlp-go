// Package embed turns text into a fixed-dimension dense []float32 vector — the
// missing front-half of the dense retrieval path. The
// [github.com/sukitss/thai-nlp-go/search/vector] package matches vectors but
// ships no way to PRODUCE them; without an embedder a hybrid retriever cannot
// run in-process, it has to call out to a model server. This package fills that
// gap with two NO-LLM, no-GPU, pure-Go embedders that run entirely on the CPU
// and plug straight into a [vector.Flat]:
//
//	emb := embed.NewHashing(256)
//	mat := vector.NewFlat(vector.NewFloat32(emb.Dim()))
//	for id, doc := range corpus {
//		mat.Add(uint32(id), emb.Embed(doc))
//	}
//	hits := mat.TopK(emb.Embed(query), 10)
//
// # The two embedders, and what each is for
//
//	Hashing — feature-hashing / random-projection over character n-grams. Needs
//	          NO pretrained model: it works offline on any input, is
//	          deterministic, and is ROBUST TO SPELLING VARIANTS because words
//	          that share character n-grams land near each other in the vector
//	          space. It captures LEXICAL similarity (typos, name variants,
//	          acronyms stuck to a name — "ปลาit" ≈ "ปลา IT") — NOT deep
//	          semantics. It will not know that "รถ" and "ยานพาหนะ" mean the same
//	          thing; they share no characters. See [Hashing].
//	Static  — mean-pools pretrained word vectors (fastText / word2vec text
//	          format). This gives REAL distributional semantics, but only if the
//	          caller supplies a model — this package ships NO vectors, just the
//	          loader and the pooling. See [Static] and [LoadStatic].
//
// # No model is shipped
//
// [Hashing] needs no model at all. [Static] needs one the CALLER provides; its
// license is the caller's concern (a fastText .vec is typically CC-BY-SA). This
// package is only the mechanism.
//
// # Determinism and concurrency
//
// Both embedders are immutable after construction and safe for concurrent
// [Embedder.Embed] calls. [Hashing] is fully deterministic (a fixed seed drives
// the hash), so the same text always yields byte-identical output; [Static] is
// deterministic given a fixed model.
package embed

import "math"

// Embedder maps text to a dense unit-length vector of a fixed dimension. Both
// [Hashing] and [Static] implement it, so either drops into a [vector.Flat]
// (built with a quantizer of the matching Dim) the same way. The returned slice
// is freshly allocated and owned by the caller.
type Embedder interface {
	// Embed returns the embedding of text: a vector of length Dim(),
	// L2-normalized (unit length), or an all-zero vector when text has no
	// embeddable content (empty, or — for Static — entirely out-of-vocabulary).
	// A zero vector scores 0 against every query, which is the intended
	// "no signal" behavior.
	Embed(text string) []float32
	// Dim reports the dimensionality of every vector Embed returns.
	Dim() int
}

// l2normalize scales v to unit length in place and returns it. A zero vector is
// left as all-zeros (it carries no direction and scores 0 against any query).
// Accumulation is read in float64 for a stable norm regardless of dimension.
func l2normalize(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
	return v
}
