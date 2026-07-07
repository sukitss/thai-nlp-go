// Package fusion combines several ranked result lists into one ranking — the
// glue that turns a hybrid retriever's separate sparse and dense hit lists (from
// [github.com/sukitss/thai-nlp-go/search/invidx] and
// [github.com/sukitss/thai-nlp-go/search/vector]) into a single ordered result.
// It is pure-Go, dependency-free, and deterministic.
//
// Two fusion methods are provided, covering the two honest ways to merge lists
// whose scores are NOT on the same scale (a BM25 score and a cosine similarity
// are not comparable numbers):
//
//	RRF         — Reciprocal Rank Fusion (Cormack et al. 2009). Scores a document
//	              from its RANK in each list, not its raw score, so it needs no
//	              calibration and is immune to one list's scores being orders of
//	              magnitude larger than another's. The robust, parameter-light
//	              default for mixing sparse + dense.
//	WeightedSum — normalize each list's scores onto a common range (min-max or
//	              z-score), then add them with per-list weights. Use it when you
//	              have tuned weights and want score magnitudes (not just ranks) to
//	              carry through — e.g. to lean on the dense list when it is
//	              confident.
//
// # The Hit type and adapting the primitives
//
// This package defines its own [Hit] ({ID uint32; Score float64}) rather than
// reuse invidx.Hit (Score float64) or vector.Hit (Score float32): a fusion
// helper must not depend on either sub-package, and the two disagree on the
// score width. Fusion carries scores in float64 (the wider of the two, lossless
// for both). Convert a primitive's result with [Adapt], a one-line projector:
//
//	sparse := idx.Search(qids, weight.NewBM25(), 100)       // []invidx.Hit
//	dense  := mat.TopK(qvec, 100)                           // []vector.Hit
//	fused  := fusion.RRF([][]fusion.Hit{
//	    fusion.Adapt(sparse, func(h invidx.Hit) fusion.Hit { return fusion.Hit{ID: h.ID, Score: h.Score} }),
//	    fusion.Adapt(dense,  func(h vector.Hit) fusion.Hit { return fusion.Hit{ID: h.ID, Score: float64(h.Score)} }),
//	}, 0)
//
// # Input contract
//
// Every input list must be ordered best-first, which is exactly what every
// search/* primitive returns (invidx.Search / vector.Flat.TopK produce a strict
// score-desc, id-asc order). RRF reads each hit's POSITION as its rank, so an
// unsorted list silently produces wrong ranks — fuse the lists as the primitives
// hand them back, do not re-shuffle them. A document may appear in any subset of
// the lists; a document absent from a list contributes nothing for that list. Ids
// identify the same document across lists (as they do when both indexes are keyed
// by the same corpus ids).
//
// # Determinism
//
// Both methods return a strict total order — higher fused score first, ties
// broken by lower id — identical to the ordering the rest of the search layer
// uses, so a fused ranking is reproducible and directly comparable.
package fusion

import "sort"

// Hit is one document in a ranked list: its id and its score under whatever
// ranker produced the list (larger = more relevant). Scores across different
// lists are generally NOT comparable — that is the problem fusion solves.
type Hit struct {
	ID    uint32
	Score float64
}

// betterHit is the strict total order used for every fused result: higher score
// wins, ties break to the lower id. Distinct ids make it a strict total order,
// so a fused ranking is unique and reproducible.
func betterHit(a, b Hit) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	return a.ID < b.ID
}

// Adapt projects any ranked list into a []Hit via a per-element projector. It is
// the bridge from a primitive's own hit type (invidx.Hit, vector.Hit, or a
// caller's struct) to the common [Hit] the fusion methods take, without this
// package importing those packages. Order is preserved, so the best-first
// contract carries through.
//
//	fusion.Adapt(dense, func(h vector.Hit) fusion.Hit {
//	    return fusion.Hit{ID: h.ID, Score: float64(h.Score)}
//	})
func Adapt[T any](list []T, proj func(T) Hit) []Hit {
	if len(list) == 0 {
		return nil
	}
	out := make([]Hit, len(list))
	for i, h := range list {
		out[i] = proj(h)
	}
	return out
}

// rank turns a score map into a strict best-first ranking. It is the shared
// tail of RRF and WeightedSum: collect the accumulated per-id scores, then sort
// by the canonical (score desc, id asc) order.
func rank(scores map[uint32]float64) []Hit {
	if len(scores) == 0 {
		return nil
	}
	out := make([]Hit, 0, len(scores))
	for id, s := range scores {
		out = append(out, Hit{ID: id, Score: s})
	}
	sort.Slice(out, func(i, j int) bool { return betterHit(out[i], out[j]) })
	return out
}
