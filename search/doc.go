// Package search is the umbrella for thai-nlp-go's retrieval / matching
// primitive layer: a pure-Go, no-LLM, permissively-licensed toolkit of the
// algorithms a lexical + semantic search stack is built from, reduced to
// embeddable primitives. It has no code of its own — the primitives live in the
// sub-packages, and you import only the ones you need:
//
//	query    turn a messy user query into matchable terms (filler strip, acronym retain/expand)
//	keyword  extract the few salient keywords/keyphrases of a document (auto-tag, facet)
//	weight   pluggable term-weighting / ranking schemes (BM25 family, DFR/PL2, QL, TF-IDF)
//	invidx   in-memory inverted index + WAND top-k (sparse retrieval)
//	vector   in-memory dense-vector nearest-neighbour matching + quantization
//	sketch   probabilistic sketches: MinHash/LSH, SimHash, count-min (dedup, approx-freq)
//	fusion   combine ranked lists into one (RRF, weighted-sum) for hybrid search
//
// See search/README.md for the pipeline diagram, per-package guidance with
// measured headline numbers, and a runnable end-to-end example (Example in
// example_test.go). The layer builds on the module's [vocab] (term→id + DF/IDF)
// and [multi] (one analysis pipeline for documents and queries).
//
// Everything here is a mechanism, not a policy: persistence, distribution,
// reranker routing, and fusion policy belong to the platform that composes these
// primitives.
package search
