// Package invidx is an in-memory inverted index with a top-k query that PRUNES:
// it skips documents that provably cannot enter the top-k instead of scoring
// every candidate. It is the mechanism a Lucene/Elasticsearch segment or a
// Terrier index wraps for lexical (sparse) retrieval, reduced to a pure-Go
// primitive any caller can embed directly. It closes the sparse pipeline the
// rest of the library builds: query → terms ([vocab]) → weighting
// ([search/weight]) → this index's top-k.
//
// # What it stores
//
// Documents are bags of term ids — the dense sequential ids [vocab] assigns —
// each with a term frequency. The index keeps, per term, a postings list
// (the documents containing it and their term frequencies) sorted by document,
// plus each document's length and the corpus aggregates (document count, mean
// length, per-term document and collection frequency). Those aggregates are
// exactly the statistics a [weight.CorpusScorer] reads, so BM25 and its
// relatives score directly against the index with no separate
// [weight.Collection]. See [Index] on why the index computes them itself rather
// than borrowing the string-keyed Collection.
//
// # Two queries, one result
//
// [Index.SearchBrute] is the document-at-a-time baseline: it merges the query
// terms' postings and fully scores every document containing any query term. It
// is the correctness oracle — simple enough to trust.
//
// [Index.Search] is WAND (Broder et al. 2003): weak-AND with max-score upper
// bounds. Each term has a precomputed ceiling on the contribution it can make to
// any document (its maximum over the postings). Summing the ceilings of the
// terms that could appear in a document upper-bounds that document's score; when
// the bound cannot beat the current k-th best, WAND skips straight past the
// document without scoring it. On a large corpus this evaluates a fraction of
// the documents the brute scan touches, yet — this is the guarantee — returns
// the IDENTICAL top-k: the same ids AND the same scores, in the same order.
//
// # The scorer contract
//
// WAND's upper bounds are only valid for an ADDITIVE, NON-NEGATIVE, present-only
// scorer: a document's score is the sum over the query terms it contains of a
// per-term contribution that is >= 0, and a term the document lacks contributes
// nothing. The BM25 family ([weight.BM25], [weight.BM25Plus], [weight.BM25L])
// and [weight.TFIDF] satisfy this. [weight.QLDirichlet] does NOT — its smoothing
// gives absent terms a negative background contribution, so a document's score
// is not a sum over only its present terms — and must be run with SearchBrute.
// [weight.DFR] (PL2) is non-negative and present-only in the regime that matters
// but its surprisal can dip slightly negative for a near-expected term; the
// index's ceilings clamp at 0, which stays a valid (if looser) upper bound, so
// WAND remains correct on it. When in doubt, SearchBrute is always exact.
//
// # Determinism
//
// Ranking uses a strict total order — higher score first, ties broken by lower
// document id — identical to search/vector, so both queries are fully
// deterministic and directly comparable.
//
// # Concurrency
//
// Build the index once with a [Builder] (single goroutine). Once built it is
// read-only: SearchBrute and Search are safe for concurrent callers, EXCEPT
// that the first Search (or Prepare) for a given scorer computes and caches that
// scorer's upper bounds under a mutex — call [Index.Prepare] once up front to
// keep the query path lock-light and allocation-light.
package invidx

import "sort"

// Hit is one retrieved document: its caller-supplied id and its score under the
// query's scorer (larger = more relevant). Scores are float64, the width the
// weight schemes compute in, so a Hit reproduces a scheme's value exactly.
type Hit struct {
	ID    uint32
	Score float64
}

// betterHit is the strict total order used for ranking: higher score wins; ties
// break to the lower id. Distinct ids make it a strict total order, so the
// top-k set is unique and both query paths agree on it.
func betterHit(a, b Hit) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	return a.ID < b.ID
}

// topK is a bounded max-selection heap keeping the k best hits. It is a min-heap
// under betterHit, so hits[0] is the worst kept hit (the eviction candidate and
// the WAND threshold). Allocation-free after construction.
type topK struct {
	k    int
	hits []Hit
}

func newTopK(k int) *topK { return &topK{k: k, hits: make([]Hit, 0, k)} }

// full reports whether the heap holds k hits, i.e. whether hits[0] is a real
// threshold rather than "anything qualifies".
func (t *topK) full() bool { return len(t.hits) >= t.k }

// worst returns the score of the worst kept hit — the WAND threshold. Only
// meaningful when full.
func (t *topK) worst() float64 { return t.hits[0].Score }

func (t *topK) push(h Hit) {
	if len(t.hits) < t.k {
		t.hits = append(t.hits, h)
		t.up(len(t.hits) - 1)
		return
	}
	if betterHit(h, t.hits[0]) { // better than the worst kept
		t.hits[0] = h
		t.down(0)
	}
}

func (t *topK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !betterHit(t.hits[p], t.hits[i]) { // parent not better => heap ok
			break
		}
		t.hits[p], t.hits[i] = t.hits[i], t.hits[p]
		i = p
	}
}

func (t *topK) down(i int) {
	n := len(t.hits)
	for {
		l, r := 2*i+1, 2*i+2
		worst := i
		if l < n && betterHit(t.hits[worst], t.hits[l]) {
			worst = l
		}
		if r < n && betterHit(t.hits[worst], t.hits[r]) {
			worst = r
		}
		if worst == i {
			break
		}
		t.hits[i], t.hits[worst] = t.hits[worst], t.hits[i]
		i = worst
	}
}

func (t *topK) result() []Hit {
	out := make([]Hit, len(t.hits))
	copy(out, t.hits)
	sort.Slice(out, func(i, j int) bool { return betterHit(out[i], out[j]) })
	return out
}
