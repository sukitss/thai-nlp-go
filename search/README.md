# search — retrieval / matching primitives for Thai

A **pure-Go, no-LLM, permissively-licensed** toolkit of the algorithms a lexical
+ semantic search stack is built from — term weighting, an inverted index with
WAND pruning, in-memory dense-vector matching with quantization, probabilistic
sketches for dedup, query understanding, keyword extraction, and rank fusion —
each reduced to an embeddable primitive with no vector database, embedding
server, or GPU required. These are the mechanisms a Lucene/Elasticsearch segment
or a Qdrant collection wraps; this layer hands them to you directly so a platform
composes its own retrieval policy on top. Everything builds on the module's
[`vocab`](../vocab) (term→id + DF/IDF) and [`multi`](../multi) (one analysis
pipeline for documents and queries).

Import only what you need — each sub-package stands alone.

## The pipeline

```
                 ┌──────────────────────── ingest (once) ────────────────────────┐
   documents ──▶ multi.Analyzer.Terms ──▶ vocab (term→id, DF/IDF)
                       │                        │
                       │                        ├──▶ weight.Collection ─┐
                       │                        │                       │
                       ├──▶ keyword.Extract ──▶ tags / facets / payload │
                       │                                                 ▼
                       └──▶ invidx.Builder (sparse postings) ◀── term ids + tf
                       └──▶ vector.Flat     (dense codes)    ◀── embeddings*
                       └──▶ sketch (MinHash/SimHash) ──▶ near-dup / dedup

                 ┌──────────────────────── query (per request) ──────────────────┐
   user query ──▶ query.ParseExpand  (strip filler, retain+expand acronyms)
                       │
          ┌────────────┼─────────────────────────┐
          ▼            ▼                           ▼
   term ids       term ids                    query embedding*
          │            │                           │
          ▼            ▼                           ▼
   invidx.Search   (BM25 via weight)         vector.Flat.TopK
   ── sparse hits ──┐                        ── dense hits ──┐
                    ▼                                        ▼
                    └──────────▶ fusion.RRF / WeightedSum ◀──┘
                                        │
                                        ▼
                           (optional) vector.Flat.Rescore   ── exact re-rank of the fused
                                        │                       candidate ids
                                        ▼
                                   final ranking
```

*`vector` takes embeddings you supply — a real dense model (BGE-m3 etc.) at
ingest and query time. The layer is embedding-model-agnostic; it does not ship
one.

A runnable version of this whole flow is [`example_test.go`](example_test.go) (a
Go `Example`, so `go test ./search/` compiles and verifies it): it indexes a tiny
Thai FAQ corpus, takes the messy query `"ช่วยหาเบอร์ปลาit หน่อย"`, runs
`query.ParseExpand` → sparse + dense → `fusion.RRF`, and prints the top hit.

## Packages

Headline numbers below are measured on the module's own Thai eval sets (real
BGE-m3 embeddings, real Thai Wikipedia, synthetic-but-permissive FAQ/office
corpora) — honest, same-arena, and reproducible from the benchmark harnesses;
they are single-machine snapshots, not CI-asserted assertions.

### [`query`](query) — user query → matchable terms · [godoc](https://pkg.go.dev/github.com/sukitss/thai-nlp-go/search/query)

Turns what a user actually types — long and conversational, or a bare acronym —
into the terms an index holds. `Parse` strips filler / question particles /
generic verbs and keeps the content entities; the crux is **acronym retention**:
a lower-case `"it"` that the stop list would drop (it *is* an English stop word)
is kept as the acronym `"IT"` the index holds, and a stuck form `"ปลาit"` is
split at the script boundary so both `"ปลา IT"` and `"ปลาit"` reach the same
terms. `Expand` adds equivalent forms (case/script variants + a Thai-enterprise
seed, e.g. `IT ⇒ ไอที / เทคโนโลยีสารสนเทศ`), and `ParseExpand` does both.

- **Constructors:** `New(opts...)` → `Parser`; `Parser.ParseExpand(text)` is the
  full query-understanding path. Options: `WithAliases` (per-tenant
  abbreviations), `WithStopwords`, `WithAnalyzer`.
- **When:** on the query side of any sparse/hybrid retriever, especially where
  short acronym queries or mixed Thai+Latin are common.
- **Measured:** on the short-acronym slice a naive lower-case-then-stopword-drop
  pipeline retains **0%** of acronyms (F1 0.460); acronym-aware parsing retains
  **100%** (F1 0.699) — the `0.00 → 1.00` acronym-recall fix that keeps a
  short query from collapsing to nothing.

### [`keyword`](keyword) — salient keywords / keyphrases · [godoc](https://pkg.go.dev/github.com/sukitss/thai-nlp-go/search/keyword)

Extracts the *few* words a human would tag a document with (auto-tag, facet,
metadata payload, highlight) — as opposed to the all-terms tokenization full-text
indexing uses. Four scorers behind one `Extractor` interface, all sharing the
same acronym-aware fold so a tag never silently loses its `"IT"`/`"POS"`.

- **Constructors:** `NewRAKE`, `NewTFIDFTopK(vocab)`, `NewYAKE`, `NewTextRank`;
  all expose `Extract(text, k) []Keyword`. `Keys(text)` returns the canonical
  content-word keys (the same projection the index side should use).
- **When:** to derive tags/facets at ingest, or to pull head entities from a long
  query. RAKE/YAKE/TextRank are corpus-free; TF-IDF needs a `vocab`.
- **Measured:** near-tie at the top on 34 hand-labeled Thai docs (RAKE F1@10
  **0.559**, TF-IDF 0.554, TextRank 0.522 @5); all four keep **100%** of gold
  acronyms vs **0%** for the ES-style lower-case baseline.

### [`weight`](weight) — pluggable term-weighting schemes · [godoc](https://pkg.go.dev/github.com/sukitss/thai-nlp-go/search/weight)

Ranking schemes over term–document statistics, so lexical retrieval is not locked
to one BM25. `Score(tf, df, n, docLen, avgDocLen)` for the df-only schemes;
`ScoreStats(Stats)` for the two that need collection frequency. `Collection`
assembles the corpus statistics from tokenized docs (or `invidx` supplies them
directly).

- **Constructors:** `NewBM25` (default `k1=1.2, b=0.75`), `NewBM25Plus`,
  `NewBM25L`, `NewDFR` (PL2), `NewQLDirichlet`, `NewTFIDF`; `NewCollection(docs)`.
- **When:** swap one line to try a scheme; use BM25+/BM25L/PL2/QL on corpora with
  long or length-skewed documents where the literature says they help.
- **Measured:** on clean short-FAQ retrieval the schemes **tie** (all R@10 1.000,
  nDCG@10 0.984–0.988) — the value here is *pluggability* (swap + re-measure), not
  a claim that a variant beats BM25 on easy corpora. Ship BM25 as the default.

### [`invidx`](invidx) — inverted index + WAND top-k (sparse) · [godoc](https://pkg.go.dev/github.com/sukitss/thai-nlp-go/search/invidx)

In-memory inverted index whose top-k query **prunes**: `Search` (WAND, Broder et
al. 2003) uses per-term max-score upper bounds to skip documents that provably
cannot enter the top-k, and returns the **identical** top-k as the full-scan
`SearchBrute` oracle — same ids, bit-identical scores, same order. It computes
the corpus statistics (df, cf, lengths) itself, so any `weight.CorpusScorer`
scores directly against it. `SearchBlockMax` (Block-Max WAND, Ding & Suel 2011)
adds per-block max impacts for tighter, position-local bounds — same exact top-k,
prunes strictly more.

- **Constructors:** `NewBuilder()` → `Add(docID, terms, tfs)` → `Build()` →
  `Index`; `Index.Search(query, scorer, k)` (WAND) or `SearchBlockMax` (BMW);
  `Count(tokenIDs)` derives terms/tfs; `Prepare(scorer)` warms the upper-bound
  cache.
- **When:** the sparse side of retrieval over a `vocab`-mapped corpus, once it is
  big enough that skipping pays. **Use `Search` (WAND) by default**; reach for
  `SearchBlockMax` only at web-scale / large k, where its extra pruning outweighs
  its per-round block lookup.
- **Measured:** on real Thai Wikipedia (**86,689 docs / 113,752 terms**) WAND is
  **1.7–6.4× faster** than brute (pruning 52–94% of candidates), matching the
  2–10× the literature reports; longer queries win more, larger k wins less. BMW
  prunes *more* than WAND at every setting but is net **slower** here (0.64–0.91×
  for |q|≥3): at ~86k WAND is already near-optimal, so the block lookup out-costs
  the extra skip — BMW's win is a large-corpus / large-k regime.

### [`vector`](vector) — dense-vector matching + quantization · [godoc](https://pkg.go.dev/github.com/sukitss/thai-nlp-go/search/vector)

In-memory nearest-neighbour matching over embedding vectors, CPU-only, no vector
DB. Two matchers share one `Matcher` interface (`Add` + `TopK`):

- **`Flat`** brute-scans every vector — **exact** with `Float32` by construction.
  `TopKParallel` shards across cores (byte-identical to serial), and `Rescore`
  re-ranks a coarse candidate set with an exact scorer (the two-stage pattern). A
  `Quantizer` trades memory for accuracy on the stored codes: `Float32` (exact),
  `Binary` (32× smaller, Hamming), `Scalar8` (4×), `PQ` (tunable, asymmetric ADC).
- **`HNSW`** is an approximate graph index whose `TopK` is **sub-linear** in the
  corpus size — it walks a navigable subgraph instead of scanning everything, so
  it pulls ahead of `Flat` as N grows. `efSearch` is the recall/latency knob.
  `WithQuantizer(NewBinarySym | NewScalar8Sym)` stores the graph over compact
  codes (built *and* searched in the code space) for 4–32× less memory and far
  cheaper build; `Float32` stays the default.

**Which matcher / quantizer:**

| corpus & goal | use |
|---|---|
| ≤ ~100k, exact ground truth | `Flat` + `Float32` |
| near-exact, 4× less RAM | `Scalar8` (on `Flat`, or `HNSW`+`WithQuantizer(NewScalar8Sym)`) |
| ≥ ~1M, sub-linear latency | `HNSW` (float32) |
| ≥ ~1M, RAM-bound / fast build | `HNSW`+`WithQuantizer(NewBinarySym)` → then `Flat.Rescore(Float32)` to recover the tail |

- **Constructors:** `NewFlat(NewFloat32(dim) | NewScalar8(dim) | NewBinary(dim) |
  TrainPQ(...))` with `Flat.TopK / Rescore / TopKParallel`; `NewHNSW(dim,
  WithEfSearch(n), WithQuantizer(NewBinarySym(dim) | NewScalar8Sym(dim)), WithM,
  WithEfConstruction, WithSeed)` with `HNSW.TopK / SetEfSearch`.
- **When:** the dense side of hybrid retrieval, or standalone semantic match —
  `Flat` for exact/small, `HNSW` once N is large enough that scanning everything
  hurts, quantized when memory or build/query throughput dominates.
- **Measured (real BGE-m3):**
  - `Flat`+`Scalar8`: **recall@10 ≈ 0.99** at 4× compression, same/faster
    latency — the near-exact default. `Binary` alone loses the tail (≈0.68) but
    scans ~26× faster at 32× smaller; **binary coarse → Float32 rerank (N≈100)
    restores recall@10 to 0.989** (N=500 → 1.000).
  - `HNSW` (float32): **recall@10 0.998–1.000**; at efSearch=10 on 7.5k vectors
    **23× faster than `Flat` single-core** (recall@10 0.998), and query latency
    stays sub-linear as N grows while `Flat` grows linear.
  - `HNSW`+`WithQuantizer(NewBinarySym)`: **builds 9–11× faster** (Hamming
    popcount ≫ float dot) at **32× less memory**, queries 8–12× faster, recall@10
    ≈ 0.67 — pair with `Flat.Rescore` to recover ~0.99. `NewScalar8Sym` keeps
    recall@10 ≈ 0.988 at 4× smaller.

### [`sketch`](sketch) — MinHash / SimHash / count-min · [godoc](https://pkg.go.dev/github.com/sukitss/thai-nlp-go/search/sketch)

Probabilistic primitives for near-duplicate detection and approximate frequency
at a scale where exhaustive comparison or a full vocabulary is too expensive.
`MinHash` (+ optional `LSH` banding) estimates Jaccard and generates near-dup
candidate pairs sub-linearly; `SimHash` is a 64-bit fingerprint compared by
Hamming distance; `CountMin` counts a stream in fixed memory and never
under-estimates.

- **Constructors:** `NewMinHash(numHashes, seed)` / `NewLSH(bands, rows, seed)`;
  `NewSimHash(seed)`; `NewCountMin(w, d, seed)` / `NewCountMinParams(eps, delta,
  seed)`; shingles via `WordShingles` / `CharNGrams`.
- **When:** dedup a large corpus (long documents, crawled docs), coarse-filter
  near-dups, or approximate DF when a full `vocab` is too big.
- **Measured (real Thai near-dup set):** **MinHash+LSH dedup F1 0.989**
  (precision 0.978, recall 1.000) at 128 hashes, signatures ~30× smaller than the
  shingle sets; SimHash-64 is a cheap 8-byte coarse gate (precision 0.974, recall
  0.62); CountMin never under-estimates (0 violations), worth it once the key
  space is genuinely huge.

### [`fusion`](fusion) — combine ranked lists (hybrid) · [godoc](https://pkg.go.dev/github.com/sukitss/thai-nlp-go/search/fusion)

The glue that merges the sparse and dense hit lists into one ranking. `RRF`
(Reciprocal Rank Fusion) scores each document from its **rank** in each list —
`Σ 1/(k + rank)`, `k` default 60 — so it needs no calibration and is immune to
the lists being on wildly different scales (BM25 vs cosine): the robust default.
`WeightedSum` min-max/z-score normalizes each list then adds with per-list
weights, for when tuned magnitudes should carry through. Both return the strict
`(score desc, id asc)` order the rest of the layer uses.

- **Constructors:** `RRF(lists, k)`, `WeightedSum(lists, weights, MinMax|ZScore)`;
  `Adapt(list, proj)` bridges `invidx.Hit` / `vector.Hit` into `fusion.Hit`.
- **When:** any hybrid retriever combining ≥2 rankings; start with `RRF` (no
  tuning), move to `WeightedSum` only when you have weights to tune.

### [`embed`](embed) — no-LLM in-process embedders (dense) · [godoc](https://pkg.go.dev/github.com/sukitss/thai-nlp-go/search/embed)

Turns text into a `[]float32` you can feed straight to [`vector`](vector) — with
no model server, GPU, or network, so a RAG stack runs fully offline. `Hashing`
feature-hashes char n-grams into a fixed-width vector: purely lexical, so it is
robust to OOV terms and typos (`เนตเวิรค์ช้า` still matches where BM25 returns
nothing) but semantically blind (`cos(รถ, ยานพาหนะ) ≈ 0`). `Static` loads a
pre-trained fastText/word2vec `.vec` file for real semantics (`cos(รถ, ยานพาหนะ)
≈ 0.996`) and **ships no model** — you point it at a file whose license is yours
to accept. Both satisfy one `Embedder{Embed, Dim}` interface, so you can swap or
average them.

- **Constructors:** `NewHashing(dim, ngram)`, `LoadStatic(r)` / `LoadStaticFile(path)`.
- **When:** any dense retrieval without an embedding service. `Hashing` for a
  zero-dependency lexical fallback; `Static` when you can supply a vector file
  and need semantic matching. Fuse the sparse and dense hits with [`fusion`](fusion)
  for the best of both.

## Design notes

- **⚠️ Analyzer alignment (the silent zero-recall trap):** the terms you *index*
  and the terms you *query* must fold identically, or matches vanish with no
  error. Index documents with [`keyword`](keyword)`.Keys` and parse queries with
  [`query`](query)`.Parse` — both apply the same acronym-aware fold (`IT`≡`it`≡`ไอที`),
  so they line up by construction. If you tokenize one side by hand, mirror the
  exact same casing/acronym fold on the other.
- **Bridging hit lists into fusion:** [`fusion`](fusion)`.Adapt(list, proj)` lifts
  any `invidx.Hit` / `vector.Hit` slice into `[]fusion.Hit` in one line —
  `fusion.Adapt(vHits, func(h vector.Hit) fusion.Hit { return fusion.Hit{ID: h.ID, Score: float64(h.Score)} })`.
- **Deterministic, strict ordering** everywhere: every result is `(score desc, id
  asc)`, so rankings are reproducible and directly comparable across primitives.
- **Low-alloc hot paths**, pure Go, no cgo/BLAS/ONNX; SIMD-ish work goes through
  `math/bits`.
- **Mechanism, not policy:** persistence (Qdrant), distribution, reranker
  routing, chunking and fusion *policy* belong to the platform, not this layer.

All Apache-2.0, same as the rest of the module. See the top-level
[README](../README.md) and [ARCHITECTURE.md](../ARCHITECTURE.md).
