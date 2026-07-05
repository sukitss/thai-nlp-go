# thai-nlp-go

A fast, memory-conscious **Thai NLP toolkit** for Go — built for
high-throughput text pipelines. One module, small focused sub-packages; import
only what you need.

```go
import "github.com/sukitss/thai-nlp-go/tokenize"

seg, _ := tokenize.NewDefault()              // shared embedded dictionary
seg.SegmentNoWS("ฉันรักภาษาไทยมาก")           // ["ฉัน" "รัก" "ภาษาไทย" "มาก"]
seg.SegmentBytes("ฉันรักภาษาไทยมาก", ' ')     // []byte, straight into a BM25 index
```

## Status

| Package | What | Status |
| --- | --- | --- |
| [`dict`](dict) | Shared dictionary (mmap flat trie, one shared instance) | ✅ |
| [`script`](script) | Split mixed-language text into runs by writing system | ✅ |
| [`token`](token) | Shared token-with-byte-offsets value used by all tokenizers | ✅ |
| [`tokenize`](tokenize) | Word segmentation (PyThaiNLP **newmm** port) + char **n-gram** | ✅ |
| [`cjk`](cjk) | Chinese word segmentation (dictionary maximal-matching) + per-tenant `Session` overlay | ✅ |
| [`jp`](jp) | Japanese word segmentation (dictionary maximal-matching) + per-tenant `Session` overlay | ✅ |
| [`en`](en) | Light English/Latin word tokenization (no dictionary) | ✅ |
| [`kr`](kr) | Light Korean tokenization (eojeol + particle stem, no dictionary) | ✅ |
| [`multi`](multi) | One-call multilingual tokenization (detect + route th/cn/jp/kr/en); Analyzer options: stop words, width fold, coarse/fine `Subwords` | ✅ |
| [`normalize`](normalize) | Text normalization: Thai (PyThaiNLP-faithful) + multilingual `FoldForIndex` (full/half-width + NFC for CJK/Latin/Korean) | ✅ |
| [`stopwords`](stopwords) | Stop-word filtering: Thai/English + curated CJK/JP/KR function words + `Multilingual()` | ✅ |
| [`vocab`](vocab) | Term → sequential-id vocabulary + DF/IDF (sparse/BM25 indexing) | ✅ |
| [`sentence`](sentence) | Sentence splitting — selectable engines: `Whitespace` (fastest), `Heuristic` (ender/starter words, no model, ~14× faster than CRF), and `crf` (crfcut, most accurate) | ✅ |
| [`chunk`](chunk) | Offset-true hierarchical chunking (RAG ingestion) | ✅ |
| [`translit`](translit) | Name-variant matching (MetaSound/Udom83/LK82 + edit distance); cross-lingual pinyin→Thai bridge (聂力/Nie Li/เนี่ยหลี่ in one bucket) | ✅ |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design, and
[docs/TEST-REPORT.md](docs/TEST-REPORT.md) for a test/benchmark snapshot
(regenerate with `make report`).

## Design principle: performance and memory first

The reason this is a single module is **so components share one memory-mapped
dictionary** (`dict.Default()`) instead of each loading its own — the failure
mode we explicitly avoid (naive PyThaiNLP wrappers reload a trie per
instance/process). Hot paths offer `[]byte`/append APIs to skip per-token
allocation. Unused packages add nothing to your binary. See
[ARCHITECTURE.md](ARCHITECTURE.md).

## Install

```sh
go get github.com/sukitss/thai-nlp-go
```

Requires Go 1.24+ (via `mmap-go`). Runtime dependency: only `mmap-go`
(`regexp2` is test-only).

## Tokenizer

Faithful port of PyThaiNLP `word_tokenize(engine="newmm")` — output matches
byte-for-byte, verified by golden tests (8000 + 3008 lines) and 200k fuzz
strings against a regexp2 oracle.

```go
seg, _ := tokenize.NewDefault()
seg.SegmentNoWS("ฉัน รัก ภาษาไทย") // ["ฉัน" "รัก" "ภาษาไทย"]  (whitespace dropped)
seg.Segment("ฉัน รัก")            // ["ฉัน" " " "รัก"]         (whitespace kept)

// Zero-allocation []byte output for indexing (reuse the buffer across a corpus):
buf := make([]byte, 0, 4096)
for _, doc := range docs {
    buf = seg.AppendBytes(buf[:0], doc, ' ')
    index.Write(buf)
}

// Per-session overlay: extra words (e.g. character names) on top of the shared
// base dictionary — nothing rebuilt, sessions isolated and concurrent-safe:
sess := seg.Session([]string{"อาริน", "เวธกา"})
sess.SegmentNoWS("อารินพบเวธกา") // names stay whole

// Dictionary-free char n-gram (robust to OOV/typos; for BM25 / fuzzy matching):
ng := tokenize.NewNGram(3)
ng.Split("ฉันรักภาษาไทย")
```

#### Token offsets

Search highlighting and entity-mention extraction need to know **where** each
token sits, not just what it says. Every tokenizer can report byte offsets:
`Tokens`/`TokensNoWS` (and buffer-reusing `AppendTokens`/`AppendTokensNoWS`)
return `token.Token{Text, Start, End}` — a tiny zero-dependency leaf package —
with the guarantee `text[t.Start:t.End] == t.Text` into the exact string you
passed. The same API exists on `en`, `cjk`, `jp` and `kr` (`Tokens`, plus
`TokensDP` for cjk/jp), and `script.Run` carries a `Start` offset so per-run
token positions compose into whole-document positions. No normalization
happens inside the tokenizers: if you normalize first, offsets point into the
normalized string you passed, not the original.

```go
for _, t := range seg.TokensNoWS("ฉันรักภาษาไทย") {
    highlight(t.Start, t.End) // text[t.Start:t.End] == t.Text
}
```

Offsets are ~free: benchmarked in `tokenize/tokens_test.go`, `Tokens` runs
within a few percent of `Segment`'s throughput over the ~1 MB corpus, and the
`AppendTokens` reuse path makes fewer allocations than `Segment` (~132k vs
~219k allocs per corpus pass — one dev-machine snapshot, not asserted in CI).

#### Per-user (cached) dictionaries

`Session` rebuilds its overlay each call. For per-user "dynamic" dictionaries,
build the overlay trie once and cache it in your application (the library stays
stateless — you own the cache, eviction and persistence):

```go
base, _ := dict.Default()          // shared, loaded once for the whole process
seg := tokenize.New(base)

ov := dict.NewTrie()               // build + cache this per user (e.g. in an LRU)
for _, w := range userWords {
    ov.Add(w)
}

// per request / per goroutine — cheap:
s := seg.SessionWithDict(ov)
s.SegmentNoWS("...")
```

Concurrency: the base dictionary and the overlay `*dict.Trie` are read-only and
safe to share across goroutines. A `Session`/`SessionWithDict` Segmenter keeps
per-lookup scratch, so make **one per goroutine** (it's a tiny struct).

`dict.Trie` covers the full lifecycle of a live per-user dictionary: `Remove`
deletes a word (pruning emptied branches), `Words`/`WalkPrefix` enumerate
entries in lexicographic order (`WalkPrefix` works on both the mutable trie
and the flat mmap trie) — e.g. for glossary autocomplete, and `WriteFlat`/`FlatBytes`
serialize a trie to the same flat format as the embedded dictionary
(`FromBytes` loads it back), so a user dictionary can persist in an object
store or database blob. Tries are never safe to mutate while shared: when a
user edits their words, rebuild (or copy and extend) the overlay and **swap
the pointer** — in-flight readers keep the old trie until they pick up the
new one.

Overlays can also carry weights: for a word present in both dictionaries, an
overlay built with `AddWeighted` **shadows** the base weight (a per-tenant
glossary can re-weight a base word); an unweighted overlay has no weight
opinion, so the base weight passes through (documented on
`OverlayDict.PrefixWeights`).

## Normalize

Faithful port of PyThaiNLP `normalize()` (rule-based: strip zero-width, collapse
spaces, drop stray spaces before marks, reorder tone marks/vowels, drop repeats
and dangling marks) — verified byte-for-byte against PyThaiNLP over 2,597 cases.
A fast path returns mark-free text untouched.

```go
import "github.com/sukitss/thai-nlp-go/normalize"

normalize.Normalize("เเปลก")   // "แปลก"  (double Sara E → Sara Ae)
normalize.Normalize("นานาาา")  // "นานา"  (drop repeated vowels)
normalize.Normalize("ก    ข")  // "ก ข"   (collapse spaces)
```

Thai digits are deliberately **not** folded by `Normalize` (PyThaiNLP parity —
its `normalize()` doesn't fold them either). When an index should treat `๕`
and `5` as the same term, apply the separate step after normalizing — it
returns the input string as-is (zero allocation) when there is no Thai digit:

```go
normalize.DigitsToArabic("บทที่ ๕") // "บทที่ 5"
```

For the strongest canonicalization (exact-match / dedup / hashing) use
`Canonical`, which adds Unicode canonical mark reordering on top — something
PyThaiNLP's `normalize()` does not do:

```go
normalize.Canonical(s)  // Normalize + canonical combining-mark order
normalize.Reorder(s)    // just the reordering step
```

`Reorder` puts Thai combining marks into Unicode canonical order (stable-sort by
combining class within a cluster), so two strings that render identically but
were typed in different mark order become identical bytes. It is verified to
match `golang.org/x/text/unicode/norm.NFC` over 200k random Thai sequences and
the cases in Unicode UTC L2/18-216 — with **no runtime dependency** (hand-coded
for the Thai block; x/text is used only as a test oracle).

## Stopwords

Thai stop-word set derived from PyThaiNLP `thai_stopwords()` (its 1,030 raw
entries, cleaned to 1,027 canonical words: two duplicates that differed only by
tone/vowel encoding merged, one stray BOM-prefixed duplicate dropped — an
intentional, documented deviation), plus a common English set. Matching is
case-sensitive, so latin acronyms stay distinct (`it` is a stop word, `IT` is
not).

```go
import "github.com/sukitss/thai-nlp-go/stopwords"

sw := stopwords.Union(stopwords.Default(), stopwords.English())
sw.IsStopword("และ")                        // true
sw.Filter([]string{"ผม","และ","รัก","the"}) // ["ผม" "รัก"]

custom := stopwords.Union(stopwords.Default(), stopwords.New("อาริน", "เวธกา"))
```

For **corpus-specific** stop words, derive them by document frequency (the
classic IR criterion — words in most documents carry little signal; more robust
than ranking by raw term frequency, which pulls in common content words):

```go
b := stopwords.NewBuilder()
for _, doc := range corpus {
    b.AddDoc(seg.SegmentNoWS(doc)) // tokens of each document
}
domain := b.Build(0.5) // words appearing in ≥50% of documents
sw := stopwords.Union(stopwords.Default(), domain)
```

## Sentence & transliteration

```go
import (
    "github.com/sukitss/thai-nlp-go/sentence"
    "github.com/sukitss/thai-nlp-go/translit"
)

sentence.Split("ผมชอบกินข้าว วันนี้อากาศดี") // ["ผมชอบกินข้าว" "วันนี้อากาศดี"]

translit.Key("ทองดี") == translit.Key("ทองดา") // true — spelling variants collide
```

`sentence` splits on whitespace (rule-based, matches PyThaiNLP non-ML engines).
`translit` offers four Thai phonetic keys — `Key` (MetaSound), `Udom83`, `LK82`,
and `CompleteSoundex` (Tapsai 2020; single-syllable, or pass pre-split syllables
to `CompleteSoundexSyllables` for multi) — all faithful ports, golden-verified
(for `CompleteSoundex` the golden covers 235 single-syllable words plus 15
multi-syllable cases with pre-split syllables; automatic multi-syllable
splitting is not golden-verified) — plus `Levenshtein`/`Similarity` for fuzzy
matching. Bucket candidates by a phonetic key, then rank with edit
distance to recover variants the keys miss:

```go
translit.SoundsLike("ทองดี", "ทองดา", 0.8) // true — same-sounding name variants
translit.SoundsLike("อุจิวะ", "อุจิฮะ", 0.8) // true — keys differ but 1 edit (~0.83)
translit.Key("ทองดี") == translit.Key("ทองดา")  // a single phonetic key
translit.Similarity("อุจิวะ", "อุจิฮะ")            // 0.83 edit-distance ratio
```

`SoundsLike` combines the phonetic keys (bucketing) with edit distance (catching
near-misses the keys drop) — the practical sound-alike test for de-duplicating
name spellings.

### Sound-alike entity index

`SoundIndex` maps many spellings of an entity to one id — e.g. a character name
transliterated differently by different translators (`เนียลี่` / `เนี่ยหลี่` /
`เนี่ยลี่`). Build it from extracted entities, then a query in *any* spelling
finds the entity; feed the ids into a hybrid-search keyword filter or weight.

```go
idx := translit.NewSoundIndex()
idx.Add("nieli", "เนียลี่")     // register surface spellings for an entity
idx.Add("nieli", "เนี่ยหลี่")
idx.Alias("นี่หลี่", "nieli")   // interpret-variant the phonetic keys miss

idx.Lookup("เนี่ยลี่") // ["nieli"] — ranked by edit-distance similarity
```

Three layers: phonetic-key bucketing + edit-distance ranking + a manual alias
map. Build it fully once, then it's read-only and safe for concurrent `Lookup`
across goroutines (phonetic key computation is stateless and allocation-light).

When ids feed a search filter, an unranked list isn't enough — a weak
sound-alike can drag in the wrong entity. `LookupScored` is `Lookup` with
scores so you can threshold: each entity scores as the maximum edit-distance
`Similarity` between the query and its bucket-matched spellings (an exact
alias hit scores 1.0), and matches below `minSim` are dropped.
`LookupScored(q, 0)` returns exactly `Lookup(q)`'s ids. `Len` reports how many
distinct entity ids are registered.

```go
for _, m := range idx.LookupScored("เนี่ยลี่", 0.8) {
    // m.ID, m.Score in [0,1] — only confident sound-alikes widen the filter
}
```

**Wiring into hybrid (entity-aware) search.** The keyword/sparse side of a hybrid
retriever matches exact tokens, so a query in one spelling misses documents that
use another. Bridge them with `SoundIndex`:

```
ingest: extract entity mentions per document; idx.Add(entityID, mention)
        and tag the document with its entityIDs.
query:  for a query name, ids := idx.Lookup(name)  // all spellings of that entity
        then FILTER or up-WEIGHT documents tagged with any of those ids.
```

So a search for `เนี่ยลี่` still recalls documents that wrote `เนียลี่` or
`เนี่ยหลี่` (or `นี่หลี่` via an alias). This complements dense retrieval, which
tends to miss proper-noun spelling variants.

For higher-accuracy sentence segmentation there is an opt-in CRF sub-package —
a faithful port of PyThaiNLP's `crfcut`, still CPU-only and batch-friendly (no
LLM), matching PyThaiNLP output exactly over 3,015 test cases:

```go
import "github.com/sukitss/thai-nlp-go/sentence/crf"

crf.Split("ผมชอบกินข้าว วันนี้อากาศดีมากครับ ยินดีที่ได้รู้จัก")
// ["ผมชอบกินข้าว " "วันนี้อากาศดีมากครับ " "ยินดีที่ได้รู้จัก"]
```

It's a separate package (pulls in the tokenizer + embeds the ~2 MB CC-BY-4.0
model), so the base `sentence` package stays dependency-light.

**Retrainable (in-domain).** crfcut is highly domain-dependent, so the CRF is
retrainable on your own labelled data (`crf.Train` / `crf.Eval` / `Model.Save`,
plus the `cmd/crftrain` tool that reads CoNLL-U). Validated on Universal
Dependencies Thai (CC BY-SA): training in-domain on UD_Thai-TUD lifts held-out
sentence-boundary **E-F1 to 0.99, vs 0.78 for the embedded TED model and 0.47
for whitespace splitting** — in-domain training is the biggest quality lever
(one-off measurement via `cmd/crftrain` on UD_Thai-TUD; not reproduced in CI —
the training data is CC BY-SA and not bundled).

```go
model := crf.Train(examples, 10)          // examples: tokens + I/E labels
m := crf.Eval(model.Labels, goldExamples) // E-precision/recall/F1
```

## One call for mixed-language text

For a translated novel or a company document that mixes Thai, Chinese, Japanese,
Korean and English, `multi.Segment` does the whole job — normalize, detect each
run's language, route it to the right tokenizer, return index-ready tokens — so
you don't wire the routing yourself:

```go
import "github.com/sukitss/thai-nlp-go/multi"

multi.Segment("ผมอ่าน三国志と日本語")          // tokens across all languages, in order
buf = multi.AppendBytes(buf[:0], text, ' ')   // output into a reused buffer for indexing
```

Thai → newmm, Chinese → cjk, Japanese → jp, Korean → kr, Latin → en. CJK runs
keep kanji+kana together and route to Japanese when kana is present, else Chinese
(`multi.DefaultHan` sets the pure-Han default — a mutable process-wide global;
`Analyzer.Han` below carries the choice per value instead). Importing `multi`
embeds all language dictionaries (loaded lazily per language on first use);
import single packages if you only need some.

## One pipeline for documents and queries

A keyword/sparse index only works when documents and queries are analyzed by
**exactly** the same pipeline — any asymmetry (stop words applied on one side,
digits folded on one side, a per-user word recognized at ingest but not at
query time) silently breaks retrieval. `multi.Analyzer` holds the whole
pipeline in one value: normalize → optional Thai-digit folding → script
routing → per-run tokenizers (with a per-tenant dictionary overlay on Thai
runs) → stop-word / Latin case-fold post-filters. Configure it once per
tenant and share it: an `Analyzer` is immutable in use and **safe for
concurrent use** across request goroutines, and the zero value behaves exactly
like `multi.Segment` (asserted by tests).

```go
glossary := dict.NewTrie() // per-tenant words: character names, product codes
glossary.Add("อาริน")

a := &multi.Analyzer{
    Overlay:        glossary,
    Stop:           stopwords.Union(stopwords.Default(), stopwords.English()),
    LowerLatin:     true,             // fold Latin case in Terms output
    FoldThaiDigits: true,             // "๕" indexes as "5"
    Han:            multi.HanChinese, // ambiguous all-kanji runs → Chinese
}

a.Terms("อารินอ่าน The Book บทที่ ๕") // same call for ingest and query
buf = a.AppendTerms(buf[:0], doc, ' ') // reused buffer on the indexing path

norm, toks := a.Tokens(query)          // highlighting path
_ = norm[toks[0].Start : toks[0].End]  // == toks[0].Text, always
```

`Tokens` runs the same pipeline but keeps every token — no stop-word dropping,
no lowercasing (filtering the highlight path would hide matches) — and returns
the **normalized** string its offsets index into. Offsets are not positions in
your original argument (Normalize collapses spaces and reorders marks;
digit folding changes byte lengths): highlight against `norm`, or store `norm`
alongside the offsets. Filter order (`LowerLatin` folds first, then `Stop` is
checked against the folded term, with the acronym caveat that entails) is
documented on the struct fields.

## Chunking for RAG ingestion

`chunk` splits text into embedding-sized pieces while keeping exact byte
offsets into the source — `input[c.Start:c.End] == c.Text` always holds,
overlaps included — so every chunk stays citable and highlightable after
retrieval. Splitting is hierarchical, tuned for Thai prose (no sentence-final
punctuation; novels put one paragraph per line): paragraphs first (line
breaks), oversized paragraphs by sentence with whole sentences packed
greedily, and only a single sentence that alone exceeds the budget is
hard-cut at rune boundaries — never immediately before a Thai combining mark,
so a mark is never split from its base.

```go
import "github.com/sukitss/thai-nlp-go/chunk"

chunks, err := chunk.Split(doc, chunk.Options{
    MaxUnits:     512,
    OverlapUnits: 64,          // whole-sentence overlap, contiguous source text
    Measure:      countTokens, // your unit, e.g. an LLM token counter; nil = runes
    Sentences:    crf.Split,   // optional: higher-quality Thai boundaries
})
```

Chunk sizes are measured in units *you* define. `Measure` may be an expensive
LLM tokenizer, so the algorithm is built around calling it sparingly: once per
paragraph, once per sentence, and O(log runes) times per hard-cut piece
(binary search instead of re-measuring every prefix). Overlap is whole
previous sentences only — never partial — so overlapped `[Start,End)` ranges
are real contiguous source text. `chunk` deliberately does not import
`sentence/crf` (that would embed the ~2 MB model for everyone); pass
`crf.Split` yourself, and every `Sentences` result is validated with fallback
to the built-in whitespace splitter on any mismatch. The full contract —
whitespace rule, overlap accounting, the unsplittable-piece exception, Measure
additivity — is specified in the package documentation and asserted by tests.

## Vocabulary for sparse retrieval

`vocab` maps terms to dense sequential ids (0, 1, 2, …) and tracks document
frequency — the term-dictionary mechanism classic lexical engines
(Lucene/Elasticsearch-style term dictionaries) use behind BM25 and
sparse-vector scoring. The reason it exists instead of the tempting shortcut,
hashing terms to dimensions: a 32-bit hash over an unbounded term space
collides, and a collision makes a query for one term match documents
containing an unrelated term — silent score corruption you can't detect from
the vectors. Sequential assignment is collision-free by construction, and
`Len()` is the exact dimension count.

```go
b := vocab.NewBuilder()
for _, doc := range corpus {
    b.AddDoc(a.Terms(doc)) // DF counted once per distinct term per document
}
b.IDF("ภาษาไทย")  // BM25 idf: ln(1 + (N-df+0.5)/(df+0.5))
b.Save(w)         // versioned binary snapshot, deterministic output

v, _ := vocab.Load(r) // immutable, safe for concurrent readers
id, ok := v.ID("ภาษาไทย")
b2 := v.Extend()      // resume building — existing ids never change
```

`Load` validates the whole snapshot (magic, version, counts vs size, DF
bounds, duplicate terms, trailing bytes) and returns an error on corrupt input
instead of panicking. Where the snapshot lives and how it synchronizes across
processes is deliberately your policy — the package is the mechanism.

## Multilingual routing (manual)

Real corpora mix scripts (Thai + English + Chinese + Korean …). `script.SplitByScript`
segments text into runs by writing system in one cheap pass, so you route each run
to the right tokenizer (Thai → this library, CJK → a CJK tokenizer, Latin →
whitespace) instead of forcing one tokenizer over everything:

```go
import "github.com/sukitss/thai-nlp-go/script"

for _, r := range script.SplitByScript("ตัวอย่างenglishwording你好") {
    switch r.Script {
    case script.Thai:  // r.Text -> tokenize.SegmentNoWS
    case script.Latin: // r.Text -> whitespace
    case script.Han:   // r.Text -> a CJK tokenizer
    }
}
```

It is script itemization (Unicode UAX #24), not language detection. Decided by
direct Unicode range checks — **no tables to load, no init cost** (~µs, stateless,
concurrency-safe). Common characters (spaces/punctuation/digits) attach to their
neighbor so runs don't fragment.

Each `Run` carries `Start`, its byte offset in the input
(`input[r.Start:r.Start+len(r.Text)] == r.Text` for valid UTF-8), so per-run
token offsets compose into whole-document positions. 0.x note: the added field
breaks positional composite literals `Run{script, text}` — use field names.

Then route each run to its tokenizer — e.g. Chinese runs to [`cjk`](cjk):

```go
import "github.com/sukitss/thai-nlp-go/cjk"

cjk.Cut("我爱自然语言处理") // ["我" "爱" "自然语言" "处理"]
```

For higher segmentation quality, `cjk.CutDP` / `jp.CutDP` resolve ambiguous runs
with a frequency-weighted DAG + dynamic-programming maximum-probability path
(jieba/MeCab-style, still non-neural) instead of greedy longest-match — e.g.
`北京大学生` → `北京|大学生` not `北京大学|生`. On a small bundled jieba-reference
set (18 sentences, asserted in CI) this lifts recall from 86% (greedy `Cut`) to
95% (`CutDP`), at ~1.5× the time. The embedded
dictionaries carry per-word weights (jieba frequency for Chinese, SudachiDict
cost for Japanese, TNC frequency for Thai).

`cjk` does Chinese word segmentation by forward maximal-matching over an embedded
dictionary (the jieba word list, MIT), with single-character fallback for OOV —
enough for BM25/keyword indexing, where segmentation accuracy has only a minor
effect on retrieval. It reuses the flat-mmap trie, so it loads in **under a millisecond
with ~no RAM** (including a full structural validation pass over the trie image) (vs ~1.5s / ~200MB for eager in-RAM Go segmenters — measured once
against gse on our dev machine; our load-time benchmark is checked in, the
comparison is not run in CI) and is pure Go,
no CGo, no neural model. Japanese runs go to `jp.Cut` (same approach, SudachiDict
small, Apache-2.0); Latin runs to `en.Cut` (no dictionary, no load); Korean runs
to `kr.Cut` (eojeol split plus multi-syllable particle stemming — dictionary-free;
full morphological analysis is on the roadmap).

You choose when to spend the dictionary's memory and how much. `cjk.EmbeddedSize()`
reports the cost up front; `cjk.Default()`/`Cut` load it lazily and keep it
resident; `cjk.Load()` gives you an instance you own (drop it to free the RAM);
`cjk.Open(path)` memory-maps an external dictionary file for near-zero resident
memory. Importing a language package is itself the first switch — code you don't
import adds nothing to your binary.

### CLI

```sh
go install github.com/sukitss/thai-nlp-go/cmd/thainlp@latest

echo "ฉันรักภาษาไทย" | thainlp                          # tokenize stdin → stdout
thainlp -overlay อาริน,เวธกา < story.txt                # with a session overlay
thainlp build -dict dict/data/words_th.txt -out my.fdt  # (re)build a flat trie
```

## Performance (tokenizer)

~1 MB mixed corpus, 48-core (`make bench`):

| Benchmark | Throughput | Notes |
| --- | --- | --- |
| Serial | ~12 MB/s | `[]string` output |
| Serial (bytes) | ~13 MB/s | `[]byte` output, fewer allocations |
| Parallel | ~53 MB/s | scales across cores (dict is shared) |
| Session | ~12 MB/s | overlay overhead negligible |

Dictionary load is a one-time ~1 ms parse+validate of the embedded image (mmap-backed files load in ~0.2 ms), shared process-wide.

Throughput numbers here (and in [docs/TEST-REPORT.md](docs/TEST-REPORT.md)) are
`make report` snapshots from the dev machine — tracked for regressions between
snapshots, not asserted in CI.

## Development

```sh
make test   # golden vs PyThaiNLP + 200k fuzz + unit tests (all packages)
make bench  # benchmarks
make dict   # rebuild dict/data/words_th.fdt
make help   # all targets
```

## Credits & license

A Go port of **[PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)** — all credit
for the tokenization algorithm and dictionary goes to the PyThaiNLP project and
its contributors. Offered back to the community in the same spirit.

- **Code:** Apache-2.0 (same as PyThaiNLP) — see [`LICENSE`](LICENSE) / [`NOTICE`](NOTICE).
- **Dictionary** (`dict/data/words_th.txt`): PyThaiNLP corpus, **CC0-1.0** (public domain).

Both are free for commercial use; please keep the `NOTICE` attribution.
