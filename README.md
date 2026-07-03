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
| [`tokenize`](tokenize) | Word segmentation (PyThaiNLP **newmm** port) + char **n-gram** | ✅ |
| [`cjk`](cjk) | Chinese word segmentation (dictionary maximal-matching) | ✅ |
| [`jp`](jp) | Japanese word segmentation (dictionary maximal-matching) | ✅ |
| [`en`](en) | Light English/Latin word tokenization (no dictionary) | ✅ |
| [`kr`](kr) | Light Korean tokenization (eojeol + particle stem, no dictionary) | ✅ |
| [`multi`](multi) | One-call multilingual tokenization (detect + route th/cn/jp/kr/en) | ✅ |
| [`normalize`](normalize) | Text normalization (PyThaiNLP-faithful) | ✅ |
| [`stopwords`](stopwords) | Thai/English stop-word filtering | ✅ |
| [`sentence`](sentence) | Whitespace sentence splitting (rule-based) | ✅ |
| [`translit`](translit) | Name-variant matching (MetaSound phonetic key) | ✅ |

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

Thai stop-word set faithful to PyThaiNLP `thai_stopwords()` (1,027 words), plus a
common English set. Matching is case-sensitive, so latin acronyms stay distinct
(`it` is a stop word, `IT` is not).

```go
import "github.com/sukitss/thai-nlp-go/stopwords"

sw := stopwords.Union(stopwords.Default(), stopwords.English())
sw.IsStopword("และ")                        // true
sw.Filter([]string{"ผม","และ","รัก","the"}) // ["ผม" "รัก"]

custom := stopwords.Union(stopwords.Default(), stopwords.New("อาริน", "เวธกา"))
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
to `CompleteSoundexSyllables` for multi) — all faithful ports, golden-verified —
plus `Levenshtein`/`Similarity` for fuzzy matching. Bucket candidates by a phonetic key, then rank with edit
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
for whitespace splitting** — in-domain training is the biggest quality lever.

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
buf = multi.AppendBytes(buf[:0], text, ' ')   // zero-allocation output for indexing
```

Thai → newmm, Chinese → cjk, Japanese → jp, Korean → kr, Latin → en. CJK runs
keep kanji+kana together and route to Japanese when kana is present, else Chinese
(`multi.DefaultHan` sets the pure-Han default). Importing `multi` embeds all
language dictionaries (loaded lazily per language on first use); import single
packages if you only need some.

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

Then route each run to its tokenizer — e.g. Chinese runs to [`cjk`](cjk):

```go
import "github.com/sukitss/thai-nlp-go/cjk"

cjk.Cut("我爱自然语言处理") // ["我" "爱" "自然语言" "处理"]
```

`cjk` does Chinese word segmentation by forward maximal-matching over an embedded
dictionary (the jieba word list, MIT), with single-character fallback for OOV —
enough for BM25/keyword indexing, where segmentation accuracy has only a minor
effect on retrieval. It reuses the flat-mmap trie, so it loads in **microseconds
with ~no RAM** (vs ~1.5s / ~200MB for eager in-RAM Go segmenters) and is pure Go,
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

Dictionary load is a one-time ~microsecond mmap, shared process-wide.

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
