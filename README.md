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
| [`tokenize`](tokenize) | Word segmentation (PyThaiNLP **newmm** port) + char **n-gram** | ✅ |
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
`translit.Key` is a MetaSound phonetic key (faithful port) for name-variant
matching.

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
