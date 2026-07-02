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
| [`normalize`](normalize) | Unicode/Thai text normalization | 🔲 stub |
| [`stopwords`](stopwords) | Thai/English stop-word filtering | 🔲 stub |
| [`sentence`](sentence) | Sentence boundary detection | 🔲 stub |
| [`translit`](translit) | Name-variant / transliteration matching | 🔲 stub |

Stubs define their intended API and are being hardened one package at a time.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and conventions.

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
