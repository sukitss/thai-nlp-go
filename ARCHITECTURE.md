# Architecture

`thai-nlp-go` is a **mono-module**: one Go module, many small sub-packages.
Import only what you use.

```
thai-nlp-go/
├── dict/         shared dictionary layer  ← the heart of the perf/mem design
│   ├── trie.go        pointer Trie (build/overlay)
│   ├── flattrie.go    FlatTrie: mmap-able, zero-copy, near-instant load
│   ├── dict.go        Prefixer interface + Default() shared instance + embed
│   └── data/          words_th.txt (source) + words_th.fdt (compiled)
├── tokenize/     word segmentation (newmm port) + char n-gram
├── normalize/    (planned)  Unicode/Thai normalization
├── stopwords/    (planned)  stop-word filtering
├── sentence/     (planned)  sentence boundary detection
├── translit/     (planned)  name-variant / transliteration matching
└── cmd/thainlp/  CLI
```

## Why mono-module (not one repo per tool)

Driven by **performance and memory**:

- **Shared dictionary.** Several components use the same word list (tokenizer
  today; spell-check, word-based sentence splitting later). In separate repos
  each would embed and load its own trie — duplicate RAM, duplicate startup. In
  one module they all call `dict.Default()` and share a **single** memory-mapped
  instance.
- **No version drift.** Stages that must agree (e.g. normalization feeding a
  later hash) stay coherent when they ship as one versioned module rather than
  separately-versioned dependencies.
- **Still lean.** Go's `//go:embed` binds data to a package, and unused packages
  are not linked in. Importing only `normalize` does **not** pull in the
  tokenizer's 2.8 MB dictionary.

## The dictionary layer (`dict`)

- `FlatTrie` is a CSR serialization of the trie, stored as `words_th.fdt` and
  loaded by `mmap` + zero-copy cast — ~microseconds, and read-only so many
  goroutines (and, via the OS page cache, many processes) share one copy.
- `Default()` loads the embedded dictionary **once** (`sync.Once`) and hands the
  same `*FlatTrie` to every caller. This is the single most important rule in the
  module: never build your own copy of the base dictionary when you can share it.
- `Prefixer` is the one-method interface every dictionary satisfies, so custom or
  overlaid dictionaries drop in anywhere.

## Conventions for new packages

- Pure, deterministic functions where possible; `normalize` must be idempotent.
- Provide a `[]byte`/append-style path on hot code (avoid per-token string alloc).
- Ship a **golden test** against a reference implementation (PyThaiNLP) and a
  **benchmark** (`-benchmem`). See `tokenize/segment_test.go` and
  `tokenize/bench_test.go` as the template.
- If you must load a resource/model, load it once and share it (mirror `dict`).
