// Package dict is the shared dictionary layer for thai-nlp-go.
//
// It holds the trie types (pointer Trie and the memory-mappable FlatTrie) and,
// crucially, a single shared instance of the default Thai word list. Every
// component that needs the base dictionary (tokenizer, and future spell / word
// segmentation helpers) calls Default and receives the SAME memory-mapped trie —
// loaded once, shared read-only across the whole process. This is deliberate:
// loading a 62k-word trie per component (as naive wrappers around PyThaiNLP do)
// wastes memory and startup time. Keep it shared.
package dict

import (
	_ "embed"
	"sync"
)

// Prefixer reports, for text[start:], the rune-lengths of every dictionary word
// that is a prefix there, in ascending order. out is reset and reused. Trie,
// FlatTrie and any custom dictionary implement it.
type Prefixer interface {
	PrefixLens(text []rune, start int, out []int) []int
}

// Weighter is the weighted companion of Prefixer: PrefixWeights reports the
// same rune-lengths with each matched word's weight alongside (0 where the
// dictionary carries no weight), for DAG/maximum-probability segmentation.
// outLen and outW are reset, kept in sync and reused. Trie and FlatTrie
// implement it.
type Weighter interface {
	PrefixWeights(text []rune, start int, outLen, outW []int32) ([]int32, []int32)
}

// embeddedFDT is the flat-trie form of the PyThaiNLP words_th dictionary
// (62,102 words), built with BuildFlatFromTrie from data/words_th.txt.
// Source: https://github.com/PyThaiNLP/pythainlp (corpus/words_th.txt, CC0-1.0).
// Regenerate with `make dict` after editing data/words_th.txt.
//
//go:embed data/words_th.fdt
var embeddedFDT []byte

var (
	defaultOnce sync.Once
	defaultTrie *FlatTrie
	defaultErr  error
)

// Default returns the process-wide shared default dictionary (the embedded
// PyThaiNLP words_th list), loading it at most once. The returned *FlatTrie is
// read-only and safe for concurrent use; callers MUST NOT mutate or Close it.
//
// Sharing one instance is the whole point — do not build your own copy of the
// base dictionary when you can reuse this.
func Default() (*FlatTrie, error) {
	defaultOnce.Do(func() {
		defaultTrie, defaultErr = FromBytes(embeddedFDT)
	})
	return defaultTrie, defaultErr
}

// EmbeddedBytes returns the raw embedded flat-trie bytes (advanced use).
//
// This is the live go:embed slice, NOT a copy (copying ~4 MB per call would
// defeat the shared-dictionary design). Treat it as strictly read-only:
// mutating it corrupts the process-wide dictionary that Default serves.
func EmbeddedBytes() []byte { return embeddedFDT }
