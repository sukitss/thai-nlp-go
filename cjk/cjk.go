// Package cjk provides fast, low-memory Chinese word segmentation for RAG
// indexing (BM25/keyword), in pure Go with no CGo and no neural model.
//
// It uses forward maximal matching over an embedded dictionary (a flat,
// memory-mapped trie — the same structure as the Thai tokenizer), with a
// single-character fallback for out-of-vocabulary runs. For information
// retrieval this is a well-established, effective indexing unit: studies show
// segmentation accuracy has only a minor, non-monotonic effect on retrieval, so
// a heavier probabilistic segmenter (jieba HMM / MeCab Viterbi) is not needed at
// index time. Non-Han characters (Latin, digits, …) are emitted as their own
// maximal runs.
//
// The dictionary loads in microseconds via mmap and adds no measurable RAM
// (vs. ~1.5s / ~200MB for eager in-RAM Go segmenters). The word list is the
// jieba dictionary (MIT); see NOTICE.
package cjk

import (
	_ "embed"
	"sync"
	"unicode"

	"github.com/sukitss/thai-nlp-go/dict"
)

//go:embed data/zh.fdt
var zhFDT []byte

// Segmenter segments Chinese text. Create one with New (custom dictionary) or
// use the package-level Cut, which lazily loads the shared embedded dictionary.
// A Segmenter is safe for concurrent use (its dictionary is read-only).
type Segmenter struct {
	d dict.Prefixer
}

var (
	once      sync.Once
	shared    *Segmenter
	sharedErr error
)

func load() (*Segmenter, error) {
	once.Do(func() {
		ft, err := dict.FromBytes(zhFDT)
		if err != nil {
			sharedErr = err
			return
		}
		shared = &Segmenter{d: ft}
	})
	return shared, sharedErr
}

// New returns a Segmenter over a custom dictionary (e.g. dict.LoadDict or a
// prebuilt dict.FlatTrie).
func New(d dict.Prefixer) *Segmenter { return &Segmenter{d: d} }

// Default returns a Segmenter backed by the shared embedded Chinese dictionary.
func Default() (*Segmenter, error) { return load() }

// Cut segments Chinese text with the shared dictionary. It panics only if the
// embedded dictionary fails to load (a build/data error, not a runtime input
// error); use Default if you want to handle that error.
func Cut(text string) []string {
	s, err := load()
	if err != nil {
		panic("cjk: " + err.Error())
	}
	return s.Cut(text)
}

// Cut segments text into tokens by forward maximal matching, with single-rune
// fallback for OOV Han and maximal runs for non-Han (Latin/digit/…) characters.
// Whitespace separates tokens and is dropped.
func (s *Segmenter) Cut(text string) []string {
	rs := []rune(text)
	n := len(rs)
	out := make([]string, 0, n/2+1)
	buf := make([]int, 0, 8)
	for i := 0; i < n; {
		r := rs[i]
		switch {
		case unicode.IsSpace(r):
			i++
		case isHan(r):
			// longest dictionary word starting at i (fall back to 1 rune)
			buf = s.d.PrefixLens(rs, i, buf)
			best := 1
			for _, L := range buf {
				if L > best {
					best = L
				}
			}
			out = append(out, string(rs[i:i+best]))
			i += best
		default:
			// maximal run of same-class non-Han, non-space characters
			j := i + 1
			for j < n && !isHan(rs[j]) && !unicode.IsSpace(rs[j]) && sameClass(r, rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
		}
	}
	return out
}

func isHan(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) || (r >= 0xF900 && r <= 0xFAFF)
}

// sameClass groups a non-Han run: letters together, digits together, everything
// else one rune at a time.
func sameClass(a, b rune) bool {
	if unicode.IsLetter(a) && unicode.IsLetter(b) {
		return true
	}
	if unicode.IsDigit(a) && unicode.IsDigit(b) {
		return true
	}
	return false
}
