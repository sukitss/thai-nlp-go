// Package jp provides fast, low-memory Japanese word segmentation for RAG
// indexing (BM25/keyword), in pure Go with no CGo and no neural model.
//
// It uses forward maximal matching over an embedded dictionary (a flat,
// memory-mapped trie — the same structure as the Thai and Chinese tokenizers),
// with a single-character fallback for out-of-vocabulary runs. Japanese mixes
// kanji, hiragana and katakana with no spaces; plain longest-match is weaker
// than a MeCab-style connection-cost model but is a well-established, effective
// unit for information retrieval, where segmentation accuracy has only a minor
// effect on results — so no Viterbi/HMM is needed at index time.
//
// The dictionary (SudachiDict small, Apache-2.0) compiles to a ~9MB flat trie
// that mmaps in microseconds with ~no resident RAM. Non-Japanese characters
// (Latin, digits, …) are emitted as their own maximal runs. See NOTICE.
package jp

import (
	_ "embed"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/dict"
)

//go:embed data/ja.fdt
var jaFDT []byte

// Segmenter segments Japanese text. Create one with New (custom dictionary),
// Load/Open (explicit resource control), or use the package-level Cut. Safe for
// concurrent use (its dictionary is read-only).
type Segmenter struct{ d dict.Prefixer }

var (
	once      sync.Once
	shared    *Segmenter
	sharedErr error
)

func load() (*Segmenter, error) {
	once.Do(func() {
		ft, err := dict.FromBytes(jaFDT)
		if err != nil {
			sharedErr = err
			return
		}
		shared = &Segmenter{d: ft}
	})
	return shared, sharedErr
}

// New returns a Segmenter over a custom dictionary.
func New(d dict.Prefixer) *Segmenter { return &Segmenter{d: d} }

// EmbeddedSize reports the embedded dictionary's byte size — the approximate
// resident memory Default/Cut will allocate. Check it before loading.
func EmbeddedSize() int { return len(jaFDT) }

// Load parses the embedded dictionary into a NEW Segmenter you own (drop the
// reference to free the memory); it does not populate the shared instance.
func Load() (*Segmenter, error) {
	ft, err := dict.FromBytes(jaFDT)
	if err != nil {
		return nil, err
	}
	return &Segmenter{d: ft}, nil
}

// Open memory-maps a prebuilt flat-trie dictionary file — near-zero resident
// memory (shared via the OS page cache).
func Open(path string) (*Segmenter, error) {
	ft, err := dict.OpenFlat(path)
	if err != nil {
		return nil, err
	}
	return &Segmenter{d: ft}, nil
}

// Default returns a process-wide shared Segmenter backed by the embedded
// dictionary, loaded at most once (lazy); its memory stays resident. Use
// Load/Open to control or release it.
func Default() (*Segmenter, error) { return load() }

// Cut segments Japanese text with the shared dictionary (loading on first use).
// Panics only if the embedded dictionary fails to load.
func Cut(text string) []string {
	s, err := load()
	if err != nil {
		panic("jp: " + err.Error())
	}
	return s.Cut(text)
}

// Cut segments text by forward maximal matching, with single-rune fallback for
// OOV Japanese and maximal runs for non-Japanese (Latin/digit/…). Whitespace
// separates tokens and is dropped.
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
		case isJapanese(r):
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
			j := i + 1
			for j < n && !isJapanese(rs[j]) && !unicode.IsSpace(rs[j]) && sameClass(r, rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
		}
	}
	return out
}

// AppendBytes appends the segmented tokens of text (joined by sep) to dst and
// returns the extended slice — no per-token string allocation. Reuse dst across
// a corpus (dst[:0]) for zero steady-state allocation when building an index.
func (s *Segmenter) AppendBytes(dst []byte, text string, sep byte) []byte {
	rs := []rune(text)
	n := len(rs)
	buf := make([]int, 0, 8)
	first := true
	emit := func(a, b int) {
		if !first {
			dst = append(dst, sep)
		}
		first = false
		for k := a; k < b; k++ {
			dst = utf8.AppendRune(dst, rs[k])
		}
	}
	for i := 0; i < n; {
		r := rs[i]
		switch {
		case unicode.IsSpace(r):
			i++
		case isJapanese(r):
			buf = s.d.PrefixLens(rs, i, buf)
			best := 1
			for _, L := range buf {
				if L > best {
					best = L
				}
			}
			emit(i, i+best)
			i += best
		default:
			j := i + 1
			for j < n && !isJapanese(rs[j]) && !unicode.IsSpace(rs[j]) && sameClass(r, rs[j]) {
				j++
			}
			emit(i, j)
			i = j
		}
	}
	return dst
}

// isJapanese reports kanji, hiragana, katakana (incl. half-width katakana) and
// the iteration/長音 marks — the runs the dictionary segments.
func isJapanese(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF: // CJK unified (kanji)
		return true
	case r >= 0x3400 && r <= 0x4DBF: // CJK ext A
		return true
	case r >= 0x3040 && r <= 0x309F: // hiragana
		return true
	case r >= 0x30A0 && r <= 0x30FF: // katakana
		return true
	case r >= 0xFF66 && r <= 0xFF9D: // half-width katakana
		return true
	case r == 0x3005 || r == 0x30FC: // 々 (kanji iteration), ー (chōonpu)
		return true
	}
	return false
}

func sameClass(a, b rune) bool {
	if unicode.IsLetter(a) && unicode.IsLetter(b) {
		return true
	}
	if unicode.IsDigit(a) && unicode.IsDigit(b) {
		return true
	}
	return false
}
