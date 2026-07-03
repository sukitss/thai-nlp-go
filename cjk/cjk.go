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
//
// Tokens/TokensDP also report each token's byte offsets, relative to the exact
// string passed to that call — no normalization happens inside these
// functions, so if you normalize first, offsets point into the string you
// passed.
package cjk

import (
	_ "embed"
	"sync"
	"unicode"
	"unicode/utf8"

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

// EmbeddedSize reports the byte size of the embedded Chinese dictionary — the
// approximate resident memory that Default (or the first Cut) will allocate.
// Call it to decide, before loading, whether to pay that cost.
func EmbeddedSize() int { return len(zhFDT) }

// Load parses the embedded dictionary into a NEW Segmenter you own — an explicit
// alternative to Default/Cut for when you want to control exactly WHEN the ~
// EmbeddedSize() bytes are allocated (and to release them by dropping the
// reference). Unlike Default it does not populate the process-wide shared
// instance, so each call allocates its own copy.
func Load() (*Segmenter, error) {
	ft, err := dict.FromBytes(zhFDT)
	if err != nil {
		return nil, err
	}
	return &Segmenter{d: ft}, nil
}

// Open memory-maps a prebuilt flat-trie dictionary file (dict.OpenFlat) into a
// Segmenter — near-zero resident memory (shared via the OS page cache), for when
// you ship the dictionary as a file rather than paying for the embedded copy.
func Open(path string) (*Segmenter, error) {
	ft, err := dict.OpenFlat(path)
	if err != nil {
		return nil, err
	}
	return &Segmenter{d: ft}, nil
}

// Default returns a process-wide shared Segmenter backed by the embedded
// dictionary, loading it at most once (lazy). Convenient, but the ~
// EmbeddedSize() bytes it allocates stay resident for the process lifetime; use
// Load or Open if you want to control or release that memory.
func Default() (*Segmenter, error) { return load() }

// Cut segments Chinese text with the shared Default dictionary (loading it on
// first use). Convenience wrapper; it panics only if the embedded dictionary
// fails to load (a build/data error). Use Default/Load/Open to handle the error
// or control memory.
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
		case isHan(r):
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
			for j < n && !isHan(rs[j]) && !unicode.IsSpace(rs[j]) && sameClass(r, rs[j]) {
				j++
			}
			emit(i, j)
			i = j
		}
	}
	return dst
}

// CutDP segments text like Cut but resolves ambiguous Han runs with a DAG +
// dynamic-programming maximum-probability path over dictionary word weights
// (jieba-style, non-neural), instead of greedy longest-match. This fixes cases
// where greedy matching makes a locally-long but globally-worse choice. It
// requires a weighted dictionary (Default is weighted); with an unweighted dict
// it falls back to fewest-tokens. Non-Han runs behave exactly like Cut.
func (s *Segmenter) CutDP(text string) []string {
	rs := []rune(text)
	n := len(rs)
	out := make([]string, 0, n/2+1)
	for i := 0; i < n; {
		r := rs[i]
		switch {
		case unicode.IsSpace(r):
			i++
		case isHan(r):
			j := i
			for j < n && isHan(rs[j]) {
				j++
			}
			out = s.dpRun(rs, i, j, out)
			i = j
		default:
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

// dpRun runs the max-probability DP over rs[a:b] (a Han run) and appends the
// best segmentation's tokens to out. Cost of an unknown single char is a large
// negative weight so real words are always preferred.
func (s *Segmenter) dpRun(rs []rune, a, b int, out []string) []string {
	starts, ok := s.dpStarts(rs, a, b)
	if !ok { // unweighted custom dict: fall back to longest-match on this run
		return append(out, s.Cut(string(rs[a:b]))...)
	}
	for t, st := range starts {
		en := b - a
		if t+1 < len(starts) {
			en = starts[t+1]
		}
		out = append(out, string(rs[a+st:a+en]))
	}
	return out
}

// dpStarts computes the max-probability segmentation of the Han run rs[a:b]
// and returns each token's start rune index relative to a, ascending (token t
// spans starts[t]..starts[t+1], the last ending at b-a). ok is false when the
// dictionary is unweighted — callers fall back to greedy longest-match.
func (s *Segmenter) dpStarts(rs []rune, a, b int) (starts []int, ok bool) {
	ft, ok := s.d.(interface {
		PrefixWeights(text []rune, start int, outLen, outW []int32) ([]int32, []int32)
	})
	if !ok {
		return nil, false
	}
	m := b - a
	const negInf = int64(-1) << 60
	const unkPenalty = -100000 // single OOV char: heavily penalised vs any word
	best := make([]int64, m+1)
	prev := make([]int32, m+1) // token start index (relative to a) chosen at each pos
	for k := 1; k <= m; k++ {
		best[k] = negInf
	}
	var lens, ws []int32
	for i := 0; i < m; i++ {
		if best[i] == negInf && i != 0 {
			continue
		}
		// single-char fallback edge (always available)
		if v := best[i] + unkPenalty; v > best[i+1] {
			best[i+1] = v
			prev[i+1] = int32(i)
		}
		lens, ws = ft.PrefixWeights(rs, a+i, lens, ws)
		for t := range lens {
			end := i + int(lens[t])
			if end > m {
				continue // dict word crosses the run end (e.g. into digits): not a candidate
			}
			if v := best[i] + int64(ws[t]); v > best[end] {
				best[end] = v
				prev[end] = int32(i)
			}
		}
	}
	// backtrack (yields starts descending), then reverse to ascending
	starts = make([]int, 0, 8)
	for k := m; k > 0; {
		p := int(prev[k])
		starts = append(starts, p)
		k = p
	}
	for i, j := 0, len(starts)-1; i < j; i, j = i+1, j-1 {
		starts[i], starts[j] = starts[j], starts[i]
	}
	return starts, true
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

// CutDP segments Chinese text with the shared dictionary using the DAG + DP
// maximum-probability path (higher quality than greedy Cut on ambiguous runs).
func CutDP(text string) []string {
	s, err := load()
	if err != nil {
		panic("cjk: " + err.Error())
	}
	return s.CutDP(text)
}
