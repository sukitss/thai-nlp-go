// Package kr provides light Korean word tokenization for RAG indexing, in pure
// Go with no dictionary and no load (so it costs no memory and is concurrency-
// safe).
//
// Korean writes with spaces between eojeol (word + attached particles), so
// whitespace already segments it — but a content word usually carries a
// grammatical particle (josa), e.g. 서울에서 = 서울 ("Seoul") + 에서 ("from").
// Splitting on whitespace alone would index 서울에서 as one token, so a query for
// 서울 would miss it. Cut therefore emits the eojeol AND, when it ends in a known
// MULTI-syllable particle, the particle-stripped stem — so a query matches
// either form.
//
// Only multi-syllable particles are stripped: they are unambiguously
// grammatical, so stripping never mangles a content word. Single-syllable
// particles (을/를/이/가/은/는/…) are intentionally left attached, because
// without a dictionary they cannot be told apart from a word's own final
// syllable (e.g. 사과 "apple" ends in 과; stripping would wrongly yield 사). Full
// particle analysis needs a morphological dictionary (mecab-ko) and is on the
// roadmap. This package stays dictionary-free, so it costs no memory and is
// concurrency-safe.
package kr

import (
	"sort"
	"strings"
	"unicode"
)

// josa is a closed set of common MULTI-syllable Korean particles, checked as
// eojeol suffixes (longest first). Multi-syllable only, by design (see package
// doc): stripping these never mangles a content word.
var josa = []string{
	"으로써", "으로서", "으로부터", "에서부터", "에게서", "이라고",
	"에서는", "에게는", "으로는", "이라는", "께서는",
	"에서", "에게", "께서", "한테", "으로", "이나", "이란", "이라", "라도", "든지",
	"처럼", "보다", "마다", "조차", "밖에", "부터", "까지", "마저", "대로", "이랑",
}

func init() { sort.Slice(josa, func(i, j int) bool { return len(josa[i]) > len(josa[j]) }) }

// Cut tokenizes Korean text. Each whitespace-separated eojeol is emitted; if it
// ends in a known particle and leaves a non-empty Hangul stem, the stem is
// emitted too. Non-Korean runs (Latin/digit) pass through as their own tokens.
// Duplicate tokens within the result are removed. Returns nil for empty input.
func Cut(text string) []string {
	fields := strings.FieldsFunc(text, unicode.IsSpace)
	if len(fields) == 0 {
		return nil
	}
	out := make([]string, 0, len(fields)+2)
	seen := make(map[string]struct{}, len(fields)+2)
	add := func(tok string) {
		if tok == "" {
			return
		}
		if _, ok := seen[tok]; ok {
			return
		}
		seen[tok] = struct{}{}
		out = append(out, tok)
	}
	for _, f := range fields {
		add(f)
		if stem := stripJosa(f); stem != "" && stem != f {
			add(stem)
		}
	}
	return out
}

// stripJosa returns f with a trailing particle removed, if f is Hangul and ends
// in a known particle leaving at least one Hangul syllable; otherwise "".
func stripJosa(f string) string {
	rs := []rune(f)
	if len(rs) < 2 || !isHangul(rs[len(rs)-1]) {
		return ""
	}
	for _, p := range josa {
		if len(p) < len(f) && strings.HasSuffix(f, p) {
			stem := f[:len(f)-len(p)]
			sr := []rune(stem)
			if len(sr) >= 1 && allHangul(sr) {
				return stem
			}
		}
	}
	return ""
}

func isHangul(r rune) bool {
	return (r >= 0xAC00 && r <= 0xD7A3) || (r >= 0x1100 && r <= 0x11FF) || (r >= 0x3130 && r <= 0x318F)
}

func allHangul(rs []rune) bool {
	for _, r := range rs {
		if !isHangul(r) {
			return false
		}
	}
	return true
}
