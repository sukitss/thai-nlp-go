// Package multi is the one-call multilingual entry point: give it mixed-language
// text (a translated novel mixing Thai, Chinese, Japanese, Korean and English,
// say) and it detects each run's language, routes it to the right tokenizer, and
// returns index-ready tokens — so callers don't reimplement script routing.
//
//	toks := multi.Segment("ผมอ่าน三国志と日本語")   // []string across all languages
//	buf   = multi.AppendBytes(buf[:0], text, ' ')   // index output into a reused buffer
//
// Routing: Thai → tokenize (newmm), Chinese → cjk, Japanese → jp, Korean → kr,
// Latin → en. CJK is grouped so kanji+kana stay together; a CJK run containing
// kana is treated as Japanese, otherwise as Chinese (set DefaultHan to change
// the pure-Han default). Text is Unicode-normalized (normalize.Normalize) first
// so the index is consistent.
//
// Importing this package pulls in all language dictionaries (Thai + Chinese +
// Japanese ≈ 18 MB embedded, mmap-loaded lazily on first use per language). If
// you need only one or two languages, import those packages directly instead.
package multi

import (
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/cjk"
	"github.com/sukitss/thai-nlp-go/en"
	"github.com/sukitss/thai-nlp-go/jp"
	"github.com/sukitss/thai-nlp-go/kr"
	"github.com/sukitss/thai-nlp-go/normalize"
	"github.com/sukitss/thai-nlp-go/tokenize"
)

var (
	thOnce sync.Once
	thSeg  *tokenize.Segmenter
)

func thai_() *tokenize.Segmenter {
	thOnce.Do(func() { thSeg, _ = tokenize.NewDefault() })
	return thSeg
}

// lang classes for routing.
type lang uint8

const (
	other lang = iota
	thai
	cjkHan // Han (+ kana); disambiguated to Chinese/Japanese by kana presence
	hangul
	latin
)

// DefaultHan selects the tokenizer for a CJK run that has NO kana (pure Han,
// ambiguous between Chinese and Japanese). "cn" (default) or "jp".
var DefaultHan = "cn"

// Segment normalizes text, splits it into language runs, tokenizes each with the
// right tokenizer, and returns all tokens in order. Returns nil for empty input.
func Segment(text string) []string {
	text = normalize.Normalize(text)
	if text == "" {
		return nil
	}
	var out []string
	forEachRun(text, func(l lang, run string, hasKana bool) {
		out = append(out, tokenizeRun(l, run, hasKana)...)
	})
	return out
}

// AppendBytes is Segment writing tokens straight into dst (joined by sep) — the
// indexing path. Each run's tokenizer still allocates its tokens as []string
// internally; what this saves over Segment is the final result slice and the
// join copy, by writing into a reusable buffer. Reuse dst (dst[:0]).
func AppendBytes(dst []byte, text string, sep byte) []byte {
	text = normalize.Normalize(text)
	if text == "" {
		return dst
	}
	first := true
	forEachRun(text, func(l lang, run string, hasKana bool) {
		for _, tok := range tokenizeRun(l, run, hasKana) {
			if !first {
				dst = append(dst, sep)
			}
			first = false
			for _, r := range tok {
				dst = utf8.AppendRune(dst, r)
			}
		}
	})
	return dst
}

func tokenizeRun(l lang, run string, hasKana bool) []string {
	switch l {
	case thai:
		return thai_().SegmentNoWS(run)
	case cjkHan:
		if hasKana || DefaultHan == "jp" {
			return jp.Cut(run)
		}
		return cjk.Cut(run)
	case hangul:
		return kr.Cut(run)
	case latin:
		return en.Cut(run)
	default:
		return en.Cut(run) // digits/symbols/other letters: whitespace-ish split
	}
}

// forEachRun groups text into maximal same-language runs (Common characters —
// space/punct/digit — attach to the current run) and calls fn per run.
func forEachRun(text string, fn func(l lang, run string, hasKana bool)) {
	rs := []rune(text)
	n := len(rs)
	i := 0
	for i < n {
		if isCommon(rs[i]) {
			i++
			continue
		}
		l := classify(rs[i])
		j := i
		hasKana := false
		for j < n {
			r := rs[j]
			if isCommon(r) {
				j++
				continue
			}
			if isKana(r) && l == cjkHan {
				hasKana = true
				j++
				continue
			}
			if classify(r) != l {
				break
			}
			j++
		}
		fn(l, string(rs[i:j]), hasKana)
		i = j
	}
}

func isCommon(r rune) bool {
	return !unicode.IsLetter(r) || unicode.Is(unicode.Common, r)
}

func isKana(r rune) bool {
	return (r >= 0x3040 && r <= 0x30FF) || (r >= 0xFF66 && r <= 0xFF9D)
}

func classify(r rune) lang {
	switch {
	case r >= 0x0E00 && r <= 0x0E7F:
		return thai
	case (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x3040 && r <= 0x30FF) || (r >= 0xFF66 && r <= 0xFF9D) ||
		r == 0x3005 || r == 0x30FC: // Han + kana → one CJK class
		return cjkHan
	case (r >= 0xAC00 && r <= 0xD7A3) || (r >= 0x1100 && r <= 0x11FF) || (r >= 0x3130 && r <= 0x318F):
		return hangul
	case (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= 0x00C0 && r <= 0x024F):
		return latin
	}
	return other
}
