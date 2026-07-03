// Package script splits mixed-language text into runs by writing system, so a
// pipeline can route each run to the right tokenizer (Thai → thai-nlp-go, CJK →
// a CJK tokenizer, Latin → whitespace, …). It is the cheap pre-step before word
// segmentation for multilingual corpora (documents mixing th/en/cn/kr/jp).
//
// It is fast and dependency-free by design: script is decided by direct Unicode
// range checks (plus stdlib unicode.IsLetter for "some other script"), so there
// are NO large tables to load and no init cost. SplitByScript is a pure function
// (stateless, safe for concurrent use) and allocates only its result.
//
// Note: this is script itemization (Unicode UAX #24), not language detection —
// "Han" covers Chinese and Japanese kanji, "Latin" covers many languages. For
// tokenizer routing that is usually enough; add language detection if you must
// tell e.g. Chinese from Japanese.
package script

import (
	"strings"
	"unicode"
)

// Script is a coarse writing-system category for tokenizer routing.
type Script uint8

const (
	Other  Script = iota // a letter of some other script (Arabic, Cyrillic, …)
	Common               // punctuation/space/digit/symbol/emoji — attaches to a neighbor
	Thai
	Latin
	Han    // CJK ideographs (Chinese, Japanese kanji)
	Hangul // Korean
	Kana   // Japanese hiragana/katakana
)

func (s Script) String() string {
	switch s {
	case Thai:
		return "Thai"
	case Latin:
		return "Latin"
	case Han:
		return "Han"
	case Hangul:
		return "Hangul"
	case Kana:
		return "Kana"
	case Common:
		return "Common"
	default:
		return "Other"
	}
}

// Run is a maximal slice of text in one script.
type Run struct {
	Script Script
	Text   string
}

// ScriptOf returns the script category of a single rune.
func ScriptOf(r rune) Script {
	switch {
	case r >= 0x0E00 && r <= 0x0E7F:
		return Thai
	case (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= 0x00C0 && r <= 0x024F):
		return Latin // Basic Latin letters + Latin-1 Supplement/Extended
	case (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) || (r >= 0xF900 && r <= 0xFAFF):
		return Han
	case (r >= 0xAC00 && r <= 0xD7A3) || (r >= 0x1100 && r <= 0x11FF) || (r >= 0x3130 && r <= 0x318F):
		return Hangul
	case (r >= 0x3040 && r <= 0x309F) || (r >= 0x30A0 && r <= 0x30FF):
		return Kana
	case unicode.IsLetter(r):
		return Other // a letter of another script — gets its own run
	default:
		return Common // space/punct/digit/symbol/emoji — attaches to a neighbor
	}
}

// SplitByScript splits text into runs by script. Common characters (spaces,
// punctuation, digits, symbols) attach to the surrounding run rather than
// breaking it, so "abc 123" stays one Latin run and a leading/trailing space
// joins its neighbor. Returns nil for empty input.
func SplitByScript(text string) []Run {
	if text == "" {
		return nil
	}
	var runs []Run
	var b strings.Builder
	cur := Common // "no script decided yet" — the first real script claims the run
	flush := func() {
		if b.Len() > 0 {
			s := cur
			if s == Common {
				s = Other // a run of only common chars
			}
			runs = append(runs, Run{Script: s, Text: b.String()})
			b.Reset()
		}
	}
	for _, r := range text {
		s := ScriptOf(r)
		switch {
		case s == Common:
			b.WriteRune(r) // stick with the current run
		case cur == Common: // first real script in this run
			cur = s
			b.WriteRune(r)
		case s == cur:
			b.WriteRune(r)
		default: // script change → new run
			flush()
			cur = s
			b.WriteRune(r)
		}
	}
	flush()
	return runs
}
