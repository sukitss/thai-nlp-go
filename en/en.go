// Package en provides light word tokenization for English (and other
// space-delimited Latin-script text), for RAG indexing. It needs no dictionary
// and loads nothing — pure Unicode rules — so it costs no memory and is safe for
// concurrent use.
//
// It is the Latin-run tokenizer in the multilingual routing story: split a
// mixed document with the script package, then send Latin runs here.
//
// Tokens also reports each token's byte offsets, relative to the exact string
// passed to that call — no normalization happens inside these functions, so if
// you normalize (or lowercase) first, offsets point into the string you passed.
package en

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/token"
)

// Cut splits text into word tokens: maximal runs of letters/digits (and word-
// internal apostrophes/hyphens between alphanumerics, so "don't" and "world-wide"
// stay whole). Whitespace and other punctuation are dropped. This is the common
// unigram indexing unit for English BM25/keyword search.
func Cut(text string) []string {
	rs := []rune(text)
	n := len(rs)
	out := make([]string, 0, len(text)/6+1)
	for i := 0; i < n; {
		if !isWord(rs[i]) {
			i++
			continue
		}
		j := i + 1
		for j < n {
			if isWord(rs[j]) {
				j++
				continue
			}
			// keep a single internal ' or - if flanked by word chars
			if (rs[j] == '\'' || rs[j] == '-') && j+1 < n && isWord(rs[j+1]) {
				j += 2
				continue
			}
			break
		}
		out = append(out, string(rs[i:j]))
		i = j
	}
	return out
}

// CutLower is Cut with each token lowercased — the usual case-folded form for a
// keyword index (so "Hello" and "hello" match).
func CutLower(text string) []string {
	toks := Cut(text)
	for i, t := range toks {
		toks[i] = strings.ToLower(t)
	}
	return toks
}

// Tokens is Cut with byte offsets: the same token boundaries, each with its
// half-open byte range in text. Token.Text is the raw input slice in ORIGINAL
// casing — CutLower's lowercasing is NOT applied here (lowercase yourself if
// you need the folded form; offsets still point at the original bytes).
// text[t.Start:t.End] == t.Text always holds, and Tokens(text)[i].Text ==
// Cut(text)[i] exactly (word tokens never contain invalid UTF-8, which is
// treated as a separator by both).
func Tokens(text string) []token.Token {
	out := make([]token.Token, 0, len(text)/6+1)
	for i := 0; i < len(text); {
		r, sz := utf8.DecodeRuneInString(text[i:])
		if !isWord(r) {
			i += sz
			continue
		}
		j := i + sz
		for j < len(text) {
			r2, sz2 := utf8.DecodeRuneInString(text[j:])
			if isWord(r2) {
				j += sz2
				continue
			}
			// keep a single internal ' or - if flanked by word chars
			if (r2 == '\'' || r2 == '-') && j+sz2 < len(text) {
				if r3, sz3 := utf8.DecodeRuneInString(text[j+sz2:]); isWord(r3) {
					j += sz2 + sz3
					continue
				}
			}
			break
		}
		out = append(out, token.Token{Text: text[i:j], Start: i, End: j})
		i = j
	}
	return out
}

func isWord(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }
