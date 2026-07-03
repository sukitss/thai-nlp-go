// Package en provides light word tokenization for English (and other
// space-delimited Latin-script text), for RAG indexing. It needs no dictionary
// and loads nothing — pure Unicode rules — so it costs no memory and is safe for
// concurrent use.
//
// It is the Latin-run tokenizer in the multilingual routing story: split a
// mixed document with the script package, then send Latin runs here.
package en

import (
	"strings"
	"unicode"
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

func isWord(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }
