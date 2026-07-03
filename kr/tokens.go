package kr

import (
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/token"
)

// Tokens is Cut with byte offsets: exactly Cut's tokens in order (eojeol, then
// its particle-stripped stem when one exists, de-duplicated within the result),
// each with its half-open byte range in text.
//
// Offset semantics for stems: stripJosa removes the particle from the END of
// the eojeol (a byte-suffix, via strings.HasSuffix), so the stem is always a
// byte-PREFIX of its surface eojeol. A stem token therefore covers exactly its
// own bytes — Start is the eojeol's start and End = Start + len(stem) — and
// text[t.Start:t.End] == t.Text holds for every token, stems included (Token
// .Text is always a raw slice of text, even on invalid UTF-8). When a token
// string occurs more than once (de-dup), the FIRST occurrence's offsets are
// kept, matching Cut's ordering. Offsets are relative to the string passed to
// THIS call.
func Tokens(text string) []token.Token {
	var (
		out  []token.Token
		seen map[string]struct{}
	)
	add := func(tok string, start, end int) {
		if tok == "" {
			return
		}
		if _, ok := seen[tok]; ok {
			return
		}
		if seen == nil {
			seen = make(map[string]struct{}, 8)
		}
		seen[tok] = struct{}{}
		out = append(out, token.Token{Text: tok, Start: start, End: end})
	}
	// fields by Unicode whitespace, tracked with byte offsets — the offset-
	// aware equivalent of strings.FieldsFunc(text, unicode.IsSpace), which
	// also returns raw subslices of text.
	for i := 0; i < len(text); {
		r, sz := utf8.DecodeRuneInString(text[i:])
		if unicode.IsSpace(r) {
			i += sz
			continue
		}
		j := i + sz
		for j < len(text) {
			r2, sz2 := utf8.DecodeRuneInString(text[j:])
			if unicode.IsSpace(r2) {
				break
			}
			j += sz2
		}
		f := text[i:j]
		add(f, i, j)
		if stem := stripJosa(f); stem != "" && stem != f {
			add(stem, i, i+len(stem)) // stem is a byte-prefix of the eojeol
		}
		i = j
	}
	return out
}
