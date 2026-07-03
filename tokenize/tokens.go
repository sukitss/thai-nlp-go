// Package tokenize provides Thai word tokenization (newmm maximal matching,
// ported from PyThaiNLP) — see segmenter.go for the algorithm.
//
// The Tokens/AppendTokens variants also report each token's position: offsets
// are BYTE offsets into the exact string passed to that call (End exclusive).
// No normalization happens inside these functions — if you normalize first,
// offsets point into the normalized string you passed.
package tokenize

import (
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/token"
)

// Tokens returns the same segmentation as Segment (whitespace tokens kept),
// with each token's byte offsets in text. Token.Text is the raw input slice:
// text[t.Start:t.End] == t.Text always holds, tokens are contiguous and cover
// the whole input. For valid UTF-8 input Tokens(text)[i].Text ==
// Segment(text)[i] exactly; on invalid UTF-8, Segment substitutes U+FFFD while
// Tokens keeps the original bytes (the offsets stay exact either way).
func (s *Segmenter) Tokens(text string) []token.Token {
	return s.AppendTokens(nil, text)
}

// AppendTokens appends Tokens(text) to dst and returns the extended slice,
// reusing dst's capacity across calls (dst[:0]) for zero steady-state
// allocation of the result slice. Offsets are relative to text, not to any
// earlier call's input.
func (s *Segmenter) AppendTokens(dst []token.Token, text string) []token.Token {
	if text == "" {
		return dst
	}
	rs := []rune(text)
	spans := s.onecut(rs)
	// onecut spans are contiguous and cover [0,len(rs)), so a single running
	// byte index converts rune spans to byte offsets in one pass (no rescans).
	b := 0
	for _, sp := range spans {
		start := b
		for k := sp.s; k < sp.e; k++ {
			_, sz := utf8.DecodeRuneInString(text[b:])
			b += sz
		}
		dst = append(dst, token.Token{Text: text[start:b], Start: start, End: b})
	}
	return dst
}

// TokensNoWS returns the same segmentation as SegmentNoWS (pure-space tokens
// dropped, surrounding ASCII spaces trimmed), with each token's byte offsets
// in text. The same Text/offset guarantees as Tokens apply; tokens are in
// order but not contiguous (dropped whitespace leaves gaps).
func (s *Segmenter) TokensNoWS(text string) []token.Token {
	return s.AppendTokensNoWS(nil, text)
}

// AppendTokensNoWS appends TokensNoWS(text) to dst and returns the extended
// slice, reusing dst's capacity across calls.
func (s *Segmenter) AppendTokensNoWS(dst []token.Token, text string) []token.Token {
	if text == "" {
		return dst
	}
	rs := []rune(text)
	spans := s.onecut(rs)
	b := 0
	for _, sp := range spans {
		start := b
		for k := sp.s; k < sp.e; k++ {
			_, sz := utf8.DecodeRuneInString(text[b:])
			b += sz
		}
		a, e := trimSpaces(rs, sp.s, sp.e)
		if a < e {
			// trimmed runes are ASCII ' ' (1 byte each), so offsets shift by count
			ts := start + (a - sp.s)
			te := b - (sp.e - e)
			dst = append(dst, token.Token{Text: text[ts:te], Start: ts, End: te})
		}
	}
	return dst
}
