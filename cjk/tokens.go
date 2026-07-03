package cjk

import (
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/token"
)

// Tokens is Cut with byte offsets: the same token boundaries (greedy
// longest-match; whitespace skipped), each with its half-open byte range in
// text. Token.Text is the raw input slice: text[t.Start:t.End] == t.Text
// always holds. For valid UTF-8 input Tokens(text)[i].Text == Cut(text)[i]
// exactly; on invalid UTF-8, Cut substitutes U+FFFD while Tokens keeps the
// original bytes (offsets stay exact either way). Offsets are relative to the
// string passed to THIS call.
func (s *Segmenter) Tokens(text string) []token.Token {
	rs := []rune(text)
	n := len(rs)
	out := make([]token.Token, 0, n/2+1)
	buf := make([]int, 0, 8)
	b := 0 // running byte index into text, in lockstep with rune index i
	for i := 0; i < n; {
		r := rs[i]
		switch {
		case unicode.IsSpace(r):
			b = byteAdvance(text, b, 1)
			i++
		case isHan(r):
			buf = s.d.PrefixLens(rs, i, buf)
			best := 1
			for _, L := range buf {
				if L > best {
					best = L
				}
			}
			e := byteAdvance(text, b, best)
			out = append(out, token.Token{Text: text[b:e], Start: b, End: e})
			b = e
			i += best
		default:
			j := i + 1
			for j < n && !isHan(rs[j]) && !unicode.IsSpace(rs[j]) && sameClass(r, rs[j]) {
				j++
			}
			e := byteAdvance(text, b, j-i)
			out = append(out, token.Token{Text: text[b:e], Start: b, End: e})
			b = e
			i = j
		}
	}
	return out
}

// TokensDP is CutDP with byte offsets: identical token boundaries (DAG + DP
// max-probability path over Han runs, greedy elsewhere), with the same
// Text/offset guarantees as Tokens.
func (s *Segmenter) TokensDP(text string) []token.Token {
	rs := []rune(text)
	n := len(rs)
	out := make([]token.Token, 0, n/2+1)
	b := 0
	for i := 0; i < n; {
		r := rs[i]
		switch {
		case unicode.IsSpace(r):
			b = byteAdvance(text, b, 1)
			i++
		case isHan(r):
			j := i
			for j < n && isHan(rs[j]) {
				j++
			}
			out, b = s.dpTokens(text, rs, i, j, b, out)
			i = j
		default:
			j := i + 1
			for j < n && !isHan(rs[j]) && !unicode.IsSpace(rs[j]) && sameClass(r, rs[j]) {
				j++
			}
			e := byteAdvance(text, b, j-i)
			out = append(out, token.Token{Text: text[b:e], Start: b, End: e})
			b = e
			i = j
		}
	}
	return out
}

// dpTokens emits the DP segmentation of the Han run rs[a:rb] as tokens with
// byte offsets, starting at byte offset b; with an unweighted dictionary it
// falls back to greedy longest-match confined to the run (the same boundaries
// as CutDP's fallback). Returns the extended slice and the byte offset after
// the run.
func (s *Segmenter) dpTokens(text string, rs []rune, a, rb, b int, out []token.Token) ([]token.Token, int) {
	starts, ok := s.dpStarts(rs, a, rb)
	if !ok {
		buf := make([]int, 0, 8)
		for i := a; i < rb; {
			buf = s.d.PrefixLens(rs, i, buf)
			best := 1
			for _, L := range buf {
				if L > best && i+L <= rb {
					best = L
				}
			}
			e := byteAdvance(text, b, best)
			out = append(out, token.Token{Text: text[b:e], Start: b, End: e})
			b = e
			i += best
		}
		return out, b
	}
	for t, st := range starts {
		en := rb - a
		if t+1 < len(starts) {
			en = starts[t+1]
		}
		e := byteAdvance(text, b, en-st)
		out = append(out, token.Token{Text: text[b:e], Start: b, End: e})
		b = e
	}
	return out, b
}

// byteAdvance returns the byte offset in text after decoding k runes starting
// at byte offset b (each invalid byte counts as one rune, matching []rune).
func byteAdvance(text string, b, k int) int {
	for ; k > 0 && b < len(text); k-- {
		_, sz := utf8.DecodeRuneInString(text[b:])
		b += sz
	}
	return b
}

// Tokens segments Chinese text with the shared Default dictionary and reports
// byte offsets (see Segmenter.Tokens). Panics only if the embedded dictionary
// fails to load.
func Tokens(text string) []token.Token {
	s, err := load()
	if err != nil {
		panic("cjk: " + err.Error())
	}
	return s.Tokens(text)
}

// TokensDP segments Chinese text with the shared dictionary using DAG + DP and
// reports byte offsets (see Segmenter.TokensDP).
func TokensDP(text string) []token.Token {
	s, err := load()
	if err != nil {
		panic("cjk: " + err.Error())
	}
	return s.TokensDP(text)
}
