// Package token defines the shared token-with-offsets value returned by the
// tokenizer packages (tokenize, en, cjk, jp, kr), so callers know WHERE each
// token sits in the input — e.g. for RAG highlighting or entity-mention
// offsets. It is a leaf package with zero dependencies.
//
// Offsets are BYTE offsets into the exact string passed to the call that
// produced the token. No normalization happens inside the tokenizers; if you
// normalize first, the offsets point into the normalized string you passed,
// not the original.
package token

// Token is one token plus its half-open byte range [Start, End) in the input
// string of the call that produced it. Producers guarantee
// input[Start:End] == Text unless their doc comment states an exception
// (e.g. invalid UTF-8 substitution or a case-folding transform).
type Token struct {
	Text  string
	Start int // byte offset of the token's first byte in the input
	End   int // byte offset just past the token's last byte (End exclusive)
}
