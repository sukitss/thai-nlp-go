package normalize

import (
	"golang.org/x/text/unicode/norm"
	"golang.org/x/text/width"
)

// Multilingual normalization for non-Thai scripts. Thai has its own rules in
// Normalize/Reorder/Canonical — these functions deliberately do not apply them.
// The goal is the same as the Thai "IT == it" fix but for CJK/Latin variants: a
// term indexed in one width/composition form must match a query typed in
// another, otherwise recall is lost silently (no error, just misses).
//
// Covered here (from Unicode data, no lookup tables):
//   - width folding: full-width ASCII/digits/punctuation → half-width (Ａ→A,
//     ３→3, ideographic space → space) and half-width kana → full-width (ｶ→カ)
//   - NFC canonical composition: Korean jamo and combining marks compose, so
//     NFD and NFC inputs compare equal
//
// NOT covered (need a mapping table / change meaning — see FoldForIndex doc):
//   - simplified ↔ traditional Chinese (繁體/简体): needs an OpenCC-style table
//   - hiragana ↔ katakana (が/ガ): kept distinct, they carry different meaning

// FoldWidth collapses full-width/half-width variants to one canonical width.
func FoldWidth(s string) string { return width.Fold.String(s) }

// NFC returns the Unicode NFC (canonical composition) form.
func NFC(s string) string { return norm.NFC.String(s) }

// FoldForIndex is the index/query normalization for non-Thai scripts: width
// fold then NFC. Apply it to CJK/Latin/Korean text before tokenizing so an
// index and a query unify regardless of width or composition. It does not fold
// simplified↔traditional Chinese or hiragana↔katakana. For Thai text use
// Normalize (this runs NFC, which reorders Thai combining marks — fine for
// matching, but Normalize is the intended Thai entry point).
func FoldForIndex(s string) string {
	return norm.NFC.String(width.Fold.String(s))
}
