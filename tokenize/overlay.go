package tokenize

import "github.com/sukitss/thai-nlp-go/dict"

// OverlayDict layers a small per-session word list on top of a shared read-only
// base dictionary. Lookups return the union of both, so session-specific words
// (custom names, product codes, a domain glossary) are recognized without
// rebuilding or copying the large base. Create one via Segmenter.Session.
//
// An OverlayDict is single-writer/single-reader: it keeps scratch buffers to
// avoid per-lookup allocation, so it is NOT safe for concurrent use. Give each
// goroutine its own Session (they all share the base safely).
type OverlayDict struct {
	base    dict.Prefixer
	overlay *dict.Trie
	ovbuf   []int // scratch: overlay lengths
	merged  []int // scratch: merged result
}

// NewOverlayDict wraps base with an overlay trie of extra words.
func NewOverlayDict(base dict.Prefixer, overlay *dict.Trie) *OverlayDict {
	return &OverlayDict{base: base, overlay: overlay}
}

// PrefixLens returns the ascending, de-duplicated union of the base and overlay
// prefix lengths. When the overlay contributes nothing it returns the base
// result directly (the common, hot case).
func (o *OverlayDict) PrefixLens(text []rune, start int, out []int) []int {
	base := o.base.PrefixLens(text, start, out)
	o.ovbuf = o.overlay.PrefixLens(text, start, o.ovbuf[:0])
	if len(o.ovbuf) == 0 {
		return base
	}
	o.merged = mergeAscUnique(o.merged[:0], base, o.ovbuf)
	return o.merged
}

// mergeAscUnique merges two ascending, individually-unique int slices into dst,
// dropping duplicates. dst is expected to be a reset (len 0) scratch slice.
func mergeAscUnique(dst, a, b []int) []int {
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			dst = append(dst, a[i])
			i++
		case a[i] > b[j]:
			dst = append(dst, b[j])
			j++
		default:
			dst = append(dst, a[i])
			i++
			j++
		}
	}
	dst = append(dst, a[i:]...)
	dst = append(dst, b[j:]...)
	return dst
}
