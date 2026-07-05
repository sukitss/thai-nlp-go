package tokenize

import "github.com/sukitss/thai-nlp-go/dict"

// OverlayDict and NewOverlayDict moved to the dict package (they are generic —
// a base Prefixer plus an overlay Trie, nothing Thai-specific — so cjk/jp can
// reuse them for per-tenant glossaries too). These aliases keep the original
// tokenize API working.

// OverlayDict is an alias of dict.OverlayDict.
type OverlayDict = dict.OverlayDict

// NewOverlayDict wraps base with an overlay trie of extra words. Alias of
// dict.NewOverlayDict.
func NewOverlayDict(base dict.Prefixer, overlay *dict.Trie) *dict.OverlayDict {
	return dict.NewOverlayDict(base, overlay)
}
