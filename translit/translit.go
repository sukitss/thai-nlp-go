// Package translit will match name variants that are transliterated differently
// but refer to the same thing — e.g. Thai spellings of the same foreign name.
// Useful for de-duplicating spelling variants, name search and entity matching.
//
// STATUS: planned. Approaches: a Thai phonetic key (soundex-like) and/or fuzzy
// match with an alias dictionary. Start with heuristics and measure. Mind
// performance/memory if a key index is built.
package translit

// Key returns a phonetic/normalization key such that name variants share a key.
//
// TODO: implement. This stub returns the input unchanged (no folding).
func Key(name string) string { return name }
