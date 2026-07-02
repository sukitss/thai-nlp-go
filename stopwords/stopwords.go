// Package stopwords will provide Thai (and mixed Thai/English) stop-word
// filtering for lexical stages — e.g. BM25/keyword weighting and keyword
// extraction, where very common words are noise rather than signal.
//
// STATUS: planned. Intended: an embedded default set (Thai from PyThaiNLP +
// common English), overridable via config, and acronym-aware so latin acronyms
// like "IT" are not treated the same as the word "it". Load the set once and
// share it (see the shared-resource pattern in package dict).
package stopwords

// Set is a stop-word set.
//
// TODO: implement (embed list + IsStopword + filter helpers). This stub treats
// nothing as a stop word.
type Set struct{}

// Default returns the default Thai+English stop-word set.
func Default() *Set { return &Set{} }

// IsStopword reports whether w is a stop word.
func (s *Set) IsStopword(w string) bool { return false }
