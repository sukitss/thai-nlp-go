// Package stopwords provides Thai (and common English) stop-word sets and
// filtering for lexical stages such as BM25/keyword weighting and keyword
// extraction, where very common words are noise rather than signal.
//
// The Thai set is derived from PyThaiNLP's thai_stopwords() but cleaned for
// robustness: every word is Unicode-normalized (see the normalize package) so
// tone-vowel encodings are canonical, duplicates that differed only by encoding
// (e.g. composed "ำ" vs decomposed "ํ"+"า") are merged, and a stray BOM entry is
// dropped — 1,027 words vs the raw 1,030. Normalize your input the same way for
// consistent matching (tokens from normalized text already are). The English set
// is a compact list of common function words (override it if you need a specific
// one). Sets are read-only after construction and safe for concurrent reads; the
// shared Default/English instances are loaded once.
//
// Matching is exact and case-sensitive, which keeps latin acronyms distinct
// from stop words: "it" is a stop word, "IT" is not. Lower-case your ordinary
// tokens (but not acronyms) before checking if you want case-insensitive
// behavior for words.
package stopwords

import (
	"bufio"
	_ "embed"
	"sort"
	"strings"
	"sync"
)

//go:embed data/stopwords_th.txt
var thaiData string

//go:embed data/stopwords_en.txt
var englishData string

// Set is an immutable-by-convention collection of stop words. Reads are safe for
// concurrent use; build custom sets with New or Union rather than mutating one.
type Set struct {
	m map[string]struct{}
}

// New builds a Set from the given words (blank entries are ignored).
func New(words ...string) *Set {
	s := &Set{m: make(map[string]struct{}, len(words))}
	for _, w := range words {
		if w = strings.TrimSpace(w); w != "" {
			s.m[w] = struct{}{}
		}
	}
	return s
}

// Union returns a new Set containing every word from all the given sets.
func Union(sets ...*Set) *Set {
	out := &Set{m: map[string]struct{}{}}
	for _, s := range sets {
		if s == nil {
			continue
		}
		for w := range s.m {
			out.m[w] = struct{}{}
		}
	}
	return out
}

var (
	thaiOnce, enOnce sync.Once
	thaiSet, enSet   *Set
)

// Default returns the shared Thai stop-word set (PyThaiNLP thai_stopwords).
// Treat it as read-only; use Union or New to derive custom sets.
func Default() *Set {
	thaiOnce.Do(func() { thaiSet = parse(thaiData) })
	return thaiSet
}

// English returns the shared common-English stop-word set.
func English() *Set {
	enOnce.Do(func() { enSet = parse(englishData) })
	return enSet
}

func parse(data string) *Set {
	s := &Set{m: map[string]struct{}{}}
	sc := bufio.NewScanner(strings.NewReader(data))
	for sc.Scan() {
		if w := strings.TrimSpace(sc.Text()); w != "" {
			s.m[w] = struct{}{}
		}
	}
	return s
}

// IsStopword reports whether w is in the set (exact, case-sensitive).
func (s *Set) IsStopword(w string) bool {
	_, ok := s.m[w]
	return ok
}

// Filter returns tokens with stop words removed. It allocates a new slice only
// if something is dropped; otherwise it returns tokens unchanged.
func (s *Set) Filter(tokens []string) []string {
	drop := false
	for _, t := range tokens {
		if s.IsStopword(t) {
			drop = true
			break
		}
	}
	if !drop {
		return tokens
	}
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if !s.IsStopword(t) {
			out = append(out, t)
		}
	}
	return out
}

// Len reports the number of words in the set.
func (s *Set) Len() int { return len(s.m) }

// Builder derives a corpus-specific stop-word set by document frequency: words
// that appear in a large fraction of documents carry little signal (the classic
// Luhn/IR criterion) and make good domain stop words. This is the principled
// "frequency-based" stop-word method — more robust than ranking by raw term
// frequency, which pulls in high-frequency content words. Use it to augment (via
// Union) the general-purpose Default set with terms specific to your corpus.
//
// Not safe for concurrent Add; build fully, then the resulting Set is read-only.
type Builder struct {
	df   map[string]int // document frequency per token
	docs int
	seen map[string]struct{} // tokens seen in the current document
}

// NewBuilder returns an empty document-frequency Builder.
func NewBuilder() *Builder {
	return &Builder{df: map[string]int{}, seen: map[string]struct{}{}}
}

// AddDoc records one document's tokens (duplicates within the doc count once).
func (b *Builder) AddDoc(tokens []string) {
	for k := range b.seen {
		delete(b.seen, k)
	}
	for _, t := range tokens {
		if _, ok := b.seen[t]; ok {
			continue
		}
		b.seen[t] = struct{}{}
		b.df[t]++
	}
	b.docs++
}

// Build returns the Set of tokens whose document frequency is at least
// minDocFraction of all documents (0..1). A typical value is 0.4–0.6. Returns an
// empty set if no documents were added.
func (b *Builder) Build(minDocFraction float64) *Set {
	s := &Set{m: map[string]struct{}{}}
	if b.docs == 0 {
		return s
	}
	threshold := minDocFraction * float64(b.docs)
	for w, c := range b.df {
		if float64(c) >= threshold {
			s.m[w] = struct{}{}
		}
	}
	return s
}

// Docs reports how many documents were added.
func (b *Builder) Docs() int { return b.docs }

// Words returns the set's words, sorted (for inspection/testing).
func (s *Set) Words() []string {
	out := make([]string, 0, len(s.m))
	for w := range s.m {
		out = append(out, w)
	}
	sort.Strings(out)
	return out
}
