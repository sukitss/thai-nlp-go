// Package keyword extracts the FEW salient words or phrases that represent a
// document — for auto-tagging, faceting, metadata payloads, and highlight — as
// opposed to the ALL-terms tokenization used for full-text/BM25 indexing (that
// is what the multi/tokenize packages give you). A search index wants every
// term; a tag wants the three or four that a human would write on a sticky note.
//
// Every extractor implements one interface:
//
//	type Extractor interface{ Extract(text string, k int) []Keyword }
//
// and shares one language front-end: normalize → script-route → tokenize (via
// [multi.Analyzer]) → classify each token as a content word, a stop word, a
// number, or punctuation. Only content words become keywords; stop words,
// numbers and punctuation are dropped (and, for the phrase extractors, act as
// phrase delimiters). The extractors differ only in how they SCORE the content
// words:
//
//	TFIDFTopK — tf(term) × idf(term) from a supplied corpus vocabulary. The
//	            simplest, fastest baseline. Needs a corpus (vocab) to know which
//	            words are globally common (low idf) versus document-specific.
//	RAKE      — Rapid Automatic Keyword Extraction: candidate PHRASES are the
//	            runs of content words between delimiters; each word scores
//	            deg(w)/freq(w) and a phrase scores the sum. Corpus-free; captures
//	            multi-word keyphrases ("ปลา IT", "ศึกชิงเจ้าสำนัก").
//	YAKE      — statistical, corpus-free: casing, position, frequency dispersion
//	            and sentence spread. Lower raw weight = better; Extract returns it
//	            inverted so larger Score = better like the others.
//	TextRank  — a co-occurrence graph over content words, ranked by PageRank
//	            centrality. Corpus-free.
//
// # Acronym-aware casing (the point of the driving use case)
//
// A person or thing tagged with an English acronym stuck to a Thai name —
// "ปลา IT", "กุ้ง POS" — must survive extraction: the acronym is the salient
// discriminator, not noise. The failure mode this package exists to fix is the
// naive pipeline that lower-cases everything before dropping stop words: "IT"
// folds to "it", "it" is an English stop word, and the tag silently loses its
// most distinctive token.
//
// The fix, per the stopwords package's contract (matching is case-sensitive:
// "it" is a stop word, "IT" is not), is a casing fold that is acronym-aware:
//
//	fold("IT")       = "IT"        // all-caps, ≥2 letters → ACRONYM, kept verbatim
//	fold("POS")      = "POS"       // acronym
//	fold("it")       = "it"        // → English stop word → dropped
//	fold("It")       = "it"        // sentence-initial cap folds to the stop word
//	fold("Password") = "password"  // ordinary word, case-folded so casing merges
//	fold("ปลา")      = "ปลา"       // no case → unchanged
//
// So "IT"/"POS" are retained while the pronoun "it" and sentence-initial "It"/
// "The" are dropped. Because the tokenizer already splits at a script boundary,
// both the spaced "ปลา IT" and the stuck "ปลาit" surface "ปลา" plus the Latin
// run; when that run is the upper-case acronym it is kept. A [Keyword]'s Text is
// this folded form (verbatim acronyms, lower-cased ordinary words), so tags are
// deterministic regardless of the source casing.
//
// # Concurrency
//
// An Extractor is immutable after construction and safe for concurrent Extract
// calls (the underlying [multi.Analyzer], stopwords.Set and vocab.Vocab are all
// read-only). Extract itself allocates per call and keeps no state.
package keyword

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/multi"
	"github.com/sukitss/thai-nlp-go/stopwords"
)

// Keyword is one extracted term or phrase with its salience score. Larger Score
// means more salient; every extractor normalizes its scoring to that convention
// so results are comparable within one extractor (not across extractors, whose
// scales differ). Text is the normalized surface: for a single word it is the
// acronym-aware folded form (see the package doc); for a phrase it is the exact
// original substring spanning the phrase, so internal spacing and acronym casing
// are preserved ("ปลา IT", "ศึกชิงเจ้าสำนัก").
type Keyword struct {
	Text  string
	Score float64
}

// Extractor turns a document into its top-k salient keywords, best first. k <= 0
// returns nil; k larger than the number of candidates returns all of them.
// Implementations are safe for concurrent use.
type Extractor interface {
	Extract(text string, k int) []Keyword
}

// Option configures an extractor at construction. Options are shared across all
// extractors (they all use the same language front-end).
type Option func(*config)

type config struct {
	analyzer   *multi.Analyzer
	stop       *stopwords.Set
	minWordLen int  // minimum rune length of a content word (acronyms exempt)
	dropNumber bool // treat pure-number tokens as delimiters, not keywords
}

func defaultConfig() config {
	return config{
		analyzer:   &multi.Analyzer{}, // zero Analyzer: normalize + route + tokenize, no Stop/LowerLatin (we fold ourselves)
		stop:       stopwords.Multilingual(),
		minWordLen: 2,
		dropNumber: true,
	}
}

func (c config) apply(opts []Option) config {
	for _, o := range opts {
		o(&c)
	}
	if c.analyzer == nil {
		c.analyzer = &multi.Analyzer{}
	}
	if c.stop == nil {
		c.stop = stopwords.New() // empty set: keep everything
	}
	if c.minWordLen < 1 {
		c.minWordLen = 1
	}
	return c
}

// WithAnalyzer sets the tokenization front-end. Use it to add a per-tenant
// dictionary Overlay (character names, product names) or CJK routing. The
// extractor drives the Analyzer through Tokens (offsets, no filtering) and does
// its OWN acronym-aware stop-word handling, so leave the Analyzer's Stop and
// LowerLatin at their zero values — setting LowerLatin would fold "IT" to the
// stop word "it" and defeat acronym survival. Default: the zero Analyzer.
func WithAnalyzer(a *multi.Analyzer) Option { return func(c *config) { c.analyzer = a } }

// WithStopwords sets the stop-word set that delimits phrases and is dropped from
// keywords. Matching is acronym-aware: a token is folded (see the package doc)
// and then looked up, so "it" is dropped but "IT" is not. Default:
// stopwords.Multilingual().
func WithStopwords(s *stopwords.Set) Option { return func(c *config) { c.stop = s } }

// WithMinWordLen drops content words shorter than n runes (acronyms are always
// kept, so "IT"/"POS" survive n=2). Default 2, which removes stray single Thai
// characters that are not stop words.
func WithMinWordLen(n int) Option { return func(c *config) { c.minWordLen = n } }

// WithNumbers controls whether pure-number tokens can become keywords. Default
// false (numbers are dropped and act as phrase delimiters), which keeps phone
// numbers and IDs out of tags. Pass true to keep numeric tokens.
func WithNumbers(keep bool) Option { return func(c *config) { c.dropNumber = !keep } }

// kind classifies a token for extraction.
type kind uint8

const (
	kindContent kind = iota // a keyword-eligible content word
	kindDelim               // stop word, punctuation, number, or too-short: dropped, breaks phrases
)

// term is one analyzed token: its scoring key (folded), its original surface,
// its byte range in the normalized text, its classification, and what the gap
// before it contained. The gap flags matter because the tokenizers DROP
// punctuation that hugs a Latin/number run ("IT)" tokenizes to just "IT", the
// ")" surviving only as a gap) — so punctuation is not always a delimiter token
// and phrase/sentence breaks must also read the inter-token gaps.
type term struct {
	key     string // acronym-aware folded form; the scoring/lookup identity
	surface string // original substring (norm[start:end]); preserves case/spacing
	start   int
	end     int
	kind    kind
	hard    bool // the gap before this token held punctuation ⇒ a phrase break
	sent    bool // the gap before this token held a sentence terminator
}

// analyze runs the shared front-end and classifies every token. It returns the
// normalized text (for phrase reconstruction) and the token stream in order,
// including delimiters (the phrase extractors need their positions).
func (c *config) analyze(text string) (string, []term) {
	norm, toks := c.analyzer.Tokens(text)
	if len(toks) == 0 {
		return norm, nil
	}
	out := make([]term, len(toks))
	prevEnd := 0
	for i, t := range toks {
		out[i] = c.classify(t.Text, t.Start, t.End)
		if i > 0 {
			gap := norm[prevEnd:t.Start]
			out[i].hard = hasPunct(gap)
			out[i].sent = isSentenceBreak(gap)
		}
		prevEnd = t.End
	}
	return norm, out
}

// hasPunct reports whether gap holds any non-whitespace rune — dropped
// punctuation that should break a candidate phrase.
func hasPunct(gap string) bool {
	for _, r := range gap {
		if !unicode.IsSpace(r) {
			return true
		}
	}
	return false
}

// classify decides a single token's kind and computes its folded key.
func (c *config) classify(surface string, start, end int) term {
	t := term{surface: surface, start: start, end: end, kind: kindDelim}
	hasLetter, hasDigit := false, false
	for _, r := range surface {
		if unicode.IsLetter(r) {
			hasLetter = true
		} else if unicode.IsDigit(r) {
			hasDigit = true
		}
	}
	switch {
	case !hasLetter && !hasDigit:
		return t // punctuation / symbol → delimiter
	case !hasLetter:
		if c.dropNumber {
			return t // pure number → delimiter
		}
		t.key = surface
		t.kind = kindContent
		return t
	}
	key, acronym := fold(surface)
	if c.stop.IsStopword(key) {
		return t // stop word (acronym-aware: "it" yes, "IT" no)
	}
	if !acronym && utf8.RuneCountInString(key) < c.minWordLen {
		return t // too short to be a tag
	}
	t.key = key
	t.kind = kindContent
	return t
}

// fold returns a token's acronym-aware scoring key and whether it is an acronym.
// A Latin token that is ALL-UPPERCASE with at least two letters is an acronym,
// kept verbatim ("IT", "POS", "USA"); every other token is lower-cased, which
// folds sentence-initial "It"/"The" onto their stop words and merges "Password"
// with "password" while leaving script-less text (Thai, digits) unchanged.
func fold(tok string) (key string, acronym bool) {
	var upper, lower, letters int
	for _, r := range tok {
		if !unicode.IsLetter(r) {
			continue
		}
		letters++
		switch {
		case unicode.IsUpper(r):
			upper++
		case unicode.IsLower(r):
			lower++
		}
	}
	if upper >= 2 && lower == 0 {
		return tok, true // acronym: verbatim
	}
	return strings.ToLower(tok), false
}

// Keys returns the acronym-aware content-word keys a document reduces to under
// the given options: the tokens the extractors actually score over, with stop
// words, punctuation and (by default) numbers dropped and ordinary words folded
// to lower case while acronyms are kept verbatim. Order is document order,
// duplicates included.
//
// It exposes the shared front-end for callers that want the canonical token set
// directly — to build a matching corpus vocabulary, to debug what an extractor
// sees, or to expand a predicted keyphrase into its component words for
// evaluation. It is the same analysis NewTFIDFTopK/NewRAKE/… apply internally.
func Keys(text string, opts ...Option) []string {
	cfg := defaultConfig().apply(opts)
	_, terms := cfg.analyze(text)
	return contentKeys(terms)
}

// contentKeys returns just the folded keys of the content words, in order —
// the common projection the frequency-based extractors start from.
func contentKeys(terms []term) []string {
	out := make([]string, 0, len(terms))
	for _, t := range terms {
		if t.kind == kindContent {
			out = append(out, t.key)
		}
	}
	return out
}

// topK sorts scored keywords and returns the best k, best first. The order is a
// strict, deterministic total order: higher score first, ties broken by Text
// ascending, so results never depend on map iteration order.
func topK(kws []Keyword, k int) []Keyword {
	sort.Slice(kws, func(i, j int) bool {
		if kws[i].Score != kws[j].Score {
			return kws[i].Score > kws[j].Score
		}
		return kws[i].Text < kws[j].Text
	})
	if k < len(kws) {
		kws = kws[:k]
	}
	return kws
}
